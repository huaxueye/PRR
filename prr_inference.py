import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from transformers import AutoTokenizer
from model.modeling_llada_with_attn import LLaDAModelLM

# ================= 1. Head Model Definition =================
class EnhancedTemperatureHead(nn.Module):
    def __init__(self, input_dim=4107, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.pre_norm = nn.LayerNorm(input_dim)
        self.project_in = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.project_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.pre_norm(x)
        x = self.project_in(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x + self.res_block(x)
        return self.project_out(x).squeeze(-1)

# ================= 2. Helper Functions =================

def compute_features_v2(logits, global_mask, block_mask, 
                       global_step, total_steps, 
                       block_start, block_len, total_len,
                       step_in_block, steps_per_block,
                       last_flip_step):
    """
    Computes 11 features:
    1. Global Commit Rate (1)
    2. Global Steps (1)
    3. Block Start Pos (1)
    4. Block Length (1)
    5. Block Commit Rate (1)
    6. Block Token Pos (1)
    7. Block Step (1)
    8. Block Last Flip (1)
    9. Top1 (1)
    10. Margin (1)
    11. Entropy (1)
    """
    B, L, V = logits.shape
    device = logits.device
    dtype = torch.bfloat16
    
    # Uncertainty Features
    _logits = logits.float()
    _probs = F.softmax(_logits, dim=-1)
    _top2 = _probs.topk(2, dim=-1)[0]
    _top1_prob = _top2[:, :, 0]
    _margin = _top1_prob - _top2[:, :, 1]
    _entropy = -torch.sum(_probs * torch.log(_probs + 1e-6), dim=-1)
    _entropy = _entropy / (torch.log(torch.tensor(V, device=device, dtype=_entropy.dtype)) + 1e-6)
    
    # 1. Global Commit Rate
    global_mask_count = global_mask.sum(dim=1, keepdim=True).float() # (B, 1)
    global_commit_rate = (total_len - global_mask_count) / total_len
    global_commit_rate = global_commit_rate.expand(B, L)
    
    # 2. Global Steps
    global_step_val = global_step / total_steps
    global_step_feat = torch.full((B, L), global_step_val, device=device, dtype=dtype)
    
    # 3. Block Start Pos
    block_start_val = block_start / total_len
    block_start_feat = torch.full((B, L), block_start_val, device=device, dtype=dtype)
    
    # 4. Block Length
    block_len_val = block_len / total_len
    block_len_feat = torch.full((B, L), block_len_val, device=device, dtype=dtype)
    
    # 5. Block Commit Rate
    block_mask_count = block_mask.sum(dim=1, keepdim=True).float()
    block_commit_rate = (block_len - block_mask_count) / block_len
    block_commit_rate = block_commit_rate.expand(B, L)
    
    # 6. Block Token Pos
    token_pos = torch.arange(L, device=device, dtype=dtype) / L
    token_pos_feat = token_pos.unsqueeze(0).expand(B, L)
    
    # 7. Block Step
    block_step_val = step_in_block / steps_per_block
    block_step_feat = torch.full((B, L), block_step_val, device=device, dtype=dtype)
    
    # 8. Block Last Flip
    last_flip_feat = (step_in_block - last_flip_step) / steps_per_block
    
    # Stack all 11 features
    features = torch.stack([
        global_commit_rate.to(dtype), 
        global_step_feat,   
        block_start_feat,   
        block_len_feat,     
        block_commit_rate.to(dtype),  
        token_pos_feat,     
        block_step_feat,    
        last_flip_feat.to(dtype),     
        _top1_prob.to(dtype), 
        _margin.to(dtype),    
        _entropy.to(dtype)    
    ], dim=-1)
    
    return features.to(dtype)

def get_num_transfer_tokens(block_mask_index, steps):
    device = block_mask_index.device
    dtype = torch.long
    total = block_mask_index.sum(dim=1)
    base  = torch.div(total, steps, rounding_mode='floor')
    rem   = total - base * steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)
    cols = torch.arange(steps, device=device).unsqueeze(0)
    add_mask = cols < rem.unsqueeze(1)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)
    return num_transfer_tokens

def get_head_score(logits, hidden_states, head_model, extra_features):
    """Compute stability score from head model using pre-computed features"""
    if head_model is None:
        return None
        
    target_device = head_model.project_in.weight.device
    
    head_input = torch.cat([
        hidden_states.to(target_device).to(dtype=torch.bfloat16), 
        extra_features.to(target_device).to(dtype=torch.bfloat16)
    ], dim=-1)
    
    logits_score = head_model(head_input)
    stability_score = torch.sigmoid(logits_score)
    return stability_score

# ================= 3. Core Logic: Temperature Scaling Strategy =================

def get_transfer_index_temperature(
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: torch.Tensor,
    head_model: nn.Module = None,
    hidden_states: torch.Tensor = None,
    extra_features: torch.Tensor = None,
    temp_alpha: float = 0.5,      # Controls how much Head affects confidence
    temp_threshold: float = 0.95, # Threshold for sharpened confidence
    use_head: bool = True,
    strategy: str = 'temperature' # 'temperature', 'weighted', 'gating'
):
    x0 = torch.argmax(logits, dim=-1)
    
    # 1. Calculate Base Confidence (Approximated)
    # Use Top-50 for LogSumExp Approximation (Very Accurate for peaked distributions)
    logits_f = logits.float()
    top_logits, _ = logits_f.topk(50, dim=-1)
    lse = torch.logsumexp(top_logits, dim=-1, keepdim=True)
    
    x0_logits = torch.gather(logits_f, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    x0_log_p = x0_logits - lse.squeeze(-1)
    x0_p = torch.exp(x0_log_p) # Base Confidence
    
    # 2. Apply Strategy
    final_score = x0_p
    should_accelerate = torch.zeros_like(mask_index, dtype=torch.bool)

    if use_head and head_model is not None:
        head_score = get_head_score(logits, hidden_states, head_model, extra_features)
        # Move back to correct device
        head_score = head_score.to(x0_p.device)
        
        if strategy == 'temperature':
            # Temperature Scaling: Conf_new = Conf_old ^ (1 - alpha * Head_Score)
            # Only boosts confidence, never penalizes (unless alpha < 0)
            power = 1.0 - temp_alpha * head_score
            final_score = torch.pow(x0_p, power)
        
        elif strategy == 'weighted':
            # Improved Weighted Strategy (Accelerate Only)
            # Formula: Score = p + alpha * max(0, s - 0.5) * p * (1 - p)
            # This only boosts confidence if Head score > 0.5. No penalty for uncertainty.
            # This ensures we only accelerate generation (early exit) and never decelerate compared to baseline.
            correction = temp_alpha * torch.clamp(head_score - 0.5, min=0.0) * x0_p * (1.0 - x0_p)
            final_score = x0_p + correction
            final_score = final_score.clamp(0.0, 1.0)
            
        elif strategy == 'gating':
            # Gating/Product: Score = Model * Head
            # Strongest penalty. Alpha is ignored here (or acts as a global scaler?)
            # Let's use alpha to mix gating: Score = Model * (1 - alpha * (1 - Head)) ? 
            # No, simple product is best for 'gating'. Let's stick to simple product.
            final_score = x0_p * head_score

        # Acceleration Criteria
        should_accelerate = (final_score > temp_threshold)

    # 3. Selection Logic
    neg_inf = torch.tensor(float('-inf'), device=x0.device)
    masked_score = torch.where(mask_index, final_score, neg_inf)
    
    # Sort
    _, idx = torch.sort(masked_score, dim=1, descending=True)
    B, L = masked_score.shape
    cols = torch.arange(L, device=masked_score.device).unsqueeze(0).expand(B, L)
    
    # Standard Schedule Selection
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)
    select_sorted = cols < k_expanded
    
    transfer_int = torch.zeros(B, L, device=masked_score.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    standard_transfer_index = transfer_int.bool() & mask_index
    
    # Union with Acceleration (Dynamic Budget)
    final_transfer_index = standard_transfer_index | (should_accelerate & mask_index)
    
    return x0, final_transfer_index

# ================= 4. Generation Loop =================

@torch.no_grad()
def generate_with_temperature(
    model, prompt, head_model=None, 
    steps=128, gen_length=256, block_length=32, 
    mask_id=126336,
    temp_alpha=0.5,
    temp_threshold=0.95,
    strategy='temperature',
    early_exit_ratio=0.0
):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    total_len = Lp + gen_length
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    x = torch.full((B, total_len), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt.clone()

    nfe = 0
    use_head = (head_model is not None)
    
    if use_head:
        print(f"[DEBUG] Strategy: {strategy}, Alpha={temp_alpha}, Threshold={temp_threshold}")

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length
        blk_len = e - s

        # Step 0
        block_mask_index = (x[:, s:e] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        # Track last flip step (relative to block start, 0 to steps_per_block-1)
        last_flip_step = torch.zeros((B, blk_len), device=model.device, dtype=torch.bfloat16)

        # 1) Prefill
        out_full = model(x, use_cache=True, output_hidden_states=True)
        past_key_values = out_full.past_key_values
        nfe += 1

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True
        
        global_mask_index = (x == mask_id)
        
        current_logits = out_full.logits[:, s:e, :]
        last_hidden = out_full.hidden_states[-1][:, s:e, :]

        # Compute Features Step 0
        global_step = nb * steps_per_block
        current_features = compute_features_v2(
            current_logits, global_mask_index, block_mask_index,
            global_step, steps,
            s, blk_len, total_len,
            0, steps_per_block,
            last_flip_step
        )

        x0, transfer_index = get_transfer_index_temperature(
            current_logits, block_mask_index, x[:, s:e], num_transfer_tokens[:, 0],
            head_model=head_model, hidden_states=last_hidden,
            extra_features=current_features,
            temp_alpha=temp_alpha, temp_threshold=temp_threshold, use_head=use_head,
            strategy=strategy
        )

        old_blk = x[:, s:e].clone()
        x[:, s:e] = torch.where(transfer_index, x0, old_blk)
        
        # Update Last Flip for next step
        changed = (x[:, s:e] != old_blk)
        last_flip_step = torch.where(changed, torch.tensor(0.0, device=model.device, dtype=torch.bfloat16), last_flip_step)

        # 2) Loop
        for i in range(1, steps_per_block):
            masks_left = (x[:, s:e] == mask_id).sum().item()
            total_block_tokens = B * block_length
            mask_ratio = masks_left / total_block_tokens
            
            if masks_left == 0:
                # print(f"[Block {nb}] Finished naturally at Step {i}")
                break
            
            # Early Exit
            if use_head and mask_ratio <= early_exit_ratio:
                # print(f"[Block {nb}] >>> Aggressive Exit at Step {i} (Masks left: {masks_left}/{total_block_tokens})")
                
                out_final = model(x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position)
                nfe += 1
                
                final_x0 = torch.argmax(out_final.logits, dim=-1)
                remaining_mask = (x[:, s:e] == mask_id)
                x[:, s:e] = torch.where(remaining_mask, final_x0, x[:, s:e])
                break

            # Standard Forward
            out_blk = model(
                x[:, s:e], past_key_values=past_key_values, 
                use_cache=True, replace_position=replace_position,
                output_hidden_states=True
            )
            nfe += 1
            
            logits_blk = out_blk.logits
            hidden_blk = out_blk.hidden_states[-1] 

            mask_blk = (x[:, s:e] == mask_id)
            
            # Compute Features
            global_step = nb * steps_per_block + i
            global_mask = (x == mask_id)
            
            current_features = compute_features_v2(
                logits_blk, global_mask, mask_blk,
                global_step, steps,
                s, blk_len, total_len,
                i, steps_per_block,
                last_flip_step
            )

            x0_blk, transfer_idx_blk = get_transfer_index_temperature(
                logits_blk, mask_blk, x[:, s:e], num_transfer_tokens[:, i],
                head_model=head_model, hidden_states=hidden_blk,
                extra_features=current_features,
                temp_alpha=temp_alpha, temp_threshold=temp_threshold, use_head=use_head, strategy=strategy
            )

            blk_old_i = x[:, s:e].clone()
            x[:, s:e] = torch.where(transfer_idx_blk, x0_blk, blk_old_i)
            
            # Update Last Flip
            changed = (x[:, s:e] != blk_old_i)
            last_flip_step = torch.where(changed, torch.tensor(float(i), device=model.device, dtype=torch.bfloat16), last_flip_step)

    return x, nfe

# ================= 5. Run Script (If run directly) =================

if __name__ == '__main__':
    # This block is for testing purposes.
    # Please update the paths below to point to your local model and head checkpoint.
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = "GSAI-ML/LLaDA-1.5" # Update this
    HEAD_PATH = "head_checkpoint.pt" # Update this
    
    if not os.path.exists(HEAD_PATH):
        print(f"Warning: Head checkpoint not found at {HEAD_PATH}. Please provide a valid path.")
    
    print(f"Loading Model: {MODEL_PATH}")
    model = LLaDAModelLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    print(f"Loading Head: {HEAD_PATH}")
    head_model = EnhancedTemperatureHead(input_dim=4107, hidden_dim=1024).to(DEVICE)
    state_dict = torch.load(HEAD_PATH, map_location=DEVICE)
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    head_model.load_state_dict(state_dict)
    head_model.to(torch.bfloat16).eval()

    prompt = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    m = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt").to(DEVICE)

    print("\n--- Temperature Strategy (Alpha=0.5, Th=0.95) ---")
    start = time.time()
    out, nfe = generate_with_temperature(
        model, prompt_ids, head_model=head_model, 
        steps=256, gen_length=256, block_length=32,
        temp_alpha=1.0, temp_threshold=0.9
    )
    end = time.time()
    
    text = tokenizer.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)
    print("\n[Result]:")
    print(text)
    print(f"Time: {end-start:.2f}s | NFE: {nfe}")
