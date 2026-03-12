'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoConfig
from prr_inference import generate_with_temperature, EnhancedTemperatureHead
from model.modeling_llada_with_attn import LLaDAModelLM
import json
import time
import sys
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_head")
class LLaDAHeadEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        head_path=None,
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        temp_alpha=0.5,
        temp_threshold=0.95,
        strategy='temperature',
        device="cuda",
        remasking='low_confidence',
        use_head=True,
        threshold=None, # Base threshold for schedule if used
        factor=None,
        show_speed=True,
        save_dir=None,
        **kwargs,
    ):
        super().__init__()
        
        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        config = AutoConfig.from_pretrained(model_path)
        config.flash_attention = True
        self.model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            self.model = self.model.to(self.device)
            self._rank = 0
            self._world_size = 1

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # --- Config ---
        self.batch_size = int(batch_size)
        self.max_length = max_length
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.temp_alpha = temp_alpha
        self.temp_threshold = temp_threshold
        self.strategy = strategy
        self.early_exit_ratio = kwargs.get('early_exit_ratio', 0.0)
        self.cfg = kwargs.get('cfg', 0.0)
        
        # Handle boolean parsing from string (lm_eval passes strings)
        if isinstance(use_head, str):
            self.use_head = (use_head.lower() == 'true')
        else:
            self.use_head = bool(use_head)
            
        self.remasking = remasking
        self.threshold = threshold
        self.factor = factor
        self.show_speed = show_speed
        self.save_dir = save_dir
        
        # Determine instruct mode
        force_instruct = kwargs.get('force_instruct', 'False')
        if isinstance(force_instruct, str):
            force_instruct = (force_instruct.lower() == 'True')
            
        if force_instruct:
            self.is_instruct = True
        else:
            self.is_instruct = True if 'instruct' in model_path.lower() else False
            
        print(f"[*] Model Path: {model_path}")
        print(f"[*] Use Head: {self.use_head}")
        print(f"[*] Is Instruct: {self.is_instruct}")
        
        self.mc_num = mc_num
        assert mc_num % self.batch_size == 0
        self.is_check_greedy = is_check_greedy
        
        # --- Load Head ---
        self.head_path = head_path
        self.head = None
        if self.head_path and self.use_head:
            print(f"[*] Loading Multimodal Head from: {self.head_path}")
            # Ensure dims match phase3_inference_multimodal definition
            # Load head to the same device as the model (or first device)
            self.head = EnhancedTemperatureHead(input_dim=4107, hidden_dim=1024).to(self.device)
            try:
                state_dict = torch.load(head_path, map_location=self.device)
                if all(k.startswith("module.") for k in state_dict.keys()):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
                self.head.load_state_dict(state_dict, strict=True)
                self.head.eval()
                self.head.to(torch.bfloat16)
                print(f"[*] Head weights loaded successfully.")
            except Exception as e:
                print(f"[!] Error loading head weights: {e}")
                raise e

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    @torch.no_grad()
    def generate_until(self, requests):
        output = []
        num_tokens = 0
        num_nfe = 0
        processed_count = 0
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f'rank_{rank}.jsonl')
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, 'r', encoding='utf-8') as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")
        
        batched_requests = [[]]
        
        print(f"DEBUG: Received {len(requests)} requests. Processed count: {processed_count}")
        if len(requests) <= processed_count and processed_count > 0:
            print("WARNING: Request count is less than or equal to processed count. Resume logic may skip all tasks.")
            print("If you are testing with --limit, please clear the previous results directory or use a limit > processed_count.")
            
        for i, req in enumerate(tqdm(requests, desc="Batching...")):
            if i < processed_count:
                continue
            batched_requests[-1].append(req)
            if len(batched_requests[-1]) == self.batch_size:
                batched_requests.append([])
        
        if len(batched_requests[-1]) == 0:
            batched_requests.pop()

        start_time = time.time()

        for batch in tqdm(batched_requests, desc="Generating..."):
            batched_input_ids = []
            max_len = 0
            pad_len = []
            
            # 1. Prepare Inputs
            for req in batch:
                question = req.args[0]
                
                # Use chat template ONLY if instructed
                if self.is_instruct:
                    m = [{"role": "user", "content": question}]
                    try:
                        user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                    except:
                         user_input = question
                else:
                    user_input = question
                    
                input_ids = self.tokenizer(user_input)['input_ids']

                batched_input_ids.append(input_ids)
                max_len = max(max_len, len(input_ids))
                pad_len.append(max_len - len(input_ids))
            
            # Padding
            batched_input_ids = [torch.cat([torch.full((1, max_len - len(input_ids)), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device), torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)], dim=1) for input_ids in batched_input_ids]
            batched_input_ids = torch.cat(batched_input_ids, dim=0).to(self.device)
            
            if self.batch_size == 1:
                attention_mask = None
            else:
                attention_mask = torch.zeros((batched_input_ids.shape[0], 1, max_len+self.gen_length, max_len+self.gen_length), device=self.device, dtype=torch.bool)
                for i in range(len(pad_len)):
                    attention_mask[i, :, pad_len[i]:, pad_len[i]:] = True

            input_ids = batched_input_ids
            stop_tokens = req.args[1]['until']


            generated_answer, nfe = generate_with_temperature(
                self.model, input_ids,
                head_model=self.head if (self.use_head and self.head is not None) else None,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                mask_id=self.mask_id,
                temp_alpha=float(self.temp_alpha),
                temp_threshold=float(self.temp_threshold),
                early_exit_ratio=float(self.early_exit_ratio),
                strategy=self.strategy
            )

            # 3. Post-process
            if self.is_instruct and 'task_id' in req.doc and str(req.doc['task_id']).lower().startswith('humaneval'):
                generated_answer_ids = generated_answer[:, input_ids.shape[1]:]
                if self.show_speed:
                    num_tokens += (generated_answer_ids != 126081).sum()
                    num_nfe += nfe * generated_answer_ids.shape[0]
                batched_generated_answer = [self.tokenizer.decode(generated_answer_ids[i], skip_special_tokens=True) for i in range(len(generated_answer_ids))]
            else:
                batched_generated_answer = []
                for i in range(len(generated_answer)):
                    generated_answer_i = self.tokenizer.decode(generated_answer[i][input_ids.shape[1]:], skip_special_tokens=False)
                    for stop_seq in stop_tokens:
                        if stop_seq in generated_answer_i:
                            generated_answer_i = generated_answer_i.split(stop_seq)[0]
                    generated_answer_ids = torch.tensor(self.tokenizer(generated_answer_i)["input_ids"])
                    if self.show_speed:
                        num_tokens += (generated_answer_ids != 126081).sum()
                        num_nfe += nfe
                    generated_answer_i = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
                    batched_generated_answer.append(generated_answer_i)

            output.extend(batched_generated_answer)

            if self.save_dir is not None:
                 with open(save_path, 'a', encoding='utf-8') as f:
                    for generated_answer in batched_generated_answer:
                        f.write(json.dumps(generated_answer, ensure_ascii=False) + '\n')
            
            for i in range(len(batched_generated_answer)):
                print('=' * 20)
                print('question: ', question)
                print('answer: ', batched_generated_answer[i])
                print('nfe: ', nfe)
                print('avg nfe: ', num_nfe / len(output))
                print('=' * 20, end='\n\n')

        end_time = time.time()
        if self.show_speed:
            print(f"Total tokens: {num_tokens}")
            print(f"Total time: {end_time - start_time:.2f}s")
            print(f"Tokens/sec: {num_tokens / (end_time - start_time):.2f}")
            print(f"Total NFE: {num_nfe}")
            
        return output

def _ensure_ifeval_nltk_data():
    argv = " ".join(sys.argv).lower()
    if "--tasks" not in argv or "ifeval" not in argv:
        return
    try:
        import nltk
    except Exception:
        return
    download_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
    try:
        os.makedirs(download_dir, exist_ok=True)
        os.environ.setdefault("NLTK_DATA", download_dir)
        if download_dir not in nltk.data.path:
            nltk.data.path.insert(0, download_dir)
    except Exception:
        download_dir = None
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
        return
    except LookupError:
        pass
    if download_dir is not None:
        for pkg in ("punkt_tab", "punkt"):
            try:
                nltk.download(pkg, download_dir=download_dir, quiet=True)
            except Exception:
                pass
    else:
        for pkg in ("punkt_tab", "punkt"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
        return
    except LookupError:
        pass
    try:
        from nltk.tokenize import TreebankWordTokenizer
        tokenizer = TreebankWordTokenizer()

        def _fallback_word_tokenize(text):
            return tokenizer.tokenize(text)

        nltk.word_tokenize = _fallback_word_tokenize
        try:
            import nltk.tokenize
            nltk.tokenize.word_tokenize = _fallback_word_tokenize
        except Exception:
            pass
        print("[NLTK] punkt_tab unavailable; using TreebankWordTokenizer fallback for IFEval")
    except Exception:
        return

if __name__ == "__main__":
    _ensure_ifeval_nltk_data()
    cli_evaluate()
