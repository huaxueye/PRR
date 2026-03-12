import torch
import os
import argparse
from prr_inference import EnhancedTemperatureHead

def main():
    parser = argparse.ArgumentParser(description="Check Head Checkpoint Dimensions")
    parser.add_argument("head_path", type=str, help="Path to the head checkpoint")
    args = parser.parse_args()

    head_path = args.head_path
    if not os.path.exists(head_path):
        print(f"Error: File not found at {head_path}")
        return

    print(f"Loading checkpoint from {head_path}...")
    try:
        state_dict = torch.load(head_path, map_location="cpu")
        
        # Check if keys have 'module.' prefix
        if all(k.startswith("module.") for k in state_dict.keys()):
            print("Detected 'module.' prefix in keys (DataParallel), removing...")
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # Print shape of first layer weight to infer input dim
        if 'project_in.weight' in state_dict:
            weight_shape = state_dict['project_in.weight'].shape
            print(f"project_in.weight shape: {weight_shape}")
            input_dim = weight_shape[1]
            print(f"Inferred input_dim: {input_dim}")
        else:
            print("Could not find project_in.weight in state_dict.")
            print("Keys:", state_dict.keys())
            return

        # Try to load into model
        print("Attempting to load into EnhancedTemperatureHead...")
        model = EnhancedTemperatureHead(input_dim=input_dim, hidden_dim=1024)
        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded state_dict!")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    main()
