from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--peft_model", type=str)  # Directory containing checkpoints
    parser.add_argument("--hub_id", type=str)

    return parser.parse_args()

def get_most_recent_checkpoint_dir(directory):
    # List all subdirectories starting with 'checkpoint'
    dirs = [os.path.join(directory, d) for d in os.listdir(directory) if d.startswith("checkpoint") and os.path.isdir(os.path.join(directory, d))]
    # Sort them by their modification time
    dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    # Return the most recent directory
    return dirs[0] if dirs else None

def main():
    args = get_args()

    print(f"[1/5] Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=None,  # Load without device_map to avoid meta tensors
    ).to('cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Find the most recent checkpoint directory
    recent_checkpoint_dir = get_most_recent_checkpoint_dir(args.peft_model)
    if not recent_checkpoint_dir:
        raise FileNotFoundError(f"No checkpoint directories found in {args.peft_model}")

    print(f"[2/5] Loading adapter from the most recent checkpoint: {recent_checkpoint_dir}")
    adapter_model = AutoModelForCausalLM.from_pretrained(
        recent_checkpoint_dir,
        device_map=None,  # Load without device_map to avoid meta tensors
    ).to('cpu')
    
    print("[3/5] Merge base model and adapter")
    for param_base, param_adapter in zip(base_model.parameters(), adapter_model.parameters()):
        param_base.data.copy_(param_adapter.data)

    # Move the merged model to GPU if available
    base_model = base_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[4/5] Saving model and tokenizer in {args.hub_id}")
    base_model.save_pretrained(f"{args.hub_id}")
    tokenizer.save_pretrained(f"{args.hub_id}")

    print(f"[5/5] Uploading to Hugging Face Hub: {args.hub_id}")
    base_model.push_to_hub(f"{args.hub_id}", use_temp_dir=False)
    tokenizer.push_to_hub(f"{args.hub_id}", use_temp_dir=False)
    
    print("Merged model uploaded to Hugging Face Hub!")

if __name__ == "__main__":
    main()