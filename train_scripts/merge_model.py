
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", type=str, default="qwen-chess-0.5b-sft")
    parser.add_argument("--output", type=str, default="qwen-chess-0.5b-merged")
    args = parser.parse_args()
    
    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    
    print("Merging model...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done!")

if __name__ == "__main__":
    merge_model()
