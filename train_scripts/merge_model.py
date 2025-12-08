
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_model():
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path = "qwen-chess-0.5b-sft"
    output_dir = "qwen-chess-0.5b-merged"
    
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging model...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

if __name__ == "__main__":
    merge_model()
