
import argparse
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def train(args):
    model_name = args.model
    data_file = args.data
    output_dir = args.output

    print(f"Loading tokenizer/model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_length

    dataset = load_dataset("json", data_files=data_file, split="train")

    if args.dtype == "auto":
        dtype = torch.bfloat16 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else torch.float32
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=dtype == torch.bfloat16,
        report_to="none",
        warmup_ratio=0.03,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training for chess model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model")
    parser.add_argument("--data", type=str, default="train_data_boychesser.jsonl", help="JSONL training data")
    parser.add_argument("--output", type=str, default="qwen-chess-0.5b-sft", help="Output directory")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable gradient checkpointing")
    parser.add_argument("--dtype", type=str, choices=["auto", "bfloat16", "float16", "float32"], default="auto", help="Torch dtype to use for model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
