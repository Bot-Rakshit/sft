
import argparse
import torch
import chess
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = "mps" if torch.backends.mps.is_available() else "cpu"

def load_model(model_path):
    global model, tokenizer
    print(f"Loading model from {model_path} on {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device
        )
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.json
        messages = data.get('messages', [])
        
        # Construct prompt from messages
        # The env sends: system (optional), user (FEN+moves)
        # We need to format this for the model
        
        # Simple concatenation for now, or use apply_chat_template if supported
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=150, 
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the assistant's part (naive splitting, might need robustness)
        if "assistant" in response_text:
            content = response_text.split("assistant")[-1].strip()
        else:
            # Fallback if template doesn't explicitly mark assistant in decoded text
            # (Qwen usually does <|im_start|>assistant...)
            # Let's try to remove the input prompt from the output
            content = response_text[len(text):].strip()
            
        # Clean up if needed
        # The environment expects the content to contain <think> and <uci_move>
        
        response = {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": 0,
            "model": "local-qwen-chess",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": inputs.input_ids.shape[1],
                "completion_tokens": outputs.shape[1] - inputs.input_ids.shape[1],
                "total_tokens": outputs.shape[1]
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during generation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="qwen-chess-0.5b-merged", help="Path to the merged model")
    parser.add_argument("--port", type=int, default=5001, help="Port to run server on")
    args = parser.parse_args()
    
    load_model(args.model_path)
    app.run(host='0.0.0.0', port=args.port, debug=False)
