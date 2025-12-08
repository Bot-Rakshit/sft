
import os
import torch
import chess
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class TrainedAgent:
    def __init__(self, model_path, base_model_name):
        print(f"Loading base model: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            torch_dtype=torch.float16, # Use float16 for inference
            device_map="auto"
        )
        
        print(f"Loading adapter: {model_path}")
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        
    def get_move(self, fen, legal_moves):
        # Format prompt
        legal_moves_str = " ".join(legal_moves)
        prompt = f"""You are an expert chess player. Here is the position in FEN format:
{fen}

Legal moves: {legal_moves_str}

Select the best move. Keep your thinking to 2 sentences or less, then output your chosen move.
Format:
<think>brief thinking (2 sentences max)</think>
<uci_move>your_move</uci_move>"""
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=100, 
                temperature=0.1, # Low temp for deterministic play
                do_sample=True
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract response part (after the prompt)
        if "assistant" in response:
             response = response.split("assistant")[-1].strip()
             
        # Simple parsing logic
        import re
        move_match = re.search(r"<uci_move>(.*?)</uci_move>", response)
        if move_match:
            move = move_match.group(1).strip()
            return move, response
        else:
            # Fallback parsing
            print(f"Failed to parse response: {response}")
            return None, response

def evaluate_model():
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path = "qwen-chess-0.5b-sft"
    
    agent = TrainedAgent(adapter_path, base_model)
    
    # Test on a few positions
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", # Start
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", # Spanish Opening
        "rnbqkb1r/pp2pppp/3p1n2/2p5/2PPP3/2N5/PP3PPP/R1BQKBNR b KQkq - 0 4" # Sicilian
    ]
    
    print("\n=== Running Local Tests ===")
    for i, fen in enumerate(test_fens):
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        
        print(f"\nPosition {i+1}: {fen}")
        move, raw_response = agent.get_move(fen, legal_moves)
        
        print(f"Agent Response: {raw_response[:100]}...")
        print(f"Selected Move: {move}")
        
        if move in legal_moves:
            print("✅ Legal Move")
        else:
            print(f"❌ Illegal Move (Expected one of: {legal_moves[:5]}...)")

if __name__ == "__main__":
    evaluate_model()
