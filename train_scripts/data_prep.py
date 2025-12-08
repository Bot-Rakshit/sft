
import torch
import datasets
from datasets import load_dataset

def prepare_chess_data():
    print("Loading Lichess/chess-position-evaluations dataset...")
    # Streaming mode is crucial for large datasets
    dataset = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)
    
    # We will create a generator to extract a small subset (e.g., 1000 samples for initial testing)
    data_list = []
    count = 0
    max_samples = 2000

    print(f"Extracting {max_samples} samples...")
    for sample in dataset:
        # Sample has keys: 'fen', 'line', 'depth', 'knodes', 'cp', 'mate'
        fen = sample['fen']
        line = sample['line'] 
        cp = sample['cp']
        mate = sample['mate']
        
        # Get the best move (first move in line)
        if isinstance(line, str):
            best_move = line.split()[0]
        else:
            continue

        # Construct a dummy rationale
        if mate is not None:
            eval_text = f"Mate in {mate}"
        elif cp is not None:
            eval_text = f"Evaluation {cp/100.0:+.2f}"
        else:
            eval_text = "Unknown evaluation"

        rationale = f"The position evaluation is {eval_text}. The best move identified by engine analysis is {best_move}."
        
        # Prompt format
        # We need to match the input format expected by the challenge
        # But wait, the TRAINING data needs to include the "response".
        # The input to the model will be FEN + legal moves.
        # The dataset doesn't strictly provide legal moves list, so we might need python-chess to generate them if we want to be robust.
        # BUT, for SFT, we can just fine-tune on:
        # User: FEN: ... Legal moves: ... -> Assistant: <think>...</think><uci_move>...</uci_move>
        
        # To make it robust, let's use python-chess to get legal moves
        import chess
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        legal_moves_str = " ".join(legal_moves)
        side = "White" if board.turn == chess.WHITE else "Black"
        
        # Create the chat format
        # We will use a simple chat template structure for the dataset
        messages = [
            {"role": "user", "content": f"You are an expert chess player. Here is the position in FEN format:\n{fen}\n\nLegal moves: {legal_moves_str}\n\nSelect the best move. Keep your thinking to 2 sentences or less, then output your chosen move.\nFormat:\n<think>brief thinking (2 sentences max)</think>\n<uci_move>your_move</uci_move>"},
            {"role": "assistant", "content": f"<think>{rationale}</think><uci_move>{best_move}</uci_move>"}
        ]
        
        data_list.append({"messages": messages})
        count += 1
        if count >= max_samples:
            break
            
    print(f"Extracted {len(data_list)} samples.")
    
    # Save to JSONL
    import json
    with open("train_data_mini.jsonl", "w") as f:
        for item in data_list:
            f.write(json.dumps(item) + "\n")
    print("Saved to train_data_mini.jsonl")

if __name__ == "__main__":
    prepare_chess_data()
