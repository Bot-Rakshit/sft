import argparse
import json
import chess
import chess.engine
import sys

def count_material(board):
    """Count material for current side to move"""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }
    
    stm = board.turn
    stm_material = sum(len(board.pieces(pt, stm)) * val for pt, val in piece_values.items())
    opp_material = sum(len(board.pieces(pt, not stm)) * val for pt, val in piece_values.items())
    
    return stm_material - opp_material

def get_mobility(board):
    """Count legal moves"""
    return len(list(board.legal_moves))

def analyze_position(board, engine, depth=7):
    """Analyze position and get top 5 moves with evals"""
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=5)
        
        top_moves = []
        for entry in info:
            move = entry["pv"][0]
            score = entry["score"].white()
            
            if score.is_mate():
                eval_cp = 10000 if score.mate() > 0 else -10000
            else:
                eval_cp = score.score()
            
            top_moves.append({
                "move": move.uci(),
                "eval_cp": eval_cp
            })
        
        return top_moves
    except Exception as e:
        return []

def create_training_example(board, fen, phase, top_moves, best_move):
    """Create training example in chat format"""
    legal_moves = [m.uci() for m in board.legal_moves]
    legal_moves_str = " ".join(legal_moves)
    
    material = count_material(board)
    mobility = get_mobility(board)
    
    top_moves_str = " | ".join([f"{m['move']}:{m['eval_cp']}" for m in top_moves])
    
    prompt = f"""You are an expert chess player. Here is the position in FEN format:
{fen}

Legal moves: {legal_moves_str}

Position analysis:
- Game phase: {phase}
- Material advantage: {material:+.0f}
- Mobility (legal moves): {mobility}
- Top moves with evaluations: {top_moves_str}

Select the best move. Keep your thinking brief, then output your chosen move.
Format:
<think>brief analysis</think>
<uci_move>your_move</uci_move>"""

    best_eval = next((m['eval_cp'] for m in top_moves if m['move'] == best_move), top_moves[0]['eval_cp'] if top_moves else 0)
    response = f"<think>Best move {best_move} with eval {best_eval:+}cp. Material {material:+}, mobility {mobility}.</think><uci_move>{best_move}</uci_move>"
    
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }

def analyze_positions(input_file, output_file, stockfish_path, depth=7, start_idx=0, end_idx=None):
    """Analyze positions and create training data"""
    print(f"Loading positions from {input_file}...")
    
    with open(input_file, 'r') as f:
        positions = [json.loads(line) for line in f]
    
    if end_idx is None:
        end_idx = len(positions)
    
    positions = positions[start_idx:end_idx]
    
    print(f"Analyzing {len(positions)} positions (indices {start_idx} to {end_idx-1})")
    print(f"Using Stockfish at: {stockfish_path}")
    print(f"Depth: {depth}")
    
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    examples = []
    
    for i, pos_data in enumerate(positions):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(positions)} analyzed...", flush=True)
        
        try:
            fen = pos_data['fen']
            phase = pos_data['phase']
            board = chess.Board(fen)
            
            top_moves = analyze_position(board, engine, depth)
            
            if not top_moves:
                continue
            
            best_move = top_moves[0]['move']
            example = create_training_example(board, fen, phase, top_moves, best_move)
            examples.append(example)
            
        except Exception as e:
            print(f"Error on position {i}: {e}", file=sys.stderr, flush=True)
            continue
    
    engine.quit()
    
    print(f"Writing {len(examples)} examples to {output_file}...")
    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print("Done!")
    print(f"Successfully created {len(examples)} training examples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input positions file")
    parser.add_argument("--output", type=str, required=True, help="Output training data file")
    parser.add_argument("--stockfish", type=str, required=True, help="Path to Stockfish")
    parser.add_argument("--depth", type=int, default=7, help="Analysis depth")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    args = parser.parse_args()
    
    analyze_positions(args.input, args.output, args.stockfish, args.depth, args.start, args.end)
