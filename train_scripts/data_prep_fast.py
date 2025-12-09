import argparse
import json
import random
import sys
import chess
import chess.engine

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
    """Count legal moves (mobility)"""
    return len(list(board.legal_moves))

def get_game_phase(board):
    """Determine game phase based on material"""
    total_pieces = len(board.piece_map())
    if total_pieces >= 28:
        return "opening"
    elif total_pieces >= 16:
        return "middlegame"
    else:
        return "endgame"

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
        print(f"Error analyzing position: {e}", file=sys.stderr, flush=True)
        return []

def generate_diverse_position(engine):
    """Generate a diverse chess position by playing random moves"""
    board = chess.Board()
    
    num_moves = random.choice([
        random.randint(0, 8),
        random.randint(8, 25),
        random.randint(25, 60)
    ])
    
    for _ in range(num_moves):
        if board.is_game_over():
            break
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        
        if random.random() < 0.7:
            move = random.choice(legal_moves)
        else:
            try:
                result = engine.play(board, chess.engine.Limit(time=0.01))
                move = result.move
            except:
                move = random.choice(legal_moves)
        
        board.push(move)
    
    if board.is_game_over() or len(list(board.legal_moves)) < 3:
        return generate_diverse_position(engine)
    
    return board

def create_training_example(board, top_moves, best_move):
    """Create a training example in chat format"""
    fen = board.fen()
    legal_moves = [m.uci() for m in board.legal_moves]
    legal_moves_str = " ".join(legal_moves)
    
    material = count_material(board)
    mobility = get_mobility(board)
    game_phase = get_game_phase(board)
    
    top_moves_str = " | ".join([f"{m['move']}:{m['eval_cp']}" for m in top_moves])
    
    prompt = f"""You are an expert chess player. Here is the position in FEN format:
{fen}

Legal moves: {legal_moves_str}

Position analysis:
- Game phase: {game_phase}
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

def generate_dataset(output_file, num_positions=100000, stockfish_path=None):
    """Generate comprehensive training dataset"""
    
    if stockfish_path is None:
        stockfish_path = "/usr/games/stockfish"
    
    print(f"Starting data generation for {num_positions} positions...", flush=True)
    print(f"Using Stockfish at: {stockfish_path}", flush=True)
    
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    all_examples = []
    
    phase_targets = {
        "opening": int(num_positions * 0.3),
        "middlegame": int(num_positions * 0.5),
        "endgame": int(num_positions * 0.2)
    }
    
    phase_counts = {phase: 0 for phase in phase_targets.keys()}
    
    print(f"Target distribution: {phase_targets}", flush=True)
    
    generated = 0
    attempts = 0
    max_attempts = num_positions * 2
    
    while generated < num_positions and attempts < max_attempts:
        attempts += 1
        
        if attempts % 100 == 0:
            print(f"Progress: {generated}/{num_positions} positions generated ({attempts} attempts)", flush=True)
            print(f"Phase distribution: {phase_counts}", flush=True)
        
        board = generate_diverse_position(engine)
        phase = get_game_phase(board)
        
        if phase_counts[phase] >= phase_targets[phase]:
            continue
        
        try:
            top_moves = analyze_position(board, engine, depth=7)
            
            if not top_moves:
                continue
            
            best_move = top_moves[0]['move']
            example = create_training_example(board, top_moves, best_move)
            
            all_examples.append(example)
            phase_counts[phase] += 1
            generated += 1
            
        except Exception as e:
            print(f"Error generating position: {e}", file=sys.stderr, flush=True)
            continue
    
    engine.quit()
    
    random.shuffle(all_examples)
    
    print(f"Writing {len(all_examples)} examples to {output_file}...", flush=True)
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Dataset created successfully!", flush=True)
    print(f"Total examples: {len(all_examples)}", flush=True)
    print(f"Phase distribution: {phase_counts}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="train_data_fast.jsonl")
    parser.add_argument("--num-positions", type=int, default=10000)
    parser.add_argument("--stockfish", type=str, default=None, help="Path to Stockfish")
    args = parser.parse_args()
    
    generate_dataset(args.output, args.num_positions, args.stockfish)
