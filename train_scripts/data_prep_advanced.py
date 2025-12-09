import argparse
import json
import random
import chess
import chess.engine
import chess.pgn
import csv
import zstandard as zstd
from io import StringIO
from pathlib import Path

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
        print(f"Error analyzing position: {e}")
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

def extract_puzzles(puzzle_file, engine, output_file, rating_ranges):
    """Extract puzzles from Lichess database"""
    print(f"Extracting puzzles from {puzzle_file}...")
    
    puzzles_by_rating = {r: [] for r in rating_ranges.keys()}
    
    dctx = zstd.ZstdDecompressor()
    
    with open(puzzle_file, 'rb') as compressed:
        with dctx.stream_reader(compressed) as reader:
            text_stream = StringIO(reader.read().decode('utf-8'))
            csv_reader = csv.DictReader(text_stream)
            
            for row in csv_reader:
                rating = int(row['Rating'])
                fen = row['FEN']
                moves = row['Moves'].split()
                
                for rating_range, (min_r, max_r, target_count) in rating_ranges.items():
                    if min_r <= rating < max_r and len(puzzles_by_rating[rating_range]) < target_count:
                        try:
                            board = chess.Board(fen)
                            
                            if len(list(board.legal_moves)) < 3:
                                continue
                            
                            top_moves = analyze_position(board, engine, depth=7)
                            
                            if top_moves:
                                best_move = moves[0] if moves[0] in [m.uci() for m in board.legal_moves] else top_moves[0]['move']
                                example = create_training_example(board, top_moves, best_move)
                                puzzles_by_rating[rating_range].append(example)
                        except Exception as e:
                            print(f"Error processing puzzle: {e}")
                            continue
                
                if all(len(puzzles_by_rating[r]) >= rating_ranges[r][2] for r in rating_ranges):
                    break
    
    all_puzzles = []
    for rating_range, puzzles in puzzles_by_rating.items():
        print(f"Extracted {len(puzzles)} puzzles from {rating_range}")
        all_puzzles.extend(puzzles)
    
    return all_puzzles

def generate_dataset(output_file, num_positions=100000, stockfish_path=None):
    """Generate comprehensive training dataset"""
    
    if stockfish_path is None:
        stockfish_path = "/usr/games/stockfish"
    
    print(f"Starting data generation for {num_positions} positions...")
    print(f"Using Stockfish at: {stockfish_path}")
    
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    puzzle_file = Path("lichess_db_puzzle.csv.zst")
    
    total_puzzle_target = min(8000, int(num_positions * 0.08))
    
    rating_ranges = {
        "below_1000": (0, 1000, int(total_puzzle_target * 0.125)),
        "1000_1500": (1000, 1500, int(total_puzzle_target * 0.125)),
        "1500_2000": (1500, 2000, int(total_puzzle_target * 0.1875)),
        "2000_2500": (2000, 2500, int(total_puzzle_target * 0.25)),
        "2500_plus": (2500, 4000, int(total_puzzle_target * 0.3125))
    }
    
    all_examples = []
    
    if puzzle_file.exists():
        puzzles = extract_puzzles(puzzle_file, engine, output_file, rating_ranges)
        all_examples.extend(puzzles)
        print(f"Added {len(puzzles)} puzzle positions")
    else:
        print(f"Puzzle file not found: {puzzle_file}")
    
    num_random = num_positions - len(all_examples)
    print(f"Generating {num_random} diverse positions...")
    
    phase_targets = {
        "opening": int(num_random * 0.3),
        "middlegame": int(num_random * 0.5),
        "endgame": int(num_random * 0.2)
    }
    
    phase_counts = {phase: 0 for phase in phase_targets.keys()}
    
    for i in range(num_random):
        if i % 1000 == 0:
            print(f"Generated {i}/{num_random} diverse positions...")
        
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
            
        except Exception as e:
            print(f"Error generating position {i}: {e}")
            continue
    
    engine.quit()
    
    random.shuffle(all_examples)
    
    print(f"Writing {len(all_examples)} examples to {output_file}...")
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Dataset created successfully!")
    print(f"Total examples: {len(all_examples)}")
    print(f"Phase distribution: {phase_counts}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="train_data_advanced.jsonl")
    parser.add_argument("--num-positions", type=int, default=100000)
    parser.add_argument("--stockfish", type=str, default=None, help="Path to Stockfish")
    args = parser.parse_args()
    
    generate_dataset(args.output, args.num_positions, args.stockfish)
