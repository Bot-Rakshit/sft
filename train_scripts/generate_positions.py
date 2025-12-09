import argparse
import json
import random
import chess
from collections import defaultdict

def get_game_phase(board, fen):
    """Determine game phase based on move number and piece count"""
    parts = fen.split()
    move_number = int(parts[-1]) if len(parts) >= 6 else 1
    total_pieces = len(board.piece_map())
    
    if move_number <= 15 and total_pieces >= 28:
        return "opening"
    elif total_pieces <= 12:
        return "endgame"
    else:
        return "middlegame"

def create_endgame_position():
    """Create an endgame position by starting from a middlegame and removing pieces"""
    board = chess.Board()
    
    for _ in range(random.randint(20, 40)):
        if board.is_game_over():
            break
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        board.push(random.choice(legal_moves))
    
    if board.is_game_over():
        return None, None
    
    pieces_to_remove = random.randint(16, 22)
    piece_map = board.piece_map()
    squares = list(piece_map.keys())
    random.shuffle(squares)
    
    removed = 0
    for square in squares:
        piece = piece_map[square]
        if piece.piece_type != chess.KING and removed < pieces_to_remove:
            board.remove_piece_at(square)
            removed += 1
    
    if board.is_valid() and not board.is_game_over() and len(list(board.legal_moves)) >= 3:
        fen = board.fen()
        phase = get_game_phase(board, fen)
        return fen, phase
    
    return None, None

def generate_position(target_phase=None):
    """Generate a chess position by playing random moves"""
    if target_phase == "endgame":
        return create_endgame_position()
    
    board = chess.Board()
    
    if target_phase == "opening":
        target_moves = random.randint(0, 15)
    else:
        target_moves = random.choice([
            random.randint(0, 15),
            random.randint(10, 30),
            random.randint(25, 70)
        ])
    
    for _ in range(target_moves):
        if board.is_game_over():
            break
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        
        move = random.choice(legal_moves)
        board.push(move)
    
    if board.is_game_over() or len(list(board.legal_moves)) < 3:
        return None, None
    
    fen = board.fen()
    phase = get_game_phase(board, fen)
    
    return fen, phase

def generate_positions(output_file, num_positions=100000):
    """Generate positions with target distribution"""
    print(f"Generating {num_positions} diverse positions...")
    
    target_distribution = {
        "opening": int(num_positions * 0.3),
        "middlegame": int(num_positions * 0.5),
        "endgame": int(num_positions * 0.2)
    }
    
    phase_counts = defaultdict(int)
    positions = []
    
    attempts = 0
    max_attempts = num_positions * 3
    
    print(f"Target: {dict(target_distribution)}")
    
    while len(positions) < num_positions and attempts < max_attempts:
        attempts += 1
        
        if attempts % 1000 == 0:
            print(f"Progress: {len(positions)}/{num_positions} (attempts: {attempts})")
            print(f"Current: {dict(phase_counts)}")
        
        needed_phase = None
        for phase in ["opening", "middlegame", "endgame"]:
            if phase_counts[phase] < target_distribution[phase]:
                needed_phase = phase
                break
        
        if needed_phase:
            fen, phase = generate_position(target_phase=needed_phase)
        else:
            break
        
        if fen is None or phase is None:
            continue
        
        if phase_counts[phase] >= target_distribution[phase]:
            continue
        
        positions.append({"fen": fen, "phase": phase})
        phase_counts[phase] += 1
    
    print(f"\nGenerated {len(positions)} positions")
    print(f"Final distribution: {dict(phase_counts)}")
    
    random.shuffle(positions)
    
    print(f"Writing to {output_file}...")
    with open(output_file, 'w') as f:
        for pos in positions:
            f.write(json.dumps(pos) + '\n')
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="positions_100k.jsonl")
    parser.add_argument("--num-positions", type=int, default=100000)
    args = parser.parse_args()
    
    generate_positions(args.output, args.num_positions)
