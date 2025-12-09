import argparse
import json
import os
import random
import sys
from typing import Optional, List

import chess
import chess.engine
import chess.pgn
from tqdm import tqdm


# Simple Boychesser-style heuristics to keep prompt shape consistent
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.2,
    chess.BISHOP: 3.3,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.5,
}

CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]
EXTENDED_CENTER = CENTER_SQUARES + [
    chess.C3, chess.C4, chess.C5, chess.C6,
    chess.D3, chess.E3, chess.F3, chess.F4, chess.F5, chess.F6,
    chess.E6, chess.D6
]


def material_score(board: chess.Board, color: bool) -> float:
    score = 0.0
    for p, v in PIECE_VALUES.items():
        score += len(board.pieces(p, color)) * v
        score -= len(board.pieces(p, not color)) * v
    return score


def mobility(board: chess.Board, color: bool) -> int:
    tmp = board.copy()
    tmp.turn = color
    return len(list(tmp.legal_moves))


def center_control(board: chess.Board, color: bool) -> int:
    return sum(1 for sq in EXTENDED_CENTER if board.is_attacked_by(color, sq))


def king_shield(board: chess.Board, color: bool) -> int:
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    direction = 1 if color == chess.WHITE else -1
    rank = chess.square_rank(king_sq)
    file = chess.square_file(king_sq)
    shield = 0
    for df in (-1, 0, 1):
        f = file + df
        r = rank + direction
        if 0 <= f < 8 and 0 <= r < 8:
            sq = chess.square(f, r)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                shield += 1
    return shield


def passed_pawns(board: chess.Board, color: bool) -> int:
    pawns = board.pieces(chess.PAWN, color)
    count = 0
    for pawn_sq in pawns:
        file = chess.square_file(pawn_sq)
        rank = chess.square_rank(pawn_sq)
        direction = 1 if color == chess.WHITE else -1
        blocked = False
        r = rank + direction
        while 0 <= r < 8 and not blocked:
            for df in (-1, 0, 1):
                f = file + df
                if 0 <= f < 8:
                    sq = chess.square(f, r)
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color != color:
                        blocked = True
                        break
            r += direction
        if not blocked:
            count += 1
    return count


def heuristic_summary(board: chess.Board) -> str:
    side = board.turn
    mat = material_score(board, side)
    mob = mobility(board, side) - mobility(board, not side)
    center = center_control(board, side) - center_control(board, not side)
    shield = king_shield(board, side) - king_shield(board, not side)
    passed = passed_pawns(board, side) - passed_pawns(board, not side)
    side_name = "White" if side == chess.WHITE else "Black"
    return (
        f"Side to move: {side_name}. Material (stm): {mat:+.2f}; "
        f"Mobility diff: {mob:+d}; Center control diff: {center:+d}; "
        f"King shield diff: {shield:+d}; Passed pawns diff: {passed:+d}."
    )


def clamp_cp(score) -> int:
    if score is None:
        return 0
    return int(max(-1000, min(1000, score)))


def stockfish_eval(engine, board: chess.Board, depth: int, movetime_ms: int) -> int:
    limit = chess.engine.Limit(time=movetime_ms / 1000.0, depth=depth if depth else None)
    info = engine.analyse(board, limit)
    rel = info["score"].relative
    cp = rel.score(mate_score=1000) if rel is not None else 0
    return clamp_cp(cp)


def reward_for_move(engine, board: chess.Board, move: chess.Move, depth: int, movetime_ms: int):
    eval_before = stockfish_eval(engine, board, depth, movetime_ms)
    board.push(move)
    eval_after = stockfish_eval(engine, board, depth, movetime_ms)
    board.pop()
    # eval_after is from opponent POV (side-to-move after push), so negate to mover POV
    eval_after_mover = -eval_after
    cpl_loss = max(0, eval_before - eval_after_mover)
    reward = -cpl_loss
    return reward, eval_before, eval_after_mover


def build_example(board: chess.Board, move: chess.Move, reward: float, eval_before: int, eval_after: int) -> Optional[dict]:
    legal_moves = [m.uci() for m in board.legal_moves]
    if move.uci() not in legal_moves:
        return None
    side = "White" if board.turn else "Black"
    heur = heuristic_summary(board)

    user = (
        f"You are an expert chess player. Here is the position in FEN format:\n{board.fen()}\n\n"
        f"Side to move: {side}\n"
        f"Legal moves: {' '.join(legal_moves)}\n"
        f"Heuristics (inspired by Boychesser): {heur}\n\n"
        "Select the best move. Keep your thinking to 2 sentences or less, then output your chosen move.\n"
        "Format:\n<think>brief thinking (2 sentences max)</think>\n<uci_move>your_move</uci_move>"
    )
    assistant = f"<think>{heur}</think><uci_move>{move.uci()}</uci_move>"

    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "reward": reward,
        "eval_before": eval_before,
        "eval_after": eval_after,
        "move_uci": move.uci(),
    }


def sample_positions_from_game(game: chess.pgn.Game, engine, depth: int, movetime_ms: int, ply_stride: int, max_positions: int) -> List[dict]:
    board = game.board()
    samples = []
    for ply_idx, move in enumerate(game.mainline_moves()):
        if ply_idx % ply_stride != 0:
            board.push(move)
            continue
        try:
            reward, ev_b, ev_a = reward_for_move(engine, board, move, depth, movetime_ms)
            ex = build_example(board, move, reward, ev_b, ev_a)
            if ex:
                ex["ply"] = ply_idx
                samples.append(ex)
        except Exception:
            # Skip problematic positions
            pass
        board.push(move)
        if len(samples) >= max_positions:
            break
    return samples


def process_pgns(args):
    random.seed(args.seed)

    pgn_files = []
    if os.path.isdir(args.pgn_dir):
        for root, _, files in os.walk(args.pgn_dir):
            for f in files:
                if f.lower().endswith(".pgn"):
                    pgn_files.append(os.path.join(root, f))
    else:
        print(f"PGN dir not found: {args.pgn_dir}")
        sys.exit(1)

    if not pgn_files:
        print("No PGN files found.")
        sys.exit(1)

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)

    total_written = 0
    with open(args.output, "w") as out:
        pbar_files = tqdm(pgn_files, desc="PGN files")
        for pgn_path in pbar_files:
            if total_written >= args.max_total_positions:
                break
            with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
                game_idx = 0
                while True:
                    if total_written >= args.max_total_positions:
                        break
                    if args.max_games > 0 and game_idx >= args.max_games:
                        break
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    game_idx += 1
                    samples = sample_positions_from_game(
                        game,
                        engine,
                        depth=args.depth,
                        movetime_ms=args.movetime_ms,
                        ply_stride=args.ply_stride,
                        max_positions=args.max_positions_per_game,
                    )
                    if samples:
                        random.shuffle(samples)
                        for ex in samples:
                            if total_written >= args.max_total_positions:
                                break
                            out.write(json.dumps(ex) + "\n")
                            total_written += 1
                    pbar_files.set_postfix({"written": total_written})

    engine.quit()
    print(f"Saved {total_written} examples to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare RL-style chess data from PGNs with Stockfish rewards")
    parser.add_argument("--pgn-dir", type=str, required=True, help="Directory containing PGN files")
    parser.add_argument("--output", type=str, default="rl_games_positions.jsonl", help="Output JSONL path")
    parser.add_argument("--stockfish-path", type=str, default="/opt/homebrew/bin/stockfish", help="Path to stockfish binary")
    parser.add_argument("--depth", type=int, default=7, help="Stockfish depth for reward eval")
    parser.add_argument("--movetime-ms", type=int, default=1000, help="Stockfish movetime per eval (ms)")
    parser.add_argument("--ply-stride", type=int, default=4, help="Sample every N plies to reduce data")
    parser.add_argument("--max-positions-per-game", type=int, default=30, help="Cap positions sampled per game")
    parser.add_argument("--max-total-positions", type=int, default=100000, help="Global cap on output rows")
    parser.add_argument("--max-games", type=int, default=0, help="Optional cap on games per PGN file (0 = no cap)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_pgns(args)
