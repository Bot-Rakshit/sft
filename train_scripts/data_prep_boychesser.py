import argparse
import json
from typing import List, Optional

import chess
from datasets import load_dataset


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.2,
    chess.BISHOP: 3.3,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.5,
}

CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]
EXTENDED_CENTER = CENTER_SQUARES + [chess.C3, chess.C4, chess.C5, chess.C6, chess.D3, chess.E3, chess.F3, chess.F4, chess.F5, chess.F6, chess.E6, chess.D6]


def material_score(board: chess.Board, color: bool) -> float:
    score = 0.0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, color)) * value
        score -= len(board.pieces(piece_type, not color)) * value
    return score


def mobility(board: chess.Board, color: bool) -> int:
    copy_board = board.copy()
    copy_board.turn = color
    return len(list(copy_board.legal_moves))


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


def heuristic_summary(board: chess.Board, cp: Optional[int], mate: Optional[int]) -> str:
    side = board.turn
    mat = material_score(board, side)
    mob = mobility(board, side) - mobility(board, not side)
    center = center_control(board, side) - center_control(board, not side)
    shield = king_shield(board, side) - king_shield(board, not side)
    passed = passed_pawns(board, side) - passed_pawns(board, not side)

    if mate is not None:
        eval_text = f"Mate in {mate}"
    elif cp is not None:
        eval_text = f"Engine eval {cp/100.0:+.2f}"
    else:
        eval_text = "Engine eval unknown"

    parts = [
        eval_text,
        f"Material (stm): {mat:+.2f}",
        f"Mobility diff: {mob:+d}",
        f"Center control diff: {center:+d}",
        f"King shield diff: {shield:+d}",
        f"Passed pawns diff: {passed:+d}",
    ]
    return " | ".join(parts)


def create_example(sample) -> Optional[dict]:
    fen = sample.get("fen")
    line = sample.get("line")
    cp = sample.get("cp")
    mate = sample.get("mate")

    if not isinstance(fen, str) or not isinstance(line, str):
        return None

    best_move = line.split()[0]
    try:
        board = chess.Board(fen)
    except Exception:
        return None

    legal_moves: List[str] = [m.uci() for m in board.legal_moves]
    if best_move not in legal_moves:
        return None

    legal_moves_str = " ".join(legal_moves)
    side = "White" if board.turn == chess.WHITE else "Black"
    heuristics = heuristic_summary(board, cp, mate)

    user_content = (
        f"You are an expert chess player. Here is the position in FEN format:\n{fen}\n\n"
        f"Side to move: {side}\n"
        f"Legal moves: {legal_moves_str}\n"
        f"Heuristics (inspired by Boychesser): {heuristics}\n\n"
        "Select the best move. Keep your thinking to 2 sentences or less, then output your chosen move.\n"
        "Format:\n<think>brief thinking (2 sentences max)</think>\n<uci_move>your_move</uci_move>"
    )

    assistant_content = f"<think>{heuristics}</think><uci_move>{best_move}</uci_move>"

    return {"messages": [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]}


def prepare_data(max_samples: int, output_path: str):
    dataset = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)

    written = 0
    with open(output_path, "w") as f:
        for sample in dataset:
            example = create_example(sample)
            if example is None:
                continue
            f.write(json.dumps(example) + "\n")
            written += 1
            if written % 1000 == 0:
                print(f"Wrote {written} examples...")
            if written >= max_samples:
                break
    print(f"Saved {written} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Boychesser-inspired chess data")
    parser.add_argument("--max-samples", type=int, default=50000, help="Number of samples to extract")
    parser.add_argument("--output", type=str, default="train_data_boychesser.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    prepare_data(args.max_samples, args.output)


if __name__ == "__main__":
    main()
