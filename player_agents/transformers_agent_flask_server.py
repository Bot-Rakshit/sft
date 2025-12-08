
import argparse
import os
import sys
import re

import chess
import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Global variables
model = None
tokenizer = None

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.2,
    chess.BISHOP: 3.3,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.5,
}

CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]
EXTENDED_CENTER = CENTER_SQUARES + [chess.C3, chess.C4, chess.C5, chess.C6, chess.D3, chess.E3, chess.F3, chess.F4, chess.F5, chess.F6, chess.D6, chess.E6]


def material_score(board: chess.Board, color: bool) -> float:
    score = 0.0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, color)) * value
        score -= len(board.pieces(piece_type, not color)) * value
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


def maybe_augment_messages_with_heuristics(messages):
    if not messages:
        return messages
    user = messages[-1]
    content = user.get("content")
    if not isinstance(content, str):
        return messages

    fen_pattern = r"([rnbqkRNBQK1-8/]+\s[wb]\s[-KQkq]+\s[-a-h1-8]+\s\d+\s\d+)"
    match = re.search(fen_pattern, content)
    if not match:
        return messages
    fen = match.group(1)
    try:
        board = chess.Board(fen)
    except Exception:
        return messages

    heuristics = heuristic_summary(board)
    augmented = messages.copy()
    augmented[-1] = dict(user)
    augmented[-1]["content"] = content + f"\n\nHeuristics (Boychesser-inspired): {heuristics}"
    return augmented

def load_model(model_name):
    global model, tokenizer
    print(f"Loading model: {model_name}")
    try:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        sys.exit(1)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.get_json(force=True, silent=True) or {}
        messages = data.get('messages') or []
        if not messages:
            return jsonify({"error": "'messages' field is required"}), 400
        
        augmented_messages = maybe_augment_messages_with_heuristics(messages)
        text = tokenizer.apply_chat_template(augmented_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        max_tokens = int(data.get('max_tokens', 150))
        temperature = float(data.get('temperature', 0.1))
        do_sample = temperature > 0
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        # Only decode generated tokens to avoid echoing the prompt
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        content = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
        response = {
            "id": "chatcmpl-transformers",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "chess-agent",
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
        print(f"Error generating response: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    if model is None:
        return jsonify({"status": "loading"}), 503
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # AIcrowd usually sets environment variables or we pass them via args
    # But since we control the launch script, we can hardcode or pass args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bot-rakshit/qwen-chess-0.5b-sft-v1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    
    load_model(args.model)
    app.run(host='0.0.0.0', port=args.port, debug=False)
