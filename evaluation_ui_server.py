import json
import os
from typing import List, Dict, Any

import chess
from flask import Flask, jsonify, send_from_directory


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

app = Flask(__name__)


def _list_game_logs() -> List[str]:
    if not os.path.isdir(LOGS_DIR):
        return []
    files = [f for f in os.listdir(LOGS_DIR) if f.endswith(".json")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(LOGS_DIR, x)), reverse=True)
    return files


def _load_game(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


@app.route("/api/games")
def api_games() -> Any:
    games = []
    for fname in _list_game_logs():
        full_path = os.path.join(LOGS_DIR, fname)
        try:
            data = _load_game(full_path)
        except Exception:
            continue

        games.append(
            {
                "id": fname,
                "timestamp": data.get("timestamp"),
                "opponent": data.get("opponent"),
                "player_color": data.get("player_color"),
                "result": data.get("result"),
                "moves_played": data.get("moves_played"),
                "player_acpl": data.get("player_acpl"),
                "opponent_acpl": data.get("opponent_acpl"),
            }
        )
    return jsonify({"games": games})


@app.route("/api/games/<path:game_id>")
def api_game_detail(game_id: str) -> Any:
    # Prevent directory traversal
    if "/" in game_id or ".." in game_id:
        return jsonify({"error": "invalid game id"}), 400
    full_path = os.path.join(LOGS_DIR, game_id)
    if not os.path.isfile(full_path):
        return jsonify({"error": "not found"}), 404

    data = _load_game(full_path)

    # Reconstruct positions as FENs from move history for convenience
    move_history = data.get("move_history", [])
    move_comments = data.get("move_comments", [])
    board = chess.Board()
    positions = [board.fen()]
    for uci in move_history:
        try:
            move = chess.Move.from_uci(uci)
            board.push(move)
            positions.append(board.fen())
        except Exception:
            break

    data_out = {
        "id": game_id,
        "timestamp": data.get("timestamp"),
        "opponent": data.get("opponent"),
        "player_color": data.get("player_color"),
        "result": data.get("result"),
        "moves_played": data.get("moves_played"),
        "player_acpl": data.get("player_acpl"),
        "opponent_acpl": data.get("opponent_acpl"),
        "white_acpl": data.get("white_acpl"),
        "black_acpl": data.get("black_acpl"),
        "move_history": move_history,
        "move_comments": move_comments,
        "positions": positions,
    }
    return jsonify(data_out)


@app.route("/")
def index() -> Any:
    # Simple HTML UI with a chessboard and game inspector
    return send_from_directory(BASE_DIR, "evaluation_ui_static.html")


if __name__ == "__main__":
    port = int(os.environ.get("EVAL_UI_PORT", "7000"))
    app.run(host="0.0.0.0", port=port, debug=False)
