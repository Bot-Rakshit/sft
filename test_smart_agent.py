#!/usr/bin/env python3
"""Test smart agent with safety checks vs Stockfish"""
import sys
sys.path.insert(0, '/Users/rakshit/workspace/repos/chess/global-chess-challenge-2025-starter-kit')

from smart_agent import SmartChessAgent
import chess
import chess.engine

def get_position_eval(board, engine, depth=10):
    """Get position evaluation"""
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white()
        if score.is_mate():
            return 10000 if score.mate() > 0 else -10000
        return score.score()
    except:
        return 0

def calculate_acpl(move_evals):
    """Calculate ACPL"""
    if not move_evals:
        return 0.0
    
    losses = []
    for i in range(len(move_evals) - 1):
        eval_before = move_evals[i]
        eval_after = move_evals[i + 1]
        
        if i % 2 == 1:
            eval_before = -eval_before
            eval_after = -eval_after
        
        loss = max(0, eval_before - eval_after)
        losses.append(loss)
    
    return sum(losses) / len(losses) if losses else 0.0

def play_game(agent, stockfish_path, stockfish_skill=1, analysis_depth=10):
    """Play one game"""
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": stockfish_skill})
    
    move_evals = []
    moves_played = []
    
    print(f"\nGame: Smart Agent (White) vs Stockfish Level {stockfish_skill} (Black)")
    print("=" * 60)
    
    move_num = 1
    
    while not board.is_game_over() and len(moves_played) < 100:
        eval_before = get_position_eval(board, engine, analysis_depth)
        move_evals.append(eval_before)
        
        if board.turn == chess.WHITE:
            move_str = agent.get_move(board, use_safety_checks=True)
            if move_str is None:
                break
            
            try:
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    print(f"Illegal move: {move_str}")
                    break
            except:
                break
            
            board.push(move)
            print(f"{move_num}. Agent: {move_str} (eval: {eval_before:+})")
        else:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            move = result.move
            board.push(move)
            print(f"{move_num}... Stockfish: {move.uci()} (eval: {eval_before:+})")
            move_num += 1
        
        moves_played.append(move_str if board.turn == chess.BLACK else move.uci())
    
    final_eval = get_position_eval(board, engine, analysis_depth)
    move_evals.append(final_eval)
    
    engine.quit()
    
    acpl = calculate_acpl(move_evals)
    
    print("=" * 60)
    print(f"Result: {board.result()}")
    print(f"Moves: {len(moves_played)}")
    print(f"ACPL: {acpl:.2f}")
    
    return {
        "result": board.result(),
        "acpl": acpl,
        "moves": len(moves_played)
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen-chess-0.5b-108k-merged-new")
    parser.add_argument("--stockfish", type=str, default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--num-games", type=int, default=3)
    args = parser.parse_args()
    
    print("Initializing Smart Agent with:")
    print(f"  - Base model: {args.model}")
    print(f"  - Tactical safety checks: ON")
    print(f"  - 1-ply search validation: ON")
    print(f"  - Blunder detection: ON")
    print()
    
    agent = SmartChessAgent(args.model, args.stockfish, use_search=True)
    
    results = []
    total_acpl = 0
    wins = 0
    draws = 0
    losses = 0
    
    for game_num in range(args.num_games):
        print(f"\n{'='*60}")
        print(f"GAME {game_num + 1}/{args.num_games}")
        print(f"{'='*60}")
        
        result = play_game(agent, args.stockfish, stockfish_skill=1, analysis_depth=10)
        results.append(result)
        total_acpl += result["acpl"]
        
        if result["result"] == "1-0":
            wins += 1
        elif result["result"] == "1/2-1/2":
            draws += 1
        else:
            losses += 1
    
    agent.close()
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Record: {wins}W-{losses}L-{draws}D")
    print(f"Win rate: {wins/args.num_games*100:.1f}%")
    print(f"Average ACPL: {total_acpl/args.num_games:.2f}")
    print(f"{'='*60}")
