#!/usr/bin/env python3
import argparse
import chess
import chess.engine
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import sys

class ChessAgent:
    def __init__(self, model_path):
        print(f"Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
    
    def get_move(self, board):
        fen = board.fen()
        legal_moves = [m.uci() for m in board.legal_moves]
        legal_moves_str = " ".join(legal_moves)
        
        prompt = f"""You are an expert chess player. Here is the position in FEN format:
{fen}

Legal moves: {legal_moves_str}

Select the best move. Keep your thinking to 2 sentences or less, then output your chosen move.
Format:
<think>brief thinking (2 sentences max)</think>
<uci_move>your_move</uci_move>"""
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        # Parse move
        move_match = re.search(r"<uci_move>(.*?)</uci_move>", response)
        if move_match:
            move_str = move_match.group(1).strip()
            # Handle Chinese characters or malformed output
            if move_str in legal_moves:
                return move_str
        
        # Fallback: try to find any legal move in response
        for move in legal_moves:
            if move in response:
                return move
        
        # Last resort: random legal move
        print(f"Warning: Could not parse move from response, using first legal move")
        return legal_moves[0] if legal_moves else None

def get_position_eval(board, engine, depth=10):
    """Get position evaluation in centipawns from white's perspective"""
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white()
        
        if score.is_mate():
            # Mate in N moves
            mate_in = score.mate()
            return 10000 if mate_in > 0 else -10000
        else:
            return score.score()
    except:
        return 0

def calculate_acpl(move_evals):
    """Calculate Average Centipawn Loss"""
    if not move_evals:
        return 0.0
    
    losses = []
    for i in range(len(move_evals) - 1):
        eval_before = move_evals[i]
        eval_after = move_evals[i + 1]
        
        # Flip sign for black's perspective
        if i % 2 == 1:
            eval_before = -eval_before
            eval_after = -eval_after
        
        loss = max(0, eval_before - eval_after)
        losses.append(loss)
    
    return sum(losses) / len(losses) if losses else 0.0

def play_game(agent, stockfish_path, stockfish_skill=1, analysis_depth=10):
    """Play one game and return ACPL"""
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    # Set Stockfish skill level (0-20)
    engine.configure({"Skill Level": stockfish_skill})
    
    move_evals = []
    moves_played = []
    
    print(f"\nStarting game: Agent (White) vs Stockfish Level {stockfish_skill} (Black)")
    print("=" * 60)
    
    move_num = 1
    
    while not board.is_game_over() and len(moves_played) < 100:
        # Get evaluation before move
        eval_before = get_position_eval(board, engine, analysis_depth)
        move_evals.append(eval_before)
        
        if board.turn == chess.WHITE:
            # Agent's turn
            move_str = agent.get_move(board)
            if move_str is None:
                print("Agent resigned (no valid move)")
                break
            
            try:
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    print(f"Illegal move by agent: {move_str}")
                    break
            except:
                print(f"Invalid move format: {move_str}")
                break
            
            board.push(move)
            print(f"{move_num}. Agent: {move_str} (eval: {eval_before:+})")
        else:
            # Stockfish's turn
            result = engine.play(board, chess.engine.Limit(time=0.1))
            move = result.move
            board.push(move)
            print(f"{move_num}... Stockfish: {move.uci()} (eval: {eval_before:+})")
            move_num += 1
        
        moves_played.append(move_str if board.turn == chess.BLACK else move.uci())
    
    # Final evaluation
    final_eval = get_position_eval(board, engine, analysis_depth)
    move_evals.append(final_eval)
    
    engine.quit()
    
    # Calculate ACPL
    acpl = calculate_acpl(move_evals)
    
    print("=" * 60)
    print(f"Game over: {board.result()}")
    print(f"Total moves: {len(moves_played)}")
    print(f"Average Centipawn Loss (ACPL): {acpl:.2f}")
    print(f"Final position evaluation: {final_eval:+}")
    
    return {
        "result": board.result(),
        "moves": moves_played,
        "acpl": acpl,
        "evals": move_evals,
        "final_eval": final_eval
    }

def main():
    parser = argparse.ArgumentParser(description="Test chess model vs Stockfish")
    parser.add_argument("--model", type=str, default="qwen-chess-0.5b-merged", help="Path to model")
    parser.add_argument("--stockfish", type=str, default="/opt/homebrew/bin/stockfish", help="Path to Stockfish")
    parser.add_argument("--skill-level", type=int, default=1, help="Stockfish skill level (0-20)")
    parser.add_argument("--analysis-depth", type=int, default=10, help="Stockfish analysis depth")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to play")
    args = parser.parse_args()
    
    print(f"Testing: {args.model}")
    print(f"Stockfish: {args.stockfish} (Skill Level {args.skill_level})")
    print(f"Analysis depth: {args.analysis_depth}")
    print(f"Number of games: {args.num_games}")
    print()
    
    agent = ChessAgent(args.model)
    
    results = []
    total_acpl = 0
    wins = 0
    draws = 0
    losses = 0
    
    for game_num in range(args.num_games):
        print(f"\n{'='*60}")
        print(f"GAME {game_num + 1}/{args.num_games}")
        print(f"{'='*60}")
        
        result = play_game(agent, args.stockfish, args.skill_level, args.analysis_depth)
        results.append(result)
        total_acpl += result["acpl"]
        
        if result["result"] == "1-0":
            wins += 1
        elif result["result"] == "1/2-1/2":
            draws += 1
        else:
            losses += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Games played: {args.num_games}")
    print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")
    print(f"Win rate: {wins/args.num_games*100:.1f}%")
    print(f"Average ACPL: {total_acpl/args.num_games:.2f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
