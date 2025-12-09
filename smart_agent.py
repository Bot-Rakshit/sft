#!/usr/bin/env python3
"""
Smart Chess Agent - Adds intelligence on top of base SFT model
No retraining needed - pure inference improvements
"""
import chess
import chess.engine
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from typing import Optional, List, Tuple

class SmartChessAgent:
    def __init__(self, model_path: str, stockfish_path: str, use_search: bool = True):
        """
        Args:
            model_path: Path to SFT model
            stockfish_path: Path to Stockfish
            use_search: Enable 1-ply search for move validation
        """
        print(f"Loading SFT model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        self.use_search = use_search
        if use_search:
            print(f"Initializing Stockfish: {stockfish_path}")
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        else:
            self.engine = None
        
        self.move_history = []
    
    def _get_base_model_move(self, board: chess.Board, temperature: float = 0.3) -> Optional[str]:
        """Get move from base SFT model"""
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
                temperature=temperature,
                do_sample=True,
                top_k=10
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        # Parse move
        move_match = re.search(r"<uci_move>(.*?)</uci_move>", response)
        if move_match:
            move_str = move_match.group(1).strip()
            if move_str in legal_moves:
                return move_str
        
        # Fallback: find any legal move in response
        for move in legal_moves:
            if move in response:
                return move
        
        return None
    
    def _evaluate_move(self, board: chess.Board, move_uci: str) -> int:
        """Evaluate position after move (in centipawns)"""
        if not self.engine:
            return 0
        
        try:
            test_board = board.copy()
            test_board.push(chess.Move.from_uci(move_uci))
            
            info = self.engine.analyse(test_board, chess.engine.Limit(depth=8))
            score = info["score"].white()
            
            if score.is_mate():
                return 10000 if score.mate() > 0 else -10000
            return score.score()
        except:
            return 0
    
    def _is_hanging_piece(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move hangs a piece"""
        if not self.engine:
            return False
        
        try:
            # Evaluate current position
            current_eval = self._evaluate_position(board)
            
            # Make move
            test_board = board.copy()
            test_board.push(move)
            
            # Evaluate after move
            after_eval = self._evaluate_position(test_board)
            
            # Flip if black's turn
            if board.turn == chess.BLACK:
                current_eval = -current_eval
                after_eval = -after_eval
            
            # Loss > 200cp = hanging piece
            loss = current_eval - after_eval
            return loss > 200
        except:
            return False
    
    def _evaluate_position(self, board: chess.Board) -> int:
        """Quick position evaluation"""
        if not self.engine:
            return 0
        
        try:
            info = self.engine.analyse(board, chess.engine.Limit(depth=8))
            score = info["score"].white()
            
            if score.is_mate():
                return 10000 if score.mate() > 0 else -10000
            return score.score()
        except:
            return 0
    
    def _get_top_moves(self, board: chess.Board, n: int = 3) -> List[str]:
        """Get top N moves from engine"""
        if not self.engine:
            return []
        
        try:
            info = self.engine.analyse(board, chess.engine.Limit(depth=10), multipv=n)
            return [entry["pv"][0].uci() for entry in info]
        except:
            return []
    
    def _avoid_repetition(self, board: chess.Board, move_uci: str) -> bool:
        """Check if move leads to repetition"""
        test_board = board.copy()
        try:
            test_board.push(chess.Move.from_uci(move_uci))
            return test_board.is_repetition(count=2)
        except:
            return False
    
    def get_move(self, board: chess.Board, use_safety_checks: bool = True) -> Optional[str]:
        """
        Get best move with safety checks and search
        
        Args:
            board: Current board state
            use_safety_checks: Enable tactical safety checks
        
        Returns:
            UCI move string
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # Try to get move from base model
        base_move = self._get_base_model_move(board)
        
        # If no valid move from model, use engine
        if base_move is None:
            if self.engine:
                result = self.engine.play(board, chess.engine.Limit(time=0.1))
                return result.move.uci()
            return legal_moves[0].uci()
        
        # Safety checks
        if use_safety_checks and self.engine:
            move_obj = chess.Move.from_uci(base_move)
            
            # Check 1: Does this hang a piece?
            if self._is_hanging_piece(board, move_obj):
                print(f"  [Safety] Base move {base_move} hangs piece, trying alternatives...")
                
                # Get top engine moves
                top_moves = self._get_top_moves(board, n=5)
                for alt_move in top_moves:
                    alt_obj = chess.Move.from_uci(alt_move)
                    if not self._is_hanging_piece(board, alt_obj):
                        print(f"  [Safety] Using engine move {alt_move} instead")
                        return alt_move
                
                # If all hang pieces, use least bad
                print(f"  [Safety] All moves hang pieces, using base move")
            
            # Check 2: Avoid repetition
            if self._avoid_repetition(board, base_move):
                print(f"  [Safety] Move {base_move} causes repetition, trying alternatives...")
                top_moves = self._get_top_moves(board, n=3)
                for alt_move in top_moves:
                    if not self._avoid_repetition(board, alt_move):
                        print(f"  [Safety] Using non-repetitive move {alt_move}")
                        return alt_move
            
            # Check 3: Validate with 1-ply search
            if self.use_search:
                current_eval = self._evaluate_position(board)
                move_eval = self._evaluate_move(board, base_move)
                
                # Flip for black
                if board.turn == chess.BLACK:
                    current_eval = -current_eval
                    move_eval = -move_eval
                
                centipawn_loss = current_eval - move_eval
                
                # If blunder (>150cp loss), try engine move
                if centipawn_loss > 150:
                    print(f"  [Search] Base move loses {centipawn_loss}cp, using engine move")
                    top_moves = self._get_top_moves(board, n=1)
                    if top_moves:
                        return top_moves[0]
        
        self.move_history.append(base_move)
        return base_move
    
    def close(self):
        """Close engine connection"""
        if self.engine:
            self.engine.quit()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--stockfish", type=str, default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--fen", type=str, help="Test position FEN")
    args = parser.parse_args()
    
    agent = SmartChessAgent(args.model, args.stockfish, use_search=True)
    
    if args.fen:
        board = chess.Board(args.fen)
    else:
        board = chess.Board()
    
    print(f"\nPosition: {board.fen()}")
    print(f"Legal moves: {[m.uci() for m in board.legal_moves]}")
    
    move = agent.get_move(board, use_safety_checks=True)
    print(f"\nSelected move: {move}")
    
    agent.close()
