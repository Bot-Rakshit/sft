import json
import argparse

def simplify_training_data(input_file, output_file):
    """Convert complex training data to simpler format matching original working model"""
    print(f"Simplifying {input_file}...")
    
    simplified = 0
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line_num, line in enumerate(fin):
            if line_num % 10000 == 0:
                print(f"Processed {line_num} examples...")
            
            try:
                data = json.loads(line)
                
                user_content = data['messages'][0]['content']
                assistant_content = data['messages'][1]['content']
                
                lines = user_content.split('\n')
                fen_line = None
                legal_moves_line = None
                
                for i, l in enumerate(lines):
                    if l.strip() and not l.startswith('You are') and not l.startswith('Legal') and not l.startswith('Position') and not l.startswith('-') and not l.startswith('Select') and not l.startswith('Format') and not l.startswith('<'):
                        if fen_line is None:
                            fen_line = l.strip()
                    if l.startswith('Legal moves:'):
                        legal_moves_line = l.strip()
                
                if not fen_line or not legal_moves_line:
                    continue
                
                move_match = assistant_content.split('<uci_move>')
                if len(move_match) < 2:
                    continue
                move = move_match[1].split('</uci_move>')[0].strip()
                
                eval_match = assistant_content.split('eval ')
                if len(eval_match) >= 2:
                    eval_str = eval_match[1].split('cp')[0].strip()
                    eval_val = float(eval_str) / 100.0
                else:
                    eval_val = 0.0
                
                simple_prompt = f"""You are an expert chess player. Here is the position in FEN format:
{fen_line}

{legal_moves_line}

Select the best move. Keep your thinking to 2 sentences or less, then output your chosen move.
Format:
<think>brief thinking (2 sentences max)</think>
<uci_move>your_move</uci_move>"""

                simple_response = f"<think>The position evaluation is Evaluation {eval_val:+.2f}. The best move identified by engine analysis is {move}.</think><uci_move>{move}</uci_move>"
                
                simplified_data = {
                    "messages": [
                        {"role": "user", "content": simple_prompt},
                        {"role": "assistant", "content": simple_response}
                    ]
                }
                
                fout.write(json.dumps(simplified_data) + '\n')
                simplified += 1
                
            except Exception as e:
                print(f"Error on line {line_num}: {e}")
                continue
    
    print(f"\nDone! Simplified {simplified} examples")
    print(f"Output: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    simplify_training_data(args.input, args.output)
