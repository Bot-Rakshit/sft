import argparse
import json
import csv
import zstandard as zstd
from io import StringIO

def extract_puzzles(puzzle_file, output_file, rating_ranges):
    """Extract puzzles from Lichess database"""
    print(f"Extracting puzzles from {puzzle_file}...")
    print(f"Target distribution:")
    for rating_range, (min_r, max_r, target_count) in rating_ranges.items():
        print(f"  {rating_range}: {target_count} puzzles ({min_r}-{max_r} rating)")
    
    puzzles_by_rating = {r: [] for r in rating_ranges.keys()}
    
    dctx = zstd.ZstdDecompressor()
    
    total_processed = 0
    
    with open(puzzle_file, 'rb') as compressed:
        with dctx.stream_reader(compressed) as reader:
            text_stream = StringIO(reader.read().decode('utf-8'))
            csv_reader = csv.DictReader(text_stream)
            
            for row in csv_reader:
                total_processed += 1
                
                if total_processed % 10000 == 0:
                    total_collected = sum(len(puzzles_by_rating[r]) for r in rating_ranges)
                    print(f"Processed {total_processed} puzzles, collected {total_collected}...")
                
                try:
                    rating = int(row['Rating'])
                    fen = row['FEN']
                    themes = row.get('Themes', '')
                    
                    for rating_range, (min_r, max_r, target_count) in rating_ranges.items():
                        if min_r <= rating < max_r and len(puzzles_by_rating[rating_range]) < target_count:
                            puzzle_data = {
                                "fen": fen,
                                "rating": rating,
                                "themes": themes,
                                "phase": "puzzle"
                            }
                            puzzles_by_rating[rating_range].append(puzzle_data)
                            break
                
                except Exception as e:
                    continue
                
                if all(len(puzzles_by_rating[r]) >= rating_ranges[r][2] for r in rating_ranges):
                    print("All puzzle targets reached!")
                    break
    
    all_puzzles = []
    for rating_range, puzzles in puzzles_by_rating.items():
        print(f"Extracted {len(puzzles)} puzzles from {rating_range}")
        all_puzzles.extend(puzzles)
    
    print(f"\nWriting {len(all_puzzles)} puzzles to {output_file}...")
    with open(output_file, 'w') as f:
        for puzzle in all_puzzles:
            f.write(json.dumps(puzzle) + '\n')
    
    print("Done!")
    return all_puzzles

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle-file", type=str, default="lichess_db_puzzle.csv.zst")
    parser.add_argument("--output", type=str, default="puzzles_8k.jsonl")
    args = parser.parse_args()
    
    rating_ranges = {
        "below_1000": (0, 1000, 1000),
        "1000_1500": (1000, 1500, 1000),
        "1500_2000": (1500, 2000, 1500),
        "2000_2500": (2000, 2500, 2000),
        "2500_plus": (2500, 4000, 2500)
    }
    
    extract_puzzles(args.puzzle_file, args.output, rating_ranges)
