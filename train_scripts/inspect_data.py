
from datasets import load_dataset
dataset = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)
print(next(iter(dataset)).keys())
print(next(iter(dataset)))
