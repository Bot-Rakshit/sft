#!/bin/bash
echo "Cleaning up unnecessary files..."

# Delete failed/bad models
rm -rf qwen-chess-0.5b-merged-bc-500
rm -rf qwen-chess-0.5b-108k-merged
rm -rf qwen-chess-3b-merged-bc-80k

# Delete unmerged adapters (keeping only merged models)
rm -rf qwen-chess-0.5b-sft*

# Delete old training data
rm -f train_data_boychesser*.jsonl
rm -f train_data_batch_*.jsonl
rm -f train_data_mini.jsonl
rm -f train_data_1k_test.jsonl
rm -f test_analyzed.jsonl

# Delete intermediate files
rm -f positions_*.jsonl
rm -f puzzles_*.jsonl
rm -f combined_*.jsonl

# Delete log files
rm -f *.log

# Keep only essential files:
# - qwen-chess-0.5b-merged (best working model)
# - train_data_108k_simple.jsonl (for retraining)
# - train_data_108k_complete.jsonl (backup)

echo "Cleanup complete!"
echo ""
echo "Kept models:"
ls -d qwen-chess-* 2>/dev/null
