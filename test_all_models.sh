#!/bin/bash

echo "Testing All Chess Models"
echo "========================"
echo ""

MODELS=(
    "qwen-chess-0.5b-merged:Qwen/Qwen2.5-0.5B-Instruct"
    "qwen-chess-0.5b-merged-bc-500:Qwen/Qwen2.5-0.5B-Instruct"
    "qwen-chess-0.5b-108k-merged:Qwen/Qwen2.5-0.5B-Instruct"
    "qwen-chess-3b-merged-bc-80k:Qwen/Qwen2.5-3B-Instruct"
)

for MODEL_INFO in "${MODELS[@]}"; do
    IFS=':' read -r MODEL BASE <<< "$MODEL_INFO"
    
    if [ -d "$MODEL" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Testing: $MODEL"
        echo "Base: $BASE"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        python3 train_scripts/test_model.py \
            --model "$MODEL" \
            --base-model "$BASE" 2>&1 | grep -E "(Position|Agent Response|Selected Move|Legal Move|Illegal Move)" | head -20
        
        echo ""
        echo ""
    else
        echo "⚠️  Model not found: $MODEL"
        echo ""
    fi
done

echo "Testing Complete!"
