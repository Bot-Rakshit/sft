MODEL_NAME_OR_PATH="bot-rakshit/qwen-chess-0.5b-sft-v1"

# The evaluator runs this script. We need to ensure it uses OUR model.
# Qwen 2.5 0.5B might have issues with some vLLM versions if not configured right.
# We explicitly set the model name to what AIcrowd expects.

vllm serve $MODEL_NAME_OR_PATH \
    --served-model-name aicrowd-chess-model \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --disable-log-stats \
    --host 0.0.0.0 \
    --port 5000
