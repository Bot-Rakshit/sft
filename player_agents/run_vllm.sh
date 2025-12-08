#!/bin/bash
set -euo pipefail

# Fallback wrapper: instead of vLLM, run the transformers Flask server.
# Dependencies are assumed preinstalled via requirements.txt.

MODEL_NAME="bot-rakshit/qwen-chess-0.5b-sft-v1"
echo "Starting transformers agent (vllm fallback) with model ${MODEL_NAME} on port 5000"
python3 player_agents/transformers_agent_flask_server.py --model "$MODEL_NAME" --port 5000