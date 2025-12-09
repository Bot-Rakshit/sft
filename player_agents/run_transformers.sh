#!/bin/bash
set -euo pipefail

# Dependencies are preinstalled via requirements.txt; avoid runtime pip (offline sandbox).

MODEL_NAME="bot-rakshit/qwen-chess-0.5b-108k-v1"
echo "Starting transformers agent with model ${MODEL_NAME} on port 5000"
python3 player_agents/transformers_agent_flask_server.py --model "$MODEL_NAME" --port 5000
