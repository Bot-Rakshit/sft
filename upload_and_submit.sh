#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "AIcrowd Chess Challenge Submission"
echo "=========================================="
echo ""

# Configuration
MODEL_DIR="qwen-chess-0.5b-108k-merged-new"
HF_REPO_NAME="bot-rakshit/qwen-chess-0.5b-108k-v1"
CHALLENGE="global-chess-challenge-2025"

# Check if model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model directory '$MODEL_DIR' not found!"
    echo "Available models:"
    ls -d qwen-chess-* 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "📦 Step 1: Uploading model to HuggingFace"
echo "Model: $MODEL_DIR"
echo "HF Repo: $HF_REPO_NAME"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Install huggingface_hub if needed
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    pip3 install --user huggingface_hub
fi

# Upload to HuggingFace
echo "Uploading model to HuggingFace..."
python3 << EOF
from huggingface_hub import HfApi, login
import os

api = HfApi()

# Login (will prompt if needed)
try:
    login()
except:
    print("Please login to HuggingFace:")
    login()

# Upload model
print(f"Uploading {os.path.abspath("$MODEL_DIR")} to $HF_REPO_NAME...")
api.upload_folder(
    folder_path="$MODEL_DIR",
    repo_id="$HF_REPO_NAME",
    repo_type="model",
    commit_message="Upload 108K trained chess model"
)
print("✅ Upload complete!")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Upload failed!"
    exit 1
fi

echo ""
echo "📤 Step 2: Submitting to AIcrowd"
echo ""

# Update submission script
cat > aicrowd_submit_updated.sh << EOF
#!/bin/bash

aicrowd login

# Configuration variables
CHALLENGE="$CHALLENGE"
HF_REPO="$HF_REPO_NAME"
HF_REPO_TAG="main"
PROMPT_TEMPLATE="player_agents/llm_agent_prompt_template.jinja"

echo "Submitting $HF_REPO"
echo "If the repo is private, make sure to add access to aicrowd."
echo "Details can be found in docs/huggingface-gated-models.md"

# Submit the model
aicrowd submit-model \\
    --challenge "\$CHALLENGE" \\
    --hf-repo "\$HF_REPO" \\
    --hf-repo-tag "\$HF_REPO_TAG" \\
    --prompt_template_path "\$PROMPT_TEMPLATE" \\
    --inference-config-path "player_agents/run_transformers.sh"
EOF

chmod +x aicrowd_submit_updated.sh

echo "✅ Submission script created: aicrowd_submit_updated.sh"
echo ""
echo "To submit, run:"
echo "  ./aicrowd_submit_updated.sh"
echo ""
echo "Or manually run:"
echo "  aicrowd submit-model \\"
echo "    --challenge '$CHALLENGE' \\"
echo "    --hf-repo '$HF_REPO_NAME' \\"
echo "    --hf-repo-tag 'main' \\"
echo "    --prompt_template_path 'player_agents/llm_agent_prompt_template.jinja' \\"
echo "    --inference-config-path 'player_agents/run_transformers.sh'"
