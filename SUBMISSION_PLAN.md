# Global Chess Challenge Submission Plan

## 1. Tracks and Prizes
The challenge has a single leaderboard with a total prize pool of **$17,000** + **$8,000** compute credits.
*   **1st Place**: $10,000 + $5,000 credits
*   **2nd Place**: $5,000 + $2,000 credits
*   **3rd Place**: $2,000 + $1,000 credits

There are two main research tracks (approaches), but they compete for the same prizes based on the evaluation metrics (TrueSkill, Legality, etc.).

### Track 1: Data-Centric Finetuning (SFT)
*   **Approach**: Supervised Learning on chess databases (Lichess).
*   **Goal**: Train the model to predict the next move and explanation given a board position.
*   **Pros**: Easier to start, stable training.
*   **Cons**: Limited by the quality of the training data (imitation learning).

### Track 2: RLVR (Reinforcement Learning with Verifiable Rewards)
*   **Approach**: Use RL (PPO/GRPO) with Stockfish as a verifier.
*   **Goal**: Optimize the model to maximize a reward function (legality, engine evaluation, checkmate).
*   **Pros**: Can potentially surpass the training data quality.
*   **Cons**: Harder to tune, computationally expensive.

## 2. First Submission (Starter Kit)
We have verified the local environment using the Random Agent. Now, let's make the first submission to get on the leaderboard.

### Step 1: Prerequisites
1.  **Sign up** at [AIcrowd Global Chess Challenge](https://www.aicrowd.com/challenges/global-chess-challenge-2025).
2.  **Accept Rules** on the challenge page.
3.  **Get API Key** from your AIcrowd profile.
4.  **Hugging Face Account**: You need a model on HF.

### Step 2: Configuration
We will use a small public model (e.g., `Qwen/Qwen2.5-0.5B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct`) for the first submission to test the pipeline.

**Edit `aicrowd_submit.sh`**:
```bash
# Configuration variables
CHALLENGE="global-chess-challenge-2025"
HF_REPO="Qwen/Qwen2.5-1.5B-Instruct"  # Using a small instruct model
HF_REPO_TAG="main"
PROMPT_TEMPLATE="player_agents/llm_agent_prompt_template.jinja"
```

### Step 3: Submit
Run the submission script:
```bash
bash aicrowd_submit.sh
```
(You will be prompted to login if you haven't already).

## 3. Local Training on MacBook M4
We have successfully set up a local training pipeline for Apple Silicon (MPS).

### Setup
1.  Run `train_scripts/setup_env.sh` to install dependencies (torch, transformers, trl, peft).
2.  Run `python3 train_scripts/data_prep.py` to generate a small training dataset from Lichess.

### Training
Run the training script:
```bash
python3 train_scripts/train.py
```
*   **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
*   **Hardware**: M4 (MPS acceleration) with `bfloat16` precision.
*   **Method**: LoRA fine-tuning.
*   **Time**: ~45 minutes for 1 epoch on 2000 samples.

### Next Steps
1.  Complete the training run.
2.  Evaluate the trained adapter using `local_evaluation.py`.
3.  Upload the merged model to Hugging Face.
4.  Submit the fine-tuned model to the leaderboard.

## 4. Future Improvements (RLVR)
