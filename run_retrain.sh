#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# Script to run retraining on the Supercomputer
# It handles environment setup and execution

# 1. Source environment (Try standard paths, but don't fail)
if [ -f "/home/vavaghad/miniconda/bin/activate" ]; then
    source /home/vavaghad/miniconda/bin/activate
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
    source "$HOME/miniconda3/bin/activate"
elif [ -f "$HOME/anaconda3/bin/activate" ]; then
    source "$HOME/anaconda3/bin/activate"
fi

# Activate base env if possible without erroring
conda activate base 2>/dev/null || true

# Just to be safe, print python version
echo "Using Python: $(which python3)"
python3 --version

# 2. Setup Env (Ensures dependencies)
# Run with bash to prevent permission issues if executable bit not set
bash setup_env.sh

# 2.5 CLEANUP & CONFIG
echo "Cleaning up zombi processes on GPU..."
echo "GPU State BEFORE cleanup:"
nvidia-smi || echo "nvidia-smi not found"

# Force Kill (SIGKILL) any existing main.py processes owned by this user
echo "Killing old main.py processes..."
pkill -9 -u "$(whoami)" -f main.py || true
sleep 5 # Give them time to yield resources

echo "GPU State AFTER cleanup:"
nvidia-smi || echo "nvidia-smi not found"

# Optimize memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. Set Token (Passed from App or Environment)
# Do NOT hardcode secrets here.
# export HF_TOKEN="hf_..." 

export HF_REPO_ID="NeuralNinjasConnector/Connector-Llama"

# 3. Run Training
# Check if passed argument is --resume
if [ "$1" == "--resume" ]; then
    echo "Resuming Training from Checkpoint..."
    # Don't delete checkpoints folder if resuming!
    python3 -u main.py \
        --hf-repo-id "NeuralNinjasConnector/Connector-Llama" \
        --num-epochs 1 \
        --resume-training
else
    echo "Starting FRESH Training (Cleaning old checkpoints)..."
    rm -rf checkpoints/
    python3 -u main.py \
        --hf-repo-id "NeuralNinjasConnector/Connector-Llama" \
        --num-epochs 1
fi

echo "Retraining Complete!"
