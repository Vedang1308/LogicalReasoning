#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# Script to run retraining on the Supercomputer
# It handles environment setup and execution

# 1. Source environment (just in case)
source /home/vavaghad/miniconda/bin/activate
# Or if using setup_env.sh
./setup_env.sh

# 2. Set Token (replace with your token if not set)
# export HF_TOKEN="hf_..." 

# 4. Run Training
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
