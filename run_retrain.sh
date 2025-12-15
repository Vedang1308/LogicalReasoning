#!/bin/bash
# Script to run retraining on the Supercomputer
# It handles environment setup and execution

# 1. Source environment (just in case)
source /home/vavaghad/miniconda/bin/activate
# Or if using setup_env.sh
./setup_env.sh

# 2. Set Token (replace with your token if not set)
# export HF_TOKEN="hf_..." 

# 3. Clean old checkpoints (optional, but good for fresh start)
echo "Removing old local checkpoints..."
rm -rf checkpoints/

# 4. Run Training
# --hf-repo-id: Where to push the result
# --num-epochs: 1 epoch is likely enough given we are just finetuning
echo "Starting Retraining (Boost 1.01x, LR 2e-6)..."
python3 -u main.py \
    --hf-repo-id "NeuralNinjasConnector/Connector-Llama" \
    --num-epochs 1

echo "Retraining Complete!"
