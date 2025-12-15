#!/bin/bash
set -e

# Configuration
# Configuration
# INTENTION: Set your Hugging Face token in your environment, e.g. export HF_TOKEN=...
if [ -z "$HF_TOKEN" ]; then
    echo "Using public access (or cached token) since HF_TOKEN is not set."
fi
REPO_ID="NeuralNinjasConnector/Connector-Llama"
FIXED_MODEL_DIR="./output/connector_model/fixed_remote"

echo "======================================================================"
echo "RUNNING REMOTE FIX PIPELINE"
echo "Repo: $REPO_ID"
echo "Output: $FIXED_MODEL_DIR"
echo "======================================================================"

# 1. Install necessary libraries if missing
echo "[1/3] Checking dependencies..."
# 1. Install necessary libraries if missing
echo "[1/3] Checking dependencies..."
# Use python3 -m pip to ensure we install for the python interpreter we are using
python3 -m pip install --user -q -r requirements.txt
python3 -m pip install --user -q huggingface_hub[cli,hf_transfer]

# Enable fast downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# 2. Run Fix Script (Download + Fix)
echo ""
echo "[2/3] Downloading and fixing model..."
# Note: Input path is the remote Repo ID. The script now handles this.
python3 fix_embeddings.py \
  --input_path "$REPO_ID" \
  --output_path "$FIXED_MODEL_DIR"

# 3. Run Evaluation on Fixed Model
echo ""
echo "[3/3] Evaluating fixed model..."
python3 baseline/logiqa_baseline.py --model_name "$FIXED_MODEL_DIR"

echo ""
echo "======================================================================"
echo "PIPELINE COMPLETE"
echo "If successful, check the accuracy above."
echo "======================================================================"
