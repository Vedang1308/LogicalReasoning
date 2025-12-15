#!/bin/bash
set -e

# Configuration
# Configuration
# export HF_TOKEN="your_token_here" # Do not hardcode secrets!
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN is not set. Please set it using: export HF_TOKEN=your_token"
    # Optional: prompt user for token if not set? For now, just warn.
fi
BASELINE_MODEL="meta-llama/Llama-3.2-3B"
FIXED_MODEL="./output/connector_model/fixed_remote"
OUTPUT_DIR="./baseline_results"

echo "======================================================================"
echo "RUNNING FULL MODEL COMPARISON"
echo "Machine: $(uname -s) $(uname -m)"
echo "Baseline: $BASELINE_MODEL"
echo "Fixed:    $FIXED_MODEL"
echo "======================================================================"

# 1. Dependency Check
echo ""
echo "[1/4] Checking dependencies..."
python3 -m pip install --user -q -r requirements.txt
python3 -m pip install --user -q huggingface_hub[cli,hf_transfer]
export HF_HUB_ENABLE_HF_TRANSFER=1

# 2. Evaluate Baseline
echo ""
echo "[2/4] Evaluating BASELINE model ($BASELINE_MODEL)..."
# Check if we already have results to save time, otherwise run
if [ -f "$OUTPUT_DIR/logiqa_results_Llama-3.2-3B.json" ]; then
    echo "Using existing baseline results."
else
    python3 -u baseline/logiqa_baseline.py --model_name "$BASELINE_MODEL" --output_dir "$OUTPUT_DIR"
fi

# 3. Evaluate Fixed Model
echo ""
echo "[3/4] Evaluating FIXED model ($FIXED_MODEL)..."
# Ensure the model exists (the remote download script should have been run)
if [ ! -d "$FIXED_MODEL" ]; then
    echo "⚠️  Fixed model not found! Running remote fix pipeline first..."
    ./run_remote_fix.sh
fi
python3 -u baseline/logiqa_baseline.py --model_name "$FIXED_MODEL" --output_dir "$OUTPUT_DIR"

# 4. Compare Results
echo ""
echo "[4/4] Final Comparison:"
echo "======================================================================"
echo "  MODEL                       | ACCURACY | SAMPLES"
echo "------------------------------|----------|---------"

# Extract accuracy using python one-liner
BASE_ACC=$(python3 -c "import json; print(f\"{json.load(open('$OUTPUT_DIR/logiqa_results_Llama-3.2-3B.json'))['accuracy']:.4f}\")" 2>/dev/null || echo "N/A")
FIX_ACC=$(python3 -c "import json; print(f\"{json.load(open('$OUTPUT_DIR/logiqa_results_fixed_remote.json'))['accuracy']:.4f}\")" 2>/dev/null || echo "N/A")

BASE_COUNT=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/logiqa_results_Llama-3.2-3B.json'))['total_examples'])" 2>/dev/null || echo "0")
FIX_COUNT=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/logiqa_results_fixed_remote.json'))['total_examples'])" 2>/dev/null || echo "0")

echo "  Baseline (Llama-3.2-3B)     | $BASE_ACC   | $BASE_COUNT"
echo "  Fixed (Connector Boost)     | $FIX_ACC   | $FIX_COUNT"
echo "======================================================================"

# Improvement check
if [ "$BASE_ACC" != "N/A" ] && [ "$FIX_ACC" != "N/A" ]; then
    DIFF=$(echo "$FIX_ACC - $BASE_ACC" | bc -l)
    if (( $(echo "$DIFF > 0" | bc -l) )); then
        echo "✅ IMPROVEMENT: +$DIFF"
    else
        echo "⚠️  NO IMPROVEMENT: $DIFF"
    fi
fi
echo ""
