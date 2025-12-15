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

# Extract accuracy for Baseline model
BASELINE_ACC=$(python3 -c "import json; print(f\"{json.load(open('$OUTPUT_DIR/logiqa_results_Llama-3.2-3B.json'))['accuracy']:.4f}\")" 2>/dev/null || echo "N/A")
BASELINE_SAMPLES=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/logiqa_results_Llama-3.2-3B.json'))['total_examples'])" 2>/dev/null || echo "0")


# 3. Evaluate Dynamic Boost Model
echo ""
echo "[3/4] Evaluating DYNAMIC model (Baseline + Dynamic 1.1x Hook)..."
python3 -u baseline/logiqa_baseline.py \
    --model_name meta-llama/Llama-3.2-3B \
    --use_dynamic_boost \
    --boost_factor 1.1 \
    --output_dir baseline_results \
    --max_samples 1000 # Evaluate full set

# Extract accuracy for Dynamic model
DYNAMIC_ACC=$(grep "Accuracy:" baseline_results/summary.txt | tail -n 1 | awk '{print $2}')
DYNAMIC_SAMPLES=$(grep "Accuracy:" baseline_results/summary.txt | tail -n 1 | awk -F'[(/]' '{print $3}')

# --- 4. COMPARE RESULTS ---
echo ""
echo "[4/4] Final Comparison:"
echo "======================================================================"
echo "  MODEL                       | ACCURACY | SAMPLES"
echo "------------------------------|----------|---------"
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
