#!/bin/bash

echo "============================================================"
echo "   Auto-Setup for NLP Environment (Conda/Python)"
echo "============================================================"

# Check Python version
PY_VER=$(python3 -c"import sys; print(sys.version_info.major, sys.version_info.minor)")
MAJOR=$(echo $PY_VER | cut -d' ' -f1)
MINOR=$(echo $PY_VER | cut -d' ' -f2)

echo "Current Python: $MAJOR.$MINOR"

if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 8 ]; then
    echo "✅ Python version looks good ($MAJOR.$MINOR >= 3.8)"
else
    echo "❌ Python version is too old ($MAJOR.$MINOR). PyTorch 2.0+ requires Python 3.8+"
    echo "   Attempting to find Conda..."
    
    if command -v conda &> /dev/null; then
        echo "✓ Conda found."
        echo "Creating new environment 'nlp_fix_env' with Python 3.10..."
        conda create -n nlp_fix_env python=3.10 -y
        
        echo ""
        echo "============================================================"
        echo "SETUP COMPLETE!"
        echo "Run the following commands to start:"
        echo ""
        echo "    conda activate nlp_fix_env"
        echo "    ./run_comparison.sh"
        echo "============================================================"
        exit 0
    else
        echo "⚠️  Conda not found!"
        echo "Please load a newer python module manually, e.g.:"
        echo "    module load python/3.10"
        echo "    module load cuda/11.8"
        exit 1
    fi
fi

# If we are here, Python is good. Try installing dependencies.
echo "Installing dependencies..."
# Install requirements
python3 -m pip install --user -r requirements.txt
python3 -m pip install --user streamlit plotly altair
python3 -m pip install --user huggingface_hub[cli,hf_transfer]

echo ""
echo "Ready! Run:"
echo "    ./run_comparison.sh"
