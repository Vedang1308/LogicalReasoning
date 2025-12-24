# Connector Aware Pretraining of LLM

## Project Overview
This project focuses on enhancing logical reasoning capabilities in language models through a "Gentle Retraining" approach and dynamic inference adjustments. The system uses a **Llama-3.2-3B** base model and applies targeted training to improve performance on logical reasoning tasks (LogiQA).

## Development Phases

### 1. Initial Project Scope
The core research focused on the fundamental question of improving logical reasoning without sacrificing general capability.
*   **Methodology**: Design of the "Gentle Retraining" methodology using LoRA.
*   **Evaluation**: Baseline comparisons and training loop implementation on the LogiQA dataset.

### 2. Ongoing Improvements
Following the initial research, the project has been expanded with engineering tools to facilitate easier usage and analysis.
*   **Web Dashboard**: A Streamlit-based control center (`app.py`) for monitoring training and visualizing results.
*   **Dynamic Inference**: Real-time token boosting during inference to emphasize logical connectors.
*   **Engineering Utilities**: Automated environment setup, resume capabilities, and improved error handling.

## Installation

### Prerequisites
*   python 3.10+
*   Conda (Recommended)

### Setup
Run the setup script to configure the environment:

```bash
./setup_env.sh
```

Activate the environment:
```bash
conda activate nlp_fix_env
```

## Usage

### Web Interface
To launch the control dashboard:

```bash
./run_app.sh
```

If running remotely, use SSH tunneling to access `localhost:8501`.

### Command Line Tools
*   **Training**: `./run_retrain.sh`
*   **Evaluation**: `./run_comparison.sh`

## Project Structure
*   `app.py`: Dashboard application.
*   `pretrain/`: Core training logic and model definitions.
*   `baseline/`: LogiQA evaluation scripts.
*   `checkpoints/`: Directory for saving model weights.
*   `setup_env.sh`: Environment setup automation.
