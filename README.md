# Connector Aware Pretraining of LLM (CSE 576 Topics in NLP)

##  Project Overview
This is a specialized NLP project focused on enhancing logical reasoning capabilities in language models through a "Gentle Retraining" approach and dynamic inference adjustments. The system uses a **Llama-3.2-3B** base model and applies targeted training to improve performance on logical reasoning tasks (specifically LogiQA).

This repository contains the complete codebase for:
- **web Interface**: A streamlined Streamlit dashboard for controlling training, visualization, and interaction.
- **Training Pipeline**: Custom training loop for "Gentle Retraining" to preserve general capabilities while boosting logic.
- **Evaluation**: Tools to compare the base model against the retrained model.

##  Features

###  Interactive Dashboard (`app.py`)
The heart of the project is the **NeuralNinjas Control Center**, a single-screen interface that allows you to:
- **Monitor Training**: View real-time logs and status of training jobs.
- **Control Execution**: seamless "Resume" and "Fresh Start" capabilities for training runs.
- **Evaluate**: Run and visualize head-to-head comparisons between models.
- **Playground**: Type text and see the "Dynamic Logic Boost" in action with real-time token highlighting.

###  Training & Modeling
- **Base Model**: `meta-llama/Llama-3.2-3B`
- **Methodology**: 
    - **Gentle Retraining**: Low-rank adaptation (LoRA) or selective fine-tuning to inject logic capabilities without catastrophic forgetting.
    - **Dynamic Inference**: At inference time, logical connectors (e.g., "therefore", "because") are dynamically boosted to emphasize reasoning paths.

##  Installation & Setup

### Prerequisites
- **Python 3.8+** (Python 3.10 recommended)
- **Conda** (Optional but recommended for environment management)

### Automatic Setup
We provide a setup script to configure your environment automatically.

```bash
./setup_env.sh
```

This script will:
1. Check your Python version.
2. Create a Conda environment `nlp_fix_env` (if needed).
3. Install all dependencies from `requirements.txt`.

### Activation
After setup, activate the environment:
```bash
conda activate nlp_fix_env
```

##  Usage

### Running the Web Interface
To start the Control Center dashboard:

```bash
./run_app.sh
```

**Remote Access (SSH)**:
If you are running this on a remote server (e.g., a university supercomputer), use SSH tunneling to view the dashboard on your local machine:

```bash
# Run this on your LOCAL machine
ssh -L 8501:localhost:8501 your_username@remote_host_address
```
Then open `http://localhost:8501` in your browser.

### Command Line Utilities
While the Dashboard is the preferred way to interact, you can also run individual components manually:

- **Run Comparison/Evaluation**:
  ```bash
  ./run_comparison.sh
  ```
- **Run Training**:
  ```bash
  ./run_retrain.sh
  ```

##  Project Structure

- **`app.py`**: Main Streamlit application entry point.
- **`pretrain/`**: Contains training logic (`trainer.py`) and model definitions.
- **`baseline/`**: Baseline model evaluation code (`logiqa_baseline.py`).
- **`utils/`**: Helper scripts and configuration files.
- **`checkpoints/`**: Directory where model weights and training metadata are saved.
- **`setup_env.sh`**: Environment installation script.

---
*Created for CSE 576 Topics in NLP*
