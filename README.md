# 🧠 NeuralNinjas: NeuroLogic Agent
### Connector Aware Pretraining of LLM (CSE 576 Topics in NLP)

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research-orange)
![Base Model](https://img.shields.io/badge/base%20model-Llama--3.2--3B-purple)

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Team](#-team)

---

## 🚀 Project Overview

**NeuroLogic Agent** is a specialized NLP research project focused on enhancing logical reasoning capabilities in language models. We employ a novel **"Gentle Retraining"** approach combined with **Dynamic Inference** adjustments to boost performance on logical reasoning tasks (LogiQA) without compromising general language understanding.

This repository hosts the complete ecosystem:
*   **Web Control Center**: A centralized dashboard for training and evaluation.
*   **Custom Training Loop**: Implementation of targeted fine-tuning.
*   **Evaluation Suite**: Tools for benchmarking and model comparison.

---

## ✨ Features

### 🎮 interactive Dashboard (`app.py`)
A single-screen command center designed for efficiency:
*   **Real-time Monitoring**: Stream training logs directly to the UI.
*   **Execution Control**: Seamlessly Pause, Resume, or Restart training runs.
*   **Visual Evaluation**: Plot comparative performance metrics (Base vs. Retrained).
*   **Logic Playground**: Type sentences and watch the "Dynamic Logic Boost" highlight reasoning connectors in real-time.

### 🧠 Training & Modeling
*   **Base Model**: `meta-llama/Llama-3.2-3B`
*   **Techniques**:
    *   **Gentle Retraining**: Using Low-Rank Adaptation (LoRA) to inject logic awareness.
    *   **Dynamic Inference**: Runtime boosting of logical connectors (e.g., "therefore", "unless") to sharpen reasoning paths.

---

## 🛠 Installation & Setup

### Prerequisites
*   **Python 3.10+**
*   **Conda** (Recommended)

### Quick Start
We provide an automated setup script to handle environment creation and dependency installation.

```bash
# 1. Clone the repository
git clone https://github.com/Vedang1308/LogicalReasoning.git
cd LogicalReasoning

# 2. Run the setup script
./setup_env.sh
```

**Manual Activation**:
```bash
conda activate nlp_fix_env
```

---

## 🖥 Usage

### 1. Web Interface (Recommended)
Launch the comprehensive dashboard:

```bash
./run_app.sh
```

**Remote Access (SSH)**:
If running on a headless server, forward the port to your local machine:
```bash
ssh -L 8501:localhost:8501 your_user@remote_host
# Open http://localhost:8501 in your browser
```

### 2. Command Line Tools
For headless operation or batch processing:

| Script | Purpose |
| :--- | :--- |
| `./run_retrain.sh` | Start or resume the Gentle Retraining pipeline. |
| `./run_comparison.sh` | Run the evaluation suite on LogiQA. |

---

## 📂 Project Structure

```text
CSE_576_TOPICS_IN_NLP_MAIN/
├── app.py                  # Streamlit Dashboard Entry Point
├── setup_env.sh            # Environment Setup Script
├── requirements.txt        # Python Dependencies
├── pretrain/               # Training Source Code
│   ├── trainer.py          # Custom Training Loop
│   └── ...
├── baseline/               # Evaluation Source Code
│   ├── logiqa_baseline.py  # Zero-shot Evaluator
│   └── ...
├── checkpoints/            # Model Weights & Metadata
└── utils/                  # Helper Scripts
```

---

## 👥 Team
**Course**: CSE 576 Topics in NLP  
**Project**: NeuroLogic Agent

*Created with ❤️ by the NeuralNinjas Team*
