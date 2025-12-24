# 🧠 NeuroLogic Agent: Connector Aware Pretraining
### CSE 576 Topics in NLP (Development Snapshot)

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research-orange)

---

## 📅 Project Timeline & Scope

**This repository contains the complete development history of the NeuroLogic Agent project, spanning two distinct phases:**

### 1. 🎓 The Semester Project (Academic Core)
The original academic work completed during the CSE 576 course. This phase focused on the fundamental research question: *Can we improve logical reasoning in small LLMs without sacrificing general capability?*
*   **Deliverable**: The final project report (uploaded separately).
*   **Core Contribution**: Design of the "Gentle Retraining" methodology using LoRA.
*   **Baseline**: Evaluation on the LogiQA dataset using standard metrics.

### 2. 🚀 Post-Semester Extensions (NeuralNinjas)
**Current Status**: Active Development
After the semester ended, the project was continued to transform the research code into a robust engineering product. All "NeuralNinjas" branding, advanced UI components, and interactive tools belong to this phase.
*   **Web Dashboard**: The `streamlist` based Control Center (`app.py`).
*   **Dynamic Inference**: Real-time token boosting for logical connectors.
*   **Engineering Polish**: Advanced error handling, resume capability, and Docker/Environment automation.

---

## ✨ Features (Full System)

### 🎨 Interactive Dashboard (Post-Semester)
The **NeuralNinjas Control Center** (`app.py`) provides a visual interface for the underlying research code:
*   **Monitor Training**: Real-time logs and loss curves.
*   **Control Execution**: Pause/Resume training jobs seamlessly.
*   **Playground**: Interactive text area demonstrating the **Dynamic Logic Boost** with visual highlighting.

### ⚙️ Training Pipeline (Core + Extensions)
*   **Base Model**: `meta-llama/Llama-3.2-3B`
*   **Methodology**:
    *   **Gentle Retraining**: Fine-tuning on logical connector-heavy datasets.
    *   **Dynamic Inference**: Runtime probability adjustment for reasoning tokens (e.g., "therefore", "thus").

---

## 🛠 Installation & Setup

### Prerequisites
*   **Python 3.10+** (Managed via Conda)
*   **CUDA/GPU Support** (Recommended for Training)

### Automatic Setup
Run the setup script to configure the environment:

```bash
./setup_env.sh
```

### Activation
```bash
conda activate nlp_fix_env
```

---

## 🖥 Usage

### Running the Dashboard
To launch the post-semester "NeuralNinjas" interface:

```bash
./run_app.sh
```

**Remote Access (SSH)**:
```bash
ssh -L 8501:localhost:8501 your_user@remote_host
```

### Command Line Tools
For pure research/training reproduction (Semester Scope):
*   **Training**: `./run_retrain.sh`
*   **Evaluation**: `./run_comparison.sh`

---

## 📂 Project Structure

```text
CSE_576_TOPICS_IN_NLP_MAIN/
├── app.py                  # [Extension] Streamlit Dashboard
├── pretrain/               # [Core] Training logic & Model definitions
│   └── trainer.py
├── baseline/               # [Core] LogiQA Evaluation
│   └── logiqa_baseline.py
├── checkpoints/            # [Mixed] Saved Models
└── setup_env.sh            # [Extension] Automation Script
```

---

## 👥 Team
**Course**: CSE 576 Topics in NLP  
**Project**: Connector Aware Pretraining
