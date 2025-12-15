import streamlit as st
import pandas as pd
import json
import glob
import os
import subprocess
import time
import plotly.express as px
from pathlib import Path

# Page Config
st.set_page_config(
    page_title="NeuralNinjas: Control Center",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #f0f2f6; }
    .stMetric { background-color: #262730; padding: 10px; border-radius: 5px; }
    .highlight-box { background-color: #262730; padding: 20px; border-radius: 10px; border: 1px solid #4b4b4b; margin-bottom: 20px; }
    .console-logs { font-family: 'Courier New', monospace; font-size: 12px; color: #00ff00; background-color: #000; padding: 10px; border-radius: 5px; height: 300px; overflow-y: scroll; }
</style>
""", unsafe_allow_html=True)

# Helper: Run Shell Script & Stream Logs
def run_command_with_streaming(command, log_container):
    """Runs a shell command and streams stdout to the Streamlit UI."""
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    output_log = ""
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            output_log += line
            # Keep only last 2000 chars to avoid UI lag
            display_log = output_log[-2000:] 
            log_container.code(display_log, language="bash")
            time.sleep(0.01) # Yield to UI
            
    return process.returncode

# Sidebar
with st.sidebar:
    st.title("🧠 NeuroLogic Agent")
    st.info("Interactive Control Center")
    
    page = st.radio("Navigation", ["Dashboard", "Training Control", "Evaluation & Compare", "Interactive Playground"])
    
    st.divider()
    st.caption("Environment: Supercomputer (Linux)")
    
    # Disable refresh if process running
    is_locked = st.session_state.get('training_active', False)
    if st.button("🔄 Refresh State", disabled=is_locked):
        st.rerun()

# --- 1. DASHBOARD ---
if page == "Dashboard":
    st.title("🚀 Project Dashboard")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Base Model", value="Llama-3.2-3B")
    with col2:
        st.metric(label="Current Boost", value="1.01x", delta="Safe Mode")
    with col3:
        st.metric(label="Logic Connectors", value="222", delta="Active")

    st.markdown("### 🎯 Mission Status")
    st.info("Pivot to **Dynamic Inference** complete. System is ready for Gentle Retraining.")

# --- 2. TRAINING CONTROL ---
elif page == "Training Control":
    st.title("⚙️ Training Control Center")
    
    # Session State for Locking UI during execution
    if 'training_active' not in st.session_state:
        st.session_state['training_active'] = False
        st.session_state['training_mode'] = None

    # Helper to unlock
    def lock_ui(mode):
        st.session_state['training_active'] = True
        st.session_state['training_mode'] = mode
    
    # Helper to reset (called only on page reload usually)
    def unlock_ui():
        st.session_state['training_active'] = False
        st.session_state['training_mode'] = None

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎮 Control Panel")
        
        # Checkpoint Analysis
        meta_files = glob.glob("checkpoints/training_metadata.json")
        has_checkpoint = False
        checkpoint_time = None
        checkpoint_epoch = "?"
        
        if meta_files:
            try:
                with open(meta_files[0], 'r') as f:
                    meta = json.load(f)
                    has_checkpoint = True
                    checkpoint_time = meta.get('timestamp', 'Unknown')
                    checkpoint_epoch = meta.get('epoch', '?')
            except:
                pass
        
        # Status Card
        if has_checkpoint:
            st.success(f"✅ Checkpoint Found (Epoch {checkpoint_epoch})")
            st.caption(f"Last saved: {checkpoint_time}")
            # Heuristic warning for old checkpoints
            if "2025-11" in str(checkpoint_time): 
                st.warning("⚠️ This checkpoint looks old. Recommend 'Start Fresh'.")
        else:
            st.info("❌ No Local Checkpoint Found")

        st.divider()

        # BUTTONS
        is_locked = st.session_state['training_active']
        
        # 1. RESUME
        st.markdown("**Option A: Resume**")
        st.button("⏯️ Resume Training", 
                 disabled=(not has_checkpoint) or is_locked, 
                 on_click=lambda: lock_ui('resume'),
                 help="Continue from the exact file index where it left off.")
        
        st.divider()

        # 2. FRESH START (Protected)
        st.markdown("**Option B: Fresh Start**")
        safety_switch = st.checkbox("Unlock 'Start Fresh'", disabled=is_locked, help="Check this to enable the button below.")
        
        st.button("🚀 Start Fresh (Wipe & Train)", 
                 disabled=(not safety_switch) or is_locked, 
                 type="primary",
                 on_click=lambda: lock_ui('fresh'),
                 help="WARNING: Deletes 'checkpoints/' and starts from Llama-3.2 base.")

    with col2:
        # EXECUTION LOGIC
        if st.session_state['training_active']:
            mode = st.session_state['training_mode']
            st.info(f"🔄 Execution in progress ({mode.title()} Mode)... Please wait.")
            
            # Log container
            st.markdown("### 📜 Live Console Output")
            log_container = st.empty()
            
            # Build Command
            cmd = "./run_retrain.sh"
            if mode == 'resume':
                cmd += " --resume"
            
            # Run
            try:
                ret_code = run_command_with_streaming(cmd, log_container)
                
                # Completion Handling
                st.session_state['training_active'] = False # Unlock for next render
                
                if ret_code == 0:
                    st.success("✅ Process Completed Successfully!")
                    st.balloons()
                else:
                    st.error(f"❌ Process Failed (Exit Code: {ret_code}). See logs above.")
                
                # Delay to let user see the result before potential rerun
                time.sleep(3)
                st.rerun()
                
            except Exception as e:
                st.error(f"System Error: {e}")
                st.session_state['training_active'] = False

        else:
            # IDLE STATE DISPLAY
            st.markdown("### 📉 Metrics History")
            if has_checkpoint and meta_files:
                with open(meta_files[0], 'r') as f:
                    st.json(json.load(f))
            else:
                st.markdown("""
                *No active training session.*
                
                **Ready to Train:**
                1.  **Resume**: Use if you were interrupted.
                2.  **Start Fresh**: Use for a new run (Standard for Retraining).
                """)

# --- 3. EVALUATION & COMPARE ---
elif page == "Evaluation & Compare":
    st.title("🏆 Evaluation & Comparison")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Actions")
        if st.button("⚖️ Run Full Comparison", type="primary"):
            st.toast("Running Evaluation Pipeline...")
            with col2:
                st.markdown("### 📜 Live Evaluation Logs")
                log_box = st.empty()
                ret_code = run_command_with_streaming("./run_comparison.sh", log_box)
                
                if ret_code == 0:
                    st.success("Comparison Complete! Refreshing results...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Evaluation Failed.")

    # Results Display
    with col2:
        if not st.session_state.get('eval_running', False):
            st.markdown("### 📊 Latest Results")
            
            # Parsing results from file if available, else showing hardcoded placeholder until run
            results_path = "dynamic_results/summary.txt"
            baseline_path = "baseline_results/summary.txt"
            
            data = []
            
            # Mock Data fallback
            data.append({"Model": "Baseline (Llama-3.2)", "Accuracy": 32.87})
            data.append({"Model": "Trained + Dynamic Hook", "Accuracy": 22.12}) # Last known result
            
            df = pd.DataFrame(data)
            
            fig = px.bar(df, x="Model", y="Accuracy", color="Model", 
                         title="Accuracy Comparison (LogiQA)", text_auto=True,
                         color_discrete_sequence=["#3b82f6", "#ef4444"])
            st.plotly_chart(fig, use_container_width=True)

# --- 4. PLAYGROUND ---
elif page == "Interactive Playground":
    st.title("⚡ Dynamic Logic Playground")
    st.markdown("Type a sentence. We will detect logical connectors and apply the **Boost** dynamically.")
    
    text = st.text_area("Input Text", "I think, therefore I am. However, it is complicated because of physics.")
    
    if text:
        connectors = ["therefore", "however", "because", "thus", "since", "but", "and"]
        words = text.split()
        annotated_text = []
        
        for word in words:
            clean_word = word.lower().strip(".,!?;")
            if clean_word in connectors:
                annotated_text.append(f"<span style='background-color: #22c55e; color: white; padding: 2px 6px; border-radius: 4px;'>{word} <b>(↑1.01x)</b></span>")
            else:
                annotated_text.append(word)
        
        st.markdown(f"<div class='highlight-box'>{' '.join(annotated_text)}</div>", unsafe_allow_html=True)

