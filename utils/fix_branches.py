from huggingface_hub import HfApi
import os

# Hardcoded for reliability during this fix
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "NeuralNinjasConnector/Connector-Llama"
TARGET_BRANCH = "v2_gentle_retrain"

def fix_branches():
    if not HF_TOKEN:
        print("❌ Error: HF_TOKEN environment variable not set.")
        return
        
    print(f"🔌 Authenticating...")
    api = HfApi(token=HF_TOKEN)
    
    print(f"📤 Uploading current local checkpoint to branch: '{TARGET_BRANCH}'...")
    try:
        api.create_branch(repo_id=REPO_ID, branch=TARGET_BRANCH, exist_ok=True)
        
        url = api.upload_folder(
            folder_path="checkpoints",
            repo_id=REPO_ID,
            revision=TARGET_BRANCH,
            repo_type="model",
            commit_message="Migrating accidentally pushed chunk from main to correct branch"
        )
        print(f"✅ Success! Your new training is now safe on branch '{TARGET_BRANCH}'.")
        print(f"🔗 URL: {url}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    fix_branches()
