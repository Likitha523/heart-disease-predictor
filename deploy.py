import os
from huggingface_hub import HfApi, create_repo

token = os.environ.get("HF_TOKEN")
username = os.environ.get("HF_USERNAME")
repo_name = os.environ.get("HF_REPO_NAME", "heart-disease-predictor")

if not token or not username:
    print("Error: HF_TOKEN or HF_USERNAME environment variables are not set.")
    exit(1)

repo_id = f"{username}/{repo_name}"
print(f"Deploying to Hugging Face Space: {repo_id}...")

api = HfApi(token=token)

print("Creating Space on Hugging Face...")
create_repo(repo_id=repo_id, repo_type="space", space_sdk="streamlit", token=token, exist_ok=True)

print("Uploading files...")
api.upload_folder(
    folder_path=".",
    repo_id=repo_id,
    repo_type="space",
    allow_patterns=["app.py", "requirements.txt", "train_model.py", "README.md", "*.pkl"],
)

print(f"\n✅ Deployment complete! View your app at: https://huggingface.co/spaces/{repo_id}")
