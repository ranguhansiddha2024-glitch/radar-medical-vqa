"""Download the research-only X-Raydar multimodal dataset from Hugging Face."""
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    print("Dataset terms: non-commercial research/evaluation only. Read the dataset card before use.")
    print("The dataset is about 2.81 GB; ensure you have enough disk space and a stable connection.")
    path = snapshot_download(
        repo_id="dnamodel/xraydar-multimodal",
        repo_type="dataset",
        local_dir="data/xraydar-multimodal",
    )
    print(f"Dataset downloaded to: {path}")
