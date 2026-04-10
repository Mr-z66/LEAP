from pathlib import Path

from modelscope import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_TO_DOWNLOAD = {
    "Qwen2.5-1.5B": "qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-7B": "qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-32B": "qwen/Qwen2.5-32B-Instruct",
}


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading models into: {MODELS_DIR}")
    for name, repo_id in MODELS_TO_DOWNLOAD.items():
        save_path = MODELS_DIR / name
        print(f"\n[{name}] downloading from {repo_id}")
        snapshot_download(model_id=repo_id, local_dir=str(save_path))
        print(f"[{name}] download completed: {save_path}")


if __name__ == "__main__":
    main()
