import os
from modelscope import snapshot_download

base_dir = os.path.join(os.getcwd(), "models")
models_to_download = {
    "Qwen2.5-1.5B": "qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-7B": "qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-32B": "qwen/Qwen2.5-32B-Instruct"
}

print("🚀 切换阿里魔搭源，准备起飞...")
for name, repo_id in models_to_download.items():
    print(f"\n[{name}] 开始拉取...")
    save_path = os.path.join(base_dir, name)
    snapshot_download(model_id=repo_id, local_dir=save_path)
    print(f"✅ [{name}] 下载圆满完成！")
