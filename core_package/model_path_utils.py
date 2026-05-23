import os


TOKENIZER_MARKERS = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "tokenizer.model")


def is_hf_model_dir(path):
    if not path or not os.path.isdir(path):
        return False
    filenames = set(os.listdir(path))
    return "config.json" in filenames and any(marker in filenames for marker in TOKENIZER_MARKERS)


def resolve_local_hf_model_path(path):
    if not path or not os.path.isdir(path) or is_hf_model_dir(path):
        return path

    candidates = []
    for root, dirnames, _ in os.walk(path):
        dirnames[:] = [name for name in dirnames if not name.startswith(".") and "temp" not in name.lower()]
        if is_hf_model_dir(root):
            rel = os.path.relpath(root, path)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            candidates.append((depth, root))

    if not candidates:
        return path

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def log_resolved_hf_model_path(label, path):
    resolved_path = resolve_local_hf_model_path(path)
    if resolved_path != path:
        print(f"Resolved {label} model path: {path} -> {resolved_path}")
    return resolved_path
