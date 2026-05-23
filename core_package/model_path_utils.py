import os
import re


TOKENIZER_MARKERS = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "tokenizer.model")


def normalize_model_name(name):
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def is_hf_model_dir(path):
    if not path or not os.path.isdir(path):
        return False
    filenames = set(os.listdir(path))
    return "config.json" in filenames and any(marker in filenames for marker in TOKENIZER_MARKERS)


def resolve_local_hf_model_path(path):
    if not path or not os.path.isdir(path) or is_hf_model_dir(path):
        return path

    search_roots = [path]
    parent = os.path.dirname(path.rstrip("/\\"))
    if parent and os.path.isdir(parent):
        search_roots.append(parent)

    target_name = os.path.basename(path.rstrip("/\\"))
    normalized_target_name = normalize_model_name(target_name)
    candidates = []
    for search_root in search_roots:
        for root, dirnames, _ in os.walk(search_root):
            dirnames[:] = [name for name in dirnames if not name.startswith(".") and "temp" not in name.lower()]
            if not is_hf_model_dir(root):
                continue

            rel = os.path.relpath(root, search_root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            name = os.path.basename(root)
            normalized_name = normalize_model_name(name)
            if normalized_name == normalized_target_name:
                name_penalty = 0
            elif normalized_target_name and normalized_target_name in normalize_model_name(root):
                name_penalty = 1
            else:
                name_penalty = 2
            candidates.append((name_penalty, depth, root))

    if not candidates:
        return path

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][2]

def log_resolved_hf_model_path(label, path):
    resolved_path = resolve_local_hf_model_path(path)
    if resolved_path != path:
        print(f"Resolved {label} model path: {path} -> {resolved_path}")
    return resolved_path
