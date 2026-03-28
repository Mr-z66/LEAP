import os
import re

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= Configuration =================
MODEL_PATH = os.path.join(os.getcwd(), "models", "Qwen2.5-1.5B")
SAVE_PATH = "gsm8k_15b_hidden_states.pt"
NUM_SAMPLES = 1000
MAX_NEW_TOKENS = 256
MIN_TOKENS = 5
MAX_TOKENS = 30
PUNCTUATIONS = [".", ",", "!", "?", "\n"]
# =================================================


def extract_last_number(text):
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def decode_tokens(tokenizer, token_ids):
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def compute_token_confidence(logits):
    probabilities = torch.softmax(logits.to(torch.float32), dim=-1)
    entropy = float(-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum().item())
    top_values = torch.topk(probabilities, k=2)
    top1_prob = float(top_values.values[0].item())
    top2_prob = float(top_values.values[1].item()) if top_values.values.numel() > 1 else 0.0
    return {
        "entropy": entropy,
        "top1_prob": top1_prob,
        "margin": float(top1_prob - top2_prob),
    }


def summarize_confidence(token_confidences):
    if not token_confidences:
        zero = torch.tensor([0.0], dtype=torch.float32)
        return {
            "mean_entropy": zero.clone(),
            "max_entropy": zero.clone(),
            "final_entropy": zero.clone(),
            "mean_top1_prob": zero.clone(),
            "min_top1_prob": zero.clone(),
            "final_top1_prob": zero.clone(),
            "mean_margin": zero.clone(),
            "min_margin": zero.clone(),
            "final_margin": zero.clone(),
        }

    entropy_values = np.asarray([item["entropy"] for item in token_confidences], dtype=np.float32)
    top1_values = np.asarray([item["top1_prob"] for item in token_confidences], dtype=np.float32)
    margin_values = np.asarray([item["margin"] for item in token_confidences], dtype=np.float32)

    def scalar_tensor(value):
        return torch.tensor([float(value)], dtype=torch.float32)

    return {
        "mean_entropy": scalar_tensor(np.mean(entropy_values)),
        "max_entropy": scalar_tensor(np.max(entropy_values)),
        "final_entropy": scalar_tensor(entropy_values[-1]),
        "mean_top1_prob": scalar_tensor(np.mean(top1_values)),
        "min_top1_prob": scalar_tensor(np.min(top1_values)),
        "final_top1_prob": scalar_tensor(top1_values[-1]),
        "mean_margin": scalar_tensor(np.mean(margin_values)),
        "min_margin": scalar_tensor(np.min(margin_values)),
        "final_margin": scalar_tensor(margin_values[-1]),
    }


def generate_with_hidden_states(model, tokenizer, prompt_text, max_new_tokens):
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_ids = prompt_inputs.input_ids
    past_key_values = None

    generated_token_ids = []
    generated_hidden_states = []
    generated_token_confidences = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )

        # When we feed a previously generated token, the last hidden state
        # aligns with that token and can be stored as a chunk feature source.
        if generated_token_ids:
            generated_hidden_states.append(
                outputs.hidden_states[-1][0, -1, :].detach().to(torch.float32).cpu()
            )

        logits = outputs.logits[0, -1, :]
        past_key_values = outputs.past_key_values
        generated_token_confidences.append(compute_token_confidence(logits))
        next_id = torch.argmax(logits).item()
        generated_token_ids.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

        input_ids = torch.tensor([[next_id]], device=model.device)

    non_eos_token_ids = [token_id for token_id in generated_token_ids if token_id != tokenizer.eos_token_id]

    if non_eos_token_ids and len(generated_hidden_states) < len(non_eos_token_ids):
        last_token_id = non_eos_token_ids[-1]
        with torch.no_grad():
            final_outputs = model(
                input_ids=torch.tensor([[last_token_id]], device=model.device),
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
        generated_hidden_states.append(
            final_outputs.hidden_states[-1][0, -1, :].detach().to(torch.float32).cpu()
        )

    valid_length = min(len(non_eos_token_ids), len(generated_hidden_states), len(generated_token_confidences))
    return (
        non_eos_token_ids[:valid_length],
        generated_hidden_states[:valid_length],
        generated_token_confidences[:valid_length],
    )


def build_chunks(tokenizer, token_ids, hidden_states, token_confidences):
    chunks = []
    current_chunk_token_ids = []
    current_chunk_hidden_states = []
    current_chunk_confidences = []
    chunk_start_idx = 0

    for token_idx, token_id in enumerate(token_ids):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        current_chunk_token_ids.append(token_id)
        current_chunk_hidden_states.append(hidden_states[token_idx])
        current_chunk_confidences.append(token_confidences[token_idx])

        chunk_len = len(current_chunk_token_ids)
        hit_punctuation = any(p in token_text for p in PUNCTUATIONS)

        cut_reason = None
        if hit_punctuation and chunk_len >= MIN_TOKENS:
            cut_reason = "punctuation"
        elif chunk_len >= MAX_TOKENS:
            cut_reason = "max_tokens"

        if cut_reason is None:
            continue

        chunk_token_ids = list(current_chunk_token_ids)
        chunk_hidden_states = list(current_chunk_hidden_states)
        chunk_confidences = list(current_chunk_confidences)
        chunks.append(
            {
                "chunk_id": len(chunks),
                "start_token_idx": chunk_start_idx,
                "end_token_idx": token_idx,
                "token_ids": chunk_token_ids,
                "token_count": len(chunk_token_ids),
                "chunk_text": decode_tokens(tokenizer, chunk_token_ids).strip(),
                "boundary_hidden_state": chunk_hidden_states[-1].clone(),
                "mean_hidden_state": torch.stack(chunk_hidden_states).mean(dim=0),
                "cut_reason": cut_reason,
                **summarize_confidence(chunk_confidences),
            }
        )

        current_chunk_token_ids = []
        current_chunk_hidden_states = []
        current_chunk_confidences = []
        chunk_start_idx = token_idx + 1

    if current_chunk_token_ids:
        chunk_token_ids = list(current_chunk_token_ids)
        chunk_hidden_states = list(current_chunk_hidden_states)
        chunk_confidences = list(current_chunk_confidences)
        chunks.append(
            {
                "chunk_id": len(chunks),
                "start_token_idx": chunk_start_idx,
                "end_token_idx": len(token_ids) - 1,
                "token_ids": chunk_token_ids,
                "token_count": len(chunk_token_ids),
                "chunk_text": decode_tokens(tokenizer, chunk_token_ids).strip(),
                "boundary_hidden_state": chunk_hidden_states[-1].clone(),
                "mean_hidden_state": torch.stack(chunk_hidden_states).mean(dim=0),
                "cut_reason": "tail",
                **summarize_confidence(chunk_confidences),
            }
        )

    return chunks


print(f"Loading base model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
model.eval()

print(f"Loading GSM8K train split[:{NUM_SAMPLES}]")
dataset = load_dataset("gsm8k", "main", split=f"train[:{NUM_SAMPLES}]")

all_extracted_data = []

for idx, row in enumerate(tqdm(dataset, desc="Building heuristic chunks")):
    question = row["question"]
    ground_truth_answer = row["answer"]

    messages = [
        {"role": "system", "content": "You are a helpful math assistant. Please reason step by step."},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    generated_token_ids, generated_hidden_states, generated_token_confidences = generate_with_hidden_states(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    generated_text = decode_tokens(tokenizer, generated_token_ids).strip()
    model_final_answer = extract_last_number(generated_text)
    ground_truth_final_answer = extract_last_number(ground_truth_answer)
    chunks = build_chunks(tokenizer, generated_token_ids, generated_hidden_states, generated_token_confidences)

    prefix_token_ids = []
    for chunk in chunks:
        prefix_token_ids.extend(chunk["token_ids"])
        chunk["prefix_text"] = decode_tokens(tokenizer, prefix_token_ids).strip()

    all_extracted_data.append(
        {
            "question_id": idx,
            "question": question,
            "prompt_text": prompt_text,
            "ground_truth_answer_text": ground_truth_answer,
            "ground_truth_final_answer": ground_truth_final_answer,
            "generated_text": generated_text,
            "generated_token_ids": generated_token_ids,
            "model_final_answer": model_final_answer,
            "is_final_correct": model_final_answer == ground_truth_final_answer,
            "chunking_method": "heuristic_punctuation_minmax",
            "chunking_config": {
                "min_tokens": MIN_TOKENS,
                "max_tokens": MAX_TOKENS,
                "punctuations": PUNCTUATIONS,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
            "chunks": chunks,
        }
    )

print(f"Saving {len(all_extracted_data)} trajectories to: {SAVE_PATH}")
torch.save(all_extracted_data, SAVE_PATH)
print("Done.")
