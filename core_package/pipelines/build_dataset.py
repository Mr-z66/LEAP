import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from core_package.answer_extraction import extract_final_answer


DEFAULT_MODEL_PATH = os.path.join(os.getcwd(), "models", "Qwen2.5-1.5B")
DEFAULT_SAVE_PATH = "gsm8k_15b_hidden_states.pt"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_MIN_TOKENS = 5
DEFAULT_MAX_TOKENS = 30
DEFAULT_PUNCTUATIONS = [".", ",", "!", "?", "\n"]
DEFAULT_SYSTEM_PROMPT = "You are a helpful math assistant. Please reason step by step."


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build chunked hidden-state trajectories for GSM8K-style math datasets."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local model path.")
    parser.add_argument("--save-path", default=DEFAULT_SAVE_PATH, help="Output .pt file path.")
    parser.add_argument("--dataset-name", choices=["gsm8k", "svamp", "jsonl"], default="gsm8k")
    parser.add_argument("--dataset-split", default=None, help="Dataset split to use.")
    parser.add_argument("--num-samples", type=int, default=1000, help="Maximum number of samples to process.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--input-path", default=None, help="Local json/jsonl path for svamp/jsonl modes.")
    parser.add_argument("--question-field", default="question", help="Question field for jsonl mode.")
    parser.add_argument("--answer-field", default="answer", help="Answer field for jsonl mode.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    return parser.parse_args()


def normalize_numeric_text(text: str) -> str:
    return text.replace(",", "").strip().rstrip(".")


def extract_last_number(text: str):
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def normalize_answer_text(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        text = str(value)
    else:
        text = str(value).strip()
    if not text:
        return None

    extracted = extract_final_answer(text)
    if extracted is not None:
        return extracted

    numeric = extract_last_number(text)
    if numeric is not None:
        return normalize_numeric_text(numeric)

    return text


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

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )

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


def build_chunks(tokenizer, token_ids, hidden_states, token_confidences, min_tokens, max_tokens, punctuations):
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
        hit_punctuation = any(p in token_text for p in punctuations)

        cut_reason = None
        if hit_punctuation and chunk_len >= min_tokens:
            cut_reason = "punctuation"
        elif chunk_len >= max_tokens:
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


def load_jsonl_rows(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_source_rows(args) -> Iterable[Dict]:
    if args.dataset_name == "gsm8k":
        split = args.dataset_split or f"train[:{args.num_samples}]"
        print(f"Loading GSM8K split: {split}")
        return list(load_dataset("gsm8k", "main", split=split))

    if args.input_path is None:
        raise ValueError("--input-path is required for svamp/jsonl modes.")

    rows = load_jsonl_rows(args.input_path)
    if args.num_samples is not None:
        rows = rows[: args.num_samples]

    print(f"Loading local dataset from: {args.input_path} | rows={len(rows)}")
    return rows


def format_sample(row: Dict, idx: int, args) -> Dict:
    if args.dataset_name == "gsm8k":
        question = str(row["question"]).strip()
        answer_text = str(row["answer"]).strip()
        return {
            "question_id": idx,
            "question": question,
            "ground_truth_answer_text": answer_text,
            "ground_truth_final_answer": normalize_answer_text(answer_text),
            "source_meta": {"dataset_name": "gsm8k"},
        }

    if args.dataset_name == "svamp":
        body = str(row.get("Body", "")).strip()
        question_part = str(row.get("Question", "")).strip()
        question = f"{body} {question_part}".strip()
        answer_value = row.get("Answer")
        answer_text = str(answer_value).strip()
        return {
            "question_id": idx,
            "question": question,
            "ground_truth_answer_text": answer_text,
            "ground_truth_final_answer": normalize_answer_text(answer_value),
            "source_meta": {
                "dataset_name": "svamp",
                "id": row.get("ID"),
                "type": row.get("Type"),
                "equation": row.get("Equation"),
            },
        }

    question = str(row.get(args.question_field, "")).strip()
    answer_value = row.get(args.answer_field)
    answer_text = "" if answer_value is None else str(answer_value).strip()
    return {
        "question_id": idx,
        "question": question,
        "ground_truth_answer_text": answer_text,
        "ground_truth_final_answer": normalize_answer_text(answer_value),
        "source_meta": {"dataset_name": "jsonl"},
    }


def main():
    args = parse_args()

    punctuations = list(DEFAULT_PUNCTUATIONS)
    print(f"Loading base model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    dataset_rows = load_source_rows(args)
    formatted_rows = [format_sample(row, idx, args) for idx, row in enumerate(dataset_rows)]

    all_extracted_data = []
    for row in tqdm(formatted_rows, desc="Building heuristic chunks"):
        question = row["question"]
        ground_truth_answer = row["ground_truth_answer_text"]

        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": question},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        generated_token_ids, generated_hidden_states, generated_token_confidences = generate_with_hidden_states(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            max_new_tokens=args.max_new_tokens,
        )

        generated_text = decode_tokens(tokenizer, generated_token_ids).strip()
        model_final_answer = extract_final_answer(generated_text)
        if model_final_answer is None:
            model_final_answer = extract_last_number(generated_text)
            if model_final_answer is not None:
                model_final_answer = normalize_numeric_text(model_final_answer)

        chunks = build_chunks(
            tokenizer=tokenizer,
            token_ids=generated_token_ids,
            hidden_states=generated_hidden_states,
            token_confidences=generated_token_confidences,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            punctuations=punctuations,
        )

        prefix_token_ids = []
        for chunk in chunks:
            prefix_token_ids.extend(chunk["token_ids"])
            chunk["prefix_text"] = decode_tokens(tokenizer, prefix_token_ids).strip()

        all_extracted_data.append(
            {
                "question_id": row["question_id"],
                "question": question,
                "prompt_text": prompt_text,
                "ground_truth_answer_text": ground_truth_answer,
                "ground_truth_final_answer": row["ground_truth_final_answer"],
                "generated_text": generated_text,
                "generated_token_ids": generated_token_ids,
                "model_final_answer": model_final_answer,
                "is_final_correct": model_final_answer == row["ground_truth_final_answer"],
                "chunking_method": "heuristic_punctuation_minmax",
                "chunking_config": {
                    "min_tokens": args.min_tokens,
                    "max_tokens": args.max_tokens,
                    "punctuations": punctuations,
                    "max_new_tokens": args.max_new_tokens,
                },
                "source_meta": row.get("source_meta", {}),
                "chunks": chunks,
            }
        )

    print(f"Saving {len(all_extracted_data)} trajectories to: {args.save_path}")
    torch.save(all_extracted_data, args.save_path)
    print("Done.")


if __name__ == "__main__":
    main()
