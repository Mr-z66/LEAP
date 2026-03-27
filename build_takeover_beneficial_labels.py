import argparse
import os
import re
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= Default Configuration =================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_PATH = os.path.join(PROJECT_ROOT, "gsm8k_labeled_training_data_strict.pt")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "gsm8k_takeover_beneficial_labels.pt")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-32B")
DEFAULT_CACHE_PATH = os.path.join(PROJECT_ROOT, "takeover_beneficial_cache.pt")
DEFAULT_MAX_NEW_TOKENS = 768
DEFAULT_SAVE_EVERY = 25
DEFAULT_START_QUESTION = 0
DEFAULT_SYSTEM_PROMPT = "You are a helpful math assistant. Please reason step by step."
# ========================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Build chunk-level takeover_beneficial labels.")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH, help="Path to strict labeled chunk data.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Path to save takeover_beneficial labels.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to the large model.")
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH, help="Path to cached takeover generations.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max generation tokens for takeover.")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="Save every N newly processed chunks.")
    parser.add_argument("--start-question", type=int, default=DEFAULT_START_QUESTION, help="Skip questions below this id.")
    parser.add_argument("--num-questions", type=int, default=None, help="Only process the first N questions after filtering.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output and cache files.")
    return parser.parse_args()


def extract_last_number(text):
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def extract_final_answer(text):
    if not text:
        return None

    boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed_matches:
        boxed_value = extract_last_number(boxed_matches[-1])
        if boxed_value is not None:
            return boxed_value

    explicit_patterns = [
        r"(?i)final answer\s*[:：]\s*([^\n]+)",
        r"(?i)the answer is\s*([^\n]+)",
        r"(?i)answer\s*[:：]\s*([^\n]+)",
        r"####\s*([^\n]+)",
    ]
    for pattern in explicit_patterns:
        matches = re.findall(pattern, text)
        if matches:
            explicit_value = extract_last_number(matches[-1])
            if explicit_value is not None:
                return explicit_value

    return extract_last_number(text)


def build_generation_messages(question, assistant_prefix=None):
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if assistant_prefix is not None:
        messages.append({"role": "assistant", "content": assistant_prefix})
    return messages


def build_generation_inputs(tokenizer, question, assistant_prefix):
    normalized_prefix = None
    if assistant_prefix is not None:
        normalized_prefix = assistant_prefix.rstrip()
        if not normalized_prefix:
            normalized_prefix = None

    messages = build_generation_messages(question, assistant_prefix=normalized_prefix)
    if normalized_prefix is None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ), normalized_prefix

    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        continue_final_message=True,
    ), normalized_prefix


def generate_takeover(model, tokenizer, question, assistant_prefix, max_new_tokens):
    inputs, normalized_prefix = build_generation_inputs(tokenizer, question, assistant_prefix)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_token_ids = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    full_reasoning = normalized_prefix + generated_text if normalized_prefix else generated_text
    final_answer = extract_final_answer(full_reasoning)
    return {
        "generated_text": generated_text,
        "full_reasoning": full_reasoning,
        "final_answer": final_answer,
        "generated_token_count": int(new_token_ids.shape[0]),
    }


def load_output(output_path, resume):
    if not resume or not os.path.exists(output_path):
        return []
    print(f"Resuming labels from: {output_path}")
    return torch.load(output_path)


def load_cache(cache_path, resume):
    if not resume or not os.path.exists(cache_path):
        return {}
    print(f"Resuming cache from: {cache_path}")
    return torch.load(cache_path)


def save_outputs(records, output_path, cache, cache_path):
    torch.save(records, output_path)
    torch.save(cache, cache_path)
    print(f"Checkpoint saved | labels: {output_path} | cache: {cache_path} | rows: {len(records)}")


def group_chunks_by_question(dataset, start_question, num_questions):
    grouped = defaultdict(list)
    for item in dataset:
        question_id = int(item["question_id"])
        if question_id < start_question:
            continue
        grouped[question_id].append(item)

    ordered_question_ids = sorted(grouped.keys())
    if num_questions is not None:
        ordered_question_ids = ordered_question_ids[:num_questions]

    grouped_records = []
    for question_id in ordered_question_ids:
        chunks = sorted(grouped[question_id], key=lambda row: int(row["chunk_id"]))
        grouped_records.append((question_id, chunks))
    return grouped_records


def build_processed_pairs(existing_records):
    processed = set()
    for row in existing_records:
        processed.add((int(row["question_id"]), int(row["chunk_id"])))
    return processed


def main():
    args = parse_args()

    print(f"Loading strict labeled chunk data from: {args.input_path}")
    dataset = torch.load(args.input_path)
    question_records = group_chunks_by_question(dataset, args.start_question, args.num_questions)
    print(f"Questions selected for takeover_beneficial labeling: {len(question_records)}")

    print(f"Loading large model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    output_records = load_output(args.output_path, args.resume)
    processed_pairs = build_processed_pairs(output_records)
    cache = load_cache(args.cache_path, args.resume)

    newly_processed = 0
    for question_id, chunks in tqdm(question_records, desc="Building takeover_beneficial labels"):
        question = chunks[0]["question"]
        ground_truth_final_answer = chunks[0]["ground_truth_final_answer"]
        small_is_correct = bool(chunks[0]["is_final_correct"])
        small_final_answer = chunks[0]["model_final_answer"]

        for index, chunk in enumerate(chunks):
            chunk_id = int(chunk["chunk_id"])
            pair_key = (question_id, chunk_id)
            if pair_key in processed_pairs:
                continue

            if index > 0:
                safe_chunk = chunks[index - 1]
                takeover_start_chunk_id = int(safe_chunk["chunk_id"])
                safe_prefix_text = safe_chunk["prefix_text"]
            else:
                takeover_start_chunk_id = -1
                safe_prefix_text = None

            cache_key = (question_id, takeover_start_chunk_id, args.max_new_tokens)
            if cache_key not in cache:
                cache[cache_key] = generate_takeover(
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    assistant_prefix=safe_prefix_text,
                    max_new_tokens=args.max_new_tokens,
                )

            takeover_result = cache[cache_key]
            takeover_is_correct = takeover_result["final_answer"] == ground_truth_final_answer
            beneficial = int((not small_is_correct) and takeover_is_correct)
            harmful = int(small_is_correct and (not takeover_is_correct))
            if beneficial:
                takeover_label = "beneficial"
            elif harmful:
                takeover_label = "harmful"
            else:
                takeover_label = "neutral"

            output_records.append(
                {
                    **chunk,
                    "takeover_anchor_chunk_id": takeover_start_chunk_id,
                    "takeover_safe_prefix_text": safe_prefix_text,
                    "takeover_final_answer": takeover_result["final_answer"],
                    "takeover_is_correct": takeover_is_correct,
                    "takeover_generated_token_count": takeover_result["generated_token_count"],
                    "takeover_full_reasoning": takeover_result["full_reasoning"],
                    "small_is_correct": small_is_correct,
                    "small_final_answer": small_final_answer,
                    "takeover_beneficial": beneficial,
                    "takeover_harmful": harmful,
                    "takeover_label": takeover_label,
                }
            )
            processed_pairs.add(pair_key)
            newly_processed += 1

            if args.save_every > 0 and newly_processed % args.save_every == 0:
                save_outputs(output_records, args.output_path, cache, args.cache_path)

    save_outputs(output_records, args.output_path, cache, args.cache_path)


if __name__ == "__main__":
    main()
