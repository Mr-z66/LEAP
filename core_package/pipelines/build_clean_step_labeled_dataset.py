import argparse
import json
import os
import re
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from core_package.answer_registry import check_answer_correctness, get_answer_extractor, resolve_answer_type
from core_package.config import MODELS
from core_package.pipelines.build_dataset import (
    decode_tokens,
    format_generation_question,
    format_sample,
    generate_with_hidden_states,
    load_source_rows,
    resolve_system_prompt,
    summarize_confidence,
)
from core_package.vllm_utils import build_openai_messages, infer_served_model_name, request_vllm_chat_completion


DEFAULT_OUTPUT_PATH = os.path.join("dataset", "clean_step_labeled_data.pt")
DEFAULT_SYSTEM_PROMPT = MODELS.system_prompt
DEFAULT_BOXED_SYSTEM_PROMPT = MODELS.boxed_math_system_prompt
SAFE_END_RE = re.compile(r"(?<!\d)[.!?](?:\s*)$|[。！？](?:\s*)$|\\\](?:\s*)$|\n\s*$")
AMBIGUOUS_TAIL_RE = re.compile(
    r"(\b\d+\.$|[=+\-*/,(]$|\\frac\{?$|\\sqrt\{?$|\\boxed\{?$|\\text\{?$|\\\[$|\\\($|"
    r"(?:step|calculate|determine|find|total|amount|number of)\s*:?\s*$)",
    flags=re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a clean v2 step-level labeled dataset from scratch: generate small-model reasoning, "
            "construct semantic chunks, extract final-layer hidden states, and judge explicit prefix errors."
        )
    )
    parser.add_argument("--dataset-name", choices=["gsm8k", "svamp", "math500", "livecodebench_v5", "jsonl"], default="gsm8k")
    parser.add_argument("--dataset-split", default=None, help="HF dataset split for gsm8k.")
    parser.add_argument("--input-path", default=None, help="Local json/jsonl path for svamp/math500/jsonl.")
    parser.add_argument("--question-field", default="question", help="Question field for jsonl mode.")
    parser.add_argument("--answer-field", default="answer", help="Answer field for jsonl mode.")
    parser.add_argument("--answer-type", default=None, help="Answer protocol override.")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of source questions to process.")
    parser.add_argument("--start-question", type=int, default=0, help="Skip source questions before this index.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Output clean labeled .pt path.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output .pt by question_id.")
    parser.add_argument("--save-every", type=int, default=5, help="Save every N newly processed questions.")

    parser.add_argument("--small-model-path", default=MODELS.small_model_path, help="Small model used to generate trajectories.")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)

    parser.add_argument("--min-step-tokens", type=int, default=12)
    parser.add_argument("--target-step-tokens", type=int, default=64)
    parser.add_argument("--max-step-tokens", type=int, default=120)
    parser.add_argument("--chunking-method", choices=["semantic_step_v2", "rsd_step", "rsd_step_fallback"], default="semantic_step_v2")
    parser.add_argument("--step-word", default="\n\n", help="RSD-style step delimiter for --chunking-method rsd_step.")
    parser.add_argument(
        "--force-step-tokens",
        type=int,
        default=180,
        help="Force a chunk after this many tokens if no safe boundary appears; such chunks are marked ambiguous.",
    )
    parser.add_argument("--lookahead-steps", type=int, default=1, help="Continuation chunks shown to judge for truncation context.")

    parser.add_argument("--judge-model-path", default=MODELS.large_model_path, help="Judge model path.")
    parser.add_argument("--judge-backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--max-judge-tokens", type=int, default=192)
    parser.add_argument("--include-reference-answer", action="store_true")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--vllm-model-name", default=None)
    parser.add_argument("--vllm-timeout", type=float, default=300.0)
    return parser.parse_args()


def load_existing_rows(path, resume):
    if not resume or not os.path.exists(path):
        return []
    rows = torch.load(path, weights_only=False)
    if not isinstance(rows, list):
        raise ValueError(f"Existing output must be a list: {path}")
    print(f"Resuming from {path}: rows={len(rows)}")
    return rows


def processed_question_ids(rows):
    return {int(row["question_id"]) for row in rows if "question_id" in row}


def save_rows(rows, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(rows, output_path)
    print(f"Saved: {output_path} | rows={len(rows)}")


def text_is_ambiguous(text):
    stripped = (text or "").strip()
    if not stripped:
        return True, "empty"
    if len(stripped.split()) <= 2 and not re.search(r"\d+\s*[=+\-*/]\s*\d+", stripped):
        return True, "too_short"
    if AMBIGUOUS_TAIL_RE.search(stripped):
        return True, "ambiguous_tail"
    if stripped.count("\\[") > stripped.count("\\]"):
        return True, "open_latex_block"
    if stripped.count("{") > stripped.count("}"):
        return True, "open_brace"
    return False, ""


def is_safe_boundary(tokenizer, token_ids):
    text = decode_tokens(tokenizer, token_ids)
    stripped = text.strip()
    if not stripped:
        return False
    ambiguous, _ = text_is_ambiguous(stripped)
    if ambiguous:
        return False
    if "\n\n" in text[-6:]:
        return True
    return bool(SAFE_END_RE.search(stripped))


def make_chunk(tokenizer, token_ids, hidden_states, token_confidences, start_idx, end_idx, cut_reason, ambiguous_reason=""):
    chunk_token_ids = list(token_ids[start_idx:end_idx])
    chunk_hidden_states = list(hidden_states[start_idx:end_idx])
    chunk_confidences = list(token_confidences[start_idx:end_idx])
    chunk_text = decode_tokens(tokenizer, chunk_token_ids).strip()
    if not ambiguous_reason:
        ambiguous, reason = text_is_ambiguous(chunk_text)
        ambiguous_reason = reason if ambiguous else ""
    return {
        "chunk_id": None,
        "start_token_idx": int(start_idx),
        "end_token_idx": int(end_idx - 1),
        "token_ids": chunk_token_ids,
        "token_count": int(len(chunk_token_ids)),
        "chunk_text": chunk_text,
        "boundary_hidden_state": chunk_hidden_states[-1].clone(),
        "mean_hidden_state": torch.stack(chunk_hidden_states).mean(dim=0),
        "last_non_delimiter_hidden_state": last_non_delimiter_hidden_state(tokenizer, chunk_token_ids, chunk_hidden_states),
        "cut_reason": cut_reason,
        "ambiguous_chunk": bool(ambiguous_reason),
        "ambiguous_reason": ambiguous_reason,
        **summarize_confidence(chunk_confidences),
    }


def last_non_delimiter_hidden_state(tokenizer, token_ids, hidden_states):
    for token_id, hidden_state in reversed(list(zip(token_ids, hidden_states))):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        if re.search(r"[A-Za-z0-9]", token_text):
            return hidden_state.clone()
    return hidden_states[-1].clone()


def build_semantic_step_chunks(tokenizer, token_ids, hidden_states, token_confidences, args):
    chunks = []
    start = 0
    idx = 0
    while idx < len(token_ids):
        current_len = idx + 1 - start
        candidate_ids = token_ids[start:idx + 1]
        should_cut = False
        cut_reason = ""
        ambiguous_reason = ""

        if current_len >= args.min_step_tokens and is_safe_boundary(tokenizer, candidate_ids):
            should_cut = True
            cut_reason = "semantic_boundary"
        elif current_len >= args.target_step_tokens and is_safe_boundary(tokenizer, candidate_ids):
            should_cut = True
            cut_reason = "target_safe_boundary"
        elif current_len >= args.force_step_tokens:
            should_cut = True
            cut_reason = "forced_max_tokens"
            ambiguous_reason = "forced_without_safe_boundary"

        if should_cut:
            chunks.append(
                make_chunk(
                    tokenizer,
                    token_ids,
                    hidden_states,
                    token_confidences,
                    start,
                    idx + 1,
                    cut_reason,
                    ambiguous_reason=ambiguous_reason,
                )
            )
            start = idx + 1
        idx += 1

    if start < len(token_ids):
        chunks.append(
            make_chunk(
                tokenizer,
                token_ids,
                hidden_states,
                token_confidences,
                start,
                len(token_ids),
                "tail",
            )
        )

    prefix_token_ids = []
    for chunk_id, chunk in enumerate(chunks):
        chunk["chunk_id"] = int(chunk_id)
        prefix_token_ids.extend(chunk["token_ids"])
        chunk["prefix_text"] = decode_tokens(tokenizer, prefix_token_ids).strip()
    return chunks


def build_rsd_step_chunks(tokenizer, token_ids, hidden_states, token_confidences, args):
    chunks = []
    step_ids = tokenizer.encode(args.step_word, add_special_tokens=False)
    start = 0
    idx = 0
    while idx < len(token_ids):
        should_cut = False
        if step_ids and idx + 1 - len(step_ids) >= start:
            tail = token_ids[idx + 1 - len(step_ids): idx + 1]
            should_cut = tail == step_ids

        if should_cut:
            chunks.append(
                make_chunk(
                    tokenizer,
                    token_ids,
                    hidden_states,
                    token_confidences,
                    start,
                    idx + 1,
                    "rsd_step_word",
                )
            )
            start = idx + 1
        idx += 1

    if start < len(token_ids):
        chunks.append(
            make_chunk(
                tokenizer,
                token_ids,
                hidden_states,
                token_confidences,
                start,
                len(token_ids),
                "tail",
            )
        )

    prefix_token_ids = []
    for chunk_id, chunk in enumerate(chunks):
        chunk["chunk_id"] = int(chunk_id)
        prefix_token_ids.extend(chunk["token_ids"])
        chunk["prefix_text"] = decode_tokens(tokenizer, prefix_token_ids).strip()
    return chunks


def split_span_with_semantic_fallback(tokenizer, token_ids, hidden_states, token_confidences, start, end, args, tail_reason):
    chunks = []
    if end - start <= args.max_step_tokens:
        chunks.append(
            make_chunk(
                tokenizer,
                token_ids,
                hidden_states,
                token_confidences,
                start,
                end,
                tail_reason,
            )
        )
        return chunks

    chunk_start = start
    idx = start
    while idx < end:
        current_len = idx + 1 - chunk_start
        candidate_ids = token_ids[chunk_start:idx + 1]
        should_cut = False
        cut_reason = ""
        ambiguous_reason = ""

        if current_len >= args.target_step_tokens and is_safe_boundary(tokenizer, candidate_ids):
            should_cut = True
            cut_reason = "target_semantic_fallback_boundary"
        elif current_len >= args.min_step_tokens and is_safe_boundary(tokenizer, candidate_ids):
            should_cut = True
            cut_reason = "semantic_fallback_boundary"
        elif current_len >= args.force_step_tokens:
            should_cut = True
            cut_reason = "max_tokens_fallback"
            ambiguous_reason = "forced_without_safe_boundary"

        if should_cut:
            chunks.append(
                make_chunk(
                    tokenizer,
                    token_ids,
                    hidden_states,
                    token_confidences,
                    chunk_start,
                    idx + 1,
                    cut_reason,
                    ambiguous_reason=ambiguous_reason,
                )
            )
            chunk_start = idx + 1
        idx += 1

    if chunk_start < end:
        chunks.append(
            make_chunk(
                tokenizer,
                token_ids,
                hidden_states,
                token_confidences,
                chunk_start,
                end,
                f"{tail_reason}_fallback_tail",
            )
        )
    return chunks


def build_rsd_step_fallback_chunks(tokenizer, token_ids, hidden_states, token_confidences, args):
    chunks = []
    step_ids = tokenizer.encode(args.step_word, add_special_tokens=False)
    start = 0
    idx = 0
    while idx < len(token_ids):
        should_cut = False
        if step_ids and idx + 1 - len(step_ids) >= start:
            tail = token_ids[idx + 1 - len(step_ids): idx + 1]
            should_cut = tail == step_ids

        if should_cut:
            chunks.extend(
                split_span_with_semantic_fallback(
                    tokenizer,
                    token_ids,
                    hidden_states,
                    token_confidences,
                    start,
                    idx + 1,
                    args,
                    "rsd_step_word",
                )
            )
            start = idx + 1
        idx += 1

    if start < len(token_ids):
        chunks.extend(
            split_span_with_semantic_fallback(
                tokenizer,
                token_ids,
                hidden_states,
                token_confidences,
                start,
                len(token_ids),
                args,
                "tail",
            )
        )

    prefix_token_ids = []
    for chunk_id, chunk in enumerate(chunks):
        chunk["chunk_id"] = int(chunk_id)
        prefix_token_ids.extend(chunk["token_ids"])
        chunk["prefix_text"] = decode_tokens(tokenizer, prefix_token_ids).strip()
    return chunks


def extract_json_object(raw_text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw_text or ""):
        try:
            payload, _ = decoder.raw_decode(raw_text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def to_int_label(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and int(value) in {-1, 0, 1}:
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "valid", "correct"}:
            return 1
        if normalized in {"0", "false", "error", "incorrect"}:
            return 0
        if normalized in {"-1", "ignore", "ambiguous", "unknown", "uncertain"}:
            return -1
    return -1


def clamp_confidence(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def parse_judge_response(raw_text):
    payload = extract_json_object(raw_text)
    if payload is None:
        return {
            "label": -1,
            "is_explicit_error": None,
            "is_ambiguous": True,
            "error_type": "parse_failed",
            "confidence": 0.0,
            "reason": (raw_text or "").strip(),
            "parse_status": "fallback",
        }

    label = to_int_label(payload.get("label", payload.get("is_prefix_valid", payload.get("is_prefix_correct", -1))))
    confidence = clamp_confidence(payload.get("confidence", 0.0))
    is_ambiguous = bool(payload.get("is_ambiguous", label == -1))
    is_explicit_error = payload.get("is_explicit_error")
    if isinstance(is_explicit_error, str):
        is_explicit_error = is_explicit_error.strip().lower() in {"1", "true", "yes"}
    elif is_explicit_error is not None:
        is_explicit_error = bool(is_explicit_error)

    return {
        "label": label,
        "is_explicit_error": is_explicit_error,
        "is_ambiguous": is_ambiguous,
        "error_type": str(payload.get("error_type", "none" if label == 1 else "unknown")).strip() or "unknown",
        "confidence": confidence,
        "reason": str(payload.get("reason", "")).strip(),
        "parse_status": "json",
    }


def build_clean_judge_prompt(
    question,
    prefix_text,
    current_chunk_text,
    continuation_text,
    reference_answer,
    include_reference_answer,
    answer_type="",
):
    reference_section = ""
    if include_reference_answer:
        reference_section = f"\nReference answer/checking payload (for checking only, not for demanding completeness):\n{reference_answer}\n"
    continuation_section = ""
    if continuation_text:
        continuation_section = (
            "\nOptional continuation after the current chunk. Use this only to decide whether the current chunk is "
            "truncated or ambiguous; do not judge the continuation itself:\n"
            f"{continuation_text}\n"
        )

    if answer_type == "livecodebench_codegen":
        label_definition = """Label definition:
- label = 1 if everything explicitly written so far is still valid for solving the programming problem, even if the code/reasoning is incomplete, redundant, verbose, or has not reached a final program yet.
- label = 0 only if the prefix already contains an explicit harmful error: misunderstanding the specification, choosing a wrong algorithm, contradicting constraints, introducing invalid code structure, using wrong input/output behavior, or making an assertion that would make the final program fail.
- label = -1 if the current chunk is truncated/ambiguous, ends mid-token/mid-code-block/mid-statement, the evidence is insufficient, or you are unsure.

Important rules:
1. Do not mark a prefix wrong merely because it has not yet written all code.
2. Do not mark a prefix wrong merely because a step is redundant or explanatory.
3. Do not require the final program to be complete at this prefix.
4. If a code statement/string/markdown fence appears cut off, return label = -1 rather than guessing.
5. Return JSON only."""
    else:
        label_definition = """Label definition:
- label = 1 if everything explicitly written so far is still mathematically/logically valid, even if the reasoning is incomplete, redundant, verbose, or has not reached the target yet.
- label = 0 only if the prefix already contains an explicit error: a wrong arithmetic result, wrong equation, wrong variable relation, wrong interpretation of the question, or using an irrelevant quantity as the target.
- label = -1 if the current chunk is truncated/ambiguous, ends mid-number/mid-formula/mid-LaTeX, the evidence is insufficient, or you are unsure.

Important rules:
1. Do not mark a prefix wrong merely because it has not yet used all given information.
2. Do not mark a prefix wrong merely because a step is redundant or unnecessary.
3. Do not mark a prefix wrong merely because it has not yet computed the final answer.
4. If a number/formula appears cut off, return label = -1 rather than guessing.
5. Return JSON only."""

    return f"""You are labeling reasoning prefixes for a process-routing dataset.
Your task is prefix-local explicit-error detection.

Question:
{question}
{reference_section}
Reasoning prefix up to and including the current chunk:
{prefix_text}

Current chunk:
{current_chunk_text}
{continuation_section}
{label_definition}

JSON schema:
{{
  "label": 1 or 0 or -1,
  "is_explicit_error": true or false,
  "is_ambiguous": true or false,
  "error_type": "none" | "arithmetic" | "logic" | "setup" | "semantic" | "format" | "unknown",
  "confidence": a number between 0 and 1,
  "reason": "one short sentence"
}}"""


def judge_prefix(prompt, tokenizer, model, args):
    if args.judge_backend == "hf":
        messages = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_judge_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    response = request_vllm_chat_completion(
        base_url=args.vllm_base_url,
        api_key=args.vllm_api_key,
        model_name=infer_served_model_name(args.judge_model_path, args.vllm_model_name),
        messages=build_openai_messages(None, prompt),
        max_tokens=args.max_judge_tokens,
        timeout=args.vllm_timeout,
    )
    return response["text"].strip()


def label_from_judge_result(judge_result, chunk, args):
    if chunk["ambiguous_chunk"]:
        return -1, "chunk_marked_ambiguous"
    if judge_result["parse_status"] != "json":
        return -1, "parse_failed"
    if judge_result["confidence"] < args.low_confidence_threshold:
        return -1, "low_confidence"
    if judge_result["is_ambiguous"]:
        return -1, "judge_ambiguous"
    return int(judge_result["label"]), "judge"


def dataset_args_from_main(args):
    return SimpleNamespace(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        input_path=args.input_path,
        num_samples=args.num_samples,
        question_field=args.question_field,
        answer_field=args.answer_field,
    )


def main():
    args = parse_args()
    if args.force_step_tokens < args.max_step_tokens:
        raise ValueError("--force-step-tokens must be >= --max-step-tokens")

    answer_type = resolve_answer_type(args.dataset_name, args.answer_type)
    args.system_prompt = resolve_system_prompt(answer_type, args.system_prompt)
    answer_extractor = get_answer_extractor(answer_type)

    print(f"Loading small model: {args.small_model_path}")
    small_tokenizer = AutoTokenizer.from_pretrained(args.small_model_path, local_files_only=True)
    small_model = AutoModelForCausalLM.from_pretrained(
        args.small_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    small_model.eval()

    judge_tokenizer = None
    judge_model = None
    if args.judge_backend == "hf":
        print(f"Loading judge model: {args.judge_model_path}")
        judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model_path, local_files_only=True)
        judge_model = AutoModelForCausalLM.from_pretrained(
            args.judge_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        judge_model.eval()
    else:
        print(f"Using vLLM judge: {args.vllm_base_url} | model={infer_served_model_name(args.judge_model_path, args.vllm_model_name)}")

    dataset_args = dataset_args_from_main(args)
    source_rows = list(load_source_rows(dataset_args))
    formatted_rows = [format_sample(row, idx, dataset_args, answer_type) for idx, row in enumerate(source_rows)]
    formatted_rows = [row for row in formatted_rows if int(row["question_id"]) >= args.start_question]

    labeled_rows = load_existing_rows(args.output_path, args.resume)
    done_questions = processed_question_ids(labeled_rows)
    newly_processed = 0

    for row in tqdm(formatted_rows, desc="Clean step labeling v2"):
        question_id = int(row["question_id"])
        if question_id in done_questions:
            continue

        generation_question = format_generation_question(row["question"], answer_type)
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": generation_question},
        ]
        prompt_text = small_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids, hidden_states, token_confidences = generate_with_hidden_states(
            model=small_model,
            tokenizer=small_tokenizer,
            prompt_text=prompt_text,
            max_new_tokens=args.max_new_tokens,
        )
        generated_text = decode_tokens(small_tokenizer, token_ids).strip()
        model_final_answer, has_model_answer = answer_extractor(generated_text)
        if not has_model_answer:
            model_final_answer = ""
        is_final_correct = check_answer_correctness(model_final_answer, row["ground_truth_final_answer"], answer_type)

        if args.chunking_method == "rsd_step":
            chunks = build_rsd_step_chunks(small_tokenizer, token_ids, hidden_states, token_confidences, args)
        elif args.chunking_method == "rsd_step_fallback":
            chunks = build_rsd_step_fallback_chunks(small_tokenizer, token_ids, hidden_states, token_confidences, args)
        else:
            chunks = build_semantic_step_chunks(small_tokenizer, token_ids, hidden_states, token_confidences, args)
        for idx, chunk in enumerate(chunks):
            continuation = "\n".join(
                next_chunk["chunk_text"] for next_chunk in chunks[idx + 1: idx + 1 + args.lookahead_steps]
            )
            prompt = build_clean_judge_prompt(
                question=row["question"],
                prefix_text=chunk["prefix_text"],
                current_chunk_text=chunk["chunk_text"],
                continuation_text=continuation,
                reference_answer=row["ground_truth_answer_text"],
                include_reference_answer=args.include_reference_answer,
                answer_type=answer_type,
            )
            raw_response = judge_prefix(prompt, judge_tokenizer, judge_model, args)
            judge_result = parse_judge_response(raw_response)
            label, label_source = label_from_judge_result(judge_result, chunk, args)

            labeled_rows.append(
                {
                    "question_id": question_id,
                    "chunk_id": chunk["chunk_id"],
                    "question": row["question"],
                    "prompt_text": prompt_text,
                    "generated_text": generated_text,
                    "chunk_text": chunk["chunk_text"],
                    "prefix_text": chunk["prefix_text"],
                    "start_token_idx": chunk["start_token_idx"],
                    "end_token_idx": chunk["end_token_idx"],
                    "token_ids": chunk["token_ids"],
                    "token_count": chunk["token_count"],
                    "cut_reason": chunk["cut_reason"],
                    "ambiguous_chunk": chunk["ambiguous_chunk"],
                    "ambiguous_reason": chunk["ambiguous_reason"],
                    "boundary_hidden_state": chunk["boundary_hidden_state"],
                    "mean_hidden_state": chunk["mean_hidden_state"],
                    "last_non_delimiter_hidden_state": chunk["last_non_delimiter_hidden_state"],
                    "mean_entropy": chunk.get("mean_entropy"),
                    "max_entropy": chunk.get("max_entropy"),
                    "final_entropy": chunk.get("final_entropy"),
                    "mean_top1_prob": chunk.get("mean_top1_prob"),
                    "min_top1_prob": chunk.get("min_top1_prob"),
                    "final_top1_prob": chunk.get("final_top1_prob"),
                    "mean_margin": chunk.get("mean_margin"),
                    "min_margin": chunk.get("min_margin"),
                    "final_margin": chunk.get("final_margin"),
                    "ground_truth_answer_text": row["ground_truth_answer_text"],
                    "ground_truth_final_answer": row["ground_truth_final_answer"],
                    "model_final_answer": model_final_answer,
                    "is_final_correct": bool(is_final_correct),
                    "answer_type": answer_type,
                    "source_meta": row.get("source_meta", {}),
                    "chunking_method": args.chunking_method,
                    "chunking_config": {
                        "step_word": args.step_word if args.chunking_method == "rsd_step" else None,
                        "min_step_tokens": args.min_step_tokens,
                        "target_step_tokens": args.target_step_tokens,
                        "max_step_tokens": args.max_step_tokens,
                        "force_step_tokens": args.force_step_tokens,
                        "lookahead_steps": args.lookahead_steps,
                    },
                    "judge_model_path": args.judge_model_path,
                    "judge_backend": args.judge_backend,
                    "judge_prompt": prompt,
                    "judge_raw_response": raw_response,
                    "judge_parse_status": judge_result["parse_status"],
                    "judge_confidence": judge_result["confidence"],
                    "judge_error_type": judge_result["error_type"],
                    "judge_reason": judge_result["reason"],
                    "judge_is_explicit_error": judge_result["is_explicit_error"],
                    "judge_is_ambiguous": judge_result["is_ambiguous"],
                    "label": int(label),
                    "label_source": label_source,
                }
            )

        done_questions.add(question_id)
        newly_processed += 1
        if args.save_every > 0 and newly_processed % args.save_every == 0:
            save_rows(labeled_rows, args.output_path)

    save_rows(labeled_rows, args.output_path)


if __name__ == "__main__":
    main()

