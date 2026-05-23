import argparse
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from core_package.config import MODELS
from core_package.model_path_utils import log_resolved_hf_model_path
from core_package.pipelines.build_clean_step_labeled_dataset import (
    build_clean_judge_prompt,
    judge_prefix,
    label_from_judge_result,
    parse_judge_response,
)
from core_package.vllm_utils import infer_served_model_name


def parse_args():
    parser = argparse.ArgumentParser(description="Label an existing chunked trajectory .pt file without regenerating it.")
    parser.add_argument("--input-path", required=True, help="Input trajectory .pt from build_dataset.py.")
    parser.add_argument("--output-path", required=True, help="Output labeled chunk .pt.")
    parser.add_argument("--num-samples", type=int, default=None, help="Only label the first N questions.")
    parser.add_argument("--start-question", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--lookahead-steps", type=int, default=1)
    parser.add_argument("--include-reference-answer", action="store_true")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.65)

    parser.add_argument("--judge-model-path", default=MODELS.large_model_path)
    parser.add_argument("--judge-backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--max-judge-tokens", type=int, default=192)
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--vllm-model-name", default=None)
    parser.add_argument("--vllm-timeout", type=float, default=300.0)
    return parser.parse_args()


def load_existing_rows(path, resume):
    if not resume or not os.path.exists(path):
        return []
    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise ValueError(f"Existing output must be a list: {path}")
    print(f"Resuming from {path}: rows={len(rows)}")
    return rows


def processed_question_ids(rows):
    return {int(row["question_id"]) for row in rows}


def save_rows(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(rows, path)
    print(f"Checkpoint saved: {path} | labeled chunks={len(rows)}")


def scalarize(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numel") and value.numel() == 1:
        return float(value.item())
    return value


def main():
    args = parse_args()

    print(f"Loading trajectories: {args.input_path}")
    trajectories = torch.load(args.input_path, map_location="cpu", weights_only=False)
    if args.num_samples is not None:
        trajectories = trajectories[: args.num_samples]
        print(f"Using first {len(trajectories)} questions")

    judge_tokenizer = None
    judge_model = None
    if args.judge_backend == "hf":
        judge_model_path = log_resolved_hf_model_path("judge", args.judge_model_path)
        print(f"Loading judge model: {judge_model_path}")
        judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_path, local_files_only=True)
        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        judge_model.eval()
    else:
        print(f"Using vLLM judge: {args.vllm_base_url} | model={infer_served_model_name(args.judge_model_path, args.vllm_model_name)}")

    labeled_rows = load_existing_rows(args.output_path, args.resume)
    done_questions = processed_question_ids(labeled_rows)
    newly_processed = 0

    for sample in tqdm(trajectories, desc="Label existing trajectories"):
        question_id = int(sample["question_id"])
        if question_id < args.start_question or question_id in done_questions:
            continue

        chunks = sample.get("chunks", [])
        answer_type = sample.get("answer_type", "")
        for idx, chunk in enumerate(chunks):
            continuation = "\n".join(
                next_chunk["chunk_text"] for next_chunk in chunks[idx + 1: idx + 1 + args.lookahead_steps]
            )
            prompt = build_clean_judge_prompt(
                question=sample["question"],
                prefix_text=chunk["prefix_text"],
                current_chunk_text=chunk["chunk_text"],
                continuation_text=continuation,
                reference_answer=sample.get("ground_truth_answer_text", ""),
                include_reference_answer=args.include_reference_answer,
                answer_type=answer_type,
            )
            raw_response = judge_prefix(prompt, judge_tokenizer, judge_model, args)
            judge_result = parse_judge_response(raw_response)
            label, label_source = label_from_judge_result(judge_result, chunk, args)

            labeled_rows.append(
                {
                    "question_id": question_id,
                    "chunk_id": int(chunk["chunk_id"]),
                    "question": sample["question"],
                    "prompt_text": sample.get("prompt_text", ""),
                    "generated_text": sample.get("generated_text", ""),
                    "chunk_text": chunk["chunk_text"],
                    "prefix_text": chunk["prefix_text"],
                    "start_token_idx": int(chunk["start_token_idx"]),
                    "end_token_idx": int(chunk["end_token_idx"]),
                    "token_ids": chunk.get("token_ids", []),
                    "token_count": int(chunk["token_count"]),
                    "cut_reason": chunk.get("cut_reason", ""),
                    "ambiguous_chunk": bool(chunk.get("ambiguous_chunk", False)),
                    "ambiguous_reason": chunk.get("ambiguous_reason", ""),
                    "boundary_hidden_state": chunk.get("boundary_hidden_state"),
                    "mean_hidden_state": chunk.get("mean_hidden_state"),
                    "last_non_delimiter_hidden_state": chunk.get("last_non_delimiter_hidden_state"),
                    "max_entropy_hidden_state": chunk.get("max_entropy_hidden_state"),
                    "min_top1_hidden_state": chunk.get("min_top1_hidden_state"),
                    "mean_entropy": scalarize(chunk.get("mean_entropy")),
                    "max_entropy": scalarize(chunk.get("max_entropy")),
                    "final_entropy": scalarize(chunk.get("final_entropy")),
                    "mean_top1_prob": scalarize(chunk.get("mean_top1_prob")),
                    "min_top1_prob": scalarize(chunk.get("min_top1_prob")),
                    "final_top1_prob": scalarize(chunk.get("final_top1_prob")),
                    "mean_margin": scalarize(chunk.get("mean_margin")),
                    "min_margin": scalarize(chunk.get("min_margin")),
                    "final_margin": scalarize(chunk.get("final_margin")),
                    "ground_truth_answer_text": sample.get("ground_truth_answer_text", ""),
                    "ground_truth_final_answer": sample.get("ground_truth_final_answer", ""),
                    "model_final_answer": sample.get("model_final_answer", ""),
                    "is_final_correct": bool(sample.get("is_final_correct", False)),
                    "answer_type": answer_type,
                    "source_meta": sample.get("source_meta", {}),
                    "chunking_method": sample.get("chunking_method"),
                    "chunking_config": sample.get("chunking_config"),
                    "judge_model_path": args.judge_model_path,
                    "judge_backend": args.judge_backend,
                    "judge_prompt": prompt,
                    "judge_raw_response": raw_response,
                    "judge_parse_status": judge_result["parse_status"],
                    "judge_confidence": judge_result["confidence"],
                    "judge_error_type": judge_result["error_type"],
                    "judge_reason": judge_result["reason"],
                    "label": int(label),
                    "label_source": label_source,
                }
            )

        done_questions.add(question_id)
        newly_processed += 1
        if args.save_every > 0 and newly_processed % args.save_every == 0:
            save_rows(labeled_rows, args.output_path)

    save_rows(labeled_rows, args.output_path)
    print(f"Completed question count: {len(done_questions)}")


if __name__ == "__main__":
    main()
