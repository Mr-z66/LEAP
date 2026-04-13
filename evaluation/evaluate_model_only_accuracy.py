import argparse
import json
import os
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core_package.answer_registry import check_answer_correctness, get_answer_extractor
from core_package.config import EVALUATION, MODELS

DEFAULT_LABEL_PATH = EVALUATION.label_path
DEFAULT_ARTIFACT_PATH = EVALUATION.artifact_path
DEFAULT_TRACE_PATH = EVALUATION.trace_path
DEFAULT_SYSTEM_PROMPT = MODELS.system_prompt
DEFAULT_BOXED_SYSTEM_PROMPT = MODELS.boxed_math_system_prompt
DEFAULT_ANSWER_TYPE = "legacy_math"

class TorchMLPProbe(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=0.0):
        super().__init__()
        dims = [input_dim, *hidden_layers, 1]
        layers = []
        for idx in range(len(dims) - 2):
            layers.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(dims[-2], dims[-1]))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate any single model on the same held-out questions used by routing experiments."
    )
    parser.add_argument("--label-path", default=DEFAULT_LABEL_PATH, help="Path to strict labeled dataset (.pt).")
    parser.add_argument("--model-path", required=True, help="Path to the model to evaluate.")
    parser.add_argument(
        "--artifact-path",
        default=DEFAULT_ARTIFACT_PATH,
        help="Optional artifact path to reuse held-out test_question_ids.",
    )
    parser.add_argument(
        "--trace-path",
        default=DEFAULT_TRACE_PATH,
        help="Optional routing trace JSON to recover the same held-out question ids safely.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=EVALUATION.max_new_tokens, help="Max new tokens for generation.")
    parser.add_argument("--num-test-questions", type=int, default=None, help="Optional cap on number of test questions.")
    parser.add_argument("--output-path", default=None, help="Optional JSON output path for detailed predictions.")
    parser.add_argument("--answer-type", default=DEFAULT_ANSWER_TYPE, help="Answer protocol used for extraction and correctness.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt used during generation.")
    return parser.parse_args()


def resolve_system_prompt(answer_type: str, system_prompt: str) -> str:
    if answer_type == "boxed" and system_prompt == DEFAULT_SYSTEM_PROMPT:
        return DEFAULT_BOXED_SYSTEM_PROMPT
    return system_prompt


def build_inputs(tokenizer, question: str, system_prompt: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    return inputs


def load_question_table(dataset) -> Dict[int, Dict]:
    table: Dict[int, Dict] = {}
    if dataset and isinstance(dataset[0], dict) and "chunks" in dataset[0]:
        for item in dataset:
            qid = int(item["question_id"])
            if qid in table:
                continue
            table[qid] = {
                "question_id": qid,
                "question": item["question"],
                "ground_truth_final_answer": item.get("ground_truth_final_answer"),
            }
        return table

    for item in dataset:
        qid = int(item["question_id"])
        if qid in table:
            continue
        table[qid] = {
            "question_id": qid,
            "question": item["question"],
            "ground_truth_final_answer": item.get("ground_truth_final_answer"),
        }
    return table


def load_eval_ids_from_trace(trace_path: str, question_table: Dict[int, Dict]) -> List[int]:
    if not trace_path or not os.path.exists(trace_path):
        return []
    with open(trace_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        return []

    found_ids = []
    for group in payload:
        rows = group.get("per_question_rows", [])
        for row in rows:
            qid = row.get("question_id")
            if qid is None:
                continue
            qid = int(qid)
            if qid in question_table:
                found_ids.append(qid)
        if found_ids:
            break
    return sorted(set(found_ids))


def load_eval_ids_from_artifact(artifact_path: str, question_table: Dict[int, Dict]) -> List[int]:
    if not artifact_path or not os.path.exists(artifact_path):
        return []
    try:
        artifact = torch.load(artifact_path, map_location="cpu")
    except Exception as exc:
        print(f"Warning: failed to load artifact for test ids: {exc}")
        return []
    test_ids = [int(x) for x in artifact.get("test_question_ids", [])]
    return [qid for qid in sorted(test_ids) if qid in question_table]


def resolve_eval_ids(question_table: Dict[int, Dict], artifact_path: Optional[str], trace_path: Optional[str]) -> List[int]:
    trace_ids = load_eval_ids_from_trace(trace_path, question_table)
    if trace_ids:
        print(f"Using held-out ids from trace file: {len(trace_ids)}")
        return trace_ids

    artifact_ids = load_eval_ids_from_artifact(artifact_path, question_table)
    if artifact_ids:
        print(f"Using held-out ids from artifact: {len(artifact_ids)}")
        return artifact_ids

    print("Warning: could not recover held-out ids from trace/artifact, falling back to all questions.")
    return sorted(question_table.keys())


def main():
    args = parse_args()
    args.system_prompt = resolve_system_prompt(args.answer_type, args.system_prompt)

    print(f"Loading labeled dataset from: {args.label_path}")
    dataset = torch.load(args.label_path)
    question_table = load_question_table(dataset)

    eval_ids = resolve_eval_ids(question_table, args.artifact_path, args.trace_path)
    if args.num_test_questions is not None:
        eval_ids = eval_ids[: args.num_test_questions]

    print(f"Evaluating model-only on questions: {len(eval_ids)}")
    print(f"Loading model from: {args.model_path}")
    answer_extractor = get_answer_extractor(args.answer_type)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    rows = []
    correct = 0

    for idx, qid in enumerate(eval_ids, start=1):
        rec = question_table[qid]
        inputs = build_inputs(tokenizer, rec["question"], args.system_prompt)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_ids = output_ids[0, input_ids.shape[1]:]
        reasoning = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        pred, has_answer = answer_extractor(reasoning)
        gt = rec["ground_truth_final_answer"]
        is_correct = has_answer and check_answer_correctness(pred, gt, args.answer_type)
        correct += int(is_correct)

        rows.append(
            {
                "question_id": qid,
                "pred_final_answer": pred,
                "has_extracted_answer": has_answer,
                "ground_truth_final_answer": gt,
                "is_correct": is_correct,
                "generated_token_count": int(gen_ids.shape[0]),
                "reasoning": reasoning,
            }
        )

        if idx % 10 == 0 or idx == len(eval_ids):
            print(f"Processed {idx}/{len(eval_ids)} | running_acc={correct / idx:.4f}")

    acc = correct / max(len(eval_ids), 1)
    avg_tokens = sum(row["generated_token_count"] for row in rows) / max(len(rows), 1)

    print("\nModel-only evaluation summary")
    print("=" * 50)
    print(f"Questions total: {len(eval_ids)}")
    print(f"Model path: {args.model_path}")
    print(f"Model-only accuracy: {acc:.4f}")
    print(f"Avg generated tokens: {avg_tokens:.2f}")

    if args.output_path:
        payload = {
            "questions_total": len(eval_ids),
            "model_path": args.model_path,
            "model_only_accuracy": acc,
            "avg_generated_tokens": avg_tokens,
            "eval_question_ids": eval_ids,
            "rows": rows,
        }
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved detailed output to: {args.output_path}")


if __name__ == "__main__":
    main()
