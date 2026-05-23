import base64
import json
import pickle
import re
import zlib
from typing import Any, Dict, List, Tuple


def translate_private_test_cases(encoded_data: str) -> List[Dict[str, Any]]:
    if not encoded_data:
        return []
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    return json.loads(original_data)


def parse_public_test_cases(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return json.loads(value)
    raise TypeError(f"Unsupported public_test_cases type: {type(value).__name__}")


def build_livecodebench_prompt(row: Dict[str, Any]) -> str:
    question = str(row.get("question_content", row.get("question", ""))).strip()
    starter_code = str(row.get("starter_code") or "").strip()

    prompt = (
        "You will be given a programming problem. Generate a correct Python program "
        "that matches the specification and passes all tests.\n\n"
        f"Question:\n{question}\n\n"
    )
    if starter_code:
        prompt += (
            "Use the following starter code and complete the solution. "
            "Enclose your final code in a Python markdown block.\n"
            f"```python\n{starter_code}\n```\n"
        )
    else:
        prompt += (
            "Read input from stdin and write the answer to stdout. "
            "Enclose your final code in a Python markdown block:\n"
            "```python\n# YOUR CODE HERE\n```\n"
        )
    return prompt


def build_livecodebench_answer_payload(row: Dict[str, Any], include_private: bool = True) -> str:
    public_tests = parse_public_test_cases(row.get("public_test_cases"))
    private_tests = translate_private_test_cases(row.get("private_test_cases", "")) if include_private else []
    tests = public_tests + private_tests
    payload = {
        "inputs": [test["input"] for test in tests],
        "outputs": [test["output"] for test in tests],
        "fn_name": row.get("fn_name"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def check_livecodebench_correctness(predicted_text: str, answer_payload: str, timeout: int = 6) -> bool:
    if not predicted_text or not answer_payload:
        return False

    try:
        from experimental.baselines.R2R.r2r.evaluate.codegen_metrics import codegen_metrics
    except Exception as exc:
        raise RuntimeError("LiveCodeBench checker requires the bundled R2R codegen_metrics module.") from exc

    code = extract_code_from_markdown(predicted_text)
    if not code.strip():
        return False

    sample = [{"input_output": answer_payload}]
    generations = [[code]]
    metrics, _ = codegen_metrics(
        sample,
        generations,
        k_list=[1],
        num_process_evaluate=1,
        timeout=timeout,
    )
    return bool(metrics.get("pass@1", 0.0) == 1.0)


def extract_livecodebench_code_answer(text: str) -> Tuple[str, bool]:
    if not text:
        return "", False
    code = extract_code_from_markdown(text)
    if not code.strip():
        return "", False
    return text, True


def extract_code_from_markdown(text: str) -> str:
    if not text:
        return ""

    fenced_blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_blocks:
        return fenced_blocks[-1].strip()

    generic_blocks = re.findall(r"```\s*\n(.*?)```", text, flags=re.DOTALL)
    if generic_blocks:
        return generic_blocks[-1].strip()

    return ""
