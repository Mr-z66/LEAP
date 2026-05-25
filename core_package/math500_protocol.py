import re
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple


MATH500_QWEN_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

MATH500_QWEN_USER_SUFFIX = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def append_math500_instruction(question: str) -> str:
    question = str(question).strip()
    if "\\boxed{" in question or "put your final answer within \\boxed{}" in question.lower():
        return question
    return f"{question}{MATH500_QWEN_USER_SUFFIX}"


def extract_last_boxed_content(text: str) -> Optional[str]:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    idx = start + len(marker)
    depth = 1
    while idx < len(text):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(marker):idx].strip()
        idx += 1
    return None


def extract_math500_answer(text: str) -> Tuple[str, bool]:
    answer = extract_last_boxed_content(text)
    if answer is None:
        answer = extract_unboxed_math500_answer(text)
    if answer is None:
        return "", False
    return answer.strip(), True


def extract_unboxed_math500_answer(text: str) -> Optional[str]:
    if not text:
        return None
    tail = str(text).strip()[-1200:]
    marker_pattern = re.compile(
        r"(?i)(?:final answer|the answer is|answer is|answer:|therefore|thus|hence)\s*(?:is\s*)?[:：]?\s*"
    )
    matches = list(marker_pattern.finditer(tail))
    if matches:
        span = tail[matches[-1].end():]
        span = re.split(r"[\n。.!?]", span, maxsplit=1)[0].strip()
        span = span.strip("$ ").rstrip(".。")
        if span:
            return span

    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    for line in reversed(lines[-5:]):
        cleaned = line.strip().strip("$").rstrip(".。")
        if "\\frac" in cleaned or re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", cleaned):
            return cleaned
        if re.fullmatch(r"[A-Za-z]\s*=\s*[^.。]+", cleaned):
            return cleaned
    return None


def _strip_outer_braces(text: str) -> str:
    text = text.strip()
    while text.startswith("{") and text.endswith("}"):
        depth = 0
        balanced = True
        for idx, char in enumerate(text):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and idx != len(text) - 1:
                    balanced = False
                    break
        if not balanced:
            break
        text = text[1:-1].strip()
    return text


def _split_top_level(text: str, sep: str) -> list[str]:
    parts = []
    depth = 0
    current = []
    for char in text:
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(depth - 1, 0)
        if char == sep and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def _normalize_math500_text(text: str) -> str:
    normalized = str(text).strip()
    normalized = normalized.strip("$")
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\!", "")
    normalized = normalized.replace("\\,", "")
    normalized = normalized.replace("\\;", "")
    normalized = normalized.replace("\\:", "")
    normalized = normalized.replace("\\ ", "")
    normalized = normalized.replace("\n", " ")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.rstrip(".")
    normalized = normalized.rstrip(",")
    normalized = _strip_outer_braces(normalized)
    normalized = re.sub(r"^([a-zA-Z])=", "", normalized)
    normalized = re.sub(r"\\text\{\(([A-Za-z])\)\}", r"\\text{\1}", normalized)
    normalized = re.sub(r"\\sqrt\{([^{}]+)\}", r"\\sqrt\1", normalized)
    normalized = re.sub(r"^([+-]?\d+(?:\.\d+)?)\\text\{[^{}]*\}$", r"\1", normalized)
    normalized = re.sub(r"^(.*?)(?:\\text\{degrees?\}|\\circ|\^\{\\circ\}|\^\\circ)$", r"\1", normalized)
    normalized = normalized.replace("\\%", "%")
    normalized = normalized.replace("\\$", "")
    normalized = normalized.replace("\\displaystyle", "")
    normalized = normalized.replace("\\operatorname", "")
    normalized = normalized.replace("\\mathrm", "\\text")
    normalized = re.sub(r"\\text\{([^{}]*)\}", lambda m: f"\\text{{{m.group(1).strip()}}}", normalized)
    if normalized.endswith("_8"):
        normalized = normalized[:-2]
    return normalized


def _expand_pm(text: str) -> str:
    if "\\pm" not in text:
        return text
    return text.replace("\\pm", "+") + "," + text.replace("\\pm", "-")


def _normalize_math500_collection(text: str) -> str:
    normalized = _normalize_math500_text(_expand_pm(text))
    if "," not in normalized:
        return normalized
    pieces = [_normalize_math500_text(piece) for piece in _split_top_level(normalized, ",")]
    return ",".join(sorted(piece for piece in pieces if piece))


def _decimal_from_math500_text(text: str) -> Optional[Decimal]:
    normalized = _normalize_math500_text(text)
    frac_match = re.fullmatch(r"\\frac\{?([+-]?\d+(?:\.\d+)?)\}?\{?([+-]?\d+(?:\.\d+)?)\}?", normalized)
    try:
        if frac_match:
            denominator = Decimal(frac_match.group(2))
            if denominator == 0:
                return None
            return Decimal(frac_match.group(1)) / denominator
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", normalized):
            if normalized.startswith("."):
                normalized = "0" + normalized
            elif normalized.startswith("-."):
                normalized = "-0" + normalized[1:]
            elif normalized.startswith("+."):
                normalized = "+0" + normalized[1:]
            return Decimal(normalized)
    except InvalidOperation:
        return None
    return None


def math500_answers_equal(predicted: str, actual: str) -> bool:
    predicted_text = str(predicted).strip()
    actual_text = str(actual).strip()
    if not predicted_text or not actual_text:
        return False
    if _normalize_math500_collection(predicted_text) == _normalize_math500_collection(actual_text):
        return True
    predicted_decimal = _decimal_from_math500_text(predicted_text)
    actual_decimal = _decimal_from_math500_text(actual_text)
    return predicted_decimal is not None and actual_decimal is not None and predicted_decimal == actual_decimal
