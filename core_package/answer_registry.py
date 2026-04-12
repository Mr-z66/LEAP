import re
from typing import Callable, List, Optional, Tuple

from core_package.answer_extraction import extract_final_answer


AnswerExtractor = Callable[[str], Tuple[str, bool]]
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
ANSWER_MARKERS = (
    "final answer",
    "the answer is",
    "answer is",
    "answer:",
    "####",
    "therefore",
    "thus",
    "hence",
    "so the answer is",
)
COMMON_UNITS = (
    "hour",
    "hours",
    "minute",
    "minutes",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "dollar",
    "dollars",
    "cent",
    "cents",
    "book",
    "books",
    "page",
    "pages",
    "car",
    "cars",
    "apple",
    "apples",
    "banana",
    "bananas",
    "orange",
    "oranges",
    "cookie",
    "cookies",
    "cake",
    "cakes",
    "bag",
    "bags",
    "bottle",
    "bottles",
    "box",
    "boxes",
    "shirt",
    "shirts",
    "person",
    "people",
    "child",
    "children",
    "student",
    "students",
    "toy",
    "toys",
    "ticket",
    "tickets",
    "mile",
    "miles",
    "meter",
    "meters",
    "foot",
    "feet",
)


def _normalize_answer_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_numeric_text(text: str) -> str:
    text = text.replace(",", "").strip()
    text = text.rstrip(".")
    return text


def _normalize_extracted_number(value) -> Optional[str]:
    if value is None:
        return None
    value = _normalize_numeric_text(str(value))
    if not value:
        return None
    if "." in value:
        head, tail = value.split(".", 1)
        if tail.strip("0") == "":
            return head or "0"
    return value


def _preprocess_text(text: str) -> str:
    text = text.replace(",", "")
    text = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", text)
    text = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", text).strip()


def _split_sentences(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def _extract_boxed_content(text: str) -> Optional[str]:
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


def _extract_last_number(text: str) -> Optional[str]:
    matches = NUMBER_PATTERN.findall(text.replace(",", ""))
    return matches[-1] if matches else None


def _collect_marked_spans(text: str) -> List[Tuple[int, str, str]]:
    spans: List[Tuple[int, str, str]] = []
    lower = text.lower()
    for marker in ANSWER_MARKERS:
        start = 0
        while True:
            idx = lower.find(marker, start)
            if idx < 0:
                break
            tail = text[idx: idx + 240]
            tail = re.split(r"[\n]", tail, maxsplit=1)[0]
            spans.append((idx, marker, tail.strip()))
            start = idx + len(marker)
    spans.sort(key=lambda item: item[0])
    return spans


def _extract_number_near_marker(span: str) -> Optional[str]:
    patterns = [
        r"(?i)(?:final answer|the answer is|answer is|answer:|####)\s*(?:is\s*)?(?:\$)?(-?\d+(?:\.\d+)?)",
        r"(?i)(?:therefore|thus|hence)\s*,?\s*(?:the answer is\s*)?(?:\$)?(-?\d+(?:\.\d+)?)",
        r"(?i)so the answer is\s*(?:\$)?(-?\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, span)
        if match:
            return _normalize_extracted_number(match.group(1))

    if "=" in span:
        rhs = span.rsplit("=", 1)[-1]
        value = _extract_last_number(rhs)
        if value is not None:
            return _normalize_extracted_number(value)

    return _normalize_extracted_number(_extract_last_number(span))


def _unit_bonus(after_text: str) -> int:
    stripped = after_text.strip().lower()
    for unit in COMMON_UNITS:
        if stripped.startswith(unit):
            return 4
    return 0


def _score_candidate(sentence: str, match: re.Match, sentence_index: int, total_sentences: int) -> Tuple[int, int]:
    start, end = match.span()
    value = match.group(0)
    before = sentence[max(0, start - 80):start].lower()
    after = sentence[end:end + 80].lower()
    score = 0

    if any(marker in before for marker in ANSWER_MARKERS):
        score += 10
    if re.search(r"(is|are|was|were|be|equals?|totals?|altogether|remaining|left|costs?|spent|needs?|has|have)\s*$", before):
        score += 5
    score += _unit_bonus(after)

    if "=" in before[-16:]:
        score += 2
    if any(op in before[-6:] for op in ("*", "/", "+", "-")):
        score -= 4
    if any(op in after[:6] for op in ("*", "/", "+", "-", "=")):
        score -= 5
    if "%" in before or "%" in after:
        score -= 3
    if value in {"2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"}:
        score -= 3

    score += max(total_sentences - sentence_index - 1, 0)
    return score, start


def _extract_from_sentence(sentence: str, sentence_index: int, total_sentences: int) -> Optional[str]:
    matches = list(NUMBER_PATTERN.finditer(sentence))
    if not matches:
        return None
    scored = [(_score_candidate(sentence, match, sentence_index, total_sentences), match.group(0)) for match in matches]
    scored.sort(key=lambda item: (item[0][0], item[0][1]))
    return _normalize_extracted_number(scored[-1][1])


def _extract_tail_fallback(text: str) -> Optional[str]:
    tail = text[-220:]
    sentences = _split_sentences(tail)
    if not sentences:
        return _normalize_extracted_number(_extract_last_number(tail))

    total = len(sentences)
    for rev_idx, sentence in enumerate(reversed(sentences[-3:]), start=1):
        sent_idx = total - rev_idx
        value = _extract_from_sentence(sentence, sent_idx, total)
        if value is not None:
            return value

    return _normalize_extracted_number(_extract_last_number(tail))


def _extract_final_answer_svamp(text: str) -> Optional[str]:
    if not text:
        return None

    normalized_text = _preprocess_text(text)

    boxed = _extract_boxed_content(normalized_text)
    if boxed:
        boxed_value = _extract_last_number(boxed)
        if boxed_value is not None:
            return _normalize_extracted_number(boxed_value)

    marked_spans = _collect_marked_spans(normalized_text[-1200:])
    for _, _, span in reversed(marked_spans):
        value = _extract_number_near_marker(span)
        if value is not None:
            return value

    sentences = _split_sentences(normalized_text[-800:])
    total_sentences = len(sentences)
    for idx, sentence in enumerate(sentences[-4:], start=max(total_sentences - 4, 0)):
        lower = sentence.lower()
        if any(marker in lower for marker in ANSWER_MARKERS):
            value = _extract_from_sentence(sentence, idx, total_sentences)
            if value is not None:
                return value

    return _extract_tail_fallback(normalized_text)


def extract_legacy_math_answer(text: str) -> Tuple[str, bool]:
    answer = extract_final_answer(text)
    if answer is None:
        return "", False
    return str(answer), True


def extract_svamp_numeric_answer(text: str) -> Tuple[str, bool]:
    answer = _extract_final_answer_svamp(text)
    if answer is None:
        return "", False
    return str(answer), True


def check_answer_correctness(predicted: str, actual: str, answer_type: str) -> bool:
    predicted_text = _normalize_answer_text(predicted)
    actual_text = _normalize_answer_text(actual)
    if not predicted_text or not actual_text:
        return False

    if answer_type in {"legacy_math", "svamp_numeric"}:
        return predicted_text == actual_text

    raise ValueError(f"Unsupported answer type for correctness check: {answer_type}")


def get_answer_extractor(answer_type: str) -> AnswerExtractor:
    extractors = {
        "legacy_math": extract_legacy_math_answer,
        "svamp_numeric": extract_svamp_numeric_answer,
    }
    if answer_type not in extractors:
        raise ValueError(f"Unsupported answer type: {answer_type}")
    return extractors[answer_type]


def resolve_answer_type(dataset_name: str, override: Optional[str] = None) -> str:
    if override:
        return override

    if dataset_name == "svamp":
        return "svamp_numeric"

    return "legacy_math"
