import re
from typing import List, Optional


def normalize_numeric_text(text: str) -> str:
    text = text.replace(",", "").strip()
    return text.rstrip(".")


def normalize_extracted_number(value) -> Optional[str]:
    if value is None:
        return None
    value = normalize_numeric_text(str(value))
    if not value:
        return None
    if "." in value:
        head, tail = value.split(".", 1)
        if tail.strip("0") == "":
            return head or "0"
    return value


def preprocess_text(text: str) -> str:
    text = text.replace(",", "")
    text = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", text)
    text = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def extract_boxed_content(text: str) -> Optional[str]:
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


def extract_last_number(text: str) -> Optional[str]:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def extract_fraction_denominator(text: str) -> Optional[str]:
    frac_match = re.search(r"\\frac\{[^{}]+\}\{([^{}]+)\}", text)
    if frac_match:
        inner = extract_last_number(frac_match.group(1))
        if inner is not None:
            return normalize_extracted_number(inner)
    return None


def extract_candidate_numbers(text: str):
    return list(re.finditer(r"-?\d+(?:\.\d+)?", text))


def extract_number_after_last_equals(clause: str) -> Optional[str]:
    parts = clause.split("=")
    if len(parts) < 2:
        return None
    rhs = parts[-1]
    value = extract_last_number(rhs)
    return normalize_extracted_number(value)


def extract_number_after_last_marker(clause: str) -> Optional[str]:
    lower = clause.lower()
    markers = [
        "final answer",
        "the answer is",
        "answer:",
        "therefore",
        "thus",
        "hence",
        "so,",
        "so ",
    ]
    last_idx = -1
    for marker in markers:
        idx = lower.rfind(marker)
        if idx > last_idx:
            last_idx = idx
    if last_idx < 0:
        return None
    tail = clause[last_idx:]
    value = extract_last_number(tail)
    return normalize_extracted_number(value)


def score_candidate(clause: str, match: re.Match) -> tuple[int, int]:
    start, end = match.span()
    value = match.group(0)
    before = clause[max(0, start - 80):start].lower()
    after = clause[end:end + 80].lower()
    score = 0

    if any(marker in before for marker in ("therefore", "thus", "hence", "so", "final answer", "answer is")):
        score += 6
    if "=" in before[-12:]:
        score += 8
    if re.search(r"(=|is|are|was|were|be|gives|get|gets|got|total|altogether|left|remain|remaining|spent|made|needs|need|cost|costs|worked|harvested|read|has|have)\s*$", before):
        score += 5
    if re.search(r"^\s*(hours?|minutes?|days?|weeks?|months?|years?|feet|books?|pages?|games?|bottles?|bags?|cookies?|cakes?|shirts?|chapters?|dollars?|oranges?|children|people|crayons?|tomatoes?|potatoes?)\b", after):
        score += 4
    if any(marker in before for marker in ("initially", "at first", "start with", "started with", "yesterday", "today", "this morning", "morning", "afternoon", "evening")):
        score -= 3
    if any(marker in after for marker in ("per day", "per minute", "per bag", "per person")):
        score -= 3
    if "%" in before or "%" in after:
        score -= 3
    if value in {"2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"}:
        score -= 3

    # Prefer later numbers in the concluding span when scores tie.
    return score, start


def extract_from_clause(clause: str) -> Optional[str]:
    equation_value = extract_number_after_last_equals(clause)
    if equation_value is not None:
        return equation_value

    marker_value = extract_number_after_last_marker(clause)
    if marker_value is not None:
        return marker_value

    matches = extract_candidate_numbers(clause)
    if not matches:
        return None
    scored = [(score_candidate(clause, match), match.group(0)) for match in matches]
    scored.sort(key=lambda item: (item[0][0], item[0][1]))
    return normalize_extracted_number(scored[-1][1])


def trim_to_conclusion_span(clause: str) -> str:
    lower = clause.lower()
    markers = [
        "final answer",
        "the answer is",
        "answer:",
        "therefore",
        "thus",
        "hence",
        "so,",
        "so ",
    ]
    last_idx = -1
    for marker in markers:
        idx = lower.rfind(marker)
        if idx > last_idx:
            last_idx = idx
    return clause[last_idx:] if last_idx >= 0 else clause


def extract_final_answer_svamp(text: str) -> Optional[str]:
    if not text:
        return None

    normalized_text = preprocess_text(text)

    boxed = extract_boxed_content(normalized_text)
    if boxed:
        boxed_number = extract_last_number(boxed)
        if boxed_number is not None:
            return normalize_extracted_number(boxed_number)
        denominator = extract_fraction_denominator(boxed)
        if denominator is not None:
            return denominator

    tail = normalized_text[-800:]
    sentences = split_sentences(tail)

    strong_candidates = []
    for sentence in sentences[-10:]:
        lower = sentence.lower()
        if any(marker in lower for marker in ("final answer", "the answer is", "therefore", "thus", "hence", "so ", "so,", "####")):
            strong_candidates.append(sentence)

    for clause in reversed(strong_candidates):
        value = extract_from_clause(trim_to_conclusion_span(clause))
        if value is not None:
            return value

    for clause in reversed(sentences[-4:]):
        value = extract_from_clause(trim_to_conclusion_span(clause))
        if value is not None:
            return value

    tail_equation_value = extract_number_after_last_equals(tail)
    if tail_equation_value is not None:
        return tail_equation_value

    tail_number = extract_last_number(tail)
    if tail_number is not None:
        return normalize_extracted_number(tail_number)

    full_number = extract_last_number(normalized_text)
    return normalize_extracted_number(full_number)
