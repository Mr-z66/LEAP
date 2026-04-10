import re


def extract_last_number(text: str):
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def normalize_numeric_text(text: str):
    return text.replace(",", "").strip().rstrip(".")


def preprocess_text(text: str):
    text = text.replace(",", "")
    # Join decimals like "54. 00" -> "54.00"
    text = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", text)
    # Join grouped integers like "54 000" -> "54000"
    text = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_extracted_number(value):
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


def split_sentences(text: str):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def trim_to_conclusion_span(clause: str):
    lower = clause.lower()
    markers = [
        "final answer",
        "the answer is",
        "answer",
        "conclusion",
        "therefore",
        "thus",
        "hence",
        "so",
    ]
    last_idx = -1
    for marker in markers:
        idx = lower.rfind(marker)
        if idx > last_idx:
            last_idx = idx
    return clause[last_idx:] if last_idx >= 0 else clause


def score_number_in_clause(clause: str, match: re.Match):
    start, end = match.span()
    value = match.group(0)
    before = clause[max(0, start - 40):start].lower()
    after = clause[end:end + 40].lower()
    score = 0

    if any(keyword in before for keyword in ("final answer", "therefore", "thus", "hence", "conclusion", "answer")):
        score += 3
    if re.search(r"(is|are|was|were|be|equals?|spent|spend|takes?|take|took|needs?|need|costs?|cost|will be|would be|type|types|typed|can type|would spend|would need)\s*$", before):
        score += 4
    if re.search(r"^\s*(hours?|minutes?|years?|meters?|rooms?|words?|dollars?|goldfish|cars?|questions?|pizzas?|days?)\b", after):
        score += 3
    if "%" in after or "%" in before:
        score -= 3
    if any(op in before[-6:] for op in ("*", "/", "+", "-", "=")):
        score -= 1
    if any(op in after[:6] for op in ("*", "/", "+", "-", "=")):
        score -= 1
    if value in {"2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"}:
        score -= 2

    return score, start


def extract_from_clause(clause: str):
    matches = list(re.finditer(r"-?\d+(?:\.\d+)?", clause))
    if not matches:
        return None
    scored = []
    for match in matches:
        score, start = score_number_in_clause(clause, match)
        scored.append((score, start, match.group(0)))
    scored.sort(key=lambda item: (item[0], item[1]))
    return normalize_extracted_number(scored[-1][2])


def extract_final_answer(text: str):
    if not text:
        return None

    normalized_text = preprocess_text(text)

    boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", normalized_text)
    if boxed_matches:
        boxed_value = extract_last_number(boxed_matches[-1])
        if boxed_value is not None:
            return normalize_extracted_number(boxed_value)

    tail = normalized_text[-600:]
    sentences = split_sentences(tail)

    # Strongest priority: explicit answer/conclusion sentences near the end.
    clause_candidates = []
    for sentence in sentences[-8:]:
        lower = sentence.lower()
        if any(marker in lower for marker in ("final answer", "the answer is", "therefore", "thus", "hence", "conclusion", "####")):
            clause_candidates.append(sentence)
    for clause in reversed(clause_candidates):
        value = extract_from_clause(trim_to_conclusion_span(clause))
        if value is not None:
            return value

    # Next priority: very last few sentences, which often contain the final answer
    # even without an explicit marker.
    for clause in reversed(sentences[-3:]):
        value = extract_from_clause(trim_to_conclusion_span(clause))
        if value is not None:
            return value

    tail_value = extract_last_number(normalize_numeric_text(tail))
    if tail_value is not None:
        return normalize_extracted_number(tail_value)

    full_value = extract_last_number(normalize_numeric_text(normalized_text))
    return normalize_extracted_number(full_value)
