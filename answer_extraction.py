import re


def extract_last_number(text: str):
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def normalize_numeric_text(text: str):
    return text.replace(",", "").strip().rstrip(".")


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


def extract_final_answer(text: str):
    if not text:
        return None

    normalized_text = text.replace(",", "")

    boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", normalized_text)
    if boxed_matches:
        boxed_value = extract_last_number(boxed_matches[-1])
        if boxed_value is not None:
            return normalize_extracted_number(boxed_value)

    tail = normalized_text[-600:]
    explicit_patterns = [
        r"(?i)final answer\s*[:?]\s*([^\n]+)",
        r"(?i)the answer is\s*([^\n]+)",
        r"(?i)answer\s*[:?]\s*([^\n]+)",
        r"####\s*([^\n]+)",
        r"(?i)therefore[^.\n]*?(-?\d+(?:\.\d+)?)",
        r"(?i)so[^.\n]*?(-?\d+(?:\.\d+)?)",
        r"(?i)thus[^.\n]*?(-?\d+(?:\.\d+)?)",
        r"(?i)hence[^.\n]*?(-?\d+(?:\.\d+)?)",
        r"(?i)which is\s+(-?\d+(?:\.\d+)?)",
        r"(?i)will be\s+(-?\d+(?:\.\d+)?)",
        r"(?i)is\s+(-?\d+(?:\.\d+)?)\s+(?:hours?|rooms?|years?|meters?|words?|dollars?)\b",
    ]
    for pattern in explicit_patterns:
        matches = re.findall(pattern, tail)
        if matches:
            explicit_value = extract_last_number(matches[-1])
            if explicit_value is not None:
                return normalize_extracted_number(explicit_value)

    tail_value = extract_last_number(normalize_numeric_text(tail))
    if tail_value is not None:
        return normalize_extracted_number(tail_value)

    full_value = extract_last_number(normalize_numeric_text(normalized_text))
    return normalize_extracted_number(full_value)
