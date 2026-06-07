from typing import Optional


def extract_last_complete_nonempty_boxed(text: str) -> Optional[str]:
    marker = "\\boxed{"
    search_end = len(text)

    while search_end > 0:
        start = text.rfind(marker, 0, search_end)
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
                    content = text[start + len(marker):idx].strip()
                    if content:
                        return content
                    break
            idx += 1

        search_end = start

    return None


def chunk_has_complete_nonempty_boxed(text: str) -> bool:
    return extract_last_complete_nonempty_boxed(text) is not None
