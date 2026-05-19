SVAMP_BOXED_INSTRUCTION = (
    "Solve the problem step by step. Put the final numeric answer within \\boxed{}."
)


def append_svamp_boxed_instruction(question: str) -> str:
    question = str(question).strip()
    if "\\boxed{}" in question or "\\boxed{" in question:
        return question
    return f"{question}\n\n{SVAMP_BOXED_INSTRUCTION}"
