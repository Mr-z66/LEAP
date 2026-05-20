GSM8K_BOXED_INSTRUCTION = (
    "Solve the problem step by step. Put the final numeric answer within \\boxed{}."
)


def append_gsm8k_boxed_instruction(question: str) -> str:
    question = str(question).strip()
    if "\\boxed{}" in question or "\\boxed{" in question:
        return question
    return f"{question}\n\n{GSM8K_BOXED_INSTRUCTION}"
