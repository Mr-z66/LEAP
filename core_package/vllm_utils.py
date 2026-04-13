import json
import os
import urllib.error
import urllib.request
from typing import List, Optional


def build_openai_messages(system_prompt: Optional[str], user_content: str):
    messages: List[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def infer_served_model_name(model_path: str, override: Optional[str] = None) -> str:
    if override:
        return override
    return os.path.basename(model_path.rstrip("/\\"))


def request_vllm_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    messages: List[dict],
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout: float = 300.0,
):
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"vLLM request failed: {exc}") from exc

    choices = response_payload.get("choices", [])
    if not choices:
        raise RuntimeError(f"vLLM response did not contain choices: {response_payload}")

    choice = choices[0]
    message = choice.get("message", {}) or {}
    return {
        "text": message.get("content", "") or "",
        "finish_reason": choice.get("finish_reason"),
        "raw": response_payload,
    }
