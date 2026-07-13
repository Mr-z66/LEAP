"""Persistent, rollback-capable Hugging Face decoding sessions."""

from dataclasses import dataclass

import torch


def _crop_legacy_cache(cache, length):
    cropped = []
    for layer in cache:
        cropped_layer = []
        for value in layer:
            if torch.is_tensor(value) and value.ndim >= 3:
                slices = [slice(None)] * value.ndim
                slices[-2] = slice(0, length)
                value = value[tuple(slices)]
            cropped_layer.append(value)
        cropped.append(tuple(cropped_layer))
    return tuple(cropped)


@dataclass
class CacheCheckpoint:
    token_count: int
    next_logits: torch.Tensor


class IncrementalKVSession:
    """A single-model KV cache that supports extension and rollback.

    `sync` reuses the cache when the desired prompt extends the cached token
    sequence.  A non-prefix change safely falls back to a full rebuild.
    """

    def __init__(self, model):
        self.model = model
        self.token_ids = []
        self.past_key_values = None
        self.next_logits = None
        self.stats = {
            "initial_prefill_tokens": 0,
            "delta_prefill_tokens": 0,
            "rebuild_prefill_tokens": 0,
            "decode_tokens": 0,
            "rollback_tokens": 0,
            "sync_calls": 0,
            "cache_hits": 0,
            "cache_rebuilds": 0,
        }

    def _forward(self, input_ids, output_hidden_states=False):
        with torch.no_grad():
            return self.model(
                input_ids=input_ids,
                past_key_values=self.past_key_values,
                use_cache=True,
                output_hidden_states=output_hidden_states,
            )

    def sync(self, input_ids):
        desired = [int(value) for value in input_ids[0].detach().cpu().tolist()]
        self.stats["sync_calls"] += 1
        if desired == self.token_ids and self.next_logits is not None:
            self.stats["cache_hits"] += 1
            return

        extends_cache = self.token_ids and desired[: len(self.token_ids)] == self.token_ids
        if extends_cache:
            delta = desired[len(self.token_ids) :]
            if not delta:
                self.stats["cache_hits"] += 1
                return
            delta_ids = torch.tensor([delta], device=self.model.device)
            outputs = self._forward(delta_ids)
            self.stats["delta_prefill_tokens"] += len(delta)
            self.stats["cache_hits"] += 1
        else:
            self.past_key_values = None
            outputs = self._forward(input_ids.to(self.model.device))
            if self.token_ids:
                self.stats["rebuild_prefill_tokens"] += len(desired)
                self.stats["cache_rebuilds"] += 1
            else:
                self.stats["initial_prefill_tokens"] += len(desired)

        self.past_key_values = outputs.past_key_values
        self.next_logits = outputs.logits[0, -1, :]
        self.token_ids = desired

    def checkpoint(self):
        if self.next_logits is None:
            raise RuntimeError("Cannot checkpoint an unsynchronized KV session")
        return CacheCheckpoint(len(self.token_ids), self.next_logits)

    def restore(self, checkpoint):
        removed = max(len(self.token_ids) - checkpoint.token_count, 0)
        if removed == 0:
            self.next_logits = checkpoint.next_logits
            return
        cache = self.past_key_values
        if hasattr(cache, "crop"):
            cache.crop(checkpoint.token_count)
        elif isinstance(cache, (tuple, list)):
            cache = _crop_legacy_cache(cache, checkpoint.token_count)
        else:
            raise TypeError(f"Unsupported KV cache type for rollback: {type(cache)!r}")
        self.past_key_values = cache
        self.token_ids = self.token_ids[: checkpoint.token_count]
        self.next_logits = checkpoint.next_logits
        self.stats["rollback_tokens"] += removed

    def generate(self, max_new_tokens, eos_token_id, boundary_fn, confidence_fn=None, capture_hidden=True):
        token_ids = []
        hidden_states = []
        confidences = []
        cut_reason = None
        reached_eos = False

        for _ in range(max(int(max_new_tokens), 0)):
            selection_logits = self.next_logits
            next_id = int(torch.argmax(selection_logits).item())
            if next_id == eos_token_id:
                reached_eos = True
                break
            if confidence_fn is not None:
                confidences.append(confidence_fn(selection_logits))

            token_ids.append(next_id)
            outputs = self._forward(
                torch.tensor([[next_id]], device=self.model.device),
                output_hidden_states=capture_hidden,
            )
            self.past_key_values = outputs.past_key_values
            self.next_logits = outputs.logits[0, -1, :]
            self.token_ids.append(next_id)
            self.stats["decode_tokens"] += 1
            if capture_hidden:
                hidden_states.append(outputs.hidden_states[-1][0, -1, :].detach().to(torch.float32).cpu())

            cut_reason = boundary_fn(token_ids, next_id)
            if cut_reason is not None:
                break

        return {
            "token_ids": token_ids,
            "hidden_states": hidden_states,
            "confidences": confidences,
            "cut_reason": cut_reason,
            "reached_eos": reached_eos,
        }

    def summary(self):
        return dict(self.stats)
