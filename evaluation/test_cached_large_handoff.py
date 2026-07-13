import types
import unittest
from unittest import mock

import torch

from core_package.schedulers import simulate_observe_rollback_scheduler as gsm_scheduler
from core_package.schedulers import simulate_observe_rollback_scheduler_svamp as svamp_scheduler


class FakeTokenizer:
    eos_token_id = 0

    def decode(self, token_ids, skip_special_tokens=True):
        mapping = {1: "a", 2: ".", 3: "b", 4: "."}
        return "".join(mapping.get(int(token_id), "") for token_id in token_ids)


class FakeModel:
    device = torch.device("cpu")

    def __init__(self):
        self.next_tokens = iter([1, 2, 3, 4])
        self.input_lengths = []

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        self.input_lengths.append(int(input_ids.shape[1]))
        next_token = next(self.next_tokens)
        logits = torch.full((1, 1, 5), -1000.0)
        logits[0, 0, next_token] = 0.0
        return types.SimpleNamespace(logits=logits, past_key_values=object())


def make_args():
    return types.SimpleNamespace(
        large_handoff_chunks=2,
        max_new_tokens=16,
        large_backend="hf",
        system_prompt="",
        answer_type="gsm8k_boxed_numeric",
        runtime_chunking="punctuation",
        runtime_step_word="\n\n",
        rewrite_step_force_tokens=8,
        rewrite_step_target_tokens=4,
        rewrite_step_min_tokens=2,
        rewrite_step_boundary_mode="auto",
        min_chunk_tokens=2,
        max_chunk_tokens=8,
    )


class CachedLargeHandoffTest(unittest.TestCase):
    def check_scheduler(self, scheduler):
        model = FakeModel()
        tokenizer = FakeTokenizer()
        prompt_ids = torch.tensor([[9, 8, 7, 6]])
        with mock.patch.object(
            scheduler,
            "build_generation_inputs",
            return_value=(types.SimpleNamespace(input_ids=prompt_ids), "prefix "),
        ):
            result = scheduler.run_large_handoff(
                model=model,
                tokenizer=tokenizer,
                question="q",
                assistant_prefix="prefix ",
                args=make_args(),
                num_chunks=2,
                max_total_new_tokens=16,
            )

        self.assertEqual(model.input_lengths, [4, 1, 1, 1])
        self.assertEqual(result["generated_chunks"], 2)
        self.assertEqual(result["generated_token_count"], 4)
        self.assertEqual([chunk["chunk_text"] for chunk in result["chunks"]], ["a.", "b."])
        self.assertEqual(result["full_reasoning"], "prefix a.b.")

    def test_gsm_math_scheduler_reuses_prefix_cache(self):
        self.check_scheduler(gsm_scheduler)

    def test_svamp_scheduler_reuses_prefix_cache(self):
        self.check_scheduler(svamp_scheduler)


if __name__ == "__main__":
    unittest.main()
