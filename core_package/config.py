import os
from dataclasses import dataclass, field
from typing import List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class PathsConfig:
    project_root: str = PROJECT_ROOT
    dataset_dir: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "dataset"))
    result_dir: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "result"))
    artifacts_dir: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "result", "artifacts"))
    traces_dir: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "result", "traces"))
    analysis_outputs_dir: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "result", "analysis_outputs"))
    models_dir: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "models"))


@dataclass(frozen=True)
class ModelConfig:
    small_model_name: str = "Qwen2.5-1.5B"
    large_model_name: str = "Qwen2.5-32B"
    math_small_model_name: str = "Qwen2.5-Math-1.5B-Instruct"
    math_large_model_name: str = "Qwen2.5-Math-32B-Instruct"
    system_prompt: str = "You are a helpful math assistant. Please reason step by step."
    boxed_math_system_prompt: str = (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )

    @property
    def small_model_path(self) -> str:
        return os.path.join(PATHS.models_dir, self.small_model_name)

    @property
    def large_model_path(self) -> str:
        return os.path.join(PATHS.models_dir, self.large_model_name)

    @property
    def math_small_model_path(self) -> str:
        return os.path.join(PATHS.models_dir, self.math_small_model_name)

    @property
    def math_large_model_path(self) -> str:
        return os.path.join(PATHS.models_dir, self.math_large_model_name)


@dataclass(frozen=True)
class DatasetBuildConfig:
    save_path: str = "gsm8k_15b_hidden_states.pt"
    num_samples: int = 1000
    max_new_tokens: int = 256
    min_tokens: int = 5
    max_tokens: int = 30
    punctuations: List[str] = field(default_factory=lambda: [".", ",", "!", "?", "\n"])


@dataclass(frozen=True)
class StrictLabelConfig:
    input_path: str = "gsm8k_15b_hidden_states.pt"
    output_path: str = field(default_factory=lambda: os.path.join(PATHS.dataset_dir, "gsm8k_labeled_training_data_strict.pt"))
    max_judge_tokens: int = 128
    save_every: int = 10


@dataclass(frozen=True)
class ProbeTrainConfig:
    label_path: str = field(default_factory=lambda: os.path.join(PATHS.dataset_dir, "gsm8k_labeled_training_data_strict.pt"))
    output_path: str = field(default_factory=lambda: os.path.join(PATHS.artifacts_dir, "probe_artifact_torch.pt"))
    feature_key: str = "boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob"
    label_key: str = "label"
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 55
    hidden_layers: str = "128,32"
    dropout: float = 0.1
    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    pos_weight: float = 1.0
    low_entropy_error_weight: float = 4.0
    early_stopping_patience: int = 8
    min_epochs: int = 10


@dataclass(frozen=True)
class SchedulerConfig:
    label_path: str = field(default_factory=lambda: os.path.join(PATHS.dataset_dir, "gsm8k_labeled_training_data_strict.pt"))
    probe_artifact_path: str = field(default_factory=lambda: os.path.join(PATHS.artifacts_dir, "probe_artifact_torch.pt"))
    feature_key: str = "boundary+mean+relative_position+final_entropy+final_margin+final_top1_prob"
    test_size: float = 0.2
    random_state: int = 55
    mlp_hidden_layers: str = "512,128,32"
    mlp_max_iter: int = 300
    mlp_alpha: float = 1e-4
    mlp_learning_rate_init: float = 1e-3
    thresholds: str = "0.15,0.20,0.25"
    max_new_tokens: int = 768
    min_chunk_tokens: int = 5
    max_chunk_tokens: int = 30
    tail_bonus_weight: float = 0.0
    max_handoffs: int = 2
    large_handoff_chunks: int = 2
    cooldown_chunks: int = 1


@dataclass(frozen=True)
class EvaluationConfig:
    label_path: str = field(default_factory=lambda: os.path.join(PATHS.dataset_dir, "gsm8k_labeled_training_data_strict.pt"))
    artifact_path: str = field(default_factory=lambda: os.path.join(PATHS.artifacts_dir, "probe_artifact_torch.pt"))
    trace_path: str = field(default_factory=lambda: os.path.join(PATHS.traces_dir, "observe_rollback_traces_768.json"))
    max_new_tokens: int = 768


PATHS = PathsConfig()
MODELS = ModelConfig()
DATASET_BUILD = DatasetBuildConfig()
STRICT_LABEL = StrictLabelConfig()
PROBE_TRAIN = ProbeTrainConfig()
SCHEDULER = SchedulerConfig()
EVALUATION = EvaluationConfig()
