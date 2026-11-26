from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, TypeVar

Architecture = Literal[
    "full_transformer",
    "session_transformer",
    "session_causal_transformer",
    "gated_deltanet",
    "mix_3_1",
    "para_chan_film",
    "para_chan_film_deltanet",
]
FeedForwardKind = Literal["dense", "category_ffn", "moe", "pcmoe", "category_moe"]
HistoryBackend = Literal["fla", "reference"]
TargetPolicy = Literal["clicked", "exposed"]
OptimizerKind = Literal["adam", "adamw"]
ScheduleKind = Literal["constant", "cosine"]
RopePositionMode = Literal["document_aligned", "sequential"]
PublicDataset = Literal["kuaisar", "jdsearch", "qilin"]

PUBLIC_DATASETS: tuple[PublicDataset, ...] = ("kuaisar", "jdsearch", "qilin")
DATASET_FILES: dict[PublicDataset, tuple[str, str]] = {
    "kuaisar": (
        "data/processed/kuaisar/session_sequences.jsonl",
        "data/indexes/kuaisar_session_sequences.npy",
    ),
    "jdsearch": (
        "data/processed/jdsearch/session_sequences.jsonl",
        "data/indexes/jdsearch_trainable_session_sequences.npy",
    ),
    "qilin": (
        "data/processed/qilin/session_sequences.jsonl",
        "data/indexes/qilin_trainable_session_sequences.npy",
    ),
}


@dataclass(slots=True)
class ModelSettings:
    attention_pattern: Architecture = "para_chan_film"
    query_vocabulary: int = 151_936
    semantic_levels: list[int] = field(default_factory=lambda: [256, 256, 256])
    width: int = 256
    layers: int = 8
    head_width: int = 64
    attention_heads: int = 8
    local_heads: int = 4
    history_heads: int = 4
    history_value_expansion: float = 2.0
    feedforward_width: int = 256
    feedforward: FeedForwardKind = "dense"
    routed_experts: int = 8
    routed_top_k: int = 1
    shared_experts: int = 0
    category_shared_experts: bool = False
    expert_width: int = 384
    moe_normalize_topk_probability: bool = True
    moe_balance_momentum: float = 0.9
    moe_balance_bias_lr: float = 0.01
    moe_balance_bias_max: float = 2.0
    dropout: float = 0.0
    rope_base: float = 1_000_000.0
    qk_normalization: bool = True
    use_rope: bool = True
    rope_position_mode: RopePositionMode = "document_aligned"
    fusion_bias: bool = False
    delta_output_gate: bool = False
    delta_mode: Literal["chunk", "fused_recurrent"] = "fused_recurrent"
    history_backend: HistoryBackend = "fla"
    use_triton_session_attention: bool = True
    use_flex_training_attention: bool = True
    tie_semantic_embedding: bool = True

    def validate(self) -> None:
        if self.width <= 0 or self.layers <= 0 or self.head_width <= 0:
            raise ValueError("Model width, layers, and head_width must be positive")
        if any(size <= 0 for size in self.semantic_levels):
            raise ValueError("Every semantic vocabulary level must be positive")
        if self.rope_position_mode not in {"document_aligned", "sequential"}:
            raise ValueError(f"Unsupported rope_position_mode={self.rope_position_mode!r}")
        if self.attention_pattern in {"para_chan_film", "para_chan_film_deltanet"}:
            if self.local_heads <= 0 or self.history_heads <= 0:
                raise ValueError("Parallel-channel models require positive local_heads and history_heads")
            if self.attention_heads != self.local_heads + self.history_heads:
                raise ValueError(
                    "attention_heads must equal local_heads + history_heads for parallel-channel models"
                )
            if self.attention_pattern == "para_chan_film_deltanet" and self.history_backend != "fla":
                raise ValueError("para_chan_film_deltanet requires history_backend='fla'")
        elif self.attention_pattern in {
            "full_transformer",
            "session_transformer",
            "session_causal_transformer",
        }:
            if self.attention_heads <= 0:
                raise ValueError("Transformer architectures require attention_heads > 0")
        elif self.attention_pattern in {"gated_deltanet", "mix_3_1"}:
            if self.history_heads <= 0:
                raise ValueError(f"{self.attention_pattern} requires history_heads > 0")
            if self.attention_pattern == "mix_3_1":
                if self.attention_heads <= 0:
                    raise ValueError("mix_3_1 requires attention_heads > 0")
                if self.layers % 4:
                    raise ValueError("mix_3_1 requires a layer count divisible by 4")
        if self.feedforward in {"category_moe", "moe", "pcmoe"} and (
            self.routed_experts <= 0 or not 1 <= self.routed_top_k <= self.routed_experts
        ):
            raise ValueError("Invalid MoE expert/top-k configuration")
        if self.shared_experts < 0:
            raise ValueError("shared_experts cannot be negative")
        if not 0.0 <= self.moe_balance_momentum < 1.0:
            raise ValueError("moe_balance_momentum must be in [0, 1)")
        if self.moe_balance_bias_lr < 0.0 or self.moe_balance_bias_max < 0.0:
            raise ValueError("MoE load-balancing bias settings cannot be negative")
        if self.feedforward in {"moe", "pcmoe"}:
            active_shared = self.shared_experts + int(self.category_shared_experts)
            if active_shared < 1:
                raise ValueError("MoE variants require at least one shared expert path")
        if self.feedforward == "moe" and self.category_shared_experts:
            raise ValueError("Vanilla MoE cannot use category-specific shared experts")
        if self.feedforward == "pcmoe" and not self.category_shared_experts:
            raise ValueError("PcMoE requires category-specific shared experts")


@dataclass(slots=True)
class DataSettings:
    train_file: str = ""
    index_file: str = ""
    maximum_tokens: int = 4096
    query_term_slots: int = 10
    add_query_markers: bool = True
    add_item_marker: bool = True
    positive_event_mask: int = 0
    target_policy: TargetPolicy = "clicked"
    loader_workers: int = 2
    prefetch_batches: int = 2
    pin_memory: bool = True

    @property
    def query_span(self) -> int:
        return self.query_term_slots + (2 if self.add_query_markers else 0)

    def validate(self) -> None:
        if self.maximum_tokens < 8:
            raise ValueError("maximum_tokens is too small")
        if self.query_term_slots <= 0:
            raise ValueError("query_term_slots must be positive")
        if self.loader_workers < 0:
            raise ValueError("loader_workers cannot be negative")


@dataclass(slots=True)
class TrainingSettings:
    output_dir: str = "outputs/run"
    epochs: int = 20
    batch_size: int = 1
    gradient_accumulation: int = 1
    optimizer: OptimizerKind = "adamw"
    learning_rate: float = 1e-4
    learning_rate_schedule: ScheduleKind = "cosine"
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    warmup_steps: int = 0
    maximum_steps: int = 0
    maximum_records: int = 0
    log_every: int = 10
    checkpoint_every: int = 1000
    validate_every: int = 0
    early_stop_patience: int = 2
    seed: int = 42
    precision: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    compile_model: bool = False
    resume_from: str = ""
    resume_mode: Literal["full", "weights_only"] = "full"
    gate_capture_every: int = 0

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.gradient_accumulation <= 0:
            raise ValueError("epochs, batch_size, and gradient_accumulation must be positive")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning_rate and gradient_clip must be positive")
        if self.maximum_steps < 0 or self.maximum_records < 0:
            raise ValueError("maximum_steps and maximum_records cannot be negative")


@dataclass(slots=True)
class InferenceSettings:
    checkpoint: str = ""
    beam_size: int = 20
    return_sequences: int = 20
    temperature: float = 1.0
    evaluation_limit: int = 0
    batch_size: int = 1
    use_cache: bool = True
    result_file: str = "outputs/evaluation/predictions.jsonl"

    def validate(self) -> None:
        if self.beam_size <= 0 or self.return_sequences <= 0:
            raise ValueError("beam_size and return_sequences must be positive")
        if self.return_sequences > self.beam_size:
            raise ValueError("return_sequences cannot exceed beam_size")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "lgmps"
    dataset: str = ""
    model: ModelSettings = field(default_factory=ModelSettings)
    data: DataSettings = field(default_factory=DataSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    source_path: str = field(default="", repr=False)

    def validate(self) -> None:
        if self.dataset and self.dataset not in PUBLIC_DATASETS:
            raise ValueError(f"Unsupported public dataset={self.dataset!r}")
        self.model.validate()
        self.data.validate()
        self.training.validate()
        self.inference.validate()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("source_path", None)
        return payload

    def resolve_paths(self, repository_root: Path) -> None:
        for section, names in (
            (self.data, ("train_file", "index_file")),
            (self.training, ("output_dir", "resume_from")),
            (self.inference, ("checkpoint", "result_file")),
        ):
            for name in names:
                raw = getattr(section, name)
                if not raw:
                    continue
                path = Path(raw).expanduser()
                if not path.is_absolute():
                    path = repository_root / path
                setattr(section, name, str(path.resolve()))

    def select_dataset(self, dataset: PublicDataset) -> None:
        """Bind a shared public-model configuration to one dataset and output namespace."""
        if dataset not in PUBLIC_DATASETS:
            raise ValueError(f"Unsupported public dataset={dataset!r}")
        train_file, index_file = DATASET_FILES[dataset]
        self.dataset = dataset
        self.data.train_file = train_file
        self.data.index_file = index_file
        run_directory = f"outputs/{dataset}/{self.name}"
        self.training.output_dir = run_directory
        self.inference.result_file = f"{run_directory}/evaluation/predictions.jsonl"


T = TypeVar("T")


def _strict_dataclass(cls: type[T], payload: dict[str, Any], section: str) -> T:
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"Unknown keys in {section}: {', '.join(unknown)}")
    return cls(**payload)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_experiment(
    path: str | Path,
    *,
    dataset: PublicDataset | None = None,
    resolve_paths: bool = True,
) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("Experiment JSON must contain an object")
    accepted = {"name", "dataset", "model", "data", "training", "inference"}
    unknown = sorted(set(raw) - accepted)
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {', '.join(unknown)}")
    config = ExperimentConfig(
        name=str(raw.get("name", config_path.stem)),
        dataset=str(raw.get("dataset", "")),
        model=_strict_dataclass(ModelSettings, raw.get("model", {}), "model"),
        data=_strict_dataclass(DataSettings, raw.get("data", {}), "data"),
        training=_strict_dataclass(TrainingSettings, raw.get("training", {}), "training"),
        inference=_strict_dataclass(InferenceSettings, raw.get("inference", {}), "inference"),
        source_path=str(config_path),
    )
    selected_dataset = dataset or config.dataset
    if selected_dataset:
        config.select_dataset(selected_dataset)  # type: ignore[arg-type]
    config.validate()
    if resolve_paths:
        config.resolve_paths(repository_root())
    return config
