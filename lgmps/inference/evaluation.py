from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from lgmps.configuration import ExperimentConfig
from lgmps.data import IndexedJsonlDataset, SequenceAssembler, pack_sequences
from lgmps.data.assembler import SpecialToken, TokenNamespace
from lgmps.nn import LGMPSModel

from .beam_search import generate_semantic_ids_batch
from .metrics import RetrievalMetrics
from .packed_ops import append_token


def resolve_checkpoint(config: ExperimentConfig, override: str = "") -> Path:
    if override:
        checkpoint = Path(override).expanduser().resolve()
    elif config.inference.checkpoint:
        checkpoint = Path(config.inference.checkpoint)
    else:
        output = Path(config.training.output_dir)
        checkpoint = output / "best.pt"
        if not checkpoint.exists():
            checkpoint = output / "latest.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    return checkpoint


def load_inference_model(
    config: ExperimentConfig,
    checkpoint: str | Path,
    device: torch.device,
) -> LGMPSModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("model", payload)
    model = LGMPSModel(config.model)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def _capture_research_artifacts(
    model: LGMPSModel,
    prefix,
    output_path: Path,
) -> None:
    context = append_token(
        prefix,
        namespace=TokenNamespace.SPECIAL,
        token_id=int(SpecialToken.ITEM_START),
        item_id=1,
    )
    gates: list[dict[str, Any]] = []
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=context.token_ids.device.type,
            dtype=torch.bfloat16,
            enabled=context.token_ids.is_cuda,
        ),
    ):
        model(context, label_split=None, capture_attention=True, gate_capture=gates.append)
    attention = []
    for layer in model.attention_captures():
        attention.append(None if layer is None else [matrix.cpu() for matrix in layer])
    routing = []
    for snapshot in model.routing_snapshots():
        routing.append(
            None
            if snapshot is None
            else {
                "counts": snapshot.counts.cpu(),
                "probability_mass": snapshot.probability_mass.cpu(),
                "category_counts": snapshot.category_counts.cpu(),
                "correction_bias": (
                    None if snapshot.correction_bias is None else snapshot.correction_bias.cpu()
                ),
                "load_ema": None if snapshot.load_ema is None else snapshot.load_ema.cpu(),
                "category_probability_mass": (
                    None
                    if snapshot.category_probability_mass is None
                    else snapshot.category_probability_mass.cpu()
                ),
                "contribution_counts": (
                    None if snapshot.contribution_counts is None else snapshot.contribution_counts.cpu()
                ),
                "routed_squared_norm_sum": (
                    None
                    if snapshot.routed_squared_norm_sum is None
                    else snapshot.routed_squared_norm_sum.cpu()
                ),
                "shared_squared_norm_sum": (
                    None
                    if snapshot.shared_squared_norm_sum is None
                    else snapshot.shared_squared_norm_sum.cpu()
                ),
                "shared_routed_dot_sum": (
                    None if snapshot.shared_routed_dot_sum is None else snapshot.shared_routed_dot_sum.cpu()
                ),
                "shared_routed_cosine_sum": (
                    None
                    if snapshot.shared_routed_cosine_sum is None
                    else snapshot.shared_routed_cosine_sum.cpu()
                ),
            }
        )
    torch.save(
        {
            "record_id": int(context.record_ids[0]),
            "user_id": int(context.user_ids[0]),
            "gates": [
                {
                    key: value.cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in record.items()
                }
                for record in gates
            ],
            "attention": attention,
            "routing": routing,
            "metadata": {
                "session_ids": context.session_ids.cpu(),
                "item_ids": context.item_ids.cpu(),
                "rope_positions": context.rope_positions.cpu(),
                "categories": context.categories.cpu(),
                "session_offsets": context.session_offsets.cpu(),
            },
        },
        output_path,
    )


def evaluate_checkpoint(
    config: ExperimentConfig,
    *,
    checkpoint_override: str = "",
    evaluation_limit: int | None = None,
    start_index: int = 0,
    end_index: int | None = None,
    index_stride: int = 1,
    index_offset: int = 0,
    capture_artifacts: bool = False,
    moe_intervention: str = "full",
) -> dict[str, Any]:
    if config.inference.return_sequences < 20:
        raise ValueError("Paper evaluation requires return_sequences >= 20 for exact Rec@20 and mAP@20")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint(config, checkpoint_override)
    model = load_inference_model(config, checkpoint, device)
    if moe_intervention != "full":
        model.set_moe_intervention(moe_intervention)
    records = IndexedJsonlDataset(
        config.data.train_file,
        config.data.index_file,
        dataset_name=config.dataset,
    )
    if start_index < 0:
        raise ValueError("start_index cannot be negative")
    if index_stride <= 0:
        raise ValueError("index_stride must be positive")
    if not 0 <= index_offset < index_stride:
        raise ValueError("index_offset must satisfy 0 <= index_offset < index_stride")
    resolved_end = len(records) if end_index is None else min(end_index, len(records))
    if resolved_end < start_index:
        raise ValueError("end_index cannot be smaller than start_index")
    assembler = SequenceAssembler(
        query_term_slots=config.data.query_term_slots,
        add_query_markers=config.data.add_query_markers,
        add_item_marker=config.data.add_item_marker,
        maximum_tokens=config.data.maximum_tokens,
        positive_event_mask=config.data.positive_event_mask,
        target_policy=config.data.target_policy,
    )
    result_path = Path(config.inference.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = RetrievalMetrics()
    limit = config.inference.evaluation_limit if evaluation_limit is None else evaluation_limit
    started = time.perf_counter()
    captured = False
    skipped_no_targets = 0
    with result_path.open("w", encoding="utf-8") as output:
        pending: list[tuple[Any, Any, Any]] = []

        def write_result(record, sequence, prediction, *, skipped_no_targets: bool = False) -> None:
            output.write(
                json.dumps(
                    {
                        "record_id": record.record_id,
                        "user_id": record.user_id,
                        "predicted_semantic_ids": ([] if prediction is None else prediction.semantic_ids),
                        "beam_scores": [] if prediction is None else prediction.scores,
                        "target_semantic_ids": sequence.target_semantic_ids,
                        "target_policy": config.data.target_policy,
                        "token_length": sequence.length,
                        "session_count": record.session_count,
                        "skipped_no_targets": skipped_no_targets,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

        def flush_pending() -> None:
            nonlocal captured
            if not pending:
                return
            predictions = generate_semantic_ids_batch(
                model,
                [prefix for _, _, prefix in pending],
                beam_size=config.inference.beam_size,
                return_sequences=config.inference.return_sequences,
                temperature=config.inference.temperature,
                use_cache=config.inference.use_cache,
            )
            for (record, sequence, prefix), prediction in zip(pending, predictions, strict=True):
                metrics.update(
                    prediction.semantic_ids,
                    sequence.target_semantic_ids,
                    token_length=sequence.length,
                    session_count=record.session_count,
                )
                write_result(record, sequence, prediction)
                if capture_artifacts and not captured:
                    _capture_research_artifacts(
                        model,
                        prefix,
                        result_path.with_suffix(result_path.suffix + ".artifacts.pt"),
                    )
                    captured = True
            pending.clear()

        for index in range(start_index, resolved_end):
            if index % index_stride != index_offset:
                continue
            if limit and metrics.examined + len(pending) >= limit:
                flush_pending()
            if limit and metrics.examined >= limit:
                break
            record = records[index]
            if record.session_count < 2:
                continue
            sequence = assembler.assemble(record, mode="test_prefix")
            if not sequence.target_semantic_ids:
                skipped_no_targets += 1
                continue
            prefix = pack_sequences([sequence]).to(device)
            pending.append((record, sequence, prefix))
            if len(pending) >= config.inference.batch_size:
                flush_pending()
        flush_pending()
    summary = metrics.summary()
    summary.update(
        {
            "checkpoint": str(checkpoint),
            "start_index": start_index,
            "end_index": resolved_end,
            "index_stride": index_stride,
            "index_offset": index_offset,
            "selected_index_count": sum(
                1 for index in range(start_index, resolved_end) if index % index_stride == index_offset
            ),
            "target_policy": config.data.target_policy,
            "moe_intervention": moe_intervention,
            "skipped_no_target_samples": skipped_no_targets,
            "elapsed_seconds": time.perf_counter() - started,
            "samples_per_second": metrics.examined / max(time.perf_counter() - started, 1e-9),
        }
    )
    metrics_path = result_path.with_suffix(result_path.suffix + ".metrics.json")
    temporary = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(metrics_path)
    return summary
