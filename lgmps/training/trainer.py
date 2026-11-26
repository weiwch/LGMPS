from __future__ import annotations

import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from lgmps.configuration import ExperimentConfig
from lgmps.data import IndexedJsonlDataset, SequenceAssembler
from lgmps.nn import LGMPSModel

from .artifacts import ArtifactSink
from .checkpoint import ResumeState, load_checkpoint, save_checkpoint, unwrap_model
from .data import EpochRandomSampler, TrainingSequenceDataset, collate_sequences


def _initialize_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast_settings(precision: str, device: torch.device) -> tuple[bool, torch.dtype]:
    if precision == "float16":
        return device.type == "cuda", torch.float16
    if precision == "bfloat16":
        return device.type in {"cuda", "cpu"}, torch.bfloat16
    return False, torch.float32


def _linear_warmup_cosine(step: int, *, warmup: int, total: int) -> float:
    if warmup and step < warmup:
        return max(step, 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def _constant_with_warmup(step: int, *, warmup: int) -> float:
    if warmup and step < warmup:
        return max(step, 1) / warmup
    return 1.0


def _topk_metrics(
    model: LGMPSModel,
    logits: list[torch.Tensor],
    labels: torch.Tensor,
    breakpoints: tuple[int, ...] = (1, 5, 20),
) -> dict[str, float | int]:
    result: dict[str, float | int] = {"labels": 0}
    hits = {cutoff: 0 for cutoff in breakpoints}
    for level, level_logits in enumerate(logits):
        lower = model.semantic_offsets[level] + 1
        upper = model.semantic_offsets[level + 1]
        selected = (labels >= lower) & (labels <= upper)
        if not selected.any():
            continue
        target = labels[selected] - lower
        maximum = min(max(breakpoints), level_logits.shape[-1])
        candidates = torch.topk(level_logits[selected], k=maximum, dim=-1).indices
        result["labels"] = int(result["labels"]) + int(target.numel())
        for cutoff in breakpoints:
            width = min(cutoff, maximum)
            hits[cutoff] += int((candidates[:, :width] == target[:, None]).any(dim=-1).sum().item())
    denominator = max(int(result["labels"]), 1)
    for cutoff in breakpoints:
        result[f"accuracy_at_{cutoff}"] = hits[cutoff] / denominator
    return result


class Trainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rank, self.world_size, self.local_rank = _initialize_distributed()
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            self.device = torch.device("cpu")
        _seed_everything(config.training.seed + self.rank)
        self.is_primary = self.rank == 0
        self.sink = ArtifactSink(config.training.output_dir, enabled=self.is_primary)
        self.sink.write_json("resolved_config.json", config.to_dict())

        records = IndexedJsonlDataset(
            config.data.train_file,
            config.data.index_file,
            dataset_name=config.dataset,
        )
        assembler = SequenceAssembler(
            query_term_slots=config.data.query_term_slots,
            add_query_markers=config.data.add_query_markers,
            add_item_marker=config.data.add_item_marker,
            maximum_tokens=config.data.maximum_tokens,
            positive_event_mask=config.data.positive_event_mask,
            target_policy=config.data.target_policy,
        )
        dataset: torch.utils.data.Dataset = TrainingSequenceDataset(records, assembler)
        if config.training.maximum_records:
            dataset = Subset(dataset, range(min(config.training.maximum_records, len(dataset))))
        if self.world_size > 1:
            self.sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=config.training.seed,
                drop_last=False,
            )
        else:
            self.sampler = EpochRandomSampler(dataset, config.training.seed)
        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": config.training.batch_size,
            "sampler": self.sampler,
            "num_workers": config.data.loader_workers,
            "pin_memory": config.data.pin_memory and self.device.type == "cuda",
            "collate_fn": collate_sequences,
            "persistent_workers": config.data.loader_workers > 0,
        }
        if config.data.loader_workers > 0:
            loader_kwargs["prefetch_factor"] = config.data.prefetch_batches
        self.loader = DataLoader(**loader_kwargs)

        raw_model = LGMPSModel(config.model).to(self.device)
        self.model: nn.Module = raw_model
        optimizer_class = torch.optim.Adam if config.training.optimizer == "adam" else torch.optim.AdamW
        self.optimizer = optimizer_class(
            raw_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        steps_per_epoch = math.ceil(len(self.loader) / config.training.gradient_accumulation)
        planned_steps = config.training.epochs * steps_per_epoch
        if config.training.maximum_steps:
            planned_steps = min(planned_steps, config.training.maximum_steps)
        if config.training.learning_rate_schedule == "constant":
            schedule = lambda step: _constant_with_warmup(
                step,
                warmup=config.training.warmup_steps,
            )
        else:
            schedule = lambda step: _linear_warmup_cosine(
                step,
                warmup=config.training.warmup_steps,
                total=max(planned_steps, 1),
            )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=config.training.precision == "float16" and self.device.type == "cuda",
        )
        self.state = ResumeState()
        if config.training.resume_from:
            self.state = load_checkpoint(
                config.training.resume_from,
                model=raw_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                mode=config.training.resume_mode,
            )
        if config.training.compile_model:
            self.model = torch.compile(self.model)
        if self.world_size > 1:
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank])
        self.autocast_enabled, self.autocast_dtype = _autocast_settings(
            config.training.precision,
            self.device,
        )

    @property
    def base_model(self) -> LGMPSModel:
        model = unwrap_model(self.model)
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    def _reduce_level_statistics(
        self, loss_sums: list[float], counts: list[int]
    ) -> tuple[list[float], float]:
        packed = torch.tensor([*loss_sums, *counts], dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        level_count = len(loss_sums)
        sums = packed[:level_count].cpu().tolist()
        totals = packed[level_count:].cpu().tolist()
        means = [value / count if count else float("nan") for value, count in zip(sums, totals, strict=True)]
        active = [value for value in means if math.isfinite(value)]
        return means, sum(active) / len(active) if active else float("nan")

    def _save(self, filename: str, *, next_epoch: int, next_batch_index: int) -> None:
        if not self.is_primary:
            return
        checkpoint_state = ResumeState(
            epoch=next_epoch,
            next_batch_index=next_batch_index,
            global_step=self.state.global_step,
            best_validation_loss=self.state.best_validation_loss,
            epochs_without_improvement=self.state.epochs_without_improvement,
        )
        save_checkpoint(
            Path(self.config.training.output_dir) / filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            state=checkpoint_state,
            config=self.config.to_dict(),
        )

    def _write_moe(self) -> None:
        if not self.is_primary:
            return
        layers = []
        for snapshot in self.base_model.routing_snapshots():
            if snapshot is None:
                layers.append(None)
            else:
                layer = {
                    "counts": snapshot.counts.cpu().tolist(),
                    "probability_mass": snapshot.probability_mass.cpu().tolist(),
                    "category_counts": snapshot.category_counts.cpu().tolist(),
                }
                if snapshot.correction_bias is not None:
                    layer["correction_bias"] = snapshot.correction_bias.cpu().tolist()
                if snapshot.load_ema is not None:
                    layer["load_ema"] = snapshot.load_ema.cpu().tolist()
                if snapshot.category_probability_mass is not None:
                    layer["category_probability_mass"] = snapshot.category_probability_mass.cpu().tolist()
                layers.append(layer)
        if any(layer is not None for layer in layers):
            self.sink.write_jsonl("moe_routing.jsonl", {"step": self.state.global_step, "layers": layers})

    def train(self) -> dict[str, Any]:
        settings = self.config.training
        self.optimizer.zero_grad(set_to_none=True)
        maximum_reached = False
        started = time.perf_counter()
        for epoch in range(self.state.epoch, settings.epochs):
            self.sampler.set_epoch(epoch)
            self.model.train()
            train_sums = [0.0] * len(self.config.model.semantic_levels)
            train_counts = [0] * len(train_sums)
            validation_sums = [0.0] * len(train_sums)
            validation_counts = [0] * len(train_sums)
            resume_batch = self.state.next_batch_index if epoch == self.state.epoch else 0
            for batch_index, batch in enumerate(self.loader):
                if batch_index < resume_batch:
                    continue
                batch = batch.to(self.device, non_blocking=True)
                should_capture = (
                    settings.gate_capture_every > 0
                    and (self.state.global_step + 1) % settings.gate_capture_every == 0
                )
                gate_records: list[dict[str, Any]] = []
                sync_step = (batch_index + 1) % settings.gradient_accumulation == 0 or batch_index + 1 == len(
                    self.loader
                )
                no_sync = hasattr(self.model, "no_sync") and not sync_step
                synchronization = self.model.no_sync() if no_sync else nullcontext()
                with synchronization:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self.autocast_dtype,
                        enabled=self.autocast_enabled,
                    ):
                        output = self.model(
                            batch,
                            label_split="train",
                            gate_capture=gate_records.append if should_capture else None,
                        )
                        if output.loss is None:
                            continue
                        remainder = len(self.loader) % settings.gradient_accumulation
                        final_group = remainder and batch_index >= len(self.loader) - remainder
                        accumulation_divisor = remainder if final_group else settings.gradient_accumulation
                        scaled_loss = output.loss / accumulation_divisor
                    self.scaler.scale(scaled_loss).backward()

                for level, (loss, count) in enumerate(
                    zip(output.level_losses, output.level_label_counts, strict=True)
                ):
                    train_sums[level] += float(loss.detach()) * count
                    train_counts[level] += count
                validation_loss, validation_levels, _, validation_level_counts = (
                    self.base_model.losses_for_labels(
                        output.level_logits,
                        batch.validation_labels,
                    )
                )
                for level, (loss, count) in enumerate(
                    zip(validation_levels, validation_level_counts, strict=True)
                ):
                    validation_sums[level] += float(loss.detach()) * count
                    validation_counts[level] += count

                if not sync_step:
                    continue
                self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.base_model.parameters(), settings.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.state.global_step += 1

                if self.is_primary and self.state.global_step % settings.log_every == 0:
                    topk = _topk_metrics(
                        self.base_model,
                        output.level_logits,
                        batch.train_labels,
                    )
                    self.sink.write_jsonl(
                        "loss.jsonl",
                        {
                            "epoch": epoch,
                            "batch_index": batch_index,
                            "step": self.state.global_step,
                            "loss": float(output.loss.detach()),
                            "validation_loss": None
                            if validation_loss is None
                            else float(validation_loss.detach()),
                            "learning_rate": self.scheduler.get_last_lr()[0],
                            "gradient_norm": float(gradient_norm),
                            "tokens": batch.token_count,
                            **topk,
                        },
                    )
                    self._write_moe()
                if should_capture and self.is_primary:
                    self.sink.write_gate_capture(
                        self.state.global_step,
                        gate_records,
                        {
                            "record_ids": batch.record_ids,
                            "user_ids": batch.user_ids,
                            "sequence_offsets": batch.sequence_offsets,
                            "session_ids": batch.session_ids,
                            "item_ids": batch.item_ids,
                            "rope_positions": batch.rope_positions,
                            "categories": batch.categories,
                        },
                    )
                if settings.checkpoint_every and self.state.global_step % settings.checkpoint_every == 0:
                    self._save("latest.pt", next_epoch=epoch, next_batch_index=batch_index + 1)
                if settings.maximum_steps and self.state.global_step >= settings.maximum_steps:
                    self._save("latest.pt", next_epoch=epoch, next_batch_index=batch_index + 1)
                    maximum_reached = True
                    break

            train_level_means, train_mean = self._reduce_level_statistics(train_sums, train_counts)
            validation_level_means, validation_mean = self._reduce_level_statistics(
                validation_sums,
                validation_counts,
            )
            if self.is_primary:
                improved = (
                    math.isfinite(validation_mean) and validation_mean < self.state.best_validation_loss
                )
                if improved:
                    self.state.best_validation_loss = validation_mean
                    self.state.epochs_without_improvement = 0
                    self._save("best.pt", next_epoch=epoch + 1, next_batch_index=0)
                else:
                    self.state.epochs_without_improvement += 1
                self.sink.write_jsonl(
                    "epoch_metrics.jsonl",
                    {
                        "epoch": epoch,
                        "step": self.state.global_step,
                        "train_loss": train_mean,
                        "train_level_losses": train_level_means,
                        "validation_loss": validation_mean,
                        "validation_level_losses": validation_level_means,
                        "best_validation_loss": self.state.best_validation_loss,
                    },
                )
                if maximum_reached:
                    self._save("latest.pt", next_epoch=epoch, next_batch_index=batch_index + 1)
                else:
                    self._save("latest.pt", next_epoch=epoch + 1, next_batch_index=0)
            stop = maximum_reached or self.state.epochs_without_improvement >= settings.early_stop_patience
            if self.world_size > 1:
                stop_tensor = torch.tensor(int(stop) if self.is_primary else 0, device=self.device)
                dist.broadcast(stop_tensor, src=0)
                stop = bool(stop_tensor.item())
            self.state.next_batch_index = 0
            if stop:
                break
        result = {
            "global_step": self.state.global_step,
            "best_validation_loss": self.state.best_validation_loss,
            "elapsed_seconds": time.perf_counter() - started,
            "output_dir": self.config.training.output_dir,
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.base_model.parameters() if parameter.requires_grad
            ),
        }
        if self.is_primary:
            self.sink.write_json("training_summary.json", result)
        if self.world_size > 1:
            dist.barrier()
        return result


def run_training(config: ExperimentConfig) -> dict[str, Any]:
    return Trainer(config).train()
