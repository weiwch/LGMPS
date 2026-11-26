from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(slots=True)
class ResumeState:
    epoch: int = 0
    next_batch_index: int = 0
    global_step: int = 0
    best_validation_loss: float = float("inf")
    epochs_without_improvement: int = 0


def unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = model.module if hasattr(model, "module") else model._orig_mod
    return model


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    state: ResumeState,
    config: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    torch.save(
        {
            "format_version": 1,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "resume_state": {
                "epoch": state.epoch,
                "next_batch_index": state.next_batch_index,
                "global_step": state.global_step,
                "best_validation_loss": state.best_validation_loss,
                "epochs_without_improvement": state.epochs_without_improvement,
            },
            "rng_state": _rng_state(),
            "config": config,
        },
        temporary,
    )
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    mode: str = "full",
) -> ResumeState:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(payload["model"], strict=True)
    if mode == "weights_only":
        return ResumeState()
    if optimizer is None or scheduler is None or scaler is None:
        raise ValueError("Full resume requires optimizer, scheduler, and scaler")
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload.get("scaler", {}))
    raw = payload["resume_state"]
    state = ResumeState(**raw)
    _restore_rng(payload["rng_state"])
    return state
