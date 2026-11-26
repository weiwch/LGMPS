from __future__ import annotations

import torch


def rotary_frequencies(
    positions: torch.Tensor,
    head_width: int,
    *,
    base: float = 1_000_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_width % 2:
        raise ValueError("Rotary head width must be even")
    inverse = 1.0 / (
        base ** (torch.arange(0, head_width, 2, device=positions.device, dtype=torch.float32) / head_width)
    )
    phase = positions.float().unsqueeze(-1) * inverse.unsqueeze(0)
    return phase.cos(), phase.sin()


def apply_rotary(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    *,
    base: float = 1_000_000.0,
) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError("Rotary input must have shape [tokens, heads, head_width]")
    cosine, sine = rotary_frequencies(positions, tensor.shape[-1], base=base)
    cosine = cosine[:, None, :].to(tensor.dtype)
    sine = sine[:, None, :].to(tensor.dtype)
    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    rotated_even = even * cosine - odd * sine
    rotated_odd = odd * cosine + even * sine
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
