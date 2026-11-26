from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

GateCapture = Callable[[dict[str, Any]], None]


class ReferenceGatedDeltaNet(nn.Module):
    """Readable recurrent Gated DeltaNet oracle for CPU tests and ablations."""

    def __init__(
        self,
        hidden_size: int,
        *,
        head_dim: int,
        num_heads: int,
        expand_v: float = 2.0,
        layer_idx: int = 0,
        output_multiplier: int = 1,
        output_bias: bool = False,
        zero_output: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * head_dim
        self.value_dim = num_heads * self.head_v_dim
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.output_gate = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.output_norm = nn.RMSNorm(self.head_v_dim)
        self.o_proj = nn.Linear(self.value_dim, hidden_size * output_multiplier, bias=output_bias)
        self.A_log = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.full((num_heads,), -3.0, dtype=torch.float32))
        if zero_output:
            nn.init.zeros_(self.o_proj.weight)
            if self.o_proj.bias is not None:
                nn.init.zeros_(self.o_proj.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        sequence_offsets: torch.Tensor,
        gate_capture: GateCapture | None = None,
    ) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError("ReferenceGatedDeltaNet expects packed [tokens, hidden_size] input")
        token_count = hidden.shape[0]
        q = F.normalize(self.q_proj(hidden).view(token_count, self.num_heads, self.head_dim).float(), dim=-1)
        k = F.normalize(self.k_proj(hidden).view(token_count, self.num_heads, self.head_dim).float(), dim=-1)
        v = self.v_proj(hidden).view(token_count, self.num_heads, self.head_v_dim).float()
        a_logits = self.a_proj(hidden).float()
        beta_logits = self.b_proj(hidden).float()
        alpha = torch.exp(-self.A_log.exp() * F.softplus(a_logits + self.dt_bias))
        beta = torch.sigmoid(beta_logits)
        outputs: list[torch.Tensor] = []
        boundaries = sequence_offsets.detach().cpu().tolist()
        for start, end in pairwise(boundaries):
            state = hidden.new_zeros((self.num_heads, self.head_dim, self.head_v_dim), dtype=torch.float32)
            for position in range(start, end):
                state = alpha[position, :, None, None] * state
                prediction = torch.einsum("hd,hdv->hv", k[position], state)
                residual = v[position] - prediction
                state = state + beta[position, :, None, None] * torch.einsum(
                    "hd,hv->hdv", k[position], residual
                )
                outputs.append(torch.einsum("hd,hdv->hv", q[position], state))
        recurrent = torch.stack(outputs, dim=0).to(hidden.dtype)
        gate = F.silu(self.output_gate(hidden).view(token_count, self.num_heads, self.head_v_dim))
        recurrent = self.output_norm(recurrent) * gate
        projected = self.o_proj(recurrent.flatten(1))
        if gate_capture is not None:
            gate_capture(
                {
                    "layer_idx": self.layer_idx,
                    "alpha": alpha.detach(),
                    "beta": beta.detach(),
                    "alpha_logits": a_logits.detach(),
                    "beta_logits": beta_logits.detach(),
                    "effective_beta_scale": 1.0,
                    "layout": "packed",
                    "cu_seqlens": sequence_offsets.detach(),
                    "indices": None,
                    "original_shape": (1, token_count),
                    "backend": "reference",
                }
            )
        return projected


class FLAGatedDeltaNet(nn.Module):
    """Small adapter around the pinned FLA layer with packed-sequence support."""

    def __init__(
        self,
        hidden_size: int,
        *,
        head_dim: int,
        num_heads: int,
        expand_v: float = 2.0,
        layer_idx: int = 0,
        output_multiplier: int = 1,
        output_bias: bool = False,
        zero_output: bool = False,
    ) -> None:
        super().__init__()
        try:
            from fla.layers.gated_deltanet import GatedDeltaNet
        except ImportError as error:  # pragma: no cover - depends on optional pinned checkout
            raise ImportError(
                "FLA is unavailable. Install a compatible Flash Linear Attention build "
                "or select history_backend='reference'."
            ) from error
        self.layer = GatedDeltaNet(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_v_heads=num_heads,
            layer_idx=layer_idx,
            use_short_conv=True,
            expand_v=expand_v,
        )
        if output_multiplier != 1 or output_bias:
            self.layer.o_proj = nn.Linear(
                self.layer.value_dim,
                hidden_size * output_multiplier,
                bias=output_bias,
            )
        if zero_output:
            nn.init.zeros_(self.layer.o_proj.weight)
            if self.layer.o_proj.bias is not None:
                nn.init.zeros_(self.layer.o_proj.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        sequence_offsets: torch.Tensor,
        gate_capture: GateCapture | None = None,
    ) -> torch.Tensor:
        output, _, _ = self.layer(
            hidden.unsqueeze(0),
            attention_mask=None,
            use_cache=False,
            output_attentions=False,
            cu_seqlens=sequence_offsets.to(device=hidden.device, dtype=torch.int64),
            gate_capture=gate_capture,
        )
        return output.squeeze(0)

    def prefill_cached(
        self,
        hidden: torch.Tensor,
        sequence_offsets: torch.Tensor,
        past_key_values: object,
    ) -> torch.Tensor:
        """Run one dense prefix and populate the FLA recurrent/short-conv cache."""
        if hidden.ndim != 2:
            raise ValueError("FLA cached prefill expects [tokens, hidden_size]")
        output, _, _ = self.layer(
            hidden.unsqueeze(0),
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            cu_seqlens=sequence_offsets.to(device=hidden.device, dtype=torch.int64),
        )
        return output.squeeze(0)

    def decode_cached(self, hidden: torch.Tensor, past_key_values: object) -> torch.Tensor:
        """Advance the FLA state by one token for every beam."""
        if hidden.ndim != 2:
            raise ValueError("FLA cached decode expects [beams, hidden_size]")
        output, _, _ = self.layer(
            hidden.unsqueeze(1),
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
        )
        return output.squeeze(1)


def build_gated_delta(
    backend: str,
    hidden_size: int,
    *,
    head_dim: int,
    num_heads: int,
    expand_v: float,
    layer_idx: int,
    output_multiplier: int = 1,
    output_bias: bool = False,
    zero_output: bool = False,
) -> nn.Module:
    implementation = FLAGatedDeltaNet if backend == "fla" else ReferenceGatedDeltaNet
    return implementation(
        hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        expand_v=expand_v,
        layer_idx=layer_idx,
        output_multiplier=output_multiplier,
        output_bias=output_bias,
        zero_output=zero_output,
    )
