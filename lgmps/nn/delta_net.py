from __future__ import annotations

import torch
from torch import nn


class FLADeltaNet(nn.Module):
    """Packed-sequence adapter for the pinned FLA DeltaNet layer.

    The paper's ``w/ DeltaNet`` ablation keeps the session-local branch and
    replaces only the recurrent GDN modulation branch.  DeltaNet emits one
    model-width modulation vector (gamma), unlike the GDN FiLM branch which
    emits both gamma and beta.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        head_dim: int,
        num_heads: int,
        expand_v: float = 2.0,
        layer_idx: int = 0,
        mode: str = "fused_recurrent",
        use_output_gate: bool = False,
        output_bias: bool = True,
        zero_output: bool = True,
    ) -> None:
        super().__init__()
        try:
            from fla.layers.delta_net import DeltaNet
        except ImportError as error:  # pragma: no cover - optional pinned checkout
            raise ImportError(
                "FLA DeltaNet is unavailable. Install a compatible Flash Linear Attention build."
            ) from error

        key_width = num_heads * head_dim
        value_width = int(key_width * expand_v)
        self.layer = DeltaNet(
            mode=mode,
            hidden_size=hidden_size,
            expand_k=key_width / hidden_size,
            expand_v=value_width / hidden_size,
            num_heads=num_heads,
            use_beta=True,
            use_gate=use_output_gate,
            use_short_conv=True,
            layer_idx=layer_idx,
            qk_activation="silu",
            qk_norm="l2",
        )
        if output_bias:
            self.layer.o_proj = nn.Linear(self.layer.value_dim, hidden_size, bias=True)
        if zero_output:
            nn.init.zeros_(self.layer.o_proj.weight)
            if self.layer.o_proj.bias is not None:
                nn.init.zeros_(self.layer.o_proj.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        sequence_offsets: torch.Tensor,
        gate_capture: object | None = None,
    ) -> torch.Tensor:
        del gate_capture
        output, _, _ = self.layer(
            hidden.unsqueeze(0),
            attention_mask=None,
            use_cache=False,
            output_attentions=False,
            cu_seqlens=sequence_offsets.to(device=hidden.device, dtype=torch.int64),
        )
        return output.squeeze(0)

    def prefill_cached(
        self,
        hidden: torch.Tensor,
        sequence_offsets: torch.Tensor,
        past_key_values: object,
    ) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError("FLA DeltaNet cached prefill expects [tokens, hidden_size]")
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
        if hidden.ndim != 2:
            raise ValueError("FLA DeltaNet cached decode expects [beams, hidden_size]")
        output, _, _ = self.layer(
            hidden.unsqueeze(1),
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
        )
        return output.squeeze(1)


def build_delta_net(
    hidden_size: int,
    *,
    head_dim: int,
    num_heads: int,
    expand_v: float,
    layer_idx: int,
    mode: str,
    use_output_gate: bool,
) -> FLADeltaNet:
    return FLADeltaNet(
        hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        expand_v=expand_v,
        layer_idx=layer_idx,
        mode=mode,
        use_output_gate=use_output_gate,
        output_bias=True,
        zero_output=True,
    )
