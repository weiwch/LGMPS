from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FLA kernels")
def test_fla_gate_capture_is_observational_and_matches_formula():
    pytest.importorskip("fla")
    from fla.layers.gated_deltanet import GatedDeltaNet

    torch.manual_seed(42)
    layer = (
        GatedDeltaNet(
            hidden_size=32,
            head_dim=8,
            num_heads=2,
            num_v_heads=2,
            expand_v=2,
            layer_idx=3,
        )
        .cuda()
        .bfloat16()
        .eval()
    )
    hidden = torch.randn(1, 16, 32, device="cuda", dtype=torch.bfloat16)
    captures = []
    with torch.inference_mode():
        expected, _, _ = layer(hidden)
        actual, attention, cache = layer(hidden, gate_capture=captures.append)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert attention is None and cache is None
    assert len(captures) == 1
    capture = captures[0]
    alpha_logits = layer.a_proj(hidden).float()
    beta_logits = layer.b_proj(hidden).float()
    alpha = torch.exp(-layer.A_log.float().exp() * F.softplus(alpha_logits + layer.dt_bias.float()))
    torch.testing.assert_close(capture["alpha"], alpha)
    torch.testing.assert_close(capture["beta"], torch.sigmoid(beta_logits))
    assert capture["layer_idx"] == 3
    assert not capture["alpha"].requires_grad
