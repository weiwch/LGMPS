from __future__ import annotations

import json

import torch

from lgmps.nn.feedforward import AuxiliaryLossFreeMoE, CategoryFFN, CategoryMoE
from lgmps.visualization.moe import render_moe


def test_category_moe_routing_and_renderer(tmp_path):
    torch.manual_seed(42)
    moe = CategoryMoE(width=8, expert_width=12, experts=4, top_k=2)
    hidden = torch.randn(6, 8, requires_grad=True)
    categories = torch.tensor([0, 0, 1, 1, 1, 0])
    output = moe(hidden, categories)
    output.square().mean().backward()
    snapshot = moe.last_routing
    assert snapshot is not None
    assert int(snapshot.counts.sum()) == 12
    assert int(snapshot.category_counts.sum()) == 12
    log = tmp_path / "moe.jsonl"
    log.write_text(
        json.dumps(
            {
                "step": 1,
                "layers": [
                    {
                        "counts": snapshot.counts.tolist(),
                        "probability_mass": snapshot.probability_mass.tolist(),
                        "category_counts": snapshot.category_counts.tolist(),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rendered = render_moe(log, tmp_path / "figures")
    assert len(rendered) == 1
    assert rendered[0].stat().st_size > 0


def test_auxiliary_loss_free_moe_budget_balance_and_interventions():
    torch.manual_seed(7)
    moe = AuxiliaryLossFreeMoE(
        width=8,
        expert_width=12,
        routed_experts=3,
        top_k=1,
        shared_experts=1,
    )
    assert moe.total_expert_count == 4
    assert moe.active_expert_count == 2
    hidden = torch.randn(9, 8, requires_grad=True)
    categories = torch.tensor([0, 1, 1, 0, 1, 0, 1, 1, 0])
    output = moe(hidden, categories)
    output.square().mean().backward()

    snapshot = moe.last_routing
    assert snapshot is not None
    assert int(snapshot.counts.sum()) == hidden.shape[0]
    assert int(snapshot.category_counts.sum()) == hidden.shape[0]
    assert snapshot.correction_bias is not None
    assert snapshot.load_ema is not None
    assert moe.correction_bias.requires_grad is False
    assert moe.router.weight.grad is not None

    moe.eval()
    moe.set_intervention("full")
    full = moe(hidden.detach(), categories)
    moe.set_intervention("shared_only")
    shared = moe(hidden.detach(), categories)
    moe.set_intervention("routed_only")
    routed = moe(hidden.detach(), categories)
    torch.testing.assert_close(full, shared + routed)


def test_pcmoe_and_category_ffn_use_exact_category_paths():
    torch.manual_seed(11)
    categories = torch.tensor([0, 1, 0, 1])
    hidden = torch.randn(4, 8)

    pcmoe = AuxiliaryLossFreeMoE(
        width=8,
        expert_width=12,
        routed_experts=6,
        top_k=1,
        category_shared_experts=True,
    ).eval()
    assert pcmoe.total_expert_count == 8
    assert pcmoe.active_expert_count == 2
    pcmoe.set_intervention("full")
    normal = pcmoe(hidden, categories)
    pcmoe.set_intervention("swap_categories")
    swapped = pcmoe(hidden, categories)
    assert not torch.allclose(normal, swapped)
    pcmoe.set_intervention("average_categories")
    averaged = pcmoe(hidden, categories)
    assert averaged.shape == hidden.shape
    assert not torch.allclose(normal, averaged)

    category_ffn = CategoryFFN(width=8, hidden_width=12).eval()
    output = category_ffn(hidden, categories)
    swapped_categories = category_ffn(hidden, 1 - categories)
    assert output.shape == hidden.shape
    assert not torch.allclose(output, swapped_categories)
