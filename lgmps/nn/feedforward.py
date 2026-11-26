from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden_width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Linear(width, hidden_width, bias=False)
        self.value = nn.Linear(width, hidden_width, bias=False)
        self.output = nn.Linear(hidden_width, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.output(F.silu(self.gate(hidden)) * self.value(hidden)))


@dataclass(slots=True)
class RoutingSnapshot:
    counts: torch.Tensor
    probability_mass: torch.Tensor
    category_counts: torch.Tensor
    correction_bias: torch.Tensor | None = None
    load_ema: torch.Tensor | None = None
    category_probability_mass: torch.Tensor | None = None
    contribution_counts: torch.Tensor | None = None
    routed_squared_norm_sum: torch.Tensor | None = None
    shared_squared_norm_sum: torch.Tensor | None = None
    shared_routed_dot_sum: torch.Tensor | None = None
    shared_routed_cosine_sum: torch.Tensor | None = None


class CategoryMoE(nn.Module):
    """Top-k token MoE with an explicit query/item routing prior."""

    def __init__(
        self,
        width: int,
        expert_width: int,
        experts: int,
        top_k: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.expert_count = experts
        self.top_k = top_k
        self.router = nn.Linear(width, experts, bias=False)
        self.category_bias = nn.Parameter(torch.zeros(2, experts))
        self.experts = nn.ModuleList(SwiGLU(width, expert_width, dropout) for _ in range(experts))
        self.last_routing: RoutingSnapshot | None = None

    def forward(self, hidden: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        logits = self.router(hidden).float() + self.category_bias[categories]
        top_values, top_indices = torch.topk(logits, k=self.top_k, dim=-1)
        weights = torch.softmax(top_values, dim=-1).to(hidden.dtype)
        output = torch.zeros_like(hidden)
        counts = torch.zeros(self.expert_count, dtype=torch.long, device=hidden.device)
        mass = torch.zeros(self.expert_count, dtype=torch.float32, device=hidden.device)
        category_counts = torch.zeros(2, self.expert_count, dtype=torch.long, device=hidden.device)
        for expert_index, expert in enumerate(self.experts):
            token_index, route_slot = torch.where(top_indices == expert_index)
            if token_index.numel() == 0:
                continue
            expert_output = expert(hidden[token_index])
            contribution = expert_output * weights[token_index, route_slot, None]
            output.index_add_(0, token_index, contribution.to(output.dtype))
            counts[expert_index] = token_index.numel()
            mass[expert_index] = weights[token_index, route_slot].float().sum()
            category_counts[:, expert_index] = torch.bincount(categories[token_index], minlength=2)
        self.last_routing = RoutingSnapshot(counts.detach(), mass.detach(), category_counts.detach())
        return output


class CategoryFFN(nn.Module):
    """Two dense FFNs selected deterministically by query/item token category."""

    def __init__(self, width: int, hidden_width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.experts = nn.ModuleList(SwiGLU(width, hidden_width, dropout) for _ in range(2))

    def forward(self, hidden: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_categories = categories.reshape(-1)
        output = torch.zeros_like(flat_hidden)
        for category, expert in enumerate(self.experts):
            token_indices = torch.where(flat_categories == category)[0]
            if token_indices.numel() == 0:
                continue
            output.index_copy_(
                0,
                token_indices,
                expert(flat_hidden.index_select(0, token_indices)).to(output.dtype),
            )
        return output.view_as(hidden)


class AuxiliaryLossFreeMoE(nn.Module):
    """Shared-plus-routed SwiGLU experts with auxiliary-loss-free balancing.

    The correction bias affects only expert selection. The unmodified router
    score supplies each selected expert's sigmoid gate, keeping load balancing
    outside the main gradient path as in auxiliary-loss-free routing.
    """

    _INTERVENTIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "average_categories",
            "full",
            "shared_only",
            "routed_only",
            "swap_categories",
        }
    )

    def __init__(
        self,
        width: int,
        expert_width: int,
        routed_experts: int,
        top_k: int,
        *,
        shared_experts: int = 0,
        category_shared_experts: bool = False,
        normalize_topk_probability: bool = True,
        balance_momentum: float = 0.9,
        balance_bias_lr: float = 0.01,
        balance_bias_max: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if routed_experts <= 0:
            raise ValueError("routed_experts must be positive")
        if not 1 <= top_k <= routed_experts:
            raise ValueError("top_k must be in [1, routed_experts]")
        if shared_experts < 0:
            raise ValueError("shared_experts cannot be negative")
        if shared_experts + int(category_shared_experts) < 1:
            raise ValueError("at least one shared expert path is required")
        if not 0.0 <= balance_momentum < 1.0:
            raise ValueError("balance_momentum must be in [0, 1)")
        if balance_bias_lr < 0.0 or balance_bias_max < 0.0:
            raise ValueError("load-balancing bias settings cannot be negative")

        self.routed_expert_count = routed_experts
        self.top_k = top_k
        self.normalize_topk_probability = normalize_topk_probability
        self.balance_momentum = balance_momentum
        self.balance_bias_lr = balance_bias_lr
        self.balance_bias_max = balance_bias_max
        self.category_shared_experts = category_shared_experts
        self.intervention = "full"
        self.capture_contributions = False

        self.router = nn.Linear(width, routed_experts, bias=False)
        self.routed_experts = nn.ModuleList(
            SwiGLU(width, expert_width, dropout) for _ in range(routed_experts)
        )
        self.shared_experts = nn.ModuleList(
            SwiGLU(width, expert_width, dropout) for _ in range(shared_experts)
        )
        self.category_experts = (
            nn.ModuleList(SwiGLU(width, expert_width, dropout) for _ in range(2))
            if category_shared_experts
            else nn.ModuleList()
        )

        self.register_buffer(
            "correction_bias",
            torch.zeros(routed_experts, dtype=torch.float32),
        )
        self.register_buffer(
            "load_ema",
            torch.full((routed_experts,), 1.0 / routed_experts, dtype=torch.float32),
        )
        self.last_routing: RoutingSnapshot | None = None

    @property
    def total_expert_count(self) -> int:
        return len(self.routed_experts) + len(self.shared_experts) + len(self.category_experts)

    @property
    def active_expert_count(self) -> int:
        return self.top_k + len(self.shared_experts) + int(self.category_shared_experts)

    def set_intervention(self, intervention: str) -> None:
        if intervention not in self._INTERVENTIONS:
            choices = ", ".join(sorted(self._INTERVENTIONS))
            raise ValueError(f"Unknown MoE intervention {intervention!r}; choose from {choices}")
        if intervention in {"average_categories", "swap_categories"} and not self.category_shared_experts:
            raise ValueError(f"{intervention} requires category-specific shared experts")
        self.intervention = intervention

    def enable_contribution_capture(self, enabled: bool = True) -> None:
        self.capture_contributions = enabled

    @torch.no_grad()
    def _update_balance(self, selected_experts: torch.Tensor) -> None:
        counts = torch.bincount(
            selected_experts.reshape(-1),
            minlength=self.routed_expert_count,
        ).to(device=self.correction_bias.device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        load = counts / counts.sum().clamp_min(1.0)
        self.load_ema.mul_(self.balance_momentum).add_(
            load,
            alpha=1.0 - self.balance_momentum,
        )
        target = 1.0 / self.routed_expert_count
        self.correction_bias.add_(self.balance_bias_lr * (target - self.load_ema))
        self.correction_bias.clamp_(-self.balance_bias_max, self.balance_bias_max)

    def _route(
        self,
        hidden: torch.Tensor,
        categories: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, RoutingSnapshot]:
        router_scores = self.router(hidden).float()
        gate_values = torch.sigmoid(router_scores)
        _, selected_experts = torch.topk(
            router_scores + self.correction_bias.unsqueeze(0),
            self.top_k,
            dim=-1,
        )
        routing_weights = gate_values.gather(-1, selected_experts)
        if self.normalize_topk_probability and self.top_k > 1:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-9)
        if self.training:
            self._update_balance(selected_experts)

        counts = torch.bincount(
            selected_experts.reshape(-1),
            minlength=self.routed_expert_count,
        )
        probability_mass = torch.zeros(
            self.routed_expert_count,
            device=hidden.device,
            dtype=torch.float32,
        )
        probability_mass.scatter_add_(
            0,
            selected_experts.reshape(-1),
            routing_weights.reshape(-1),
        )
        category_counts = torch.zeros(
            2,
            self.routed_expert_count,
            device=hidden.device,
            dtype=torch.long,
        )
        category_probability_mass = torch.zeros(
            2,
            self.routed_expert_count,
            device=hidden.device,
            dtype=torch.float32,
        )
        flat_categories = categories.reshape(-1)
        for category in range(2):
            category_mask = flat_categories == category
            selected = selected_experts[category_mask]
            if selected.numel():
                category_counts[category] = torch.bincount(
                    selected.reshape(-1),
                    minlength=self.routed_expert_count,
                )
                category_probability_mass[category].scatter_add_(
                    0,
                    selected.reshape(-1),
                    routing_weights[category_mask].reshape(-1),
                )
        snapshot = RoutingSnapshot(
            counts=counts.detach(),
            probability_mass=probability_mass.detach(),
            category_counts=category_counts.detach(),
            correction_bias=self.correction_bias.detach().clone(),
            load_ema=self.load_ema.detach().clone(),
            category_probability_mass=category_probability_mass.detach(),
        )
        return selected_experts, routing_weights, snapshot

    def _routed_output(
        self,
        hidden: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden)
        for expert_index, expert in enumerate(self.routed_experts):
            token_indices, route_slots = torch.where(selected_experts == expert_index)
            if token_indices.numel() == 0:
                continue
            expert_output = expert(hidden.index_select(0, token_indices))
            weights = routing_weights[token_indices, route_slots].to(expert_output.dtype)
            contribution = (expert_output * weights.unsqueeze(-1)).to(output.dtype)
            output.index_add_(0, token_indices, contribution)
        return output

    def _shared_output(self, hidden: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(hidden)
        for expert in self.shared_experts:
            output.add_(expert(hidden).to(output.dtype))
        if self.category_shared_experts:
            if self.intervention == "average_categories":
                category_outputs = [expert(hidden).to(output.dtype) for expert in self.category_experts]
                output.add_(torch.stack(category_outputs).mean(dim=0))
                return output
            effective_categories = categories.reshape(-1)
            if self.intervention == "swap_categories":
                effective_categories = 1 - effective_categories
            for category, expert in enumerate(self.category_experts):
                token_indices = torch.where(effective_categories == category)[0]
                if token_indices.numel() == 0:
                    continue
                output.index_add_(
                    0,
                    token_indices,
                    expert(hidden.index_select(0, token_indices)).to(output.dtype),
                )
        return output

    @torch.no_grad()
    def _capture_contributions(
        self,
        snapshot: RoutingSnapshot,
        routed_output: torch.Tensor,
        shared_output: torch.Tensor,
        categories: torch.Tensor,
    ) -> None:
        device = routed_output.device
        snapshot.contribution_counts = torch.zeros(2, device=device, dtype=torch.long)
        snapshot.routed_squared_norm_sum = torch.zeros(2, device=device, dtype=torch.float64)
        snapshot.shared_squared_norm_sum = torch.zeros(2, device=device, dtype=torch.float64)
        snapshot.shared_routed_dot_sum = torch.zeros(2, device=device, dtype=torch.float64)
        snapshot.shared_routed_cosine_sum = torch.zeros(2, device=device, dtype=torch.float64)
        routed = routed_output.float()
        shared = shared_output.float()
        for category in range(2):
            selected = categories == category
            routed_category = routed[selected]
            shared_category = shared[selected]
            routed_squared = routed_category.square().sum(dim=-1)
            shared_squared = shared_category.square().sum(dim=-1)
            dot = (routed_category * shared_category).sum(dim=-1)
            cosine = dot / (routed_squared.sqrt() * shared_squared.sqrt()).clamp_min(1e-12)
            snapshot.contribution_counts[category] = selected.sum()
            snapshot.routed_squared_norm_sum[category] = routed_squared.double().sum()
            snapshot.shared_squared_norm_sum[category] = shared_squared.double().sum()
            snapshot.shared_routed_dot_sum[category] = dot.double().sum()
            snapshot.shared_routed_cosine_sum[category] = cosine.double().sum()

    def forward(self, hidden: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        input_shape = hidden.shape
        flat_hidden = hidden.reshape(-1, input_shape[-1])
        flat_categories = categories.reshape(-1)
        selected_experts, routing_weights, snapshot = self._route(flat_hidden, flat_categories)
        self.last_routing = snapshot

        routed_output = None
        shared_output = None
        if self.intervention != "shared_only" or self.capture_contributions:
            routed_output = self._routed_output(flat_hidden, selected_experts, routing_weights)
        if self.intervention != "routed_only" or self.capture_contributions:
            shared_output = self._shared_output(flat_hidden, flat_categories)
        if self.capture_contributions:
            if routed_output is None or shared_output is None:
                raise RuntimeError("Contribution capture requires both MoE paths")
            self._capture_contributions(snapshot, routed_output, shared_output, flat_categories)

        output = torch.zeros_like(flat_hidden)
        if self.intervention != "shared_only" and routed_output is not None:
            output.add_(routed_output)
        if self.intervention != "routed_only" and shared_output is not None:
            output.add_(shared_output)
        return output.view(*input_shape)
