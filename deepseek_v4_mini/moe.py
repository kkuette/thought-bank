"""
DeepSeekMoE – DeepSeek-V4 §2.1

Fine-grained expert design with two classes of experts:
  • Shared experts: always activated (n_shared, typically 1)
  • Routed experts: top-k activated per token based on affinity scores

Routing score (from DeepSeek-V4):  sqrt(softplus(h W_gate))
  replacing the Sigmoid used in DeepSeek-V3.

Load balancing: sequence-wise auxiliary loss that penalises uneven expert usage
within a single sequence (light, auxiliary-loss-free strategy from V3 is the
default for large models; here we use a small explicit balance loss for clarity).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block: works on any [..., d_model] input."""

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w12 = nn.Linear(d_model, d_hidden * 2, bias=False)
        self.w3  = nn.Linear(d_hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        # SwiGLU clamping for training stability (§4.2.3):
        # linear branch ∈ [-10, 10], gate branch upper-bounded at 10
        a = a.clamp(-10.0, 10.0)
        b = b.clamp(max=10.0)
        return self.w3(self.drop(F.silu(a) * b))


class DeepSeekMoE(nn.Module):
    """
    Mixture-of-Experts FFN layer.

    Returns (output, balance_loss) where balance_loss is a scalar auxiliary
    loss to be weighted and added to the main training objective.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        n_shared: int,
        top_k_experts: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert top_k_experts <= n_experts
        self.n_routed      = n_experts
        self.n_shared      = n_shared
        self.top_k         = top_k_experts

        # Shared experts (always active – merged into a single scaled output)
        self.shared = nn.ModuleList(
            [SwiGLU(d_model, d_ff, dropout) for _ in range(n_shared)]
        )

        # Routed experts
        self.experts = nn.ModuleList(
            [SwiGLU(d_model, d_ff, dropout) for _ in range(n_experts)]
        )

        # Gating: score = sqrt(softplus(h W_gate))
        self.W_gate = nn.Linear(d_model, n_experts, bias=False)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, d_model]
        Returns: (output [B, T, d_model], balance_loss scalar)
        """
        B, T, d = x.shape
        flat = x.reshape(B * T, d)

        # ── Shared experts ────────────────────────────────────────────────────
        if self.n_shared == 1:
            shared_out = self.shared[0](flat)
        else:
            shared_out = sum(e(flat) for e in self.shared) / self.n_shared

        # ── Routing ───────────────────────────────────────────────────────────
        scores = torch.sqrt(F.softplus(self.W_gate(flat)))     # [BT, n_routed]
        top_scores, top_idx = scores.topk(self.top_k, dim=-1)  # [BT, top_k]
        # Normalise weights within the selected set
        top_w = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-8)

        # ── Dispatch & combine ────────────────────────────────────────────────
        routed_out = torch.zeros_like(flat)
        for ki in range(self.top_k):
            expert_ids = top_idx[:, ki]      # [BT]
            weights    = top_w[:, ki]        # [BT]
            for ei, expert in enumerate(self.experts):
                mask = expert_ids == ei
                if not mask.any():
                    continue
                out = expert(flat[mask])     # [n_tokens, d]
                routed_out[mask] = routed_out[mask] + weights[mask].unsqueeze(-1) * out

        # ── Sequence-wise load balance loss ───────────────────────────────────
        # Penalise deviation from uniform expert usage within the batch
        usage = F.one_hot(top_idx, self.n_routed).float().sum(dim=1)  # [BT, n_routed]
        usage_frac    = usage.mean(dim=0)                              # [n_routed]
        expected_frac = torch.full_like(usage_frac, self.top_k / self.n_routed)
        balance_loss  = ((usage_frac - expected_frac) ** 2).mean()

        out = (shared_out + routed_out).view(B, T, d)
        return out, balance_loss
