"""
Manifold-Constrained Hyper-Connections (mHC) – DeepSeek-V4 §2.2

Key idea: expand the residual stream width by n_hc and constrain the residual
mapping matrix B to the Birkhoff polytope (doubly stochastic matrices) via
Sinkhorn-Knopp.  This bounds ||B||_2 ≤ 1, making deep stacks numerically
stable without sacrificing expressivity.

Standard HC update:
    X_{l+1} = B_l X_l + C_l F_l(A_l X_l)

where X_l ∈ R^{n_hc × d} and A_l, B_l, C_l are dynamically generated from X_l.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class ManifoldHyperConnections(nn.Module):
    """
    mHC wrapper.  Call as:
        X_new = mhc(X, layer_fn)
    where X has shape [B, T, n_hc, d] and layer_fn maps [B, T, d] → [B, T, d].
    """

    def __init__(self, d_model: int, n_hc: int, sinkhorn_iters: int = 5) -> None:
        super().__init__()
        self.n_hc = n_hc
        self.d_model = d_model
        self.sinkhorn_iters = sinkhorn_iters
        flat = n_hc * d_model

        # Dynamic component generators (small projections – n_hc is typically 2–4)
        self.W_pre = nn.Linear(flat, n_hc, bias=False)
        self.W_res = nn.Linear(flat, n_hc * n_hc, bias=False)
        self.W_post = nn.Linear(flat, n_hc, bias=False)

        # Learnable static biases (initialised so B ≈ identity)
        self.S_pre  = nn.Parameter(torch.zeros(n_hc))
        self.S_res  = nn.Parameter(torch.eye(n_hc).flatten())
        self.S_post = nn.Parameter(torch.full((n_hc,), 1.0 / n_hc))

        # Gating scalars – start near 0 so dynamic component is initially small
        self.alpha_pre = nn.Parameter(torch.zeros(1))
        self.alpha_res = nn.Parameter(torch.zeros(1))
        self.alpha_post = nn.Parameter(torch.zeros(1))

        self.norm = RMSNorm(flat)

    # ── Sinkhorn-Knopp projection onto the Birkhoff polytope ─────────────────
    def _sinkhorn(self, M: torch.Tensor) -> torch.Tensor:
        """M: [BT, n_hc, n_hc], positive.  Returns doubly stochastic matrix."""
        for _ in range(self.sinkhorn_iters):
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)   # row normalise
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)   # column normalise
        return M

    def forward(self, X: torch.Tensor, layer_fn: Callable) -> torch.Tensor:
        """
        X:         [B, T, n_hc, d]
        layer_fn:  [B, T, d] → [B, T, d]
        Returns:   [B, T, n_hc, d]
        """
        B, T, n_hc, d = X.shape
        BT = B * T

        # Flatten residual state and normalise for parameter generation
        X_flat = X.reshape(BT, n_hc * d)
        X_hat = self.norm(X_flat)                                   # [BT, n_hc*d]

        # ── Generate raw (unconstrained) A, B, C ─────────────────────────────
        A_raw = self.alpha_pre.tanh() * self.W_pre(X_hat) + self.S_pre   # [BT, n_hc]
        B_raw = self.alpha_res.tanh() * self.W_res(X_hat) + self.S_res   # [BT, n_hc²]
        C_raw = self.alpha_post.tanh() * self.W_post(X_hat) + self.S_post # [BT, n_hc]

        # ── Apply constraints (DeepSeek-V4 §2.2 eqs. 6-8) ──────────────────────
        A    = torch.sigmoid(A_raw)                                # [BT, n_hc] non-neg, bounded
        C    = 2.0 * torch.sigmoid(C_raw)                         # [BT, n_hc] non-neg, ≤ 2
        B_ds = self._sinkhorn(B_raw.view(BT, n_hc, n_hc).exp())   # doubly stochastic

        # ── Compute layer input: h_in = A ⊗ X (weighted sum of n_hc streams) ─
        X_r = X.reshape(BT, n_hc, d)
        h_in = torch.einsum("bn,bnd->bd", A, X_r)                 # [BT, d]

        # ── Run the wrapped layer ─────────────────────────────────────────────
        h_out = layer_fn(h_in.view(B, T, d)).reshape(BT, d)       # [BT, d]

        # ── Update residual stream: X_{l+1} = B X_l + C ⊗ h_out ─────────────
        X_new = (
            torch.einsum("bij,bjd->bid", B_ds, X_r)               # [BT, n_hc, d]
            + torch.einsum("bn,bd->bnd", C, h_out)
        )
        return X_new.view(B, T, n_hc, d)
