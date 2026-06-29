"""
Hybrid attention – DeepSeek-V4 §2.3

Two complementary attention variants interleaved across layers:

CSA – Compressed Sparse Attention
  * Overlapping KV compression by factor m (two series Ca/Cb)
  * Sparse top-k selection of compressed blocks via a lightweight indexer
  * Sliding window branch for local fine-grained dependencies
  * Shared KV Multi-Query Attention + grouped output projection

HCA – Heavily Compressed Attention
  * Non-overlapping compression by factor m' (≫ m)
  * Dense causal attention over *all* compressed blocks (no top-k)
  * Same sliding window branch, MQA, grouped output projection

Both variants use:
  * RMSNorm on queries and KV entries before core attention
  * Partial RoPE (applied to all head dims in this mini model)
  * Attention sink (learnable denominator addend per head)
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mhc import RMSNorm


# ── RoPE utilities ────────────────────────────────────────────────────────────

def _rope_cache(
    seq_len: int, dim: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    half = dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)            # [T, half]
    return freqs.cos(), freqs.sin()             # [T, half]


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [..., T, dim]; cos/sin: [T, dim//2]"""
    T, dim = x.shape[-2], x.shape[-1]
    half = dim // 2
    c = cos[:T]                                 # [T, half]
    s = sin[:T]
    # prepend necessary dims for broadcast
    for _ in range(x.dim() - 2):
        c, s = c.unsqueeze(0), s.unsqueeze(0)
    x_e = x[..., :half]
    x_o = x[..., half:]
    return torch.cat([x_e * c - x_o * s, x_e * s + x_o * c], dim=-1)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _grouped_out_proj(
    out: torch.Tensor,          # [B, T, n_heads, d_head]
    n_groups: int,
    group_linears: nn.ModuleList,
    final_proj: nn.Linear,
) -> torch.Tensor:
    B, T, n_heads, d_head = out.shape
    hpg = n_heads // n_groups
    parts = [
        group_linears[g](out[:, :, g * hpg:(g + 1) * hpg, :].reshape(B, T, -1))
        for g in range(n_groups)
    ]
    return final_proj(torch.cat(parts, dim=-1))   # [B, T, d_model]


def _attn_sink_softmax(
    logits: torch.Tensor,       # [BT, n_heads, n_kv]
    sink_logits: torch.Tensor,  # [n_heads]
    drop: nn.Dropout,
) -> torch.Tensor:
    """Softmax with a learnable sink added to the denominator (eq. 27)."""
    a_max = logits.detach().max(dim=-1, keepdim=True).values
    e = (logits - a_max).exp()
    denom = e.sum(dim=-1) + sink_logits.exp().unsqueeze(0)     # [BT, n_heads]
    w = e / denom.unsqueeze(-1)
    return drop(w)


# ── Compressed Sparse Attention (CSA) ────────────────────────────────────────

class CompressedSparseAttention(nn.Module):
    """
    CSA: overlapping KV compression (factor m) → sparse top-k + sliding window.
    Layer index i (0-indexed) is even in the hybrid interleaving.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        csa_m: int,
        top_k: int,
        n_win: int,
        d_latent_q: int,
        n_groups: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert n_heads % n_groups == 0
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_head
        self.m, self.top_k, self.n_win = csa_m, top_k, n_win
        self.n_groups = n_groups
        hpg = n_heads // n_groups

        # Overlapping KV compression – two series (Ca/Cb = values, Za/Zb = gates)
        # Positional biases are applied to Z (gates) only, NOT to C (values) – §2.3.1 eq. 9-12
        self.W_kv_a = nn.Linear(d_model, d_head, bias=False)
        self.W_kv_b = nn.Linear(d_model, d_head, bias=False)
        self.W_z_a  = nn.Linear(d_model, d_head, bias=False)
        self.W_z_b  = nn.Linear(d_model, d_head, bias=False)
        self.pos_a  = nn.Parameter(torch.zeros(csa_m, d_head))  # bias for Za only
        self.pos_b  = nn.Parameter(torch.zeros(csa_m, d_head))  # bias for Zb only

        # Low-rank queries: d → d_latent_q → n_heads * d_head
        self.W_dq = nn.Linear(d_model, d_latent_q, bias=False)
        self.W_uq = nn.Linear(d_latent_q, n_heads * d_head, bias=False)
        # Lightning indexer (§2.3.1 eqs. 13-16): multi-head indexer queries + head weights
        # W_IUQ: latent → n_idx_heads * d_head  (shared latent c_Q from W_dq)
        # W_w  : d_model → n_idx_heads  (per-head scalar weights for score aggregation)
        self.n_idx_heads = max(1, n_heads // 4)  # lightweight: n_h/4 heads
        self.W_iq = nn.Linear(d_latent_q, self.n_idx_heads * d_head, bias=False)
        self.W_w  = nn.Linear(d_model, self.n_idx_heads, bias=False)

        # Sliding window KV (uncompressed local tokens)
        self.W_wk = nn.Linear(d_model, d_head, bias=False)
        self.W_wv = nn.Linear(d_model, d_head, bias=False)

        # Grouped output projection
        d_g = d_model // n_groups
        self.out_group = nn.ModuleList(
            [nn.Linear(hpg * d_head, d_g, bias=False) for _ in range(n_groups)]
        )
        self.out_proj = nn.Linear(n_groups * d_g, d_model, bias=False)

        # Norms (applied just before core attention – avoids exploding logits)
        self.q_norm  = RMSNorm(d_head)
        self.kv_norm = RMSNorm(d_head)

        # Attention sink (one learnable scalar per head)
        self.sink_logits = nn.Parameter(torch.zeros(n_heads))
        self.drop = nn.Dropout(dropout)

    # ── KV compression ────────────────────────────────────────────────────────

    def _compress_kv(self, H_pad: torch.Tensor) -> torch.Tensor:
        """
        Overlapping compression: block i merges Ca[i] (current) with Cb[i-1] (previous).
        H_pad: [B, T_pad, d_model]  (T_pad divisible by m)
        Returns: CComp [B, n_blocks, d_head]
        """
        B, T_pad, _ = H_pad.shape
        m = self.m
        n_blocks = T_pad // m
        H_b = H_pad.view(B, n_blocks, m, -1)

        # Values (Ca, Cb): no positional bias (eq. 9)
        # Gates (Za, Zb): add learnable positional bias (eq. 10-11)
        Ca = self.W_kv_a(H_b)                 # [B, n_blocks, m, d_head]
        Cb = self.W_kv_b(H_b)
        Za = self.W_z_a(H_b) + self.pos_a     # bias on gates only
        Zb = self.W_z_b(H_b) + self.pos_b

        # Shift Cb by 1 block; mask block-0 predecessor with -∞
        Cb_prev = torch.cat([torch.zeros_like(Cb[:, :1]), Cb[:, :-1]], dim=1)
        Zb_shift = torch.cat([Zb[:, :1], Zb[:, :-1]], dim=1)
        # block 0 gets -inf so softmax assigns zero weight to the phantom predecessor
        inf_mask = torch.zeros(n_blocks, device=H_pad.device, dtype=torch.bool)
        inf_mask[0] = True
        Zb_prev = Zb_shift.masked_fill(inf_mask.view(1, n_blocks, 1, 1), float("-inf"))

        # Concatenate along m dimension; softmax over 2m entries per feature dim
        Z_cat = torch.cat([Za, Zb_prev], dim=2)    # [B, n_blocks, 2m, d_head]
        C_cat = torch.cat([Ca, Cb_prev], dim=2)
        S     = F.softmax(Z_cat, dim=2)            # [B, n_blocks, 2m, d_head]
        return (S * C_cat).sum(dim=2)              # [B, n_blocks, d_head]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        B, T, d = H.shape
        m, n_win = self.m, self.n_win

        # Pad T to multiple of m
        pad = (-T) % m
        H_pad = F.pad(H, (0, 0, 0, pad)) if pad else H
        n_blocks = H_pad.shape[1] // m

        # 1. Compress KV
        CComp = self._compress_kv(H_pad)           # [B, n_blocks, d_head]

        # 2. Low-rank queries (shared latent c_Q for both indexer and core attention)
        cQ  = self.W_dq(H)                         # [B, T, d_latent_q]
        q   = self.W_uq(cQ).view(B, T, self.n_heads, self.d_head)

        # RoPE on queries
        cos, sin = _rope_cache(T, self.d_head, H.device)
        q = _apply_rope(q.permute(0, 2, 1, 3), cos, sin).permute(0, 2, 1, 3)
        q = self.q_norm(q)                         # [B, T, n_heads, d_head]

        # 3. Lightning indexer (§2.3.1 eqs. 13-16): multi-head with head weights + ReLU
        #    I_{t,s} = Σ_h  w_{t,h} · ReLU(q_I_{t,h} · K_IComp_s)
        n_ih = self.n_idx_heads
        qI   = self.W_iq(cQ).view(B, T, n_ih, self.d_head)    # [B, T, n_ih, d_head]
        w_h  = self.W_w(H)                                     # [B, T, n_ih] head weights
        # [B, T, n_ih, n_blocks]  via ReLU dot product
        idx_scores_h = F.relu(
            torch.einsum("bthd,bnd->bthn", qI, CComp) / math.sqrt(self.d_head)
        )
        idx_scores = torch.einsum("bth,bthn->btn", w_h, idx_scores_h)  # [B, T, n_blocks]

        # Causal block mask: token t can see block j only if j < t//m
        block_of_t = torch.arange(T, device=H.device) // m          # [T]
        block_j    = torch.arange(n_blocks, device=H.device)        # [nb]
        causal     = (block_of_t[:, None] <= block_j[None, :])      # [T, nb] True=masked
        idx_scores = idx_scores.masked_fill(causal.unsqueeze(0), float("-inf"))

        k = min(self.top_k, n_blocks)
        if k > 0:
            top_scores, top_idx = idx_scores.topk(k, dim=-1)        # [B, T, k]
            valid = (top_scores > -1e9)                              # [B, T, k]
            exp   = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_head)
            CComp_exp = CComp.unsqueeze(1).expand(-1, T, -1, -1)
            KV_sel = CComp_exp.gather(2, exp)                        # [B, T, k, d_head]
            KV_sel = self.kv_norm(KV_sel)
        else:
            k = 0
            valid  = H.new_zeros(B, T, 0, dtype=torch.bool)
            KV_sel = H.new_zeros(B, T, 0, self.d_head)

        # 4. Sliding window KV (causal: last n_win tokens before position t)
        Wk = self.kv_norm(self.W_wk(H))            # [B, T, d_head]
        Wv = self.W_wv(H)
        # Pad left by n_win; for token t, gather indices [t, t+1, ..., t+n_win-1]
        Wk_p = F.pad(Wk, (0, 0, n_win, 0))
        Wv_p = F.pad(Wv, (0, 0, n_win, 0))
        win_idx = (
            torch.arange(T, device=H.device).unsqueeze(1)
            + torch.arange(n_win, device=H.device).unsqueeze(0)
        )                                           # [T, n_win]
        KV_wk = Wk_p[:, win_idx, :]               # [B, T, n_win, d_head]
        KV_wv = Wv_p[:, win_idx, :]

        # 5. Combined keys/values: compressed (key=value) + window (separate k,v)
        K_all = torch.cat([KV_sel, KV_wk], dim=2) # [B, T, k+n_win, d_head]
        V_all = torch.cat([KV_sel, KV_wv], dim=2)
        n_kv  = k + n_win

        # 6. MQA: all n_heads share the same K/V
        q_bt  = q.reshape(B * T, self.n_heads, self.d_head)
        K_bt  = K_all.reshape(B * T, n_kv, self.d_head)
        V_bt  = V_all.reshape(B * T, n_kv, self.d_head)

        logits = torch.einsum("bhd,bnd->bhn", q_bt, K_bt) / math.sqrt(self.d_head)

        # Mask invalid (causally blocked) compressed entries
        if k > 0:
            valid_bt = valid.reshape(B * T, k).unsqueeze(1).expand(-1, self.n_heads, -1)
            logits[:, :, :k] = logits[:, :, :k].masked_fill(~valid_bt, float("-inf"))

        attn_w = _attn_sink_softmax(logits, self.sink_logits, self.drop)
        out    = torch.einsum("bhn,bnd->bhd", attn_w, V_bt).view(B, T, self.n_heads, self.d_head)

        return _grouped_out_proj(out, self.n_groups, self.out_group, self.out_proj)


# ── Heavily Compressed Attention (HCA) ───────────────────────────────────────

class HeavilyCompressedAttention(nn.Module):
    """
    HCA: non-overlapping compression (factor m' ≫ m) → dense causal attention.
    No top-k selection – all preceding compressed blocks are attended.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        hca_m: int,
        n_win: int,
        d_latent_q: int,
        n_groups: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert n_heads % n_groups == 0
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_head
        self.m_prime, self.n_win = hca_m, n_win
        self.n_groups = n_groups
        hpg = n_heads // n_groups

        # Single KV projection (no overlapping for HCA)
        self.W_kv = nn.Linear(d_model, d_head, bias=False)
        self.W_z  = nn.Linear(d_model, d_head, bias=False)
        self.pos  = nn.Parameter(torch.zeros(hca_m, d_head))

        # Low-rank queries
        self.W_dq = nn.Linear(d_model, d_latent_q, bias=False)
        self.W_uq = nn.Linear(d_latent_q, n_heads * d_head, bias=False)

        # Sliding window KV
        self.W_wk = nn.Linear(d_model, d_head, bias=False)
        self.W_wv = nn.Linear(d_model, d_head, bias=False)

        # Grouped output projection
        d_g = d_model // n_groups
        self.out_group = nn.ModuleList(
            [nn.Linear(hpg * d_head, d_g, bias=False) for _ in range(n_groups)]
        )
        self.out_proj = nn.Linear(n_groups * d_g, d_model, bias=False)

        self.q_norm  = RMSNorm(d_head)
        self.kv_norm = RMSNorm(d_head)

        self.sink_logits = nn.Parameter(torch.zeros(n_heads))
        self.drop = nn.Dropout(dropout)

    def _compress_kv(self, H_pad: torch.Tensor) -> torch.Tensor:
        """Non-overlapping compression."""
        B, T_pad, _ = H_pad.shape
        m = self.m_prime
        n_blocks = T_pad // m
        H_b = H_pad.view(B, n_blocks, m, -1)
        C = self.W_kv(H_b)                         # [B, nb, m, d_head]
        Z = self.W_z(H_b) + self.pos               # [B, nb, m, d_head]
        S = F.softmax(Z, dim=2)
        return (S * C).sum(dim=2)                  # [B, nb, d_head]

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        B, T, d = H.shape
        m, n_win = self.m_prime, self.n_win

        pad   = (-T) % m
        H_pad = F.pad(H, (0, 0, 0, pad)) if pad else H
        n_blocks = H_pad.shape[1] // m

        # 1. Compress KV
        CComp = self.kv_norm(self._compress_kv(H_pad))  # [B, nb, d_head]

        # 2. Queries
        cQ = self.W_dq(H)
        q  = self.W_uq(cQ).view(B, T, self.n_heads, self.d_head)
        cos, sin = _rope_cache(T, self.d_head, H.device)
        q = _apply_rope(q.permute(0, 2, 1, 3), cos, sin).permute(0, 2, 1, 3)
        q = self.q_norm(q)

        # 3. Dense causal attention over compressed blocks
        block_of_t = torch.arange(T, device=H.device) // m     # [T]
        block_j    = torch.arange(n_blocks, device=H.device)   # [nb]
        causal     = (block_of_t[:, None] <= block_j[None, :]) # [T, nb] True=masked

        q_bt = q.reshape(B * T, self.n_heads, self.d_head)
        CComp_bt = CComp.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, n_blocks, self.d_head)

        logits_comp = torch.einsum("bhd,bnd->bhn", q_bt, CComp_bt) / math.sqrt(self.d_head)
        causal_bt = causal.unsqueeze(0).expand(B, -1, -1).reshape(B * T, n_blocks)
        logits_comp = logits_comp.masked_fill(causal_bt.unsqueeze(1), float("-inf"))

        # 4. Sliding window
        Wk = self.kv_norm(self.W_wk(H))
        Wv = self.W_wv(H)
        Wk_p = F.pad(Wk, (0, 0, n_win, 0))
        Wv_p = F.pad(Wv, (0, 0, n_win, 0))
        win_idx = (
            torch.arange(T, device=H.device).unsqueeze(1)
            + torch.arange(n_win, device=H.device).unsqueeze(0)
        )
        KV_wk = Wk_p[:, win_idx, :].reshape(B * T, n_win, self.d_head)
        KV_wv = Wv_p[:, win_idx, :].reshape(B * T, n_win, self.d_head)

        logits_win = torch.einsum("bhd,bnd->bhn", q_bt, KV_wk) / math.sqrt(self.d_head)

        # 5. Combined attention
        logits_all = torch.cat([logits_comp, logits_win], dim=-1)  # [BT, n_h, nb+n_win]
        V_all_bt   = torch.cat([CComp_bt, KV_wv], dim=1)          # [BT, nb+n_win, d_head]

        attn_w = _attn_sink_softmax(logits_all, self.sink_logits, self.drop)
        out    = torch.einsum("bhn,bnd->bhd", attn_w, V_all_bt).view(B, T, self.n_heads, self.d_head)

        return _grouped_out_proj(out, self.n_groups, self.out_group, self.out_proj)
