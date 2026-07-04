"""
Two model variants:

DeepSeekV4Mini  (single-stream, legacy)
  Text stream only. Optional bolt-on thought memory (cross-attention read).

DualModalDeepSeekV4Mini  (fast-weight thought bank, recommended)
  Text stream [B, T, d_model] processed by CSA/HCA blocks with mHC. A rolling
  thought bank [B, M, mem_dim] is READ as FAST WEIGHTS at every text block: each
  slot parametrises a low-rank MLP layer and the text stream is passed through the
  stack of them (slot → linear → activation → dropout → next slot). After the text
  blocks a gated write head appends one new thought vector to the bank (FIFO). The
  bank is single-stream: there is no separate thought transformer — the text model
  writes the vectors and reuses them directly as weights (continual-learning goal).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DeepSeekV4MiniConfig
from .mhc import ManifoldHyperConnections, RMSNorm
from .attention import CompressedSparseAttention, HeavilyCompressedAttention
from .moe import DeepSeekMoE
from .memory import ThoughtStream


# ── Legacy bolt-on thought-memory components (DeepSeekV4Mini only) ────────────

class _MemoryCrossAttention(nn.Module):
    """Read from a thought-vector memory bank via cross-attention."""

    def __init__(self, dim: int, mem_dim: int, n_heads: int) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.q  = nn.Linear(dim, dim, bias=False)
        self.k  = nn.Linear(mem_dim, dim, bias=False)
        self.v  = nn.Linear(mem_dim, dim, bias=False)
        self.o  = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor]) -> torch.Tensor:
        if mem is None or mem.size(1) == 0:
            return x
        B, T, H = x.shape
        M = mem.size(1)
        scale = self.head_dim ** -0.5
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(mem).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(mem).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        w = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        z = (w @ v).transpose(1, 2).contiguous().view(B, T, H)
        return x + self.o(z)


class _WriteGate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, 1, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lin(h))   # [B, 1]


class _ThoughtHead(nn.Module):
    def __init__(self, dim: int, mem_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, mem_dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)                 # [B, mem_dim]


# ── Single transformer block (legacy) ─────────────────────────────────────────

class DeepSeekV4Block(nn.Module):
    """One transformer block: two mHC-wrapped sub-layers (attention + MoE).
    Even layer_idx → CSA; odd → HCA."""

    def __init__(self, cfg: DeepSeekV4MiniConfig, layer_idx: int) -> None:
        super().__init__()
        if layer_idx % 2 == 0:
            attn: nn.Module = CompressedSparseAttention(
                d_model=cfg.d_model, n_heads=cfg.n_heads, d_head=cfg.d_head,
                csa_m=cfg.csa_m, top_k=cfg.top_k_csa, n_win=cfg.n_win,
                d_latent_q=cfg.d_latent_q, n_groups=cfg.n_groups,
                dropout=cfg.dropout,
            )
        else:
            attn = HeavilyCompressedAttention(
                d_model=cfg.d_model, n_heads=cfg.n_heads, d_head=cfg.d_head,
                hca_m=cfg.hca_m, n_win=cfg.n_win,
                d_latent_q=cfg.d_latent_q, n_groups=cfg.n_groups,
                dropout=cfg.dropout,
            )

        moe = DeepSeekMoE(
            d_model=cfg.d_model, n_experts=cfg.n_experts,
            n_shared=cfg.n_shared, top_k_experts=cfg.top_k_experts,
            d_ff=cfg.d_ff, dropout=cfg.dropout,
        )

        self.norm_attn = RMSNorm(cfg.d_model)
        self.norm_moe  = RMSNorm(cfg.d_model)
        self.mhc_attn = ManifoldHyperConnections(cfg.d_model, cfg.n_hc, cfg.sinkhorn_iters)
        self.mhc_moe  = ManifoldHyperConnections(cfg.d_model, cfg.n_hc, cfg.sinkhorn_iters)
        self._attn = attn
        self._moe  = moe

    def forward(self, X: torch.Tensor):
        """X: [B, T, n_hc, d_model] → (X_new, balance_loss)"""
        X = self.mhc_attn(X, lambda h: self._attn(self.norm_attn(h)))

        bal = torch.zeros((), device=X.device)

        def _moe_fn(h: torch.Tensor):
            nonlocal bal
            out, bl = self._moe(self.norm_moe(h))
            bal = bl
            return out

        X = self.mhc_moe(X, _moe_fn)
        return X, bal


# ── Full model (legacy) ───────────────────────────────────────────────────────

class DeepSeekV4Mini(nn.Module):
    """Small DeepSeek-V4 reproduction with optional bolt-on thought memory."""

    def __init__(self, cfg: DeepSeekV4MiniConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop  = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [DeepSeekV4Block(cfg, i) for i in range(cfg.n_layers)]
        )

        self.A_out_net = nn.Linear(cfg.d_model, cfg.n_hc, bias=False)
        self.norm_out  = RMSNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight   # weight tying
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        if cfg.use_thought_memory:
            self.mem_attn  = _MemoryCrossAttention(cfg.d_model, cfg.mem_dim, cfg.n_heads)
            self.gate      = _WriteGate(cfg.d_model)
            self.thought   = _ThoughtHead(cfg.d_model, cfg.mem_dim)
            self.norm_mem  = RMSNorm(cfg.d_model)

    def forward(
        self,
        input_ids: torch.LongTensor,
        init_mem: Optional[torch.Tensor] = None,
        compute_logits: bool = True,
    ) -> dict:
        B, T = input_ids.shape
        cfg  = self.cfg

        h = self.drop(self.embed(input_ids))              # [B, T, d]
        X = h.unsqueeze(2).expand(-1, -1, cfg.n_hc, -1).contiguous()

        total_balance = torch.zeros((), device=X.device)
        for block in self.blocks:
            X, bal = block(X)
            total_balance = total_balance + bal
        total_balance = total_balance / cfg.n_layers

        X_mean = X.mean(dim=2)
        A = torch.softmax(self.A_out_net(X_mean), dim=-1)
        H = (A.unsqueeze(-1) * X).sum(dim=2)
        H = self.norm_out(H)

        p_gates  = None
        mem_bank = init_mem
        if cfg.use_thought_memory:
            h_aug_list, p_list = [], []
            for t in range(T):
                h_t   = H[:, t:t+1, :]
                h_aug = self.mem_attn(h_t, mem_bank)
                h_aug_list.append(h_aug)
                p_t   = self.gate(h_aug.squeeze(1))
                m_t   = self.thought(h_aug.squeeze(1))
                p_list.append(p_t)
                m_write = (p_t * m_t).unsqueeze(1)
                mem_bank = m_write if mem_bank is None else torch.cat([mem_bank, m_write], dim=1)
                if mem_bank.size(1) > cfg.max_mem:
                    mem_bank = mem_bank[:, -cfg.max_mem:]
            H_final = torch.cat(h_aug_list, dim=1)
            p_gates = torch.cat(p_list, dim=1).squeeze(-1)
        else:
            H_final = H

        out = {"balance_loss": total_balance, "mem_bank": mem_bank, "p_gates": p_gates}
        if compute_logits:
            out["logits"] = self.lm_head(H_final)
        else:
            out["hidden"]         = H_final
            out["lm_head_weight"] = self.lm_head.weight
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, cfg: DeepSeekV4MiniConfig) -> "DeepSeekV4Mini":
        return cls(cfg)


# ── Fast-weight text block ────────────────────────────────────────────────────

class DualModalBlock(nn.Module):
    """
    Text transformer block that reads the thought bank as FAST WEIGHTS.

    Forward:
      1. mHC(CSA or HCA)      – text self-attention
      2. Fast-weight read     – the token stream is passed through a stack of
                                low-rank MLP layers, one per bank slot (see
                                _cross_modal). Placed between attention and MoE so
                                the applied "weights" can influence FFN routing.
      3. mHC(MoE)             – text FFN

    Fast-weight read (memory-as-weights, not memory-as-data)
    ────────────────────────────────────────────────────────
    Each bank slot m_i ∈ R^mem_dim is expanded by a (frozen, learned) hypernet into
    a low-rank layer with weights A_i ∈ R^{r×d}, B_i ∈ R^{d×r}:

        y ← y + dropout( B_i · act( A_i · y ) )          (residual, per slot i)

    applied SEQUENTIALLY over the M slots → an M-layer fast-weight MLP whose weights
    the model wrote. The activation between slots is what makes the composition
    non-linear: without it, stacking/summing slots collapses to one low-rank linear
    map (the failure mode of the earlier outer-product read). The read's net effect
    is a delta added to the token: h + fw_o(y - y0), so a trivial bank ≈ identity.
    """

    def __init__(self, cfg: DeepSeekV4MiniConfig, layer_idx: int) -> None:
        super().__init__()
        d    = cfg.d_model
        n_hc = cfg.n_hc

        if layer_idx % 2 == 0:
            attn: nn.Module = CompressedSparseAttention(
                d_model=d, n_heads=cfg.n_heads, d_head=cfg.d_head,
                csa_m=cfg.csa_m, top_k=cfg.top_k_csa, n_win=cfg.n_win,
                d_latent_q=cfg.d_latent_q, n_groups=cfg.n_groups,
                dropout=cfg.dropout,
            )
        else:
            attn = HeavilyCompressedAttention(
                d_model=d, n_heads=cfg.n_heads, d_head=cfg.d_head,
                hca_m=cfg.hca_m, n_win=cfg.n_win,
                d_latent_q=cfg.d_latent_q, n_groups=cfg.n_groups,
                dropout=cfg.dropout,
            )

        moe = DeepSeekMoE(
            d_model=d, n_experts=cfg.n_experts, n_shared=cfg.n_shared,
            top_k_experts=cfg.top_k_experts, d_ff=cfg.d_ff, dropout=cfg.dropout,
        )

        self.norm_attn = RMSNorm(d)
        self.norm_moe  = RMSNorm(d)
        self.mhc_attn  = ManifoldHyperConnections(d, n_hc, cfg.sinkhorn_iters)
        self.mhc_moe   = ManifoldHyperConnections(d, n_hc, cfg.sinkhorn_iters)
        self._attn = attn
        self._moe  = moe

        # ── Fast-weight read: slot → low-rank MLP layer (A_i [r,d], B_i [d,r]) ──
        # mem_read_layers restricts which blocks read the bank (empty = all).
        # Reading at every block composes code-dependent transforms in depth —
        # code sensitivity escalates even with SN per matrix.
        _rl = list(getattr(cfg, "mem_read_layers", []) or [])
        self.read_bank = (not _rl) or (layer_idx in _rl)
        r = int(cfg.mem_read_rank)
        self.read_rank = r
        # SwiGLU read: fw_A emits TWO maps per slot (gate A_g, value A_v).
        self.fw_swiglu = bool(getattr(cfg, "mem_read_swiglu", False))
        _na = 2 if self.fw_swiglu else 1
        self.fw_A    = nn.Linear(cfg.mem_dim, _na * r * d, bias=False)  # slot → A_i
        self.fw_B    = nn.Linear(cfg.mem_dim, d * r, bias=False)  # slot → B_i
        if bool(getattr(cfg, "mem_read_spectral_norm", False)):
            from torch.nn.utils.parametrizations import spectral_norm
            self.fw_A = spectral_norm(self.fw_A)
            self.fw_B = spectral_norm(self.fw_B)
        self.fw_o    = nn.Linear(d, d, bias=False)               # output projection
        self.norm_fw = RMSNorm(d)
        self.fw_act  = nn.GELU()
        self.fw_drop = nn.Dropout(cfg.mem_read_dropout)

        # Dynamic mHC collapse for the read: same principle as A_out_net.
        self.A_cross_net = nn.Linear(d, n_hc, bias=False)

    def _cross_modal(self, h: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        """
        h    : [B, T, d]         – current text representations
        bank : [B, M, mem_dim]   – thought bank (fast-weight codes)
        Returns [B, T, d] with the fast-weight read added as a residual.
        """
        B, M, _ = bank.shape
        d = h.size(-1)
        r = self.read_rank

        _na = 2 if self.fw_swiglu else 1
        A = self.fw_A(bank).view(B, M, _na, r, d)  # [B, M, 1|2, r, d]
        Bm = self.fw_B(bank).view(B, M, d, r)      # [B, M, d, r]

        ds = d ** -0.5
        rs = r ** -0.5
        y0 = self.norm_fw(h)
        y  = y0
        for i in range(M):
            if self.fw_swiglu:
                zg = torch.einsum("brd,btd->btr", A[:, i, 0], y) * ds
                zv = torch.einsum("brd,btd->btr", A[:, i, 1], y) * ds
                z  = (F.silu(zg) * zv).clamp(-8.0, 8.0)                       # gated, clamped
            else:
                z = self.fw_act(torch.einsum("brd,btd->btr", A[:, i, 0], y) * ds)  # [B,T,r]
            upd = torch.einsum("bdr,btr->btd", Bm[:, i], z) * rs              # [B,T,d]
            y   = y + self.fw_drop(upd)
        return h + self.fw_o(y - y0)

    def forward(self, X: torch.Tensor, bank: Optional[torch.Tensor]):
        """
        X    : [B, T, n_hc, d_model]
        bank : [B, M, mem_dim] or None
        Returns (X_new, balance_loss)
        """
        # 1. Text self-attention (mHC wrapped)
        X = self.mhc_attn(X, lambda h: self._attn(self.norm_attn(h)))

        # 2. Fast-weight read (thought bank → text)
        if self.read_bank and bank is not None and bank.size(1) > 0:
            h0 = (torch.softmax(self.A_cross_net(X.mean(dim=2)), dim=-1).unsqueeze(-1) * X).sum(dim=2)
            h1 = self._cross_modal(h0, bank)          # _cross_modal normalises internally
            delta = h1 - h0
            X = X + delta.unsqueeze(2)

        # 3. MoE (mHC wrapped)
        bal = torch.zeros((), device=X.device)

        def _moe_fn(h: torch.Tensor) -> torch.Tensor:
            nonlocal bal
            out, bl = self._moe(self.norm_moe(h))
            bal = bl
            return out

        X = self.mhc_moe(X, _moe_fn)
        return X, bal


# ── Fast-weight dual-stream model ─────────────────────────────────────────────

class DualModalDeepSeekV4Mini(nn.Module):
    """
    Text stream + fast-weight thought bank (single-stream bank).

    Forward flow
    ────────────
    1. Seed the bank with random-uniform[0,1] slots on a fresh conversation
       (init_mem is None); otherwise reuse the carried-in bank.
    2. Text blocks process the input; at each block the bank is READ as fast
       weights (DualModalBlock._cross_modal).
    3. After the text blocks, the write head appends one new gated thought vector
       to the bank (FIFO-evicting the oldest beyond max_mem).
    4. LM head produces logits from the final text hidden states.

    The updated bank is returned as `mem_bank`; pass it back as `init_mem` on the
    next turn for multi-turn / streaming continual learning.
    """

    def __init__(self, cfg: DeepSeekV4MiniConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop  = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [DualModalBlock(cfg, i) for i in range(cfg.n_layers)]
        )

        self.A_out_net = nn.Linear(cfg.d_model, cfg.n_hc, bias=False)
        self.norm_out  = RMSNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        # Write head for the fast-weight bank.
        self.thought_stream = ThoughtStream(cfg)

    def forward(
        self,
        input_ids: torch.LongTensor,
        init_mem: Optional[torch.Tensor] = None,
        compute_logits: bool = True,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        B, T  = input_ids.shape
        cfg   = self.cfg
        n_hc  = cfg.n_hc

        # ── Step 1: embed text ────────────────────────────────────────────────
        h = self.drop(self.embed(input_ids))                              # [B,T,d]

        # ── Step 2: seed a fresh bank, or reuse the carried-in one ────────────
        if init_mem is None or init_mem.size(1) == 0:
            bank = self.thought_stream.seed_bank(B, h.device, h.dtype)
        else:
            bank = init_mem

        # ── Step 3: text blocks read the bank as fast weights ─────────────────
        X = h.unsqueeze(2).expand(-1, -1, n_hc, -1).contiguous()         # [B,T,n_hc,d]
        total_bal = torch.zeros((), device=X.device)
        for block in self.blocks:
            X, bal = block(X, bank)
            total_bal = total_bal + bal
        total_bal = total_bal / cfg.n_layers

        # ── Step 4: dynamic collapse mHC → text hidden states ─────────────────
        X_mean = X.mean(dim=2)                                           # [B,T,d]
        A = torch.softmax(self.A_out_net(X_mean), dim=-1)               # [B,T,n_hc]
        H_text = (A.unsqueeze(-1) * X).sum(dim=2)                       # [B,T,d]
        H_text = self.norm_out(H_text)

        # ── Step 5: gated thought write + FIFO eviction ───────────────────────
        mem_bank = self.thought_stream._write(H_text, bank, pad_mask)

        # ── Step 6: outputs ───────────────────────────────────────────────────
        out = {
            "balance_loss":     total_bal,
            "mem_bank":         mem_bank,
            "write_alpha":      self.thought_stream.last_write_alpha,       # mean α (telemetry)
            "write_penalty":    self.thought_stream.last_write_penalty,     # diff E[-log(1-α)]
            "write_alpha_mean": self.thought_stream.last_write_alpha_mean,  # diff E[α]
            "write_redundancy": self.thought_stream.last_write_redundancy,  # diff E[max cos]
        }
        if compute_logits:
            out["logits"] = self.lm_head(H_text)                           # [B,T,V]
        else:
            out["hidden"]         = H_text
            out["lm_head_weight"] = self.lm_head.weight
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
