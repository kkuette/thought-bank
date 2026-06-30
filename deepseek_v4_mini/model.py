"""
Two model variants:

DeepSeekV4Mini  (single-stream, legacy)
  Text stream only. Optional bolt-on thought memory from thought_lm_minimal.

DualModalDeepSeekV4Mini  (dual-stream, recommended)
  Text stream  [B, T, d_model]  processed by CSA/HCA blocks with mHC.
  Thought stream [B, M, mem_dim] processed by its own CSA/HCA blocks with mHC.
  The two streams interact at every text block via cross-modal attention
  (text tokens read from processed thought representations).
  After all text blocks the ThoughtStream writes a new (gated) vector to the
  bank, FIFO-evicting the oldest slot once the bank exceeds max_mem.
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


# ── Thought-memory components (ported from thought_lm_minimal) ───────────────

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


# ── Single transformer block ──────────────────────────────────────────────────

class DeepSeekV4Block(nn.Module):
    """
    One transformer block: two mHC-wrapped sub-layers (attention + MoE).
    Even layer_idx → CSA; odd → HCA.
    """

    def __init__(self, cfg: DeepSeekV4MiniConfig, layer_idx: int) -> None:
        super().__init__()
        attn_cls = CompressedSparseAttention if layer_idx % 2 == 0 else HeavilyCompressedAttention

        # Build attention sub-layer
        if layer_idx % 2 == 0:
            attn = attn_cls(
                d_model=cfg.d_model, n_heads=cfg.n_heads, d_head=cfg.d_head,
                csa_m=cfg.csa_m, top_k=cfg.top_k_csa, n_win=cfg.n_win,
                d_latent_q=cfg.d_latent_q, n_groups=cfg.n_groups,
                dropout=cfg.dropout,
            )
        else:
            attn = attn_cls(
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

        # Each sub-layer gets its own mHC wrapper
        self.mhc_attn = ManifoldHyperConnections(cfg.d_model, cfg.n_hc, cfg.sinkhorn_iters)
        self.mhc_moe  = ManifoldHyperConnections(cfg.d_model, cfg.n_hc, cfg.sinkhorn_iters)

        self._attn = attn
        self._moe  = moe

    def forward(self, X: torch.Tensor):
        """
        X: [B, T, n_hc, d_model]
        Returns: (X_new [B, T, n_hc, d_model], balance_loss scalar)
        """
        # Attention sub-layer
        X = self.mhc_attn(X, lambda h: self._attn(self.norm_attn(h)))

        # MoE sub-layer
        bal = torch.zeros((), device=X.device)

        def _moe_fn(h: torch.Tensor):
            nonlocal bal
            out, bl = self._moe(self.norm_moe(h))
            bal = bl
            return out

        X = self.mhc_moe(X, _moe_fn)
        return X, bal


# ── Full model ────────────────────────────────────────────────────────────────

class DeepSeekV4Mini(nn.Module):
    """
    Small DeepSeek-V4 reproduction with optional thought memory.

    Forward pass returns a dict:
      logits          [B, T, vocab_size]
      balance_loss    scalar  – MoE load balancing auxiliary
      mem_bank        [B, M, mem_dim] | None  – thought memory carry-out
      p_gates         [B, T]  – thought write probability (if use_thought_memory)
    """

    def __init__(self, cfg: DeepSeekV4MiniConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Token embedding
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop  = nn.Dropout(cfg.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [DeepSeekV4Block(cfg, i) for i in range(cfg.n_layers)]
        )

        # Dynamic output collapse: token-dependent softmax mixture over n_hc streams.
        # A_out_net maps the per-token mean-stream summary to mixture weights.
        self.A_out_net = nn.Linear(cfg.d_model, cfg.n_hc, bias=False)
        self.norm_out  = RMSNorm(cfg.d_model)

        # LM head (tied to embedding by default for smaller param count)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight   # weight tying

        # Embedding init: nn.Embedding defaults to N(0,1), which (with the tied
        # head and RMSNorm'd hidden states) makes logits std ~= sqrt(d_model) and
        # blows up the init CE far above ln(vocab). Scale down to std=0.02 so the
        # model starts near uniform predictions.
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        # ── Optional thought memory ───────────────────────────────────────────
        if cfg.use_thought_memory:
            self.mem_attn  = _MemoryCrossAttention(cfg.d_model, cfg.mem_dim, cfg.n_heads)
            self.gate      = _WriteGate(cfg.d_model)
            self.thought   = _ThoughtHead(cfg.d_model, cfg.mem_dim)
            self.norm_mem  = RMSNorm(cfg.d_model)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.LongTensor,
        init_mem: Optional[torch.Tensor] = None,
        compute_logits: bool = True,
    ) -> dict:
        B, T = input_ids.shape
        cfg  = self.cfg

        # 1. Embed → broadcast across n_hc streams
        h = self.drop(self.embed(input_ids))              # [B, T, d]
        X = h.unsqueeze(2).expand(-1, -1, cfg.n_hc, -1).contiguous()  # [B, T, n_hc, d]

        # 2. Transformer blocks
        total_balance = torch.zeros((), device=X.device)
        for block in self.blocks:
            X, bal = block(X)
            total_balance = total_balance + bal
        total_balance = total_balance / cfg.n_layers

        # 3. Dynamic collapse n_hc → d
        X_mean = X.mean(dim=2)                                     # [B, T, d]
        A = torch.softmax(self.A_out_net(X_mean), dim=-1)          # [B, T, n_hc]
        H = (A.unsqueeze(-1) * X).sum(dim=2)                       # [B, T, d]
        H = self.norm_out(H)

        # 4. Thought memory augmentation (sequential, causal)
        p_gates  = None
        mem_bank = init_mem

        if cfg.use_thought_memory:
            h_aug_list, p_list, mem_list = [], [], []
            for t in range(T):
                h_t = H[:, t:t+1, :]                     # [B, 1, d]
                h_aug = self.mem_attn(h_t, mem_bank)      # [B, 1, d]
                h_aug_list.append(h_aug)
                p_t   = self.gate(h_aug.squeeze(1))       # [B, 1]
                m_t   = self.thought(h_aug.squeeze(1))    # [B, mem_dim]
                p_list.append(p_t)
                m_write = (p_t * m_t).unsqueeze(1)        # [B, 1, mem_dim]
                mem_list.append(m_write)
                mem_bank = m_write if mem_bank is None else torch.cat([mem_bank, m_write], dim=1)
                if mem_bank.size(1) > cfg.max_mem:
                    mem_bank = mem_bank[:, -cfg.max_mem:]

            H_final = torch.cat(h_aug_list, dim=1)        # [B, T, d]
            p_gates = torch.cat(p_list, dim=1).squeeze(-1)  # [B, T]
        else:
            H_final = H                                   # [B, T, d]

        out = {
            "balance_loss": total_balance,
            "mem_bank":     mem_bank,
            "p_gates":      p_gates,
        }
        if compute_logits:
            out["logits"] = self.lm_head(H_final)         # [B, T, V]
        else:
            out["hidden"]         = H_final
            out["lm_head_weight"] = self.lm_head.weight
        return out

    # ── Convenience ───────────────────────────────────────────────────────────

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, cfg: DeepSeekV4MiniConfig) -> "DeepSeekV4Mini":
        return cls(cfg)


# ── Dual-stream text block ────────────────────────────────────────────────────

class DualModalBlock(nn.Module):
    """
    Text transformer block with cross-modal injection from the thought stream.

    Forward:
      1. mHC(CSA or HCA)  – text self-attention
      2. Cross-modal       – each text token attends to thought representations
                             (standard cross-attention; M ≤ max_mem so cheap)
      3. mHC(MoE)          – text FFN

    The cross-modal is placed BETWEEN the attention and FFN sub-layers so that
    thought context can influence how the FFN routes tokens.
    The thought representations H_thought are injected as a gated residual
    distributed uniformly across the n_hc residual streams.
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

        # Cross-modal: text [B,T,d] reads from thought [B,M,mem_dim]
        # Standard multi-head cross-attention (M is small)
        assert d % cfg.n_heads == 0
        self.cross_q    = nn.Linear(d, d, bias=False)
        self.cross_k    = nn.Linear(cfg.mem_dim, d, bias=False)
        self.cross_v    = nn.Linear(cfg.mem_dim, d, bias=False)
        self.cross_o    = nn.Linear(d, d, bias=False)
        self.norm_cross = RMSNorm(d)
        self.n_heads    = cfg.n_heads
        self.head_dim   = d // cfg.n_heads

        # Dynamic collapse for cross-modal: same principle as A_out_net
        self.A_cross_net = nn.Linear(d, n_hc, bias=False)

    def _cross_modal(
        self, h: torch.Tensor, H_thought: torch.Tensor
    ) -> torch.Tensor:
        """
        h         : [B, T, d]  – current text representations
        H_thought : [B, M, mem_dim]
        Returns   : [B, T, d]  – augmented text (residual added inside)
        """
        B, T, d = h.shape
        M = H_thought.size(1)
        nh, hd = self.n_heads, self.head_dim
        scale  = hd ** -0.5

        q = self.cross_q(h).view(B, T, nh, hd).transpose(1, 2)      # [B,nh,T,hd]
        k = self.cross_k(H_thought).view(B, M, nh, hd).transpose(1, 2)  # [B,nh,M,hd]
        v = self.cross_v(H_thought).view(B, M, nh, hd).transpose(1, 2)

        w = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)     # [B,nh,T,M]
        z = (w @ v).transpose(1, 2).contiguous().view(B, T, d)
        return h + self.cross_o(z)

    def forward(
        self, X: torch.Tensor, H_thought: Optional[torch.Tensor]
    ):
        """
        X         : [B, T, n_hc, d_model]
        H_thought : [B, M, mem_dim] or None
        Returns   : (X_new, balance_loss)
        """
        # 1. Text self-attention (mHC wrapped)
        X = self.mhc_attn(X, lambda h: self._attn(self.norm_attn(h)))

        # 2. Cross-modal injection (thought → text)
        if H_thought is not None:
            B, T, n_hc, d = X.shape
            h0 = (torch.softmax(self.A_cross_net(X.mean(dim=2)), dim=-1).unsqueeze(-1) * X).sum(dim=2)
            h1 = self._cross_modal(self.norm_cross(h0), H_thought)    # [B, T, d]
            delta = h1 - h0                                           # [B, T, d]
            X = X + delta.unsqueeze(2)                                # broadcast

        # 3. MoE (mHC wrapped)
        bal = torch.zeros((), device=X.device)

        def _moe_fn(h: torch.Tensor) -> torch.Tensor:
            nonlocal bal
            out, bl = self._moe(self.norm_moe(h))
            bal = bl
            return out

        X = self.mhc_moe(X, _moe_fn)
        return X, bal


# ── Dual-stream model ─────────────────────────────────────────────────────────

class DualModalDeepSeekV4Mini(nn.Module):
    """
    Dual-stream architecture: text and thought streams both using CSA/HCA.

    Forward flow
    ────────────
    1. Thought stream processes the current mem_bank → H_thought [B,M,mem_dim]
       (if bank is empty, H_thought = None and we skip cross-modal).
    2. Text blocks process input tokens; at each block the cross-modal reads
       H_thought so every layer is aware of the current thought context.
    3. After all text blocks, ThoughtStream writes a new (gated) thought vector,
       FIFO-evicting the oldest slot once the bank exceeds max_mem.
    4. LM head produces logits from the final text hidden states.

    The bank is returned as mem_bank and should be passed back as init_mem on
    the next forward call for multi-turn / streaming generation.

    Returns a dict:
      logits        [B, T, vocab_size]
      balance_loss  scalar – combined text + thought MoE auxiliary loss
      mem_bank      [B, M', mem_dim] – updated bank (M' ≤ max_mem)
    """

    def __init__(self, cfg: DeepSeekV4MiniConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop  = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [DualModalBlock(cfg, i) for i in range(cfg.n_layers)]
        )

        # Dynamic output collapse (same as DeepSeekV4Mini)
        self.A_out_net = nn.Linear(cfg.d_model, cfg.n_hc, bias=False)
        self.norm_out  = RMSNorm(cfg.d_model)

        # LM head (weight-tied to embedding)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        # Embedding init: nn.Embedding defaults to N(0,1), which (with the tied
        # head and RMSNorm'd hidden states) makes logits std ~= sqrt(d_model) and
        # blows up the init CE far above ln(vocab). Scale down to std=0.02 so the
        # model starts near uniform predictions.
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        # Thought stream
        self.thought_stream = ThoughtStream(cfg)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.LongTensor,
        init_mem: Optional[torch.Tensor] = None,
        compute_logits: bool = True,
    ) -> dict:
        B, T  = input_ids.shape
        cfg   = self.cfg
        n_hc  = cfg.n_hc

        # ── Step 1: process existing thought bank (runs ONCE before text blocks) ─
        # H_thought_pre is injected into every text layer via cross-modal.
        # After text blocks we reuse it (via _write) to avoid re-running blocks.
        H_thought_pre: Optional[torch.Tensor] = None
        thought_bal = torch.zeros((), device=input_ids.device)
        if init_mem is not None and init_mem.size(1) > 0:
            H_thought_pre, thought_bal = self.thought_stream._process_only(init_mem)

        # ── Step 2: embed text and run dual-modal text blocks ─────────────────
        h = self.drop(self.embed(input_ids))                               # [B,T,d]
        X = h.unsqueeze(2).expand(-1, -1, n_hc, -1).contiguous()          # [B,T,n_hc,d]

        total_bal = torch.zeros((), device=X.device)
        for block in self.blocks:
            X, bal = block(X, H_thought_pre)
            total_bal = total_bal + bal
        total_bal = (total_bal / cfg.n_layers) + thought_bal

        # ── Step 3: dynamic collapse mHC → text hidden states ────────────────
        X_mean = X.mean(dim=2)                                            # [B,T,d]
        A = torch.softmax(self.A_out_net(X_mean), dim=-1)                 # [B,T,n_hc]
        H_text = (A.unsqueeze(-1) * X).sum(dim=2)                        # [B,T,d]
        H_text = self.norm_out(H_text)

        # ── Step 4: gated thought write + FIFO eviction (no re-run of blocks) ──
        if init_mem is None or init_mem.size(1) == 0:
            # First forward: write the very first thought vector
            _, mem_bank, _ = self.thought_stream(H_text, init_mem)
        else:
            # Blocks already ran in _process_only; only append the new thought.
            mem_bank = self.thought_stream._write(H_text, init_mem)

        # ── Step 5: LM head ───────────────────────────────────────────────────
        out = {
            "balance_loss":  total_bal,
            "mem_bank":      mem_bank,
            "write_alpha":   self.thought_stream.last_write_alpha,    # mean α (telemetry)
            "write_penalty": self.thought_stream.last_write_penalty,  # diff budget E[-log(1-α)]
        }
        if compute_logits:
            out["logits"] = self.lm_head(H_text)                           # [B,T,V]
        else:
            # Defer the head to a memory-efficient fused cross-entropy: hand back
            # the hidden states and the tied head weight instead of [B,T,V] logits.
            out["hidden"]        = H_text
            out["lm_head_weight"] = self.lm_head.weight
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
