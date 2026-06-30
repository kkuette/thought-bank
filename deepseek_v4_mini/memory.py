"""
Dual-stream thought memory system.

Architecture
────────────
Text stream  [B, T, d_model]  ──→ CSA/HCA blocks ──→ LM head
                                        ↕  cross-modal
Thought stream [B, M, mem_dim] ──→ CSA/HCA blocks ──→ write gate → bank

Memory bank
───────────
A rolling buffer of M thought vectors.  The slot index encodes temporal order
directly: slot 0 = oldest surviving thought, slot M-1 = most recent.
`nn.Embedding(max_mem, mem_dim)` provides learned slot-position embeddings so
the thought-stream CSA/HCA can exploit this temporal structure.

Eviction (FIFO)
───────────────
The bank is a strict rolling buffer: each pass appends one thought vector and,
once the bank exceeds max_mem, the OLDEST slot is dropped (`mem_bank[:, -max_mem:]`).
No learned consolidation module — this is the clean baseline so the memory's
usefulness can be attributed to the bank content itself, not to an auxiliary
compressor. Far-past information beyond max_mem segments is lost cold.

Write as a modality choice (gated)
──────────────────────────────────
The write is no longer unconditional. `_new_thought` produces, alongside the
per-dim content gate, a scalar write-decision α = sigmoid(write_decision(ctx))
∈ [0,1] that multiplicatively scales the whole new vector. α≈0 means "nothing
worth committing this pass" (a near-zero slot, soon evicted by FIFO); α≈1 means
"commit this thought". The choice is fully differentiable and driven by the LM
loss, so the model learns when to use memory vs. let the step pass — directly
attacking the slot redundancy (near-duplicate writes) seen with forced writes.

ThoughtBlock
────────────
Same building block as the text stream (mHC + CSA/HCA + MoE) but operating
in `mem_dim` space with smaller hyperparameters.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mhc import ManifoldHyperConnections, RMSNorm
from .attention import CompressedSparseAttention, HeavilyCompressedAttention
from .moe import DeepSeekMoE
from .config import DeepSeekV4MiniConfig


# ── Thought-stream transformer block ─────────────────────────────────────────

class ThoughtBlock(nn.Module):
    """
    One transformer block for the thought stream.
    Mirrors DeepSeekV4Block but operates in mem_dim space.
    Even layer_idx → CSA; odd → HCA.
    """

    def __init__(self, cfg: DeepSeekV4MiniConfig, layer_idx: int) -> None:
        super().__init__()
        d = cfg.mem_dim

        if layer_idx % 2 == 0:
            attn: nn.Module = CompressedSparseAttention(
                d_model=d, n_heads=cfg.mem_n_heads, d_head=cfg.mem_d_head,
                csa_m=cfg.mem_csa_m, top_k=cfg.mem_top_k, n_win=cfg.mem_n_win,
                d_latent_q=cfg.mem_d_latent_q, n_groups=1,
                dropout=cfg.dropout,
            )
        else:
            attn = HeavilyCompressedAttention(
                d_model=d, n_heads=cfg.mem_n_heads, d_head=cfg.mem_d_head,
                hca_m=cfg.mem_hca_m, n_win=cfg.mem_n_win,
                d_latent_q=cfg.mem_d_latent_q, n_groups=1,
                dropout=cfg.dropout,
            )

        moe = DeepSeekMoE(
            d_model=d, n_experts=cfg.mem_n_experts, n_shared=cfg.mem_n_shared,
            top_k_experts=cfg.mem_top_k_experts, d_ff=cfg.mem_d_ff,
            dropout=cfg.dropout,
        )

        self.norm_attn = RMSNorm(d)
        self.norm_moe  = RMSNorm(d)
        self.mhc_attn  = ManifoldHyperConnections(d, cfg.n_hc, cfg.sinkhorn_iters)
        self.mhc_moe   = ManifoldHyperConnections(d, cfg.n_hc, cfg.sinkhorn_iters)
        self._attn = attn
        self._moe  = moe

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """X: [B, M, n_hc, mem_dim] → same shape + scalar balance loss"""
        X = self.mhc_attn(X, lambda h: self._attn(self.norm_attn(h)))

        bal = torch.zeros((), device=X.device)

        def _moe_fn(h: torch.Tensor) -> torch.Tensor:
            nonlocal bal
            out, bl = self._moe(self.norm_moe(h))
            bal = bl
            return out

        X = self.mhc_moe(X, _moe_fn)
        return X, bal


# ── Full thought stream ───────────────────────────────────────────────────────

class ThoughtStream(nn.Module):
    """
    Complete thought-stream processor.

    One forward call does:
      1. Add slot-index positional embeddings to mem_bank (temporal encoding).
      2. Expand to mHC residual format [B, M, n_hc, mem_dim].
      3. Run through n_mem_layers ThoughtBlocks (CSA alternates with HCA).
      4. Collapse mHC → [B, M, mem_dim]  (H_thought — for text blocks to read).
      6. Gated write: produce a new thought vector from the current text,
         scaled by a learned scalar write-decision α (modality choice).
      7. Append it and FIFO-evict the oldest slot if the bank exceeds max_mem.
      8. Return updated bank and the processed thought representations.
    """

    def __init__(self, cfg: DeepSeekV4MiniConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d    = cfg.mem_dim
        n_hc = cfg.n_hc

        # Slot-index positional embeddings (slot 0 = oldest, slot M-1 = newest)
        self.pos_embed = nn.Embedding(cfg.max_mem + cfg.consolidate_k, d)

        # Thought-stream transformer blocks
        self.blocks = nn.ModuleList(
            [ThoughtBlock(cfg, i) for i in range(cfg.n_mem_layers)]
        )

        # Dynamic mHC collapse: token-dependent softmax mixture over n_hc streams
        # Input is the mean across streams (cheap summary); output is n_hc weights.
        self.A_out_net = nn.Linear(d, n_hc, bias=False)
        self.norm_out  = RMSNorm(d)

        # Write head: attention-pooled context → per-dim gate + thought vector.
        # write_ctx_q projects each text token to a scalar score for the pool.
        # write_gate is per-dim (mem_dim) so each feature dimension can be
        # written independently — analogous to an LSTM input gate.
        self.write_ctx_q  = nn.Linear(cfg.d_model, 1, bias=False)
        self.write_gate   = nn.Linear(cfg.d_model, d, bias=True)
        self.thought_head = nn.Linear(cfg.d_model, d, bias=False)
        self.norm_write   = RMSNorm(d)

        # Write-decision head: scalar α = sigmoid(.) ∈ [0,1] scaling the whole
        # new thought vector — the model's "write or skip" modality choice.
        # bias=0 → α≈0.5 at init (neutral); the LM loss then learns when to write.
        self.write_decision = nn.Linear(cfg.d_model, 1, bias=True)

        # Telemetry: batch-mean of the last write probability α (set in
        # _new_thought). Lets train.py track the write/skip modality over training.
        self.last_write_alpha: Optional[torch.Tensor] = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _process(self, mem_bank: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run thought blocks on mem_bank. Returns (H_thought, balance_loss)."""
        B, M, d = mem_bank.shape
        n_hc     = self.cfg.n_hc

        # 1. Slot-index positional embeddings
        slots = torch.arange(M, device=mem_bank.device)
        X = mem_bank + self.pos_embed(slots).unsqueeze(0)   # [B, M, d]

        # 2. Expand to mHC format
        X = X.unsqueeze(2).expand(-1, -1, n_hc, -1).contiguous()  # [B, M, n_hc, d]

        # 3. Thought-stream transformer blocks
        total_bal = torch.zeros((), device=X.device)
        for blk in self.blocks:
            X, bal = blk(X)
            total_bal = total_bal + bal
        total_bal = total_bal / max(1, len(self.blocks))

        # 4. Dynamic mHC collapse → [B, M, d]
        # Use the per-slot mean as a cheap summary to compute mixture weights.
        X_mean = X.mean(dim=2)                                     # [B, M, d]
        A = torch.softmax(self.A_out_net(X_mean), dim=-1)          # [B, M, n_hc]
        H = (A.unsqueeze(-1) * X).sum(dim=2)                       # [B, M, d]
        H = self.norm_out(H)

        return H, total_bal

    # ── Process-only (no write, no consolidation) ─────────────────────────────

    def _process_only(
        self, mem_bank: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run thought blocks WITHOUT cross-modal update, consolidation, or write.
        Used by DualModalDeepSeekV4Mini to get H_thought BEFORE text blocks run,
        so the result can be injected into every text layer via cross-modal.

        Returns (H_thought [B, M, mem_dim], balance_loss scalar).
        """
        return self._process(mem_bank)

    # ── Write-only (consolidation + append, skips re-running thought blocks) ──

    def _write(
        self,
        H_text: torch.Tensor,           # [B, T, d_model]
        mem_bank: torch.Tensor,         # [B, M, mem_dim]
    ) -> torch.Tensor:
        """
        Append a new gated thought vector and FIFO-evict the oldest slot.

        Called by DualModalDeepSeekV4Mini after text blocks. The thought-block
        outputs read by the text stream are produced once by _process_only before
        the text blocks; this write step only needs the post-text H_text summary,
        so it does not re-run the thought blocks.

        Returns updated mem_bank [B, M', mem_dim].
        """
        cfg = self.cfg

        # Gated write + FIFO eviction
        m_new    = self._new_thought(H_text)
        mem_bank = torch.cat([mem_bank, m_new], dim=1)
        if mem_bank.size(1) > cfg.max_mem:
            mem_bank = mem_bank[:, -cfg.max_mem:, :]
        return mem_bank

    # ── Main forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        H_text: torch.Tensor,           # [B, T, d_model]  current text hidden states
        mem_bank: Optional[torch.Tensor],# [B, M, mem_dim] or None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Returns:
          H_thought  : [B, M, mem_dim] or None – processed thoughts for text to attend
          mem_bank   : [B, M', mem_dim]         – updated bank (M' ≤ max_mem)
          balance    : scalar                   – auxiliary MoE loss
        """
        cfg = self.cfg

        # ── No memories yet: skip processing, just write the first vector ─────
        if mem_bank is None or mem_bank.size(1) == 0:
            m_new    = self._new_thought(H_text)                # [B, 1, mem_dim]
            mem_bank = m_new
            return None, mem_bank, torch.zeros((), device=H_text.device)

        # ── 1. Run thought-stream blocks ──────────────────────────────────────
        H_thought, bal = self._process(mem_bank)                # [B, M, mem_dim]

        # ── 2. Gated write + FIFO eviction ────────────────────────────────────
        m_new    = self._new_thought(H_text)                     # [B, 1, mem_dim]
        mem_bank = torch.cat([mem_bank, m_new], dim=1)           # [B, M+1, mem_dim]
        if mem_bank.size(1) > cfg.max_mem:                       # drop oldest slot
            mem_bank = mem_bank[:, -cfg.max_mem:, :]

        return H_thought, mem_bank, bal

    def _new_thought(self, H_text: torch.Tensor) -> torch.Tensor:
        """
        Produce a new thought vector from the current text.

        Context: soft attention pool over all positions (vs. last-token only).
        Per-dim gate: sigmoid in [0,1]^mem_dim so each feature can be written
        independently — near-zero gate dims suppress uninformative features.
        Write-decision α: scalar sigmoid in [0,1] scaling the whole vector — the
        model's "write or skip" modality choice (α≈0 → an empty slot, evicted by
        FIFO; α≈1 → commit this thought). Fully differentiable; learned via the LM
        loss, so forced/redundant writes are no longer imposed every pass.
        """
        # Attention-pooled summary: learned scalar score per position → softmax
        scores  = self.write_ctx_q(H_text).squeeze(-1)         # [B, T]
        weights = torch.softmax(scores, dim=-1)                 # [B, T]
        h_ctx   = (weights.unsqueeze(-1) * H_text).sum(dim=1)  # [B, d_model]

        p     = torch.sigmoid(self.write_gate(h_ctx))          # [B, mem_dim] content gate
        m     = self.norm_write(self.thought_head(h_ctx))      # [B, mem_dim] thought
        alpha = torch.sigmoid(self.write_decision(h_ctx))      # [B, 1] modality choice
        # Stash the batch-mean write probability for telemetry (detached, no graph).
        # >0.5 ≈ "the model chose to commit this thought"; ≈0 ≈ "skip / empty slot".
        self.last_write_alpha = alpha.detach().mean()
        return (alpha * p * m).unsqueeze(1)                    # [B, 1, mem_dim]
