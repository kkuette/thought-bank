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

Consolidation
─────────────
When writing would exceed max_mem, the oldest `consolidate_k` vectors are
compressed into ONE new vector via cross-attention where:
  - Query  = projection of the current text summary (what matters NOW)
  - Keys/Values = the vectors being evicted (what we knew BEFORE)
The result carries information from both past and present, then the
`consolidate_k` old slots are replaced by this single condensed vector.

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


# ── Memory consolidation ──────────────────────────────────────────────────────

class MemoryConsolidator(nn.Module):
    """
    Compress the oldest `consolidate_k` thought vectors into one new vector
    that is informed by the current text context.

    Concretely:
        query  = proj(mean(H_text))           # what the model currently cares about
        keys   = old_vecs                      # what was known before
        values = old_vecs
        consolidated = cross_attn(Q, K, V)    # context-aware summary of old thoughts
    """

    def __init__(self, mem_dim: int, d_model: int, consolidate_k: int, n_heads: int = 2) -> None:
        super().__init__()
        self.k = consolidate_k
        # Project text summary to mem_dim for the query
        self.ctx_proj = nn.Linear(d_model, mem_dim, bias=False)
        self.norm_q   = RMSNorm(mem_dim)
        # Multi-head cross-attention (old thoughts → KV, text → Q)
        assert mem_dim % n_heads == 0
        self.attn   = nn.MultiheadAttention(mem_dim, n_heads, batch_first=True, dropout=0.0)
        self.norm_o = RMSNorm(mem_dim)
        self.out    = nn.Linear(mem_dim, mem_dim, bias=False)

    def forward(self, mem_bank: torch.Tensor, H_text: torch.Tensor) -> torch.Tensor:
        """
        mem_bank : [B, M, mem_dim]  (M >= max_mem, consolidation triggered)
        H_text   : [B, T, d_model]
        Returns  : [B, M - k + 1, mem_dim]
                   (oldest k slots replaced by 1 consolidated vector)
        """
        k        = self.k
        old_vecs = mem_bank[:, :k, :]   # [B, k, mem_dim]  — to compress
        kept     = mem_bank[:, k:, :]   # [B, M-k, mem_dim] — to keep

        # Text summary as query
        h_summary = H_text.mean(dim=1)                         # [B, d_model]
        query     = self.norm_q(self.ctx_proj(h_summary)).unsqueeze(1)  # [B, 1, mem_dim]

        # Cross-attention: what matters in old memories given the current context?
        consolidated, _ = self.attn(query, old_vecs, old_vecs)  # [B, 1, mem_dim]
        consolidated    = self.out(self.norm_o(consolidated))    # [B, 1, mem_dim]

        # New bank: [consolidated_summary | kept_memories]
        return torch.cat([consolidated, kept], dim=1)           # [B, M-k+1, mem_dim]


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
      5. Cross-modal update: thoughts absorb text summary via a gated residual.
      6. Consolidation: if bank is full, compress oldest k vectors using the
         current text context into one richer vector (MemoryConsolidator).
      7. Write gate: decide whether to append a new thought vector derived from
         the current last-token text hidden state.
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

        # Cross-modal: thought ← text  (gated residual from text summary)
        self.text_proj = nn.Linear(cfg.d_model, d, bias=False)
        self.text_gate = nn.Linear(d, d, bias=False)   # per-dim gate
        self.norm_tm   = RMSNorm(d)

        # Write head: attention-pooled context → per-dim gate + thought vector.
        # write_ctx_q projects each text token to a scalar score for the pool.
        # write_gate is per-dim (mem_dim) so each feature dimension can be
        # written independently — analogous to an LSTM input gate.
        self.write_ctx_q  = nn.Linear(cfg.d_model, 1, bias=False)
        self.write_gate   = nn.Linear(cfg.d_model, d, bias=True)
        self.thought_head = nn.Linear(cfg.d_model, d, bias=False)
        self.norm_write   = RMSNorm(d)

        # Consolidator
        self.consolidator = MemoryConsolidator(
            mem_dim=d, d_model=cfg.d_model,
            consolidate_k=cfg.consolidate_k,
            n_heads=max(1, cfg.mem_n_heads),
        )

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
        H_thought: torch.Tensor,        # [B, M, mem_dim]  pre-computed by _process_only
    ) -> torch.Tensor:
        """
        Apply cross-modal update to pre-computed H_thought, run consolidation
        if needed, and append a new thought vector.

        Called by DualModalDeepSeekV4Mini after text blocks, reusing H_thought
        already computed before text blocks to avoid running thought-stream
        blocks twice per forward pass.

        Returns updated mem_bank [B, M', mem_dim].
        """
        cfg = self.cfg

        # Cross-modal: thoughts absorb text summary (gated)
        h_text_summary = H_text.mean(dim=1, keepdim=True)        # [B, 1, d_model]
        h_proj = self.text_proj(h_text_summary)                   # [B, 1, mem_dim]
        gate   = torch.sigmoid(self.text_gate(self.norm_tm(H_thought)))
        # H_thought updated in place (not returned — caller already has it)
        _ = H_thought + gate * h_proj

        # Consolidation
        if mem_bank.size(1) >= cfg.max_mem:
            mem_bank = self.consolidator(mem_bank, H_text)

        # Append new thought
        m_new    = self._new_thought(H_text)
        mem_bank = torch.cat([mem_bank, m_new], dim=1)
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

        # ── 2. Cross-modal: thoughts absorb text summary (gated) ─────────────
        h_text_summary = H_text.mean(dim=1, keepdim=True)       # [B, 1, d_model]
        h_proj = self.text_proj(h_text_summary)                  # [B, 1, mem_dim]
        gate   = torch.sigmoid(self.text_gate(self.norm_tm(H_thought)))  # [B, M, mem_dim]
        H_thought = H_thought + gate * h_proj                    # broadcast over M

        # ── 3. Consolidation: compress oldest k slots if bank is full ─────────
        if mem_bank.size(1) >= cfg.max_mem:
            mem_bank = self.consolidator(mem_bank, H_text)       # [B, M-k+1, mem_dim]

        # ── 4. Write new thought vector ───────────────────────────────────────
        m_new    = self._new_thought(H_text)                     # [B, 1, mem_dim]
        mem_bank = torch.cat([mem_bank, m_new], dim=1)           # [B, M'+1, mem_dim]

        return H_thought, mem_bank, bal

    def _new_thought(self, H_text: torch.Tensor) -> torch.Tensor:
        """
        Produce a new thought vector from the current text.

        Context: soft attention pool over all positions (vs. last-token only).
        Gate: per-dim sigmoid in [0,1]^mem_dim so each feature can be written
        independently — near-zero gate dims suppress uninformative features.
        """
        # Attention-pooled summary: learned scalar score per position → softmax
        scores  = self.write_ctx_q(H_text).squeeze(-1)         # [B, T]
        weights = torch.softmax(scores, dim=-1)                 # [B, T]
        h_ctx   = (weights.unsqueeze(-1) * H_text).sum(dim=1)  # [B, d_model]

        p = torch.sigmoid(self.write_gate(h_ctx))              # [B, mem_dim]
        m = self.norm_write(self.thought_head(h_ctx))          # [B, mem_dim]
        return (p * m).unsqueeze(1)                            # [B, 1, mem_dim]
