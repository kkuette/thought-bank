from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DeepSeekV4MiniConfig:
    # ── Core dimensions ──────────────────────────────────────────────────────
    vocab_size: int = 32000
    d_model: int = 256
    n_layers: int = 6          # total blocks; even → CSA, odd → HCA
    n_heads: int = 4           # query heads per attention layer
    d_head: int = 64           # compressed KV head dimension
    dropout: float = 0.0
    max_seq_len: int = 2048

    # ── mHC: Manifold-Constrained Hyper-Connections (§2.2) ───────────────────
    n_hc: int = 2              # residual stream expansion factor
    sinkhorn_iters: int = 20   # Sinkhorn-Knopp iterations (paper uses t_max=20)

    # ── Attention (§2.3) ──────────────────────────────────────────────────────
    csa_m: int = 4             # CSA: compress every m tokens (overlapping)
    hca_m: int = 16            # HCA: compress every m' tokens (non-overlapping, m' >> m)
    top_k_csa: int = 8         # CSA: top-k compressed blocks to attend
    n_win: int = 16            # sliding window tokens for local fine-grained attention
    d_latent_q: int = 64       # low-rank query compression dimension
    n_groups: int = 2          # grouped output projection groups

    # ── MoE (§2.1, DeepSeekMoE) ───────────────────────────────────────────────
    n_experts: int = 8         # routed expert count
    n_shared: int = 1          # shared experts (always active)
    top_k_experts: int = 2     # routed experts activated per token
    d_ff: int = 512            # FFN hidden dim per expert

    # ── Fast-weight thought bank ──────────────────────────────────────────────
    # The bank is a rolling FIFO buffer of M thought vectors (dim = mem_dim). It is
    # READ as fast WEIGHTS: each slot parametrises a low-rank MLP layer applied in
    # sequence to the text stream (see DualModalBlock._cross_modal). It is WRITTEN by
    # a gated head that appends one vector per forward. There is no separate
    # thought-stream transformer — the text model writes vectors and reuses them.
    mem_dim: int = 64          # thought-vector dimension (= fast-weight code size)
    max_mem: int = 32          # max bank size (FIFO-evict the oldest beyond this)
    mem_seed_slots: int = 4    # random-uniform[0,1] slots seeding a fresh bank
    mem_read_rank: int = 16    # bottleneck rank of each per-slot fast-weight layer
    mem_read_dropout: float = 0.0  # dropout inside the fast-weight MLP layers

    # ── Model selection ───────────────────────────────────────────────────────
    # True  → DualModalDeepSeekV4Mini (text stream + fast-weight thought bank).
    # False → DeepSeekV4Mini (legacy text-only, optional bolt-on memory below).
    use_dual_stream: bool = True
    use_thought_memory: bool = False    # legacy DeepSeekV4Mini bolt-on memory only

    # ── Training ──────────────────────────────────────────────────────────────
    balance_loss_weight: float = 1e-4   # MoE balance loss weight (paper §4.2.2: 0.0001)
    # Sparsity budget on the write-decision α: adds cost · E[-log(1-α)] to the loss
    # so writing has an opportunity cost. 0.0 = no penalty (α free to saturate at 1).
    # The budget form keeps a live gradient even when α≈1 (unlike an L1 on α).
    mem_write_cost: float = 0.0
    # Novelty-gated write: penalise the cosine of each new write to the closest
    # existing bank slot (adds weight · E[max_j cos(m_new, slot_j)] to the loss).
    # Trains the write head to store DIVERSE thoughts instead of near-duplicates —
    # without it the bank collapses to ~1 effective vector. 0.0 = off.
    mem_write_diversity: float = 0.0
    # Code-space augmentation: std of Gaussian noise added to each newly written
    # vector during TRAINING only. Forces the read to decode a neighbourhood of
    # each stored code rather than the exact trained points — the lever for
    # held-out-rule generalization (the read snaps to trained codes without it).
    # Scale is relative to the RMSNorm'd write (per-dim RMS ≈ 1). 0.0 = off.
    # NEGATIVE RESULT (2026-07-03, σ=0.1, K=2 interleaved): train 0.995 / held
    # 0.000 exact — noise teaches local robustness but actively REINFORCES
    # snapping (a midpoint code falls in a neighbour's cloud, where supervision
    # says "behave like the neighbour"). Kept as a regularizer knob only.
    mem_write_noise: float = 0.0
    # Code-space mixup: with probability p per presentation lane, replace the
    # stored rule code by the MIDPOINT of the EMA codes of the two neighbouring
    # rules (s-1, s+1) while keeping the labels of s — supervises the read at
    # points BETWEEN trained codes so interpolation is learned as a law of the
    # manifold. Only fires when s-1, s+1 (and s) are all trained shifts → no
    # held-out leakage. Requires mem_teacher_forcing plumbing (rule ids).
    mem_code_mixup_p: float = 0.0
    # EMA momentum for the per-shift code dictionary the mixup draws from.
    mem_code_mixup_momentum: float = 0.99
    # Target-rate objective on the write gate: adds weight · (E[α] - target)² to the
    # loss. Unlike mem_write_cost (monotone, only pushes α→0), this has a stable
    # minimum at α=target, so it curbs BOTH α→1 (over-write, duplicate pollution)
    # and α→0 (never write) — the two ways the bank collapses. weight 0.0 = off.
    mem_write_target: float = 0.0        # target E[α] (e.g. 0.5); 0.0 disables via weight
    mem_write_target_weight: float = 0.0

    # ── Write gate ────────────────────────────────────────────────────────────
    # When False the write head is UNGATED: the appended slot is the pure normalised
    # thought (no scalar α, no per-dim content gate p). The gate otherwise attenuates
    # the written vector, diminishing the code it carries; ablating it removes a
    # coordination variable during bootstrap. Keep OFF while bootstrapping the
    # fast-weight transport, re-enable for streaming write selectivity.
    mem_write_gate: bool = True

    # ── Teacher-forced bank bootstrap (multiturn_rule) ────────────────────────
    # Breaks the "ignore-bank" fixed point: during bootstrap the read consumes a
    # CLEAN teacher code correlated with the latent rule id (a meta-training signal —
    # the id is latent at inference), while a distill loss pulls the WRITTEN slot
    # toward that teacher. The read's code is annealed teacher→written (β: 1→0).
    # Only wired for multiturn_rule (needs the ground-truth rule id per conversation).
    mem_teacher_forcing: bool = False
    mem_teacher_anneal_start: int = 300      # β=1 until this optimiser step
    mem_teacher_anneal_end: int = 500        # β linear→0 by here, then 0 (teacher gone)
    mem_teacher_distill_weight: float = 2.0  # weight on MSE(w0, teacher[s]); scaled by β

    # ── Factory helpers ───────────────────────────────────────────────────────
    @classmethod
    def tiny(cls) -> "DeepSeekV4MiniConfig":
        """~8M params – fast CPU/single-GPU experimentation."""
        return cls(
            d_model=128, n_layers=4, n_heads=2, d_head=32,
            csa_m=4, hca_m=16, top_k_csa=4, n_win=8,
            d_latent_q=32, n_groups=1, n_experts=4,
            top_k_experts=1, d_ff=256, sinkhorn_iters=3,
            mem_dim=32, max_mem=16, mem_seed_slots=4, mem_read_rank=16,
        )

    @classmethod
    def small(cls) -> "DeepSeekV4MiniConfig":
        """~50M params – single RTX 3090 training target."""
        return cls(
            d_model=256, n_layers=6, n_heads=4, d_head=64,
            csa_m=4, hca_m=32, top_k_csa=8, n_win=16,
            d_latent_q=64, n_groups=2, n_experts=8,
            top_k_experts=2, d_ff=512,
            mem_dim=64, max_mem=32, mem_seed_slots=4, mem_read_rank=16,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "DeepSeekV4MiniConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**fields)
