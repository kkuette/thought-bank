from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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

    # ── Thought Memory bank ───────────────────────────────────────────────────
    mem_dim: int = 64          # thought vector dimension
    max_mem: int = 32          # max bank size before consolidation
    consolidate_k: int = 4     # how many oldest vectors to compress into 1

    # ── Thought stream (dual-modal) ───────────────────────────────────────────
    # Mirrors the text-stream architecture at smaller scale.
    # Slot index in the bank doubles as temporal encoding (nn.Embedding).
    use_dual_stream: bool = True
    n_mem_layers: int = 2      # thought-stream transformer depth
    mem_n_heads: int = 2       # query heads in thought attention
    mem_d_head: int = 16       # KV head dim in thought attention
    mem_d_latent_q: int = 16   # low-rank query dim in thought attention
    mem_csa_m: int = 2         # CSA compression factor for thought stream
    mem_hca_m: int = 4         # HCA compression factor for thought stream
    mem_top_k: int = 4         # top-k blocks in thought CSA
    mem_n_win: int = 4         # sliding window in thought attention
    mem_n_experts: int = 2     # routed experts in thought MoE
    mem_n_shared: int = 1      # shared experts in thought MoE
    mem_top_k_experts: int = 1 # activated routed experts per thought token
    mem_d_ff: int = 64         # FFN dim in thought MoE

    # ── Legacy single-stream thought memory (DeepSeekV4Mini only) ───────────
    use_thought_memory: bool = False    # bolt-on sequential memory for the legacy model

    # ── Training ──────────────────────────────────────────────────────────────
    balance_loss_weight: float = 1e-4   # MoE balance loss weight (paper §4.2.2: 0.0001)

    # ── Factory helpers ───────────────────────────────────────────────────────
    @classmethod
    def tiny(cls) -> "DeepSeekV4MiniConfig":
        """~8M params – fast CPU/single-GPU experimentation."""
        return cls(
            d_model=128, n_layers=4, n_heads=2, d_head=32,
            csa_m=4, hca_m=16, top_k_csa=4, n_win=8,
            d_latent_q=32, n_groups=1, n_experts=4,
            top_k_experts=1, d_ff=256, sinkhorn_iters=3,
            # thought stream
            mem_dim=32, max_mem=16, consolidate_k=4,
            n_mem_layers=2, mem_n_heads=1, mem_d_head=16,
            mem_d_latent_q=16, mem_csa_m=2, mem_hca_m=4,
            mem_top_k=2, mem_n_win=4, mem_n_experts=2,
            mem_d_ff=64,
        )

    @classmethod
    def small(cls) -> "DeepSeekV4MiniConfig":
        """~50M params – single RTX 3090 training target."""
        return cls(
            d_model=256, n_layers=6, n_heads=4, d_head=64,
            csa_m=4, hca_m=32, top_k_csa=8, n_win=16,
            d_latent_q=64, n_groups=2, n_experts=8,
            top_k_experts=2, d_ff=512,
            # thought stream
            mem_dim=64, max_mem=32, consolidate_k=8,
            n_mem_layers=2, mem_n_heads=2, mem_d_head=16,
            mem_d_latent_q=16, mem_csa_m=2, mem_hca_m=8,
            mem_top_k=4, mem_n_win=4, mem_n_experts=2,
            mem_d_ff=128,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "DeepSeekV4MiniConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**fields)
