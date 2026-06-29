# deepseek_v4_mini

Small Python reproduction of the [DeepSeek-V4](https://arxiv.org/abs/2606.19348) architecture, fused with a dual-stream thought memory system.

Designed for single-GPU experimentation (~6M–32M params).

---

## Architecture overview

```
Pass k
──────

  ┌─────────────────────────────────────────────────────────┐
  │  Thought stream  [B, M, mem_dim]                        │
  │                                                         │
  │  pos_embed(slot_idx) → CSA/HCA blocks (mHC) → H_thought│
  │                                                         │
  │  Slot 0 = oldest surviving thought                      │
  │  Slot M-1 = most recent thought                         │
  └────────────────────┬────────────────────────────────────┘
                       │  cross-modal at every layer
                       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Text stream  [B, T, d_model]                           │
  │                                                         │
  │  for each layer:                                        │
  │    mHC( CSA or HCA )          ← self-attention          │
  │    cross_attn( text ← thought )                        │
  │    mHC( MoE )                 ← feed-forward            │
  └────────────────────┬────────────────────────────────────┘
                       │  H_text [B, T, d_model]
                       ▼
               ThoughtStream.write()
                       │
          ┌────────────┴────────────┐
          │  bank full?             │
          │  yes → consolidate      │  MemoryConsolidator:
          │        (oldest k → 1)   │  query  = proj(mean(H_text))
          │        then append      │  kv     = old thought vectors
          │  no  → append directly  │  output = context-aware summary
          └────────────────────────┘
                       │
              Banque [B, M', mem_dim]  →  pass k+1
```

**Even-indexed layers** use **CSA** (Compressed Sparse Attention).  
**Odd-indexed layers** use **HCA** (Heavily Compressed Attention).  
Both streams follow the same pattern at different scales.

---

## Key components

### mHC — Manifold-Constrained Hyper-Connections (`mhc.py`)

Replaces standard residual connections. The residual stream is widened by a factor `n_hc`, and the residual mapping matrix **B** is constrained to the Birkhoff polytope (doubly stochastic matrices) via Sinkhorn-Knopp iteration. This bounds `||B||₂ ≤ 1`, making deep stacks numerically stable.

```
X_{l+1} = B_l X_l + C_l F_l(A_l X_l)

X_l ∈ R^{n_hc × d}   — expanded residual stream
A_l, B_l, C_l         — dynamically generated from X_l
B_l ∈ Birkhoff polytope  (Sinkhorn-Knopp projection)
```

### CSA — Compressed Sparse Attention (`attention.py`)

Compression factor `m` with **overlapping** windows (two KV series Ca, Cb).  
A lightweight indexer selects the top-k most relevant compressed blocks for each query token. A sliding window branch (`n_win` tokens) handles local dependencies.

```
n_blocks = T // m

For token t in block i = t // m:
  - attend top-k compressed blocks from {0, …, i-1}   (global, sparse)
  - attend last n_win tokens                            (local, dense)
```

### HCA — Heavily Compressed Attention (`attention.py`)

Compression factor `m' >> m` with **non-overlapping** windows.  
Dense attention over all preceding compressed blocks — no top-k selection.  
Lower per-token cost than CSA; captures very long-range structure cheaply.

### DeepSeekMoE (`moe.py`)

Fine-grained mixture of experts with:
- **Shared experts** — always active (`n_shared`, typically 1)
- **Routed experts** — top-k activated per token based on affinity score `√(softplus(h W_gate))`

An auxiliary sequence-balance loss prevents expert collapse.

### Thought stream (`memory.py`)

A second transformer (CSA/HCA + mHC) that operates on the memory bank `[B, M, mem_dim]`. Uses **slot index as temporal encoding** — `nn.Embedding(max_mem, mem_dim)` indexed by position in the bank, so slot 0 = oldest, slot M-1 = newest.

### Memory consolidation (`memory.py`)

When the bank reaches `max_mem`, instead of dropping old vectors:

1. Take the `consolidate_k` oldest vectors.
2. Cross-attend them with a query derived from the current text summary.
3. Produce **one condensed vector** that blends past knowledge with present context.
4. Replace the `k` old slots with this single vector.

The bank therefore grows to `max_mem`, consolidates back to `max_mem - k + 1`, then grows again — staying bounded while never discarding information cold.

---

## Models

| Class | Description |
|---|---|
| `DeepSeekV4Mini` | Single text stream, optional bolt-on thought memory |
| `DualModalDeepSeekV4Mini` | Full dual-stream: text + thought, both CSA/HCA |

### Parameter counts

| Config | Text stream | Thought stream | Total |
|---|---|---|---|
| tiny | ~6M | ~50k | ~6.5M |
| small | ~31M | ~256k | ~32M |

---

## Quick start

```python
from deepseek_v4_mini import DualModalDeepSeekV4Mini, DeepSeekV4MiniConfig
import torch

cfg   = DeepSeekV4MiniConfig.tiny()
model = DualModalDeepSeekV4Mini(cfg)

ids = torch.randint(0, cfg.vocab_size, (2, 64))

# First pass — no memory
out = model(ids)
print(out["logits"].shape)   # [2, 64, 32000]
print(out["mem_bank"].shape) # [2, 1, 32]

# Subsequent passes — carry the bank across calls
out2 = model(ids, init_mem=out["mem_bank"])
```

### Training

```bash
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/tiny.yaml
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/small.yaml
```

The training script streams from HuggingFace datasets (default: `roneneldan/TinyStories`), logs to TensorBoard, and saves checkpoints. Edit the YAML to change dataset, learning rate, precision, etc.

---

## Configuration

All hyperparameters live in `DeepSeekV4MiniConfig` (`config.py`).

```python
# Programmatic config
cfg = DeepSeekV4MiniConfig(
    d_model=256, n_layers=6, n_heads=4, d_head=64,
    csa_m=4, hca_m=32, top_k_csa=8, n_win=16,
    n_experts=8, top_k_experts=2, d_ff=512,
    # Thought stream
    mem_dim=64, max_mem=32, consolidate_k=8,
    n_mem_layers=2,
)

# Or from YAML
cfg = DeepSeekV4MiniConfig.from_yaml("deepseek_v4_mini/configs/small.yaml")

# Built-in presets
cfg = DeepSeekV4MiniConfig.tiny()   # ~6M params
cfg = DeepSeekV4MiniConfig.small()  # ~32M params
```

Key parameters:

| Parameter | Description |
|---|---|
| `n_hc` | mHC residual stream width (2 = paper default) |
| `sinkhorn_iters` | Iterations for Birkhoff projection (5 = paper, 3 = faster) |
| `csa_m` | CSA compression factor (overlapping windows) |
| `hca_m` | HCA compression factor (`>> csa_m`) |
| `top_k_csa` | Compressed blocks attended per token in CSA |
| `n_win` | Sliding window size (both CSA and HCA) |
| `mem_dim` | Thought vector dimension |
| `max_mem` | Bank size that triggers consolidation |
| `consolidate_k` | Old vectors compressed into 1 at each consolidation |
| `n_mem_layers` | Depth of the thought-stream transformer |

---

## File structure

```
deepseek_v4_mini/
  config.py      — DeepSeekV4MiniConfig dataclass + YAML loader
  mhc.py         — ManifoldHyperConnections + RMSNorm
  attention.py   — CompressedSparseAttention, HeavilyCompressedAttention, RoPE
  moe.py         — SwiGLU, DeepSeekMoE
  memory.py      — MemoryConsolidator, ThoughtBlock, ThoughtStream
  model.py       — DeepSeekV4Mini, DualModalDeepSeekV4Mini
  train.py       — Training loop (HF datasets, TensorBoard, checkpointing)
  configs/
    tiny.yaml    — ~6M params, fast iteration
    small.yaml   — ~32M params, single RTX 3090
```

---

## References

- DeepSeek-V4: [arxiv 2606.19348](https://arxiv.org/abs/2606.19348)
- Hyper-Connections: Zhu et al., 2025
- DeepSeekMoE: Dai et al., 2024
- Muon optimizer: Jordan et al., 2024
- Thought memory baseline: [`thought_lm_minimal`](../thought_lm_minimal/)
