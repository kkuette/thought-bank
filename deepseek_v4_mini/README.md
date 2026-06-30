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
          │  gated write            │  m_new = α · p · m
          │   α = σ(write_decision) │   α : scalar write/skip choice
          │   p = σ(write_gate)     │   p : per-dim content gate
          │   m = thought_head(ctx) │   m : the thought vector
          │  append, then FIFO-evict│
          │  oldest if > max_mem    │
          └────────────────────────┘
                       │
              Bank [B, M', mem_dim]  →  pass k+1
              (carried across passes; persists across sequences if enabled)
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

### Gated write + FIFO eviction (`memory.py`)

After the text blocks, each segment writes one thought vector to the bank:

```
ctx   = attention-pool over the segment's H_text     # [B, d_model]
m     = norm(thought_head(ctx))                      # the thought  [B, mem_dim]
p     = sigmoid(write_gate(ctx))                     # per-dim content gate
α     = sigmoid(write_decision(ctx))                 # scalar write/skip choice
m_new = α · p · m                                     # gated thought  [B, 1, mem_dim]
```

The vector is appended; if the bank exceeds `max_mem` the **oldest slot is
FIFO-evicted**. The scalar `α` is the model's *modality choice* — write this
thought or skip it — learned end-to-end through the LM loss. (An earlier
cross-attention consolidator was removed in favour of this simpler, trainable
write; the `consolidate_k` field now only sizes the slot positional embedding.)

**Training the write head requires `mem_bptt_window ≥ 2`.** The write is a pure
*output* of a segment — the segment's own loss never depends on it (the write
happens after the LM head); its only consumer is the *next* segment's read. With
a per-segment detach (`window = 1`) the write head gets zero gradient and the
bank is filled by an untrained projection. `window ≥ 2` keeps the graph across a
boundary so segment i+1's loss trains segment i's write.

### Persistence across sequences (`train.py`)

By default the bank is reset every sequence, so it only spans the in-sequence
segments. With `data.persist: true` each batch lane streams one source file in
order and the bank is **carried across training steps** (reset at file
boundaries) — the "remember earlier context" use case. This is what makes the
memory actually useful (see the root README's Findings).

---

## Models

| Class | Description |
|---|---|
| `DeepSeekV4Mini` | Single text stream, optional bolt-on thought memory |
| `DualModalDeepSeekV4Mini` | Full dual-stream: text + thought, both CSA/HCA |

### Parameter counts

Dominated by the token embedding (`vocab_size × d_model`), so the total depends
on the tokenizer. Examples:

| Config | vocab | Total |
|---|---|---|
| `tiny()` preset | 32k | ~6.5M |
| `tiny.yaml` (DeepSeek-V3 tokenizer) | ~129k | ~19M |
| `small.yaml` | ~129k | ~32M |

---

## Quick start

```python
from deepseek_v4_mini import DualModalDeepSeekV4Mini, DeepSeekV4MiniConfig
import torch

cfg   = DeepSeekV4MiniConfig.tiny()
model = DualModalDeepSeekV4Mini(cfg)

ids = torch.randint(0, cfg.vocab_size, (2, 64))

# First pass — empty bank
out = model(ids)
print(out["logits"].shape)    # [2, 64, vocab_size]
print(out["mem_bank"].shape)  # [2, 1, mem_dim]
print(out["write_alpha"])     # mean write probability α (write/skip telemetry)

# Subsequent passes — carry the bank across calls
out2 = model(ids, init_mem=out["mem_bank"])
```

### Training

```bash
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/tiny.yaml          # TinyStories
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/code.yaml          # code, bank reset/seq
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/code_persist.yaml  # code, bank persists
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/code_persist_sparse.yaml  # + write-sparsity budget
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/synth_recall.yaml  # addressable-recall test
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/gist.yaml          # latent-context test
```

The script streams from HuggingFace datasets (default `roneneldan/TinyStories`),
or generates synthetic data for `task: associative_recall` / `task: latent_context`.
It logs to `runs/<run_name>/metrics.jsonl` and saves checkpoints.

**Memory probes** run during training (every `mem_probe_every` steps):
`mem_ablation_gap` (CE without vs with the bank), `mem_diversity` (slot spread),
`mem_write_rate` (mean α), and — for persistent runs — `persist_gap` and its
decomposition `content_gap` + `structure_gap`.

> **`content_gap` is the memory metric to trust.** `persist_gap` (bank carried
> across chunks vs reset each chunk) conflates the *content* written into slots
> with the bank *structure* (slot count + slot positional embeddings): a carried
> bank has ~`max_mem` slots, a reset one rebuilds from empty. Re-running the
> carried arm with writes **zeroed** (slots still appended, slot count identical)
> isolates them — `content_gap = CE_zero − CE_real` is the pure content benefit,
> `structure_gap = CE_reset − CE_zero` the rest. On the code dataset ~2/3 of
> `persist_gap` turned out to be structural; trust `content_gap`.

Offline PPL with/without the bank: `python -m deepseek_v4_mini.eval_memory`.

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
    mem_dim=64, max_mem=32, n_mem_layers=2,
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
| `max_mem` | Bank capacity; oldest slot is FIFO-evicted past this |
| `n_mem_layers` | Depth of the thought-stream transformer |

Memory training knobs live in the YAML `training:` / `data:` sections:

| Parameter | Description |
|---|---|
| `mem_segment_len` | Attention window per segment; smaller ⇒ more reliance on the bank |
| `mem_bptt_window` | TBPTT span; **≥2 required** to train the write head |
| `mem_probe_every` | How often to run the ablation / persistence probes |
| `mem_write_cost` | Sparsity budget on α: adds `cost · E[-log(1-α)]` so writing has a cost (0 ⇒ α saturates at 1). Needs a warmup if used. |
| `data.persist` | `true` ⇒ per-file ordered lanes + carry the bank across steps |

---

## File structure

```
deepseek_v4_mini/
  config.py      — DeepSeekV4MiniConfig dataclass + YAML loader
  mhc.py         — ManifoldHyperConnections + RMSNorm
  attention.py   — CompressedSparseAttention, HeavilyCompressedAttention, RoPE
  moe.py         — SwiGLU, DeepSeekMoE
  memory.py      — ThoughtBlock, ThoughtStream (gated write + FIFO)
  model.py       — DeepSeekV4Mini, DualModalDeepSeekV4Mini
  train.py       — Training loop, memory/persistence probes, synthetic tasks
  eval_memory.py — Offline PPL with vs without the bank
  configs/
    tiny.yaml          — TinyStories, fast iteration
    small.yaml         — TinyStories, single RTX 3090
    code.yaml              — code (Python), bank reset per sequence
    code_persist.yaml      — code (Python), bank persists across sequences
    code_persist_sparse.yaml — persistent + write-sparsity budget (mem_write_cost)
    synth_recall.yaml      — synthetic addressable-recall diagnostic
    gist.yaml              — synthetic latent-context (gist) diagnostic
```

---

## References

- DeepSeek-V4: [arxiv 2606.19348](https://arxiv.org/abs/2606.19348)
- Hyper-Connections: Zhu et al., 2025
- DeepSeekMoE: Dai et al., 2024
- Muon optimizer: Jordan et al., 2024
- Thought memory baseline: [`thought_lm_minimal`](../thought_lm_minimal/)
