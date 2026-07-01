# deepseek_v4_mini

Small Python reproduction of the [DeepSeek-V4](https://arxiv.org/abs/2606.19348) architecture, fused with a **fast-weight thought bank**: a rolling memory the model reads as *weights* (not attended data) and writes to itself, targeting continual learning at inference **without a backward pass**.

Designed for single-GPU experimentation (~6M–32M params).

---

## The idea: memory as fast weights

The thought bank `[B, M, mem_dim]` is not a KV cache the text attends to. Each slot is
expanded by a learned hypernet into a small low-rank MLP layer, and the token stream is
passed **through** that stack of layers. The model writes its own vectors into the bank and
then reuses them as the weights of its own forward pass. A rule inferred at turn 0 (e.g.
"shift every symbol by `s`") can thus be *applied* at later turns even though the answer
window contains no examples — the rule crosses the turn boundary through the bank, as a
fast weight. See [Schmidhuber 1992; Ba et al. 2016; Schlag et al. 2021].

---

## Architecture overview

```
Pass k  (one turn / segment)
────────────────────────────

  input_ids [B,T] ──► embed ──► X [B,T,n_hc,d]
                                   │
   thought bank [B,M,mem_dim] ─────┤  read as FAST WEIGHTS at every block
   (seeded random[0,1] on a fresh  │
    conversation, else carried in) │
                                   ▼
  ┌──────────────────── DualModalBlock × n_layers ───────────────────┐
  │  1. mHC( CSA even / HCA odd )          ← text self-attention      │
  │  2. fast-weight read( text ← bank )    ← slot-parametrised MLP    │
  │  3. mHC( MoE )                         ← feed-forward             │
  └──────────────────────────────┬───────────────────────────────────┘
                                 │  H_text [B,T,d]
                                 ▼
                   ThoughtStream.write()   (once, after the blocks)
                                 │   m = norm(thought_head(pool(H_text)))
                                 │   gate (optional): m_new = α·p·m
                                 │   append, FIFO-evict oldest past max_mem
                                 ▼
                   mem_bank [B,M',mem_dim]  ──►  pass k+1
                   (carry as init_mem for multi-turn continual learning)
```

The bank is **shared and static across the blocks of one forward** (all `n_layers` blocks
read the same bank); the write happens **once, after** the blocks. There is no separate
thought-stream transformer — the text model writes the vectors and reuses them directly.

**Even-indexed layers** use **CSA** (Compressed Sparse Attention); **odd-indexed layers**
use **HCA** (Heavily Compressed Attention).

---

## Fast-weight read (`model.py` · `DualModalBlock._cross_modal`)

Each slot `mᵢ ∈ R^mem_dim` is expanded by a learned hypernet into a low-rank layer
`Aᵢ ∈ R^{r×d}`, `Bᵢ ∈ R^{d×r}` (`r = mem_read_rank`), applied **sequentially** over the
`M` slots:

```
y ← norm(h)                                  # y0
for i in range(M):                           # one fast-weight layer per slot
    y ← y + dropout( Bᵢ · GELU(Aᵢ · y) )     # residual, non-linear
read = h + fw_o(y − y0)                       # net delta; trivial bank ≈ identity
```

The **GELU between slots** is load-bearing: without it, stacking/summing slots collapses to
a single low-rank *linear* map (the failure mode of the earlier outer-product read, which
could not express an input-conditioned permutation). Placed between attention and MoE so the
applied "weights" also influence expert routing.

---

## Write head + FIFO eviction (`memory.py` · `ThoughtStream`)

`memory.py` owns only the **write** side (the read lives in the block). A fresh bank is
seeded by `seed_bank()` with `mem_seed_slots` random-uniform[0,1] vectors, so the
fast-weight layers are non-zero from the first forward; later writes append on top.

```
ctx   = attention-pool over H_text (pad-masked)   # [B, d_model]
m     = norm(thought_head(ctx))                    # the thought      [B, mem_dim]
# optional gate (mem_write_gate: true):
p     = sigmoid(write_gate(ctx))                    # per-dim content gate
α     = sigmoid(write_decision(ctx))               # scalar write/skip choice
m_new = m           if gate off   else   α · p · m
```

The vector is appended; past `max_mem` the **oldest slot is FIFO-evicted**.

> **Write gate (`mem_write_gate`).** With the gate on, `α` is the model's write/skip
> *modality choice* and `p` a per-dim content gate — useful for streaming selectivity. But
> the gate **attenuates** the written vector (`α·p·m` can only shrink `m`), which slows the
> bootstrap of the fast-weight transport and dilutes the code a downstream read must recover.
> Set `mem_write_gate: false` to write the pure normalised thought while bootstrapping;
> re-enable it once transport works and selectivity matters.

**Training the write head requires `mem_bptt_window ≥ 2`.** The write is a pure *output* of
a segment — the segment's own loss never depends on it (the write happens after the LM head);
its only consumer is the *next* segment's read. With a per-segment detach (`window = 1`) the
write head gets zero gradient. `window ≥ 2` keeps the graph across a boundary so segment
`i+1`'s loss trains segment `i`'s write.

---

## Teacher-forced bank bootstrap (`train.py`, `multiturn_rule` only)

Read and write each work in isolation — the read applies any fixed code (clean, learned, or
frozen-random) to ~100%, and the write can encode the latent rule so it is decodable — yet
naïve **joint** training sticks at an "ignore-bank" fixed point (`rule_acc` at chance): at
init the read ≈ identity and the early written code is useless, so no gradient tells the read
to consume the bank, and the write never gets a read-useful gradient.

The fix is a bootstrap that breaks the fixed point. During training on `multiturn_rule`
(each conversation draws a fresh rule id `s`, a legitimate meta-training signal since `s` is
latent at inference):

```
turn 0 :  produce written slot w0 ;  distill = MSE(w0, teacher[s].detach())
read code = β · teacher[s] + (1-β) · w0        # what the read consumes downstream
β anneals 1 → 0  over [mem_teacher_anneal_start, mem_teacher_anneal_end]
```

Early on the read consumes a **clean teacher code** correlated with `s` (the read_isolation
regime → strong "use me" gradient) while distillation pulls the written slot toward it; then
the teacher is annealed away and the read applies the pure written code. On the benchmark
this takes cross-turn rule transport from **0.03 (chance) → ~0.97**, holding after the teacher
is removed — far above the in-context (ICL) ceiling. Evaluation always reads the pure written
code, so `rule_acc` measures the true objective throughout. Off by default; enable with
`mem_teacher_forcing: true`.

---

## Other components

### mHC — Manifold-Constrained Hyper-Connections (`mhc.py`)

Replaces standard residuals. The residual stream is widened by `n_hc`, and the residual
mapping matrix **B** is constrained to the Birkhoff polytope (doubly-stochastic) via
Sinkhorn-Knopp, bounding `||B||₂ ≤ 1` for stable deep stacks.

```
X_{l+1} = B_l X_l + C_l F_l(A_l X_l)     A_l, B_l, C_l dynamically generated from X_l
```

### CSA — Compressed Sparse Attention (`attention.py`)

Compression factor `m` with **overlapping** windows; a lightweight indexer selects the top-k
compressed blocks per query, plus a sliding window (`n_win`) for local dependencies.

### HCA — Heavily Compressed Attention (`attention.py`)

Compression factor `m' >> m` with **non-overlapping** windows; dense over all preceding
compressed blocks (no top-k). Cheaper than CSA; captures long-range structure.

### DeepSeekMoE (`moe.py`)

Fine-grained MoE with always-active **shared experts** (`n_shared`) and top-k **routed
experts** (affinity `√(softplus(h W_gate))`). An auxiliary sequence-balance loss prevents
expert collapse.

---

## Models

| Class | Description |
|---|---|
| `DeepSeekV4Mini` | Single text stream; optional legacy bolt-on cross-attention memory |
| `DualModalDeepSeekV4Mini` | Text stream + fast-weight thought bank (recommended) |

### Parameter counts

Dominated by the token embedding (`vocab_size × d_model`).

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

# First pass — a fresh bank is seeded with mem_seed_slots random[0,1] vectors,
# then the write head appends one thought.
out = model(ids)
print(out["logits"].shape)    # [2, 64, vocab_size]
print(out["mem_bank"].shape)  # [2, mem_seed_slots + 1, mem_dim]  → [2, 5, 32]
print(out["write_alpha"])     # mean write probability α (telemetry)

# Subsequent turns — carry the bank across calls (continual learning).
out2 = model(ids, init_mem=out["mem_bank"])
```

### Training

```bash
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/tiny.yaml           # TinyStories
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/code_persist.yaml   # code, bank persists
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/multiturn_rule.yaml # continual-rule benchmark (teacher-forced)
```

The script streams from HuggingFace datasets or generates synthetic data for the
`associative_recall` / `latent_context` / `multiturn_gist` / `multiturn_gist_kv` /
`multiturn_rule` tasks. It logs to `runs/<run_name>/metrics.jsonl` and saves checkpoints.

**Probes** run every `mem_probe_every` steps. For `multiturn_rule`, `synthetic_rule_probe`
reports `rule_acc` (accuracy on **unseen** queries via the carried bank — the verdict; chance
= `1/n_symbols`) and its no-bank ablation. For streaming runs, `content_gap` is the memory
metric to trust:

> `persist_gap` (bank carried vs reset each chunk) conflates the *content* written into slots
> with the bank *structure* (slot count + positional structure). Re-running the carried arm
> with writes **zeroed** isolates them: `content_gap = CE_zero − CE_real` is the pure content
> benefit. On the code dataset ~2/3 of `persist_gap` was structural — trust `content_gap`.

---

## Configuration

All hyperparameters live in `DeepSeekV4MiniConfig` (`config.py`).

```python
cfg = DeepSeekV4MiniConfig(
    d_model=256, n_layers=6, n_heads=4, d_head=64,
    csa_m=4, hca_m=32, top_k_csa=8, n_win=16,
    n_experts=8, top_k_experts=2, d_ff=512,
    mem_dim=64, max_mem=32, mem_seed_slots=4, mem_read_rank=16,
)
cfg = DeepSeekV4MiniConfig.from_yaml("deepseek_v4_mini/configs/small.yaml")
cfg = DeepSeekV4MiniConfig.tiny()   # ~6.5M params
```

Model knobs (`config.py`):

| Parameter | Description |
|---|---|
| `n_hc` / `sinkhorn_iters` | mHC residual width / Birkhoff projection iterations |
| `csa_m` / `hca_m` / `top_k_csa` / `n_win` | Attention compression + sparsity |
| `mem_dim` | Thought-vector / fast-weight code size |
| `max_mem` | Bank capacity; oldest slot FIFO-evicted past this |
| `mem_seed_slots` | Random-uniform[0,1] slots seeding a fresh bank |
| `mem_read_rank` | Bottleneck rank `r` of each per-slot fast-weight layer |
| `mem_read_dropout` | Dropout inside the fast-weight MLP layers |
| `mem_write_gate` | `false` ⇒ ungated write (pure thought); `true` ⇒ `α·p·m` |
| `mem_write_cost` / `mem_write_diversity` / `mem_write_target(_weight)` | Write-rate / novelty / target-rate regularisers (gate on) |
| `mem_teacher_forcing` | Enable the teacher-forced bootstrap (`multiturn_rule`) |
| `mem_teacher_anneal_start` / `_end` | β=1 until start; β linear→0 by end (teacher gone) |
| `mem_teacher_distill_weight` | Weight on `MSE(w0, teacher[s])`, scaled by β |

Training/data knobs (YAML `training:` / `data:`):

| Parameter | Description |
|---|---|
| `mem_segment_len` | Attention window per segment; smaller ⇒ more reliance on the bank |
| `mem_bptt_window` | TBPTT span; **≥2 required** to train the write head |
| `mem_probe_every` | How often to run the probes |
| `data.persist` | `true` ⇒ per-file ordered lanes + carry the bank across steps |

---

## File structure

```
deepseek_v4_mini/
  config.py      — DeepSeekV4MiniConfig dataclass + YAML loader
  mhc.py         — ManifoldHyperConnections + RMSNorm
  attention.py   — CompressedSparseAttention, HeavilyCompressedAttention, RoPE
  moe.py         — SwiGLU, DeepSeekMoE
  memory.py      — ThoughtStream: bank seeding + gated write + FIFO (write side only)
  model.py       — DeepSeekV4Mini, DualModalDeepSeekV4Mini, DualModalBlock (fast-weight read)
  train.py       — training loop, probes, synthetic tasks, teacher-forced bootstrap
  eval_memory.py — offline PPL with vs without the bank
  configs/
    tiny.yaml / small.yaml   — TinyStories
    code_persist.yaml        — code, bank persists across sequences
    synth_recall.yaml        — synthetic addressable-recall diagnostic
    gist.yaml                — synthetic latent-context (gist) diagnostic
    multiturn_rule.yaml      — continual-rule benchmark (fast-weight transport + teacher-forcing)
```

---

## References

- DeepSeek-V4: [arxiv 2606.19348](https://arxiv.org/abs/2606.19348)
- Fast weights: Schmidhuber 1992; Ba et al. 2016; Schlag et al. 2021; Test-Time Training (Sun et al. 2024)
- Hyper-Connections: Zhu et al., 2025 · DeepSeekMoE: Dai et al., 2024 · Muon: Jordan et al., 2024
- Thought memory baseline: [`thought_lm_minimal`](../thought_lm_minimal/)
```
