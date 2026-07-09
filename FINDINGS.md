# Research log — beyond the paper

The [paper](paper/paper.pdf) froze the synthetic-rule results at tag
`V0.2.2-preprint`. This file is the living log of what came after, with exact
reproduction commands for every claim. Newest entry first.

---

## 2026-07-09 — dsv6: the bank as long-context memory on real data (97M, code+web mix)

**TL;DR.** A 97M from-scratch model with an 8-slot thought bank, trained 2000
steps (~2h40 on one RTX 3090) on a 50/50 mix of Python code and web text,
predicts the opening of the *next* 512-token chunk of a held-out document
**from the bank alone** — no context window, just 8 gist vectors — at **+0.85
nats** below the no-bank baseline, flat from 1 to 10 chunks written, on both
domains simultaneously. Inference-time probes show the advantage is
**file-specific content** (not domain detection), stored as a
**recency-weighted, near-linear superposition** of per-chunk gists that the
read consumes causally and without filtering.

### 1. The task: deferred continuation

Documents are split into 512-token chunks. Each chunk is fed as one "turn";
after each turn the model writes **one gist vector** into its bank
(`max_mem: 8` slots, FIFO). The probe turn is a sequence of `<blank>` tokens:
the model must predict the opening 16 tokens of the **next chunk it has never
seen**, with *nothing* in its attention window — the only path from the
document to the prediction is the bank. The metric is

> **GAP = CE(reset bank) − CE(carried bank)**, in nats — how much the bank
> shifts the predictive distribution toward the true continuation.

This isolates the bank by construction: same model, same target, the only
difference is whether the written gists are present.

### 2. Model & training recipe (the two traps we hit)

97M-param DeepSeek-mini trunk (d=384, 6 layers, MoE) + bank (`mem_dim: 512`,
8 slots, fast-weight read rank 16). Config:
[`deepseek_v4_mini/configs/code_defer_native_v2b_mix.yaml`](deepseek_v4_mini/configs/code_defer_native_v2b_mix.yaml).
Two findings that anyone scaling this should know:

- **The Muon √cols trap.** Our Muon variant scales updates by `√cols`, so the
  effective per-matrix update RMS is `√(cols/rows)` — *shape-dependent*.
  Growing `mem_dim` 64→512 silently made the read heads train 2.8× faster and
  the write head 2.8× slower at the same `muon_lr`; the memory circuit died
  mid-run (GAP peak then collapse into the ignore-bank fixed point). A
  "validated" Muon LR does **not** transfer when a matrix dimension changes.
  Fix: per-group LR scales (`muon_ref_mem_dim` knob — read × √(ref/mem_dim),
  write × √(mem_dim/ref)).
- **Decay after death is embalming.** WSD-style decay consolidates whatever
  state the model is in. With the trap above, decay landed after the collapse
  and locked it in. With the fix, the best evals of the run come *after* the
  two step-decay drops (×0.316 at step 700, ×0.1 at 1675) — decay now
  consolidates a living memory circuit.

Training runs batched: B conversations of the same chunk-depth are windowed
over full 512-token chunks (no padding, the bank batches natively), ~3.7×
throughput over one-conversation-at-a-time on the same GPU.

### 3. Training result (2000 steps, 50/50 codeparrot + fineweb)

Per-source held-out eval at the end of training: GAP **+0.79** on code,
**+0.87** on web — the write head serves two registers at once without
interference, and the GAP survives the full LR schedule.

### 4. Depth-stratified GAP (n=48 per depth per source)

Written chunks → predict the next one, controlled depth:

| writes | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| codeparrot | +1.02 | +0.88 | +0.82 | +0.91 | +0.86 | +0.81 | +0.91 | +0.76 |
| fineweb | +0.84 | +0.97 | +0.82 | +0.87 | +0.82 | +0.84 | +0.72 | +0.79 |

**Flat to 10 writes on both domains.** No cliff at the FIFO horizon. For
scale reference, the same protocol on the 47M predecessor gave +0.43→+0.64:
doubling the bank width (`mem_dim` 64→512, with the optimizer fix) nearly
doubled the GAP.

### 5. What the bank actually stores (inference probes, zero training)

All probes predict the *same* target (the opening of a file's 4th chunk)
under controlled write sequences, paired per file, n=48 per domain,
CE in nats (GAP reference ≈ 0.83). Script:
[`deepseek_v4_mini/analysis/code_defer_bank_probes.py`](deepseek_v4_mini/analysis/code_defer_bank_probes.py).

**(a) Bank-swap — the GAP is file-specific content, not register.**
Predict file A's continuation from: A's own bank / another same-domain file's
bank / a cross-domain bank / no bank.

| | own | swap (same domain) | xdom | reset |
|---|---|---|---|---|
| codeparrot | **6.45** | 7.42 | 8.74 | 7.29 |
| fineweb | **7.49** | 8.01 | 8.96 | 8.30 |

On code, a wrong same-domain bank is worth *nothing* (swap ≈ reset): the
file-specific share is +0.97 nats (t≈9.5) — **the entire GAP**. On web,
~2/3 specific (+0.52, t≈5.6), ~1/3 register (+0.29, t≈3.2). Note greedy
decoding from the bank still only surfaces register (degenerate argmax):
content retrieval at 97M is real but below the greedy threshold — the
distribution is file-shaped, the argmax is not. And the **cross-domain bank
actively misleads** (−1.45 nats *below* reset, t≈16): the read trusts the
bank causally and does not filter it.

**(b) Duplicates — the write is near-idempotent.**
Writing the last chunk twice, (c1,c2,c2) vs (c1,c2): +0.01/+0.02, null.
Duplicating an *older* chunk, (c1,c1,c2): small significant cost
(+0.05/+0.06, t≈4) — over-weighting stale context dilutes the recent gist;
over-weighting the recent one is free.

**(c) Distractor — the thread survives interruptions.**
One foreign chunk written into an otherwise on-topic conversation, vs a
depth-matched control, as % of the GAP destroyed:

| distractor position | codeparrot | fineweb |
|---|---|---|
| middle, same domain | −1% (ns) | −3% (ns) |
| middle, cross-domain | −15% | −8% |
| last write, same domain | −28% | −20% |
| last write, cross-domain | −34% | −17% |

A mid-conversation distractor is almost fully neutralized by the next
on-topic write. As the most recent write it costs ~25-30% of the GAP — and a
*plausible* distractor (same domain) hurts as much as an absurd one: no
coherence checking in the read. Worst case still keeps 66-83% of the
advantage (−0.55/−0.68 nats below reset, t≈8-11).

**(d) Order — recency weighting, not sequence encoding.**
Reversed writes (c2,c1,c0) cost +0.49/+0.26 (t≈8/4) — but this is confounded
with *which chunk is written last*. The clean condition permutes only the
*old* writes, (c1,c0,c2) vs (c0,c1,c2), keeping the last write fixed:
**+0.03/+0.01, null (t≈1)**. The bank is a recency-weighted *bag* — no order
encoding beyond recency.

**(e) Eviction — no cliff at the FIFO boundary.**
Write all 12 chunks of a long file (`max_mem: 8` ⇒ the first gists are
physically evicted), then decode the opening of chunk *j* from the bank.
Recall GAP by lag (1 = most recent write, 9-10 = gist evicted):

| lag | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 (evicted) | 10 (evicted) |
|---|---|---|---|---|---|---|---|---|---|---|
| codeparrot | +0.67 | +0.49 | +0.59 | +0.58 | +0.45 | +0.50 | +0.41 | +0.49 | **+0.49** | **+0.41** |
| fineweb | +0.59 | +0.68 | +0.65 | +0.60 | +0.61 | +0.62 | +0.61 | +0.63 | **+0.39** | **+0.62** |

Content from evicted chunks is recalled as well as content whose gist is
still resident. Consistent with (a): each gist carries *file-level*
information, so the surviving 8 gists cover the evicted region — redundancy
of the superposition, not slot lookup. (Recalling old openings is slightly
off-task for this model — it is only trained to predict the chunk after the
last write — so read the flat *profile*, not the absolute level.)

**Unified picture.** The bank behaves as a **near-linear superposition of
file-specific gists, weighted by recency, consumed by a trusting read**:
duplicates superpose harmlessly (b), foreign gists dilute out under
subsequent writes (c), the most recent write dominates (b,c,d), the content
is genuinely file-level (a), and the read follows the bank even when it lies
(a). Every probe is a facet of the same mechanism.

**(f) The `<think>` write trigger.** P(`<think>`) at the end-of-chunk
position: 0.81 (code) / 0.91 (web), top-1 in 57/58 cases, and **zero**
argmax false alarms over ~26k mid-chunk positions. Caveat: chunks are always
exactly 512 tokens here, so this is positionally learnable; the semantic
"when to write" test only becomes meaningful with variable-length turns.

### 6. Caveats

- 16-token targets, defer CE only — no long-form generation claim.
- n=48 per cell; ±0.10-0.16 s.e. on single-condition CEs (paired deltas are
  much tighter).
- Greedy decode from the bank is register-quality, not content-quality, at
  this scale (see 5a for why the GAP is still content).
- One seed, one run per config (self-funded single-GPU budget).

### 7. Reproduce

Hardware: one 24GB GPU (RTX 3090/4090). Total: ~3h GPU for the training run,
~1h for all evals/probes. Datasets stream from HuggingFace and are cached to
`data_cache/` on first run (~10 min tokenization).

```bash
pip install -r requirements.txt

# 1. Train (2000 steps, ~2h40 on a 3090; checkpoints + logs under runs/)
python -m deepseek_v4_mini.code_defer_native \
    deepseek_v4_mini/configs/code_defer_native_v2b_mix.yaml

# 2. Depth-stratified GAP curve (table in §4)
python -m deepseek_v4_mini.code_defer_depthcurve \
    deepseek_v4_mini/configs/code_defer_native_v2b_mix.yaml \
    checkpoints/code_defer_native_v2b_mix/final.pt 48

# 3. Bank probes (tables in §5; ~40 min for all five)
PYTHONPATH=. python deepseek_v4_mini/analysis/code_defer_bank_probes.py \
    deepseek_v4_mini/configs/code_defer_native_v2b_mix.yaml \
    checkpoints/code_defer_native_v2b_mix/final.pt

# 4. Qualitative: greedy-decode continuations from the bank alone, one long
#    file per domain
python -m deepseek_v4_mini.code_defer_sample_deep \
    deepseek_v4_mini/configs/code_defer_native_v2b_mix.yaml \
    checkpoints/code_defer_native_v2b_mix/final.pt
```

Interrupted runs resume bit-exactly with `--resume` (full optimizer + RNG
state in the checkpoint); `scripts/pod_run.sh <config>` wraps the
train-resume loop for rented GPUs.

### What's next

350M on rented pods (same recipe, `deepseek_v4_mini/configs/code_defer_native_350m.yaml`),
then a diverse-mix pretrain and an RL phase (GRPO on verified deferred
continuation — the `<think>` token is reserved for the model *deciding* when
to write). The open question scaling answers: does content retrieval cross
the greedy-decoding threshold — do file identifiers start appearing in the
decoded continuations?
