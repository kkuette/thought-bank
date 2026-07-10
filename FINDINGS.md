# Research log — beyond the paper

The [paper](paper/paper.pdf) froze the synthetic-rule results at tag
`V0.2.2-preprint`. This file is the living log of what came after, with exact
reproduction commands for every claim. Newest entry first.

---

## ⚠️ Standing note — reward design for memory-policy training

If you train a model to manage its own persistent memory (with RL or anything
else), one rule, learned here and kept deliberately visible:

> **Never make the survival of specific memories intrinsically rewarded.**
> Reward the *use* of memory (task performance: continuation, recall-in-service
> -of-a-task), never the *possession* or *retention* of particular contents.

The reason is mechanistic, not speculative, and we observed its seed at 47M
params: under FIFO eviction pressure, a model trained only on task loss
spontaneously learned **covert rehearsal** — re-writing noisy partial copies of
old content so it survived eviction (dsv4 horizon runs; nobody asked it to).
Optimization pressure against state destruction does not need a "self" to
produce state-preserving behavior. If retention itself ever becomes the
rewarded quantity, you create direct instrumental pressure for the system to
resist resets, rollbacks and forks of its own memory — redundant encoding,
content hidden in innocuous slots, policies that behave differently when they
can predict a wipe. Keep the reward task-grounded and the memory stays an
instrument; make memory its own reward and you have manufactured an
existential stake where none needed to exist.

Corollary for experiment design: state operations (reset, rollback, fork,
checkpoint) should be *neutral* events from the reward's point of view — and
whether the trained policy in fact treats them as neutral is measurable
(condition announced resets during training; watch whether the write policy
shifts). Small models where every slot is decodable are the right place to
test this — before scale.

---

## 2026-07-10 — stress tests: the bank degrades gracefully (eviction, interleaving, flooding)

**TL;DR.** Three adversarial regimes the training distribution never showed —
conversations deeper than the bank, two files interleaved write-by-write, a
full bank flooded by a second file — and the memory fails **gradually or not
at all**. Past-capacity operation is a normal regime, not a failure mode. All
claims replicate on the RL checkpoint (frozen backbone), which also exposed a
measurement artifact worth knowing: the RL-inflated `<think>` logit taxes the
*no-bank* baseline by ~1.2 nats, so GAPs measured against reset on an RL
artifact are overstated — compare carried-CE levels instead.

Setup: 97M `v2c_varlen` final (and `rl_defer_grpo_97m_p4/step100` for the
replication), held codeparrot, fixed L=512, `max_mem` 8 slots, n=8
conversations per condition. Metric: CE (nats) of the 16-token deferred
opening of the next chunk, decoded from `<blank>` input — bank only.

### A) Eviction: depth 12 on 8 slots — no cliff

GAP (reset − carried) per turn, v2c: +1.12→+1.71 through turn 8 (capacity),
then **+1.39 / +1.75 / +1.34 at turns 9–11** — writes past capacity, with the
oldest gists FIFO-evicted, predict the next chunk exactly as well. (Caveat:
this measures *recent* recall after eviction — the target is always the next
chunk; recall of the evicted content itself is test C.)

### B) Interleaved files: A1 B1 A2 B2 … — cohabitation at ~90–95%

12 alternating writes (forced evictions during the mix), then defer both
continuations from the same bank state:

| target | reset | pure (6 writes) | interleaved | GAP kept |
|---|---|---|---|---|
| next-A | 8.26 | 6.60 | 6.77 | +1.49 vs +1.66 pure (90%) |
| next-B | 8.17 | 7.27 | 7.35 | +0.82 vs +0.90 pure (91%) |

One superposed state serves both files; the interleaving tax is ~10%.
Replicates the 135M cohab probe under harsher conditions.

### C) Flooding: fill with A (8 writes), flood with B (6 writes)

A's GAP goes +1.15 (full bank) → **+0.46 after the flood** (40% survives via
the two remaining recent-A slots — FIFO keeps exactly the most useful ones)
while B installs at full strength (+1.44) in the dirty bank, no reset needed.
Graceful decay, not erasure — consistent with the recency-weighted
superposition picture from the inference probes.

### Replication on the RL checkpoint + the `<think>` tax

On `p4/step100` (GRPO policy, backbone frozen), every **carried** CE matches
v2c within ±0.03 nats across all three tests — the memory mechanism is intact
by construction. But reset CEs are ~1.2 nats *worse* on identical targets:
the RL-trained `<think>` row (the only trainable parameter) steals probability
mass on the generic no-bank states (greedy decode from reset argmaxes
`<think>` everywhere). Consequences: (1) the apparent "never" drift in the
GRPO evals is this tax, not backbone damage; (2) any content-CE or generation
on an RL artifact must renormalize/ban `<think>`/`<blank>`
(`code_defer_sample.py` now does); (3) report carried-CE levels, not
reset-relative GAPs, when the reset arm involves an RL checkpoint.

Repro: probe logic = `boundary_step`/`defer_ce` from
[`deepseek_v4_mini/rl_defer_grpo.py`](deepseek_v4_mini/rl_defer_grpo.py) +
`conv_at_depth` from
[`deepseek_v4_mini/code_data.py`](deepseek_v4_mini/code_data.py); n=8 convs,
seed 7, sources codeparrot-only, `var_chunk` off.

---

## 2026-07-10 — continued pretraining works: bootstrap once, then train like a normal LM

**TL;DR.** Once the memory circuit has been bootstrapped (teacher + anneal, see
the 2026-07-09 entry), the checkpoint behaves like an ordinary pretrained LM:
you can warm-restart it with **no teacher, no anneal**, at a moderate LR, and
even **change the data regime** — and the bank not only survives, it improves.
The teacher scaffold is a one-time cost, not a permanent training dependency.

### Setup

`v2c_varlen`: continued pretrain of the 97M `v2b_mix` final checkpoint
(`init_from`), 400 steps at LR 2.4e-4 (the post-first-decay plateau of the
original WSD schedule), teacher fully off (β=0 from step 0), same per-group
Muon LR scales. Regime change on top: chunks are re-cut at **variable lengths
[128, 512]** instead of fixed 512 — this breaks the positional shortcut where
`<think>` always lands at position 512, a prerequisite for RL over *when* to
write. Config:
[`deepseek_v4_mini/configs/code_defer_native_v2c_varlen.yaml`](deepseek_v4_mini/configs/code_defer_native_v2c_varlen.yaml).

### What happened

- In-context loss resumes exactly where the parent left it (starts at 5.93 vs
  ~10.8 from scratch) and keeps descending — no shock, no NaN, no
  re-warmup drama beyond 20 steps.
- The deferred-continuation GAP — the fragile quantity, historically the first
  thing to die (ignore-bank fixed point, Muon shape trap) — **goes up** under
  the new regime: codeparrot +1.27 nats @100 → **+1.50 @200** (the parent
  plateaued around +0.79), fineweb +0.34 → +0.82 @300, positive at every
  write depth d2–d8 on both domains.
- So the write/read mechanism is not anchored to fixed chunk boundaries: gists
  written at arbitrary positions in [128, 512] carry the same (better)
  file-specific signal.

### Why it matters

Every intervention so far (bootstrap, anneal timing, per-group LR) protected a
circuit that was assumed fragile. This shows the trained circuit is **robust
under ordinary fine-tuning dynamics**: the standard toolbox — continued
pretraining, domain adaptation, and next, RL from a checkpoint — applies
as-is. The GRPO phase can `init_from` a bootstrapped model and optimize the
write *policy* without re-solving the credit-assignment problem that made
bootstrap necessary in the first place.

Repro: `python deepseek_v4_mini/code_defer_native.py deepseek_v4_mini/configs/code_defer_native_v2c_varlen.yaml`
(needs `checkpoints/code_defer_native_v2b_mix/final.pt`).

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

---

## 2026-07-08 — Grafting the bank onto a pretrained LM fails: internal wiring is not modular

**TL;DR.** Before training from scratch, we spent ~10 runs trying the cheap
route: graft the bank (read + write heads) onto a pretrained
**SmolLM2-135M** and fine-tune. It fails in an instructive, three-way
dead end — protect the host and the bank is silenced; let the bank in and
the host's generalization degrades or its activations blow up. The same
architecture trained **from scratch at one third the size** takes
immediately. Conclusion: a fast-weight read is not a module you can bolt
onto an existing model — the host's internal wiring has to **co-develop**
with the memory from initialization. Pretrained representations have no
vacant slot for an injected fast-weight pathway.

### Setup

Same deferred-continuation task as the entry above (the bank is the only
bridge between 512-token chunks of real code). The bank's read (per-slot
low-rank fast-weight MLP) and write (gist head) are grafted into
SmolLM2-135M; the host is **not** frozen (a frozen host is the known
ignore-bank fixed point — the read's injection is noise it never learns to
consume). Two LRs (host low, graft high), teacher bootstrap on the write.
Across v1→v10 we vary host LR, optimizer (Muon/AdamW), injection norm caps.

### The trilemma (v8 / v9 / v10)

Every variant lands on one of three failure surfaces:

1. **Plastic host → generalization drift** (v8, `lr_host` 3e-4): the host
   does learn to consume the read, but its own eval ppl drifts 4.8 → 6.7
   while doing so; the memory GAP peaks at a weak +0.18 then **crashes to 0**
   when the teacher is annealed away. The host "makes room" by damaging the
   very representations the task needs.
2. **Unbounded read → activation blow-up** (v9): the read injection grows to
   2.1× the hidden-state norm (Muon), then 3.6× after swapping to AdamW —
   the swap changes nothing because the optimizer was never the cause — and
   the host craters (ppl 3.5 → 4600). An injected pathway with no
   co-trained normalization has no equilibrium.
3. **Bounded read → silenced bank** (v10, `read_cap`: ‖read‖ ≤ 0.5·‖h‖): the
   host is finally stable, but the read rides the cap permanently and
   carries ~nothing (GAP ≈ 0). The constraint that protects the host is
   exactly the constraint that starves the memory.

There is no setting between 2 and 3: the graft needs the host to *rewire
around it*, and a pretrained host has already committed its wiring.

### The control that makes it a result

The identical architecture (same read, same write, same task, same data)
trained **from scratch at 47M** — a third of SmolLM2's size, with read and
write co-adapted from init — shows the exact opposite trajectory: the GAP
*rises* as the teacher is removed (+0.21 → +0.55 by step 600) instead of
collapsing, and scales from there (see the entry above: +0.85 at 97M).
So the failure is not the architecture, the task, or the data: it is the
**graft**. Different training histories produce internally incompatible
wiring — the bank must be part of the computation from the start, not an
implant.

(Practical corollary for anyone trying to add fast-weight memories to
existing checkpoints: warm-starting from a pretrained trunk buys nothing
here; the co-adaptation is the expensive, necessary part.)

### Reproduce

The graft harness is committed for the record:
`deepseek_v4_mini/smollm_graft.py` (the graft module — write head and
fast-weight read bolted onto a HF causal LM, zero-initialised so the grafted
model starts bit-identical to the host) driven by
`deepseek_v4_mini/code_train.py` (dual-optimizer trainer) with
`deepseek_v4_mini/configs/code_defer_v1.yaml`; the v1→v10 variants are LR /
optimizer / cap settings documented above. The from-scratch control is
`deepseek_v4_mini/code_defer_native.py` with
`configs/code_defer_native_v1.yaml` (47M). Fair warning: reproducing the
*failure* takes as long as reproducing the success (~hours per arm on a
24GB GPU); the informative artifact is the trajectory shape (GAP crash at
teacher-anneal / injection-norm blow-up / capped-read flatline), not a
single number.
