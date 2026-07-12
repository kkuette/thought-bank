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

## 2026-07-12 — Life without resets: no-reset learns a boundary heuristic, interleaving learns the filter, and selection is the missing third piece

**TL;DR.** Three regimes for a bank that lives across files, judged by the
same probe battery (n=48 held-out files, paired). (1) **No-reset chains
(v2d)**: free on GAP but the read learns a *boundary inference* — "off-topic
last write ⇒ a new file started" — and a mid-file distractor drops the thread
*below* empty-bank level (+0.87 vs reset; v2c survives at −0.96). (2)
**Interleaved training (v2e, "school" regime)**: F ~ U[2,4] files mixed in
one bank lifetime, same chunk budget. The content filter is acquired — a
mid-bank distractor costs **nothing** (+0.03/−0.00/+0.01 across 3 seeds, vs
+0.26 for v2c) — with the best specificity measured (+0.91) and GAP at or
above v2c on all seeds. (3) But when the deferred query is **blank**, v2e
continues the *last-written* thread, whatever it is (last-distractor cost
+2.40/+2.42/+2.48 — three seeds, remarkable constancy: a trained convention,
not noise). A new cued probe shows why nothing better could have been
learned: **thread selection is never exercised** — with any real context the
bank is barely read (~0.03 nat vs ~1.5–2 in blank mode), and with no context
recency is the only available policy. The fix in flight (v2f/G2) trains
*addressed* deferred reads: every write carries a stable synthetic file label,
and the model must continue a **non-last** thread named by a label or by a
16-token opening. Zero-shot baseline to beat (v2c): bank value under
addressed defer −0.8/−0.9 nat but junk/live-thread interference +0.4.

### v2d (no-reset chains): the prediction failed in an instructive way

Resumed v2c for 800 no-reset steps (chains of 4 files, one bank lifetime,
defer masked at junctions). Dated prediction was "distractor cost collapses".
Reality (codeparrot, n=48):

| | v2c @400 | v2d @800 |
|---|---|---|
| GAP (code/web) | +1.44 / +0.82 | +1.39 / +0.81 |
| d SPECIFIC | +0.77 | +0.79 |
| d REGISTER (swap vs reset) | +0.82 | **+0.50** |
| distractor LAST xdom | +0.60 | **+2.17** |
| worst case vs reset | **−0.96** (survives) | **+0.87** (below empty) |

The regime is free (GAP, specificity, no eviction cliff) but the read did not
learn to *filter stale content* — it learned that an off-topic last write
means the old thread is dead, because in sequential chains that correlation
is perfect. It also discounts wrong banks harder (REGISTER halved) and is
better from an empty bank: consistent with "restart detection", not
selection. The sequential *form* of no-reset is the confound.

### v2e (interleaved / "school" regime): the filter, replicated ×3

One bank lifetime = F ~ U[2,4] files whose chunks are randomly merged
(within-file order kept; deferred target = same-file successor, carried
per-seg). Total chunk budget matched to v2c/v2d (m ~ U[2,K] split across
streams) ⇒ same VRAM/compute, only the *structure* changes. 800 steps from
v2c final, seeds 0/2/3:

| seed | GAP @800 code/web | MID xdom | LAST xdom | worst vs reset |
|---|---|---|---|---|
| 0 | +1.61 / +0.76 | +0.03 (t 0.6) | +2.42 | +0.71 |
| 2 | +1.97 / +1.13 | −0.00 | +2.48 | +0.48 |
| 3 | +1.45 / +0.96 | +0.01 | +2.40 | +1.15 |

Everything replicates: mid-bank distractors are **free** (the filter over old
entries exists), specificity is the highest measured (+0.91), REGISTER is
restored to v2c level (+0.83), recall-by-lag is at the best measured level
with still no eviction cliff, and the mature sequential seeds (v2b @2000,
n=3) put the seed-noise yardstick at ±0.2 on these metrics — the v2e/v2d
last-distractor numbers are ~10σ out. The +2.4 constancy across seeds is the
signature of a *learned convention*: in the interleaved regime the blank
query is genuinely ambiguous (several live threads), and training always
paired "deferred continuation" with "thread of the last write". v2d
destroyed the thread (below reset); v2e keeps everything and *selects by
recency*. Different failure classes: only the second one is repairable by
giving the query an address.

### The cued probe: selection is never exercised

New probe (`--probes cued`): query = 16 tokens in the window instead of the
blank. Three cue types — `cont` (a2's tail, target continues it), `id` (a1's
interior, identifies without continuing), and defer-mode `open`/`lbl` (see
below). On v2e/v2c/v2d alike: with *any* in-context cue the junk-last cost
vanishes (+0.004) **but so does the bank's contribution** (clean vs reset
under cue: −0.025/−0.037/−0.014 nat, vs ~1.5–2 nat in blank mode). The
pathology is confined to the blank mode — but so is the bank's value. The
model was only ever trained to read the bank as a *substitute for missing
context*; given 16 tokens of context it stops consulting it. So
content-based thread selection was never trained, never needed, and cannot
be measured zero-shot. (In defer mode with a cue *prefix* the bank IS read —
v2c zero-shot: bank value −0.82/−0.91 — but selection leaks: junk-last and
live-thread-last cost +0.37–0.46. That leak is the target.)

### G2 (v2f, running): addressed defers — the mechanism in SFT, the policy in GRPO later

Data-only change on top of v2e: (a) every written chunk is prefixed with its
file's **stable synthetic label** (`<<FILE:483920>>`, arithmetic hash of the
opening tokens — not the real path, so addressing is measured without
semantic leakage; no dataset/domain tier: 2 values would be a domain prior
and a shortcut, the hierarchy belongs to the pages/strata designs); (b) with
p=0.5 per conv position, capped at **2 per conv**, an *addressed* defer
toward a random **non-last** live thread: cue = the thread's label (50%) or
the raw opening of its last written chunk (50%), target = that chunk's
successor — ~500 tokens away, so the gist is the only bridge. Loss on blank
positions only. Division of labor decided up front: SFT creates the
*mechanism* (dense CE — pure RL on a zero capability has no reward to
amplify: the dsv4 ignore-bank fixed point), GRPO later learns the *policy*
(which thread at the white turn, model-chosen labels), reward staying
task-grounded per the standing note.

Ops post-mortem: the first two farm attempts OOM'd at steps 20–50 on the
8 GB rigs — each addressed forward holds a full fast-weight-read graph
(hypernet cost ~independent of sequence length) until the conv's backward;
p=0.5/seg stacked up to ~8 of them on top of a regime already at 7.2/7.8 GB.
The cap (addr_max=2, sampled uniformly over eligible positions) brings the
measured 10-step peak to 6.35 GiB vs 6.19 for v2e. The addr loss was already
moving before the crash (8.04 → 7.80 by step 50, from the reset level: the
capability starts at zero, as designed).

### Side results from the same farm wave

- **Seed σ (v2b from scratch, 2000 steps, n=4 so far)**: GAP code
  +0.51/+0.74/+1.00/+1.36, web +0.81/–/+0.94/+1.00 — inter-seed σ ≈ 0.25+
  nat on code GAP. Any ablation verdict below ~0.5 nat is seed noise; probe
  metrics are tighter (last_xdom ±0.15).
- **Seed-init ablation (idea E) resolved by the eval, not the training**:
  the zeros-init arm shows "GAP +5.8" — an artifact. Its carried CE is
  identical to uniform01 and ±1 (7.04 vs 7.00 vs 7.00): the *reset baseline*
  collapses (CE 12.8) because an all-zeros bank is a degenerate read input,
  while uniform seeds act as a benign "empty" marker. Init does not change
  memory quality; it changes the ablation baseline. Compare carried, never
  GAP, across init arms. The uniform[0,1] "lottery" is acquitted on
  performance.
- **mem_dim grid (preliminary, 1 seed, params confounded)**: 512→256→128
  from scratch: GAP code ~+1.36 → +0.76 → +0.87, web ~+1.00 → +0.81 →
  +0.93. No capacity cliff down to 128; distractor robustness *improves* at
  small dims (last_xdom +0.41 → +0.30 → +0.13). 256 vs 128 indistinguishable
  at seed noise.

### Reproduce

```bash
# v2e interleaved (seed 0; s2/s3 = configs v2e_s2/v2e_s3)
python -m deepseek_v4_mini.code_defer_native deepseek_v4_mini/configs/farm/v2e_interleave.yaml
# v2f addressed (G2; seed 2 = v2f_addr_s2)
python -m deepseek_v4_mini.code_defer_native deepseek_v4_mini/configs/farm/v2f_addr.yaml
# probe battery incl. the cued probe (any checkpoint)
PYTHONPATH=. python deepseek_v4_mini/analysis/code_defer_bank_probes.py \
  deepseek_v4_mini/configs/farm/v2e_interleave.yaml \
  /mnt/tb/checkpoints/farm/v2e_interleave/final.pt \
  --probes swap,dup,distractor,order,eviction,cued --n-files 48
```

---

## 2026-07-10 — GRPO on the write decision: a 384-float policy recovers 75% of always-write for half the writes

**TL;DR.** We trained the *decision to write* with RL (GRPO), leaving the
memory mechanism itself untouched. The final recipe freezes the **entire
model** and trains only the `<think>` row of the tied LM head — **384
floats** — as a linear policy readout on frozen features. That policy
recovers **~75% of the never-write → always-write gain for ~53% of the write
budget**: about half of what pretraining wrote turns out to be skippable
boilerplate, and a linear probe on the boundary state can tell which half.
The stronger claim (`selective > always` at equal budget) was **not**
reached. Getting there took four phases, and phase 2 bought a lesson that
generalizes beyond this project: **the group-relative baseline in GRPO is
blind to any degradation that hits all rollouts equally** — full-model GRPO
quietly destroys the shared backbone while the reward looks perfectly
healthy.

### Setup

Start: the 97M `v2c_varlen` checkpoint (variable-length chunks, so the write
position is not positionally predictable — the prerequisite entry below).
At each chunk boundary the policy is `π = P(<think>)` at the boundary
position; action = write / skip; reward = **−CE(deferred continuation) −
λ·writes** (λ = 0.03). GRPO groups of 8 rollouts over the same conversation;
DAPO-style zero-variance-group filtering from day 1. Two traps solved before
any learning happened:

- **Shared seed bank per group.** Random per-rollout seed banks make the
  bank lottery dominate the within-group advantage; share the seed bank
  across the group and the advantage isolates the *decisions*.
- **Temper the Bernoulli, not the vocab softmax.** The pretrained gate fires
  at p ≈ 2·10⁻³ off-boundary; a vocabulary-softmax temperature *flattens
  toward 1/V*, which is below p — it lowers exploration. The right knob is a
  temperature in probability space on the write Bernoulli (T = 3 ⇒ p ~ 0.1,
  state-dependent).

### The four phases

1. **p1, lr 5·10⁻⁶** — learns, slowly: p(write) 0.18 → 0.5, policy pulls
   +0.4 nats over `never` at 13–18% budget. Side lesson: the eval is blind
   while p(write) < ~0.26 (greedy-ish sampling deterministically writes
   nothing, policy ≡ never); train-side p(w) and KL are the instruments.
2. **p2, lr 2·10⁻⁵ (warm restart)** — the backbone breaks: the *forced*
   baselines drift (`never` −8.6 → −10.5, `always` −7.0 → −7.5), train CE
   U-turns once KL > ~0.5. Diagnosis: the group baseline cancels any reward
   shift **common to all 8 rollouts**, so a capability loss shared by the
   whole group produces exactly zero advantage — nothing in the objective
   defends the backbone. The healthy window observed: **KL 0.04–0.4**.
3. **p3, lr 10⁻⁵ from the healthy p1 checkpoint** — best single point of the
   saga (76% of the corridor at 39% budget: policy −7.19 / always −6.62 /
   never −8.99) but the same erosion resumes. Conclusion: full-model GRPO
   erodes the backbone at any usable LR; the fix is structural, not a
   schedule.
4. **p4, freeze everything** — `train_scope: think_row`: every parameter
   frozen except the `<think>` row of the tied embedding/LM-head matrix
   (a gradient-hook mask; 384 floats), lr 3·10⁻⁴. The policy becomes a
   linear readout on frozen features; the backbone is invulnerable **by
   construction**, so the KL window stops being a cliff edge.

### Result (p4, 300 steps, ~8h on the 3090)

Forced-rollout eval (16 held conversations): policy vs `always`-write vs
`never`-write, write budget = writes taken / boundaries offered.

| eval @ | policy | always | never | writes |
|---|---|---|---|---|
| 175 | −8.42 | −7.50 | −9.97 | 28/58 |
| 200 | −8.02 | −7.85 | −10.32 | 31/58 |
| 225 | −8.68 | −7.67 | −10.34 | 27/53 |
| 250 | −7.86 | −7.41 | −10.39 | 33/59 |
| 275 | −8.47 | −7.58 | −10.90 | 28/54 |
| 300 | −8.13 | −7.38 | −10.34 | 27/50 |

Average of the last four evals: **policy −8.28 / always −7.51 / never
−10.49** — corridor 2.98 nats, policy at **74% of it for 53% of the write
budget** (75% @ 52% averaged over all six). p(write) stays state-dependent
(0.50–0.80 across evals, never saturating to 1), positive position-reward
correlation throughout, KL 0.14–0.34, essentially zero dropped groups.

Honest reading of the near-misses: the @200 eval alone says "93% of the
corridor at 53% budget" — that is batch noise, not a headline (evals are 16
conversations; `never` varies by >2 nats between batches). And `never` on an
RL checkpoint is artificially deep because of the `<think>` tax documented
in the stress-test entry — the clean comparison is **policy vs always**
(both carry the bank, same tax), which is why we report corridor position
*and* the raw pair.

### Why this matters

- The write policy is **cheap**: 384 trainable floats recover three quarters
  of the memory benefit at half the write cost. Selectivity does not need
  the model to change — the information "is this chunk worth remembering?"
  is already linearly present in the boundary state.
- **Safety-relevant negative result**: any RL fine-tuning of a
  memory-equipped model that lets gradients reach the shared trunk is
  structurally unprotected against common-mode capability loss (the group
  baseline cannot see it). Freezing the trunk and training a minimal policy
  readout is both the safe and the effective recipe at this scale.
- Follows the standing note at the top of this file: the reward is pure task
  loss minus a write *cost* — retention itself is never rewarded.

Repro: [`deepseek_v4_mini/rl_defer_grpo.py`](deepseek_v4_mini/rl_defer_grpo.py)
with [`deepseek_v4_mini/configs/rl_defer_grpo_97m.yaml`](deepseek_v4_mini/configs/rl_defer_grpo_97m.yaml)
(phases = `lr` / `init_from` / `train_scope` settings; needs
`checkpoints/code_defer_native_v2c_varlen/final.pt`). Final checkpoint:
`checkpoints/rl_defer_grpo_97m_p4/final.pt`; metrics in
`runs/rl_defer_grpo_97m_p4/metrics.jsonl`.

---

## 2026-07-10 — cross-register transfer: a natural-language gist helps generate the code it describes

**TL;DR.** Zero-shot on the 97M `v2c_varlen` checkpoint: writing a
function's **docstring** (natural language only, no code) into the bank
lowers the CE of deferred generation of that function's **body** by
**+0.68 ± 0.07 nats** vs an empty bank — and **+0.17 ± 0.07 nats vs writing
an unrelated docstring** (p ≈ 0.01, own-doc beats foreign-doc in 61/96
pairs). Prose and code cross the bank in gist space. First signal: n=96,
one direction (doc → code), single checkpoint.

Protocol (n=96 functions, one per source file, docstring ≥ 150 chars):
write one of {nothing, the function's docstring, an unrelated docstring,
def-line + docstring} into a seed bank, then defer-decode the 16-token
opening of the function body from `<blank>` input. Paired deltas (nats,
± SEM):

- **transfer** (reset − own doc): **+0.683 ± 0.067**
- **specificity** (foreign doc − own doc): **+0.169 ± 0.066** (p ≈ 0.01,
  own doc wins 61/96 pairs)
- upper reference (reset − [def line + doc]): +1.087

Reading: ~three quarters of the raw effect is register (a docstring says
"Python is coming"), but a quarter is **content-specific** — the docstring's
*meaning* reaches the code tokens. Adding the `def` line is worth ~0.4 nats
more: identifiers written to the bank lower the CE of the tokens that reuse
them, i.e. names are retrievable from the gist. Consistent with the 135M
`invar` probe below (~47% of the gist survives total identifier renaming):
the gist carries substantially more than surface strings.

Why we ran it: this is the cheap test of the bank-as-shared-abstraction-
space hypothesis — if abstractions written from one register (prose)
modulate generation in another (code), the bank is a candidate substrate
for cross-modal working memory, not just same-stream continuation. The
reverse direction (code → doc) and trained (non-zero-shot) transfer are
open.

Repro: [`deepseek_v4_mini/analysis/doc2code_probe.py`](deepseek_v4_mini/analysis/doc2code_probe.py)
(needs `checkpoints/code_defer_native_v2c_varlen/final.pt`).

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

## 2026-07-10 — third scale point: 135M, parameter-matched to SmolLM2 — the GAP keeps growing

**TL;DR.** Same recipe, same data, same bank, ~1.4× the trunk: a **135.0M**
model parameter-matched to SmolLM2-135M reaches **~+1.07 nats** GAP on code
— above the 97M at *every* write depth, still flat from 1 to 10 writes. The
scaling curve now has three points: **47M +0.43–0.64 → 97M ~+0.85 → 135M
~+1.07**. On web text the 135M plateaus at the 97M level (~+0.7): that
ceiling is **data**, not parameters (the fixed corpus is ~3 epochs over
11.7M tokens at this budget) — future scale points go single-epoch. An
8-probe battery on the checkpoint replicates all five 97M probe results and
adds three new ones: cohabitation works, zero-shot "reflection" hurts, and
**~47% of the gist survives total identifier renaming**.

### Setup

386M was the original target and spills under WSL2 (~14 GB/conv); resized to
135.0M matched to SmolLM2-135M — the natural external yardstick — keeping
`mem_read_rank × mem_dim` and `mem_dim: 512` **verbatim** (the scale point
must not change the memory mechanism; `mem_dim` is a separate sweep). fp32,
B=2 × grad-accum 4, 9 s/step, ~5h on the 3090, identical `v2b_mix`
data/recipe (data held constant deliberately — the comparison is
params-only). Config:
[`deepseek_v4_mini/configs/code_defer_native_135m_mix.yaml`](deepseek_v4_mini/configs/code_defer_native_135m_mix.yaml).

### Depth-stratified verdict (n=48 per depth per source)

Code: **+1.14 → +0.93 flat over 1→10 writes**, above the 97M at each depth.
Web: +0.73 → +0.69, flat but ~−0.1 vs the 97M — with training loss showing
epoch-3 behavior on the fineweb slice. The GAP grows with parameters when
data allows; when it doesn't, it saturates rather than degrades. Single-epoch
sizing for the next runs: `n_files` 24k ≈ ≥72k chunks for 2000 steps.

### Probe battery (8 probes, same protocol as the 97M entry)

The five standard probes **replicate**: the code GAP is entirely
file-specific (swap ≈ reset), a cross-domain bank misleads (−1.14 below
reset), duplicate writes are near-idempotent, order is a pure
recency-weighted bag, eviction has no cliff. Three new probes:

- **`cohab` — store both, select at recall: confirmed.** Interleave writes
  from files A and B into one bank, then defer both continuations from the
  same superposed state: **B keeps 82% of its mono-file GAP, A keeps 60%**
  (the asymmetry is pure recency). One bank serves two documents at once;
  selection happens at read time, not write time.
- **`reflect-k` — zero-shot reflection is negative.** Insert k "thought"
  turns (blank forwards whose bank write is kept) before the probe: CE
  *rises* monotonically with k (+0.036 / +0.086 / +0.136 on code, all
  significant). Untrained thought-gists are near-empty writes that dilute
  the bank under recency weighting. Reflection is a *trained* capability
  (consistent with the dsv5 think-cell result), not an emergent one — so
  the RL phase stays write/skip; a "think again" action needs its own
  training signal first.
- **`invar` — the gist is ~half abstraction.** Clean gradient of the same
  file-specific advantage: own bank 6.40 < re-segmented +0.27 (73% survives
  re-chunking at different boundaries) < **total identifier renaming +0.54
  (~47% survives)** < swap +1.03. Half the file-specific signal is carried
  by *what the code does*, not what its names are. This defines a new
  scaling axis to track: the abstraction fraction of the gist vs
  params/data.

Repro: probes `cohab`/`reflect`/`invar` in
[`deepseek_v4_mini/analysis/code_defer_bank_probes.py`](deepseek_v4_mini/analysis/code_defer_bank_probes.py)
(seeds 444/606/888, n=48 per domain, target = each file's chunk-3 opening,
16 tokens).

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
