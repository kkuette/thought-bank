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
(batched across conversations — cheap) reports `rule_acc` (accuracy on **unseen** queries via
the carried bank — the verdict; chance = `1/n_symbols`) and its no-bank ablation, plus,
depending on the task knobs:

| Probe metric | When | Meaning |
|---|---|---|
| `rule_HELD` | `heldout_shifts` / `train_shift_max` set | accuracy on shifts never trained — the generalization arm |
| `[horizon] acc/turn` + `rule_LATE` | `turns_per_conv ≥ 12` | per-turn accuracy profile (exposes a FIFO-eviction cliff) and last-quarter mean |
| `pre/post` + `STICK` | `switch_at` set | accuracy before/after the mid-conversation rule switch, and the fraction of post-switch answers still using the OLD rule (squatting diagnostic) |

For streaming runs, `content_gap` is the memory metric to trust:

> `persist_gap` (bank carried vs reset each chunk) conflates the *content* written into slots
> with the bank *structure* (slot count + positional structure). Re-running the carried arm
> with writes **zeroed** isolates them: `content_gap = CE_zero − CE_real` is the pure content
> benefit. On the code dataset ~2/3 of `persist_gap` was structural — trust `content_gap`.

---

## Experiments & results (`multiturn_rule` campaign)

Recipe for the campaign runs below: all-AdamW `lr 3e-4` **constant** (no warmup),
teacher-forced bootstrap annealed over steps [300,500], write gate off. Chance = 0.031,
in-window ICL ceiling ≈ 0.49. **Current default for new runs**: Muon + cosine +
early anneal [150,350] (`multiturn_rule_muon_cos_early.yaml`) — 0.974@800 on the
K=1 reference, ~2× faster than the AdamW baseline (see the optimizer rows below).

| Experiment | Config | Result |
|---|---|---|
| K=1 cross-turn transport | `multiturn_rule.yaml` | **0.948** @1500 — the bootstrap breaks the ignore-bank fixed point |
| no-distill ablation | (flag) | chance forever — distillation is the active ingredient |
| K=2 keyed routing | `multiturn_rule_k2.yaml` | **0.99** — two rules held + routed by key; the old K=2 wall was the fixed point, not addressing |
| held-out shifts (contiguous) | `multiturn_rule_k2_heldout.yaml` | 0.97 train / **0.011 held** — recognition within a closed repertoire, not open rule induction |
| held-out shifts (interleaved) | `multiturn_rule_k2_interleaved.yaml` | rule_HELD = **0.000** (snapping to trained neighbours); no spontaneous interpolation, irregular coverage hurts everywhere |
| persistence horizon (24 turns) | `multiturn_rule_horizon.yaml` | **no FIFO cliff** — rehearsal emerges from TBPTT pressure alone; cost: ~0.48 plateau (vs 0.95 @9 turns) |
| rule switch (12+12) | `multiturn_rule_switch.yaml` | **STICK = 0.000** @acc 0.795 — old rule dropped *actively* (recency override: s1 still in the bank, never used) |
| joint retain-then-drop (24+16) | `multiturn_rule_joint.yaml` | **0.747/0.746 pre/post, STICK 0.02** @1200 — maintenance through eviction THEN clean replacement; beats horizon's maintenance (0.74 vs 0.48); rehearsal uses a covert code, off the presentation manifold (`analysis/joint_inspect.py`) |
| Muon retest, constant LR | `multiturn_rule_muon.yaml` | no collapse (the historical peak-then-collapse was the pad_mask/warmup/distill bugs) and ~2× faster early (0.74@600), but the top end is **unstable**: band 0.83–0.95, touches 0.945 without holding |
| Muon + cosine decay | `multiturn_rule_muon_cos.yaml` | **0.99 @1000** (probes 850–1000: 0.951/0.984/0.971/0.990) — stable, above the AdamW baseline, ~1.75× faster |
| + early anneal [150,350] | `multiturn_rule_muon_cos_early.yaml` | **0.974 @800–900** stable — another ~150 steps saved; the gain shows up *post*-anneal (consolidation runs at high LR), the lift-off itself tracks model maturity, not the teacher schedule; **new default recipe** (~2× vs AdamW) |
| write noise σ=0.1 | `multiturn_rule_k2_inter_noise.yaml` | train 0.995 / held **0.000** — noise *reinforces* snapping (a midpoint falls in a neighbour's cloud, where supervision says "behave like the neighbour") |
| code mixup (midpoint supervision) | `..._mixup.yaml` / `_mixup2(w).yaml` | v1 (8 d=1 midpoints): memorized as extra codes; v2/v2w (62 symmetric pairs): stalls installation when injected pre-maturity — held 0.000 throughout |
| ⚠ short format disqualified | `..._short.yaml` (control) | fast-iter format (`turns_per_conv 4` = 1 query/rule/conv) caps train at **~0.48 regardless of intervention** — three cells (SN, [0,2], bare control) produced the same curve; only long-format verdicts count |
| read placement | `..._sn_late.yaml` / `_read02L.yaml` | block-3-only read **kills the bootstrap** (gradient-starved, abl_gap 0.002); blocks **[0,2]** install fully (**0.987**, ~300 steps slower than 4 blocks) — but held **0.000**: halved composition doesn't interpolate |
| spectral norm on the read hypernet | `..._snonly.yaml` / `_sn02L.yaml` | SN blocks the final consolidation step (~0.53 vs 0.99, 8 flat probes) even at 2 reads — the 0.55→0.99 installation step is a *sharpening* (32 neighbouring rules need razor code boundaries); **imposed smoothness is incompatible with what installation requires** |
| mixup post-installation | `..._read02L_mix.yaml` | mixup injected at step 1200 on a fully installed [0,2] base (0.995): train holds (0.997), mixup CE absorbed (0.99→0.3–0.6), held **0.000** — absorbed by *memorizing* the 62 midpoints, not by interpolating, even under ideal conditions |
| single read at block 0 | `..._read0L.yaml` | installs **0.987** (fastest bootstrap of the campaign) / held **0.000** — snapping is not caused by read *composition*; the block-3-only death was position + confounds |
| mem_dim 8 (capacity squeeze) | `..._dim8L.yaml` | installs **0.990** (faster than dim 32!) / held **0.000** — the lookup survives an 8-dim code; dimensional capacity is not the lock alone |
| **rule diversity: affine family** | `..._affineL.yaml` | 448 trained rules `y=(a·x+s)%32` (~620 params/rule vs ~5,900 at 25 rules — Raventós-style diversity squeeze). v1 (anneal [150,350]) starved: the teacher kick needs rule-count-proportional time. v2 (anneal [1000,1500], 4000 steps): grokking-style crack out of the ln(32) plateau at ~600, then **first non-zero held of the program** — held tracks rule_acc with no reopening gap, but the circuit engaged at ~1900 with the cosine already dying and plateaued at **0.11–0.13** |
| affine v3: warm restart (SGDR) | `..._affineL_wr.yaml` | LR restored to peak from the v2 step-2200 checkpoint (teacher OFF — β was already 0): rule_acc **0.13–0.16**, held **0.12–0.13** still tracking, abl_gap ~2.2 — confirms the v2 ceiling was *not* the LR schedule alone; diversity buys held ≈ rule_acc but both stay low at 448 affine rules |
| **task pivot: fresh-rule benchmark (dsv4 series)** | `..._s256L*.yaml` | task redefined as true continual learning: a **fresh shift rule per conversation** (S=256, 224 train / 31 held grid), unseen queries — no repertoire to memorize. At S=256 the CE **never cracks under β=1** (it only moves during/after the anneal): s256L v1 (anneal [300,600]) starved; s256L v2 (anneal [800,1300]) cracked CE to 4.86 with abl_gap **+1.9** but rule_acc at **chance** — "gap without decisions"; lrv3 (peak 1.5e-3) dead |
| dsv4 conventions (RMS-match, WSD, SwiGLU read) | `..._s256L_dsv4.yaml` | Muon RMS-match (update RMS 0.2·lr on every matrix — validated 7.5e-4 effective on the backbone, ~8× boost on the fw hypernets) + WSD schedule + gated SwiGLU read: still dead at chance — conventions alone don't unlock S=256 |
| Fourier teacher, full spectrum | `..._s256L_dsv4f.yaml` | FIXED Fourier codes (anti-collapse pressure): dead — `write_code_probe` (CPU, offline) localized the failure: **constant-writer loophole** — MSE distill vs zero-mean fixed targets is minimized by shrinking ‖w0‖→0 (cos(w,target) −0.004, distill = the constant-writer value exactly); intra-rule = inter-rule = 1.000 |
| Fourier k≤8 + cosine distill + clock curriculum | `..._s256L_dsv4g.yaml` | three fixes (low-freq codes, (1−cos) distill that pays only for alignment, 16→224 rule ramp by step 800): cosine kills the constant writer (intra/inter 1.00/1.00 → 0.29/0.49 — writes diversify) but **intra < inter at every probe**: variance is presentation noise, never rule identity; distill stuck at 0.89 (cos≈0.1). Diagnosis: the ramp is `step/800` from step 0 — **the 16-rule mastery regime lasted <50 steps, i.e. never existed**; killed @850 |
| ✔ control: old task + old teacher, current stack | `..._s32_ctrl.yaml` | S=32/25 shifts, learned-emb teacher + MSE + anneal [150,350], under the full dsv4 stack (SwiGLU, RMS-match, WSD, balance 1e-4, adam_eps): **0.935 @700 and rising** (ref. 0.974@800) — **no regression**; every S=256 negative belongs to the task or the teacher, not the code |
| ✔ control: old task + NEW teacher | `..._s32_ft.yaml` | ONE variable vs s32_ctrl: Fourier k≤8 + cosine distill. Installs identically (**0.807 @710** ↗, distill 0.09 = cos(w,teacher)≈0.9 at S=32 vs plafond cos≈0.1 at S=256) — **teacher innocented**; the S=256 wall is the task's scale. Bonus negative: held **0.000–0.005 (below chance)** — an explicitly *interpolable* teacher manifold, with the write aligned to it, still snaps: code geometry was never the read-generalization lock |
| **🎯 mastery-gated curriculum** | `..._s256L_dsv4h.yaml` | pool held at 16 rules until CE EMA < 5.0, then DOUBLES (min 150 steps/stage); anneal at pool full + 300. Stages **accelerate**: 16→32 @392, →64 @657 (265), →128 @898 (241), →224 @1048 (dwell floor); distill cosine 0.79→**0.011** (cos≈0.99 entering the anneal [1348,1848]). Post-β=0: **rule_acc 0.18–0.25 (50× chance, ablation at chance) with rule_HELD 0.16–0.27 TRACKING** — never-trained rules applied as well as trained ones: **the program's first held ≈ rule_acc**, the law learned rather than the repertoire. Consolidation at full LR plateaus in a noisy band (CE 1.9↔2.8, spike+recovery @2200) — LR-bound; cut @~2400 to hand the GPU to dsv4i |
| **early crutch exit + anneal-anchored decay** | `..._s256L_dsv4i.yaml` | ONE protocol vs dsv4h: anneal fires at the **mastery of 64 rules** while the curriculum keeps doubling — the remaining 160 rules must install **teacher-free** (TBPTT gradient on an organized circuit); plus EMA-gated full-pool fallback and WSD decay re-anchored to anneal end (+800). Verdict: trajectory identical to dsv4h through 898 (same seed = controlled comparison), anneal pulled [898,1398], **pool full @1048 in 150 steps (dwell floor) under β≈0.7**; post-β=0 installation 0.083→0.28 with held tracking throughout; decay from 2198 tolerated (no erosion). Final band @2400: **rule_acc 0.22–0.28 / held 0.21–0.29** (last probe 0.271/**0.281**, ablation 0.000) — **≥ dsv4h ~1600 steps earlier**. CAVEAT (user): no rule is fully teacher-free (the late 96 saw the β tail), so this validates the **cost claim only** (teacher time need not scale with rule count), not "never-taught rules install" — that is dsv4j's blind arm. Cut @2400 for dsv4j |
| **teacher-blind control × 4-block read** | `..._s256L_dsv4j.yaml` | protocol = dsv4i + two orthogonal questions: (1) **32 TRAIN rules excluded from blend+distill from step 0** (`teacher_blind_shifts` [4,12,…,252], probe arm `rule_BLIND`) — blind ≈ taught ⇒ the kick is per-circuit (the STRONG kill-the-crutch claim); blind at chance ⇒ per-rule; (2) read on **all 4 blocks** vs dsv4i's [0] — capacity-of-application test for the ~0.25 plateau. Registered prediction (user, pre-test): 0.25 = 1/4 = one read out of four blocks — if true the ceiling should shatter, not creep. Verdict: **KILLED @905 — stage 16 never mastered** (CE pinned at/above the ln 256 chance floor 5.52–5.62 for 900 steps, distill *rising* 0.68→0.84, no doubling; dsv4i mastered @392). First structural falsification of the diversity regime: **four injection points feeding an unorganized code poison the trunk during bootstrap** — same lesson as the teacher, organize first, extend after. Blind arm unread (never got past β=1) |
| **read graft on organized circuit** | `..._s256L_dsv4l.yaml` | warm-restart **dsv4i@2400** (rule 0.27/held 0.28) with reads enabled on all 4 blocks; checkpoint pre-patched: `fw_o` of blocks 1–3 **zeroed** so the new reads are an exact no-op at step 0 (the read is a residual delta `h + fw_o(y−y0)`) and gradient grows them from zero — LoRA-B-style graft. Teacher OFF, full pool, fresh optimizer, 1500 steps, decay @800. Sanity at launch: **CE 1.85 @step 1** = dsv4i's final level, circuit intact. 3rd point of the blocks-proportional curve. Verdict: **THE GRAFT WORKS AND THE 0.25 CEILING FALLS** — no restart shock (CE 1.85 = dsv4i's final, then monotonic ↓ to 1.59), probes climb steadily 0.30→0.32→0.35→0.36→**rule_acc 0.404 / held up to 0.383** (program records, held above train at several probes: the graft extends the LAW, not the repertoire). Cut @800 (user cutover to dsv4m) — **still rising, decay never ran**; checkpoints to step_800 preserved for a resume. Depth-of-application WAS a binding ceiling; organize-first-extend-after is now the program's twice-validated recipe (teacher for the write, graft for the read) |
| **diversity/capacity probe S=128** | `..._s128_dsv4m.yaml` | ONE variable vs dsv4i: n_symbols 256→128 (112 train / 15 held, Fourier codes 2× spaced, ce_thresh rescaled 5.0→4.3 to keep the 0.55-nat margin under ln 128). No graft, no blind — the S question stays clean. SECOND variable added at relaunch@121 (user): anneal_len 500→300. Interim (run in progress, ~step 2950/4000): **the ceiling was capacity — the task train-solves**: paliers 259/437/592, anneal [592,892], then post-β=0 climb 0.065→0.46→0.81→**0.979 train / 0.688 held @2800** (125×/88× chance — both program records; dsv4i band was 64-72×), CE 0.16, ablation ≈0. Held tracked train through ~0.6 then the gap REOPENED at the top (0.98 vs 0.69) — watch: top-end sharpening vs held's own ceiling. Diversity threshold: held massively alive at 112 rules ⇒ threshold ≤112, bracketed in (25,112]. **Post-anneal code drift** (write_code_probe step 1200 vs 2600, informational distill 0.22→0.59 while acc climbed): the write LEAVES the Fourier circle (circular ridge decode 1.0→9.2 symbols ≈ chance) and compresses into an anisotropic cone (inter-cos 0.19→0.85, intra 0.997, **1-NN ident 0.978**) — teacher = scaffolding, dismantled once β=0; distill↑ at β=0 is HEALTH (exploration), the same signal at β=1 was dsv4j's pathology. Probe lesson: circular-ridge probes measure the TEACHER's geometry — use 1-NN after any anneal |
| **family transfer, teacher-free** | `..._s128aff_dsv4n.yaml` | PREPARED (awaiting dsv4m's final.pt): warm-restart the dsv4m weights and train a NEW family y=(a·x+s) mod 128, a∈{3,5} (`affine_units` knob; 256 rules, 32 held grid-interleaved on the (a,s) torus) with **no teacher at all** — same vocab, same key tokens, only the law changes. Baseline: affine from scratch WITH teacher = 0.13 (affineL v2). Installs ≫0.13 ⇒ the bootstrap transfers across families (teacher = one-time cost per MODEL — the headline claim); stalls at the CE floor ⇒ the kick is per-family. Capacity question (user): after the run, re-probe the SHIFT family offline on the final ckpt — catastrophic forgetting at family scale. First live use of `wsd_decay_on_plateau` (ReduceLROnPlateau twin: decay fires when the post-anneal CE EMA stalls ≥300 steps, replacing the +800 clock — the protocol's last clock becomes mastery-gated) |

**Headline:** memory *policy* — retention (rehearsal past eviction) AND replacement (dropping
a superseded rule) — is task-adaptive and **emerges end-to-end**; no gate/LRU/allocation
mechanism was needed. The write generalizes (circular code manifold, held shifts placed
correctly ON-manifold); the **read** is the generalization blocker and the rehearsal-precision
bottleneck. Mechanistic evidence and per-script details: [`analysis/`](analysis/README.md).

**Read-generalization campaign (closed, negative):** neither data pressure (noise, mixup
in four variants including post-installation) nor structural smoothness (spectral norm at
4 and 2 read points) moves `rule_HELD` off 0.000. The two arms fail for dual reasons —
installation *requires* sharp decision boundaries between neighbouring codes (SN forbids
them ⇒ consolidation blocked at ~0.53), while midpoint supervision is *absorbed by
memorization* rather than interpolation once the read is sharp. On this task the transport
mechanism is a **closed-repertoire recognizer**: it generalizes perfectly to new
conversations of trained rules, but discrimination-grade sharpness and on-manifold
interpolation are locally incompatible for the current read. Open exits: FiLM-style
affine reads, larger code spacing, or embracing the recognition framing.
The `s32_ft` control later added a third negative arm: an explicitly **interpolable
teacher manifold** (fixed Fourier codes, write aligned at cos≈0.9) leaves held at
0.000 too — the code geometry was never the lock. The only lever that has ever moved
`rule_HELD` is **rule diversity** (affineL, 448 rules: held ≈ rule_acc ≈ 0.13) —
density of rules forces interpolation where smoothness of codes cannot.

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
| `mem_bptt_window` | TBPTT span; **≥2 required** to train the write head — for `multiturn_rule`, set it (and `grad_accum`) to the full conversation length (presentations + turns) so credit reaches every write |
| `mem_probe_every` | How often to run the probes |
| `data.persist` | `true` ⇒ per-file ordered lanes + carry the bank across steps |

`multiturn_rule` task knobs (`data:`):

| Parameter | Description |
|---|---|
| `n_symbols` / `n_examples` | alphabet size S (chance = 1/S) / example pairs per presentation |
| `turns_per_conv` | unseen-query turns per conversation; ≥12 enables the per-turn horizon probe (FIFO eviction of the presentation slot happens at query turn 16) |
| `n_contexts` (K) | rules per conversation, each behind a key token (keyed routing) |
| `heldout_shifts` / `train_shift_max` | held-out generalization arm: explicit list (interleaved) or contiguous tail above the max |
| `switch_at` | mid-conversation rule switch (K=1): s2 re-presented at that turn, bank carried, no reset — enables `pre/post` + `STICK` |

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
  analysis/      — offline mechanistic diagnostics + campaign results (see its README)
    code_geometry.py     — write-code manifold structure + held-code placement
    rehearsal_inspect.py — do query-turn writes re-encode the rule? (horizon)
    switch_inspect.py    — per-turn write similarity across the rule switch
    canonical_ident.py   — which rule does a write encode, vs canonical codes
  configs/
    tiny.yaml / small.yaml   — TinyStories
    code_persist.yaml        — code, bank persists across sequences
    synth_recall.yaml        — synthetic addressable-recall diagnostic
    gist.yaml                — synthetic latent-context (gist) diagnostic
    multiturn_rule.yaml              — continual-rule benchmark (the K=1 reference, 0.948)
    multiturn_rule_k2.yaml           — K=2 keyed routing (0.99)
    multiturn_rule_k2_heldout.yaml   — contiguous held-out shifts (generalization arm)
    multiturn_rule_k2_interleaved.yaml — interleaved held-out (interpolation test)
    multiturn_rule_horizon.yaml      — 24-turn persistence horizon (rehearsal emergence)
    multiturn_rule_switch.yaml       — mid-conversation rule switch (forgetting test)
    multiturn_rule_joint.yaml        — retain-then-drop joint test (24+16)
```

---

## References

- DeepSeek-V4: [arxiv 2606.19348](https://arxiv.org/abs/2606.19348)
- Fast weights: Schmidhuber 1992; Ba et al. 2016; Schlag et al. 2021; Test-Time Training (Sun et al. 2024)
- Hyper-Connections: Zhu et al., 2025 · DeepSeekMoE: Dai et al., 2024 · Muon: Jordan et al., 2024
- Thought memory baseline: [`thought_lm_minimal`](../thought_lm_minimal/)
```
