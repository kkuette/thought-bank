# Thought Bank — paper skeleton (v0, 2026-07-06)

Working title (recommended):
**"A Trained Fast-Weight Memory: Continual Rule Binding at Inference Without Backward"**

Alternatives:
- "Thought Bank: Learning a Memory Policy for Forward-Only Continual Adaptation"
- "Memory Policy Is a Trained Behaviour: Fast-Weight Banks vs Test-Time Training at 1/138th the Cost"

Central claim (fixed by user 2026-07-06): **the bank is a functional,
GENERALIZING memory** — it binds never-trained rules at inference in one
forward pass, retains them across turns and eviction pressure, and replaces
them on demand; this replicates across seeds. Selectivity of replacement is a
secondary finding (seed-dependent basin).

Venue: arXiv preprint (cs.LG / cs.NE). Single-GPU (RTX 3090), 3.08M-param
model — framing: mechanism study, not scale claim.

---

## Abstract (draft)

Continual learning at inference usually means test-time training (TTT):
gradient steps on a clone of the model. We study the alternative the fast-
weight literature has long promised: a small bank of vectors, written by the
forward pass itself, that modulates subsequent computation — no backward, no
optimizer, no weight copy. On a keyed multi-turn rule task (fresh modular
rule per conversation, unseen queries, K=2 concurrent rules), a 3M-parameter
transformer with an 8-slot bank learns to (i) install a never-trained rule
from a single 13-token presentation (held-rule accuracy 0.79–1.00 across two
seeds, chance 0.008), (ii) retain it across turns and slot eviction, and
(iii) replace it mid-conversation in one forward pass (post-switch accuracy
0.95 train / 0.78 held, old-rule persistence 0.000). Head-to-head on the same
conversations, TTT with a full LR sweep memorizes its adaptation examples
(pair accuracy 0.99) yet transfers exactly nothing to unseen queries, while
costing 138× more per rule update and destroying 62% of the untouched
concurrent rule; in-window ICL is also at chance — the bank is the model's
only functional adaptation pathway. Crucially, none of this is emergent from
the architecture: the same architecture trained without mid-conversation
rule switches perseverates completely (old-rule persistence 1.000 zero-shot).
Memory policy — what to keep, when to overwrite, how to write on a dirty
bank — is a trained behaviour, installed by randomizing conversation
structure at training time. We map the boundary (an untrained rule family
defeats bank and TTT equally: the limit is the meta-learned envelope, not
the mechanism) and report a seed-level bifurcation between selective update
and flush-and-rewrite replacement policies. Training the read/write circuit
requires breaking an ignore-the-bank fixed point; we give the recipe
(teacher-forced code bootstrap with annealed blending, mastery-gated rule
curriculum) and the diversity threshold below which the read memorizes
instead of generalizing.

~250 words. Numbers to freeze at final: held band per seed, STICK sweep.

---

## 1. Introduction

- Goal: continual learning AT INFERENCE without backward. Bank = fast
  WEIGHTS modulating the forward, not attended data (contrast: KV-cache /
  retrieval / ICL keep data in the window; TTT edits slow weights with
  gradients).
- The two-sentence pitch of the result matrix (Table 1 teaser):
  same model, same conversations — bank works, TTT memorizes-without-
  transferring, ICL at chance.
- Second contribution, arguably the more general one: **memory policy is
  trained, not architectural**. Zero-shot STICK 1.000 → trained STICK 0.000.
  The architecture provides a substrate; the training distribution decides
  what the memory DOES (retention, replacement, dirty-bank writes,
  eviction robustness). Analogy: random-crop → invariance; here structure
  randomization → memory policy.
- Third: the training problem itself is non-trivial (ignore-bank fixed
  point; joint credit assignment through write→store→read). Recipe
  contribution.
- Honest scope up front: 3M params, synthetic algebra, single 3090. This is
  a mechanism study; the currency is the controlled comparison.

## 2. Related work

- Fast weights: Schmidhuber '92; Ba et al. '16; linear-attention-as-fast-
  weights (Schlag et al. '21); Titans / memory-augmented transformers;
  DeltaNet-style write rules.
- Test-time training / adaptation: TTT layers (Sun et al.), test-time
  fine-tuning, ARC-style TTT.
- Meta-learning envelope: MAML-family, in-context meta-learning; the
  counterfactual-task literature (models degrade out-of-family) — grounds
  our subtraction boundary.
- Memory-augmented nets: NTM/DNC (gradient-trained differentiable memory —
  closest ancestors of the trained-policy claim), Hopfield-modern.
- Continual learning: catastrophic interference (our TTT arm measures it
  directly), rehearsal.

## 3. Architecture: the Thought Bank

Figure 1: architecture diagram (trunk + write head + bank + fast-weight read).

- Trunk: 4-block transformer (ThoughtBankLM, 3.08M params; CSA/HCA
  attention, MoE FFN — cite as engineering detail, not claim).
- Write: pool the segment → one vector/segment → FIFO slot (8 slots).
  Write gate OFF in the main recipe (audited: accelerates bootstrap ~30%,
  costs consolidation 4–6× — Appendix).
- Read: sequential per-slot fast-weight MLP (hypernet-LoRA + GELU) — the
  slot vector generates a rank-r delta applied to the residual stream.
  Rank-1 outer-product read provably insufficient for rule application
  (can't express a mapping) — design rationale.
- TBPTT through segments (mem_bptt_window ≥ conversation length) — the
  write head gets NO gradient otherwise.
- Cost model: install/update = one 13-token forward (80 MFLOPs at 3.08M
  params, proxy 2·P·tokens).

## 4. Benchmark: keyed fresh-rule conversations

Figure 2: task schematic (presentation segments, keyed queries, switch).

- multiturn_rule: per conversation, K=2 keys each bound to a fresh rule
  y=(x+s) mod 128; presentation = [key, x0,y0..x5,y5] (13 tokens); queries
  on UNSEEN x; rule crosses turns only via the bank (window sees one
  segment).
- Train/held split on s (mod-8 interleave → 112 train / 15 held; held
  rules NEVER trained — "fresh law" at eval).
- Structure randomization (the policy trainer): turns ~ U[8,16], up to 2
  mid-conversation re-presentations of a random key at random positions
  with a new rule (bank carried, i.e. dirty-bank writes), optimizer steps
  at conversation end. Eviction pressure: up to 20 writes > 8 slots.
- Careful claim wording (user correction): held rules are NOT "new
  knowledge" — they are fresh PARAMETER BINDINGS within a meta-learned
  family. The paper claims fresh-binding, not fresh-law-learning; §7 marks
  the family boundary.

## 5. Training the memory: breaking the ignore-bank fixed point

Figure 3: training dynamics (CE + distill + curriculum stages + anneal,
one run: dsv4w).

- The fixed point: gradient prefers ignoring the bank (read contributes
  nothing → write gets no signal → bank stays noise). Evidence: read
  applies ANY fixed code perfectly in isolation; joint training stalls.
- Recipe: (1) teacher-forced bootstrap — blend a Fourier positional code
  of the rule into the bank, β annealed 1→0; distillation (cosine) of the
  write toward the teacher code; teacher = kick, model drifts to its own
  code post-anneal (1-NN ident, ridge dies — Appendix drift analysis).
  (2) Mastery-gated curriculum: pool doubles 16→…→112 on CE mastery.
  (3) Muon + WSD; anneal timing coupled to curriculum.
- Diversity threshold: ≤25 rules → read memorizes (held 0.000 exactly,
  sharpening ⊥ interpolation); 112 rules → held ≈ train. Bracket (25,112].
- Ablations that fail: β-blend without distill (chance), from-scratch
  4-block read (never bootstraps), full structure randomization at this
  scale (dsv4v: K 2–5 × turns 8–32 never bootstraps — structural-entropy
  wall bracketed).

## 6. Results I: a functional, generalizing memory (CENTRAL CLAIM)

Table 2: main numbers, two seeds.

| capability | seed 42 (@3000) | seed 43 (@4000) |
|---|---|---|
| train rules, unseen queries | 0.951–0.987 | ~1.000 |
| HELD rules (never trained) | 0.792–0.828 | 0.997–1.000 |
| replacement (post-switch, train/held) | 0.953 / 0.777 | 0.91–1.00 |
| old-rule persistence (STICK), positions 2–14 | 0.000 | 0.008–0.012 |
| dirty-bank write identifiability (1-NN) | 0.90 | 0.95 |
| ablated bank | chance (0.008) | chance |

- Retention: no FIFO cliff; rule survives physical eviction of its slot
  (sw_long: code evicted at t8, accuracy 0.59+ persists; post-decay 0.924).
- Mechanism (Figure 4): redundant superposition — all 8 slots ≈ same
  vector (bank eff_rank 1.13/1.48 s42/s43) while RULE space stays rank
  7.3/12.4 of 32 centred (inter-rule cos ~0 after centring; 1-NN
  0.90-0.95). Not collapse: gap +4.6. Key-conditioned read disambiguates;
  redundancy buys eviction robustness. Verdict rule: eff_rank must be read
  jointly with ablation gap. [numbers re-frozen 2026-07-06 from
  superposition_probe.py on BOTH seeds — the earlier 1.08 / 14/32 / 0.17 /
  0.32 came from an uncommitted ad-hoc probe]
- Switch write is genuinely novel content (redundancy +0.50 s42 / -0.10
  s43 vs ~1.0 rehearsal — the dip gap IS the §9 bifurcation, visible at
  the write), bank restabilizes in one turn (>=0.95 next write).

## 7. Results II: head-to-head vs test-time training

Table 3 (= frozen act-1 matrix, dsv4m ckpt):

| pool | bank | TTT best (50 steps, LR swept) | ICL in-window | ablate |
|---|---|---|---|---|
| train | 0.992 | 0.008 | 0.006 | 0.006 |
| held | 0.799 | 0.002 | 0.010 | 0.010 |
| subtraction (fresh family) | 0.012 | 0.004 | 0.002 | 0.008 |

- TTT arm is HEALTHY: pair-loss 5.12→0.03, pair-acc 0.99 — it memorizes
  its 12 examples perfectly and transfers zero. Basin failure, not
  optimization failure. (Anticipates the obvious review objection.)
- ICL at chance even on train: the in-window pathway does not exist in
  this model; the bank is the ONLY route to the trunk's competence.
- Subtraction row = the fairness arm: out-of-family, gradient is ALLOWED
  to leave the envelope and still fails at this scale/budget. Boundary =
  meta-training, not forward-vs-gradient. (Gate-ON control also at chance.)

Table 4 (= frozen act-2, dsv4w@3000): replacement at inference.

| | bank | sequential TTT |
|---|---|---|
| update cost | 80 MFLOPs (1 fwd, 13 tokens) | 11,075 MFLOPs (50 steps) = 138× |
| new rule on unseen queries | 0.953 train / 0.777 held | chance (fit 0.97) |
| untouched concurrent rule | 0.977 → 0.832–0.844 | fit 1.000 → 0.383–0.344 (−62%) |
| old rule | evacuated (STICK 0.000) | 0.979 → 0.078 |

- Two qualitatively different degradation mechanisms (discussion point):
  bank collateral (−0.14) = eviction pressure (19 writes / 8 slots);
  TTT collateral (−62%) = gradient interference. One is a capacity knob,
  the other is intrinsic to the update rule.

## 8. Results III: memory policy is a trained behaviour

Figure 5: STICK vs switch position, zero-shot model vs policy-trained model.

- Zero-shot (trained on fixed 8-turn, switch-free structure): STICK 1.000 —
  total perseveration; dirty-bank write unreadable (1-NN 0.05); ONLY K=2
  key routing survives zero-shot.
- Policy-trained (structure-randomized): STICK 0.000 at every position
  2–14; dirty-bank writes clean (1-NN 0.90); old code physically evacuated;
  held rules install mid-conversation 0.74–0.80.
- Same architecture, same task family, same bank. The delta is the
  TRAINING DISTRIBUTION of conversation structure. Retention itself is
  trained too (fixed-horizon models show an exact FIFO cliff at their
  training horizon).
- Framing: the architecture is a substrate; policy comes from data — the
  memory-augmented-nets analogue of "invariances come from augmentation".

## 9. Findings and boundaries

- **Selectivity bifurcation** (secondary finding, honest reporting): on
  identical training streams, seed 42 learns selective component update
  (untouched-key accuracy 0.863 post-switch of the other key), seed 43
  learns flush-and-rewrite (0.011) while being STRONGER on everything else
  (held 1.00). Categorized eval on the true training stream: the
  non-switched key = 17% of query tokens; s43 pays chance on them for the
  entire run without leaving the basin. Two attractors of the update
  policy, decided during bootstrap; gradient does not escape. Implication:
  selective replacement needs a copy-forward circuit; average pressure is
  insufficient — event ORDERING during bootstrap is the lever (future work).
- **Out-of-family boundary**: subtraction defeats every arm (Table 3).
  Consistent with sequential/mixed family-transfer arc (all negative at
  this scale) and with the counterfactual-task literature on LLMs.
- **Composition boundary**: external chaining f(f(x)) via re-prompting
  works (0.961) but internal composition is at chance — the policy gap
  twin of STICK; "thinking via the bank" is trainable-in-principle future
  work.

## 10. Limitations

- 3.08M params, synthetic modular algebra, S=128, K=2. No natural-language
  transfer demonstrated (gist benchmark positive but preliminary).
- TTT baseline is full-parameter AdamW; stronger TTT (LoRA-TTT, more data,
  more steps) could shift Table 3/4 margins — though the fit diagnostic
  (memorize-without-transfer) suggests the failure is not budget.
- Teacher bootstrap requires knowing a code geometry for the family
  (Fourier on the circle); generality of the recipe untested beyond
  modular arithmetic (multiplicative family failed at this size).
- Two seeds for the central claim; selectivity needs a basin census
  (~5h/seed) we did not run.
- Held rules are fresh bindings within the meta-learned family, not new
  laws (wording enforced throughout).

## 11. Reproducibility

- Repo: github.com/kkuette/thought-bank (code, configs dsv4w/s43,
  analysis probes ttt_demo.py / ttt_demo_act2.py / switch_probe_k2.py).
- One 3090; policy run ≈ 5h; all probes CPU.
- End-to-end script: two training runs + three probes (TO BUILD — repro
  chantier item).
- Seeds 42/43, checkpoints at /100, deterministic generators.

---

## Figure/table inventory

| # | content | source | status |
|---|---|---|---|
| Fig 1 | architecture diagram | fig1_architecture.svg (hand SVG) | DONE |
| Fig 2 | task schematic (conv + switch) | fig2_task.svg (hand SVG) | DONE |
| Fig 3 | training dynamics dsv4w s42+s43 (CE, acc, distill, anneal) | metrics.jsonl both seeds | DONE |
| Fig 4 | superposition (slot sims, rule PCA, write redundancy, 2 seeds) | superposition_probe.py s42@3000 + s43@4000 | DONE |
| Fig 5 | switch policy: zero-shot vs s42 vs s43, traces + position sweep | switch_probe_k2 --sweep --dump ×3 | DONE |
| Tab 1 | intro teaser (bank/TTT/ICL one-liner) | Tab 3 subset | frozen |
| Tab 2 | central claim, two seeds | probes s42@3000 / s43@4000 | frozen |
| Tab 3 | act-1 matrix (pool × arm) | ttt_demo.py dsv4m | frozen |
| Tab 4 | act-2 replacement vs sequential TTT | ttt_demo_act2.py dsv4w@3000 | frozen |
| Tab 5 | ablations (recipe components) | memory files / READMEs | assemble |

## Writing order (proposed)

1. §6–§8 (results — tables frozen, prose from memory files) ← start here
2. §3–§5 (architecture/benchmark/recipe)
3. §9–§11
4. §1, §2, abstract polish
5. Figures (log parsing + probe re-runs with array dumps)
6. LaTeX port (arXiv template), bibliography
