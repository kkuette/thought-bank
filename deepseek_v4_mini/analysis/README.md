# analysis/ — offline mechanistic diagnostics

Post-hoc analyses run on `multiturn_rule` checkpoints (all CPU-friendly).
Run from the repo root with `PYTHONPATH=.`; each script takes an optional
checkpoint path as first argument.

| Script | Question | Key result |
|---|---|---|
| `code_geometry.py` | Are the written rule codes a structured manifold or a lookup table? | The write head builds a **circular manifold** mirroring the modular rule structure (cos-sim monotone in circular distance, erank 3.4/32); **held-out shifts are written ON-manifold at the correct position** — the READ, not the write, is the generalization blocker. |
| `rehearsal_inspect.py` | Do query-turn writes re-encode the rule? (horizon model) | Yes: **noisy partial copies** (sim ~0.5 to the presentation write, rule-identifiability ~0.48 vs chance 0.125). Post-eviction writes degrade (0.41/0.35) but accuracy holds — the read integrates redundant copies across slots. |
| `switch_inspect.py` | What happens to s1 in the writes after a mid-conversation rule switch? | Raw cosines are a **trap** (writes are bank-conditioned); see canonical_ident. Also measures the per-turn accuracy profile across the switch. |
| `canonical_ident.py` | Which rule does each write encode, measured against canonical codes? | In the **switch** model query writes carry **no rule identity** (0.03 = chance — 12-turn phases need no rehearsal); dirty-bank presentations stay canonical (0.56). Forgetting is a **recency override** in the read: s1's code is still in the bank at q13-q15 yet never used. |
| `joint_inspect.py` | Per-turn accuracy + write identity across the joint (rehearsal + switch) conversation | **No per-turn cliff** (acc 0.6-0.8 uniform over 40 turns) yet query writes have **no canonical identity** (chance) and are anti-correlated with the presentation — rehearsal in a **covert distributed code**, off the presentation manifold. Caveat: canonical ident only sees presentation-style codes; chance ≠ no information. |
| `write_code_probe.py` | Does the WRITE encode which rule was presented, or does identification die at the source? (S=256 series) | Localized the dsv4f failure offline: **constant writer** — intra-rule = inter-rule cosine = 1.000, ridge decode of `s` at chance, distill = the constant-writer MSE value exactly (the ‖w0‖→0 loophole). Under cosine distill (dsv4g) writes diversify (0.29/0.49) but **intra < inter persists**: presentation noise, not rule identity. Safe to run on CPU alongside a GPU training. |
| `affine_per_unit.py` | Per-(a,s) breakdown of affine-family accuracy — which rules install, which stay dead? | Companion diagnostic for the affineL runs (448-rule diversity squeeze). |
| `switch_probe_k2.py` | K=2 switch probe on GENERALIZING checkpoints: STICK, per-turn profile, bank-slot 1-NN identity vs an empirical clean-code dictionary; `--sweep` = switch-position invariance | Re-audit of the memorizing-regime switch claim. dsv4m (trained fixed 8-turn, no switch): **STICK=1.0** total perseveration, dirty-bank write unreadable (1-NN 0.05). dsv4w (structure-randomized): **STICK=0.000 at every switch position 2-14**, s2 written clean onto the dirty bank (1-NN 0.90+), s1 actively evacuated — memory policy is a TRAINED behaviour. Seed 43 replicates everything EXCEPT selectivity (untouched key 0.86 vs 0.01 on the same stream): two update-policy attractors, selective vs flush-and-rewrite. |
| `ttt_demo.py` | HEADLINE act 1 — fresh-rule adaptation: bank (forward-only) vs test-time training (per-conv AdamW on the example pairs) vs in-window ICL, same conversations, FLOPs accounting; `--sub` = out-of-family subtraction arm | On dsv4m: bank **0.992 train / 0.799 held** for 160 MFLOPs; TTT converges on its pairs (fit 0.99) and transfers **zero** at 138× the cost; ICL at chance even on trained rules — the bank is the model's only working adaptation pathway. Subtraction (fresh family): all arms at chance — the boundary is meta-training, not the mechanism. |
| `ttt_demo_act2.py` | HEADLINE act 2 — rule REPLACEMENT at inference: one-forward bank update vs sequential TTT; interference measured on the untouched key's pair-fit (nonzero baseline) | On dsv4w step_3000: bank replaces at **0.953 train / 0.777 held** for 80 MFLOPs (untouched key 0.98→0.83, eviction pressure not gradient interference); TTT fits the new pairs at 138× the cost, **destroys 62%** of the untouched key's fit, and still answers no query. |

## The experimental campaign these belong to

Benchmark: `multiturn_rule` — each conversation draws a fresh shift rule
`y=(x+s)%32` shown once (6 example pairs), then queried on UNSEEN symbols across
turns; the rule can only cross turn boundaries through the fast-weight bank.
Chance = 0.031, in-window ICL ceiling ≈ 0.49. Recipe for all runs: all-AdamW
3e-4 constant LR, no warmup, teacher-forced bootstrap annealed over steps
[300,500], write gate off.

| Experiment (config) | Result |
|---|---|
| K=1 transport (`multiturn_rule.yaml`) | **0.948** @1500 (chance→, ablation at chance) — the teacher-forced bootstrap breaks the ignore-bank fixed point |
| no-distill ablation | chance forever — distillation IS the active ingredient (β-blend alone does nothing) |
| K=2 keyed routing (`multiturn_rule_k2.yaml`) | **0.99** — two rules held simultaneously, queries routed by key token; the old K=2 wall was the fixed point, not addressing |
| held-out contiguous (`multiturn_rule_k2_heldout.yaml`, train s∈1..24) | 0.97 train / **0.011 held** — transport = closed-repertoire recognition, not open rule induction |
| held-out interleaved (`multiturn_rule_k2_interleaved.yaml`, held {4,8,...,28}) | rule_HELD = **0.000 exact** (deterministic snapping to trained neighbours); irregular coverage degrades the read even on trained shifts — no spontaneous interpolation |
| persistence horizon (`multiturn_rule_horizon.yaml`, 24 turns, slot evicted @16) | **no FIFO cliff** — rehearsal emerges from TBPTT pressure alone; cost: plateau ~0.48 (vs 0.95 @9 turns), uniform across turns |
| rule switch (`multiturn_rule_switch.yaml`, 12+12 turns) | **STICK = 0.000** at acc 0.795 — zero post-switch answers with the old rule; pre/post 0.80/0.79; forgetting is active (recency override), not just FIFO cleanup |
| joint test (`multiturn_rule_joint.yaml`, 24+16 turns) | **0.747/0.746 pre/post, STICK 0.02** @1200 (plateau) — full retain-then-drop: 24-turn maintenance THROUGH eviction, then clean replacement; maintenance beats the horizon model (0.74 vs 0.48) — switch pressure *improves* retention; rehearsal happens in a covert code (see `joint_inspect.py`) |
| Muon retest (`multiturn_rule_muon.yaml` / `_muon_cos.yaml` / `_muon_cos_early.yaml`) | constant LR: no collapse but unstable band 0.83-0.95; cosine: 0.99 @1000 stable; **+ early anneal [150,350]: 0.974 @800** — default recipe, ~2× faster than the AdamW baseline |
| read-generalization campaign (`_noise` / `_mixup*` / `_sn*` / `_read02L*` configs) | **closed, negative — held 0.000 across all arms.** Data: noise reinforces snapping; mixup memorized (even injected post-installation on a 0.995 base: train holds, mixup CE absorbed, held 0.000). Structural: block-3-only read kills the bootstrap; blocks [0,2] install 0.987 but don't interpolate; spectral norm blocks the final consolidation (~0.53) at 4 AND 2 reads — installation is a *sharpening* step, imposed smoothness forbids it. Caveat discovered en route: the fast-iter short format (turns_per_conv 4) caps train at ~0.48 regardless of intervention — 3 cells replicated the same curve; only long-format verdicts count |

Bottom line: **memory policy — retention AND replacement — is task-adaptive
and emerges end-to-end**; no gate/LRU/allocation mechanism was needed, even
under joint retain-then-replace pressure. Open costs: maintenance precision
(0.74 plateau → consolidation), read generalization (recognition, not
induction → code-space augmentation).
