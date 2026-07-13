# Experiment tree — what each test settled (and what it didn't)

Companion to [FINDINGS.md](FINDINGS.md) (the newest-first journal with full
numbers and repro commands). This file is the map: one row per experiment,
in program order, with the question it answered, the verdict, and — just as
important — **what it deliberately does not establish**. Verdicts link to the
FINDINGS entry (by date) or to the config/commit when the entry is pending.

Legend: ✅ positive · ❌ negative (published refutations included — they steer
the tree as much as the wins) · 🔶 mixed/nuanced · ⚪ neutral by design ·
🟡 in flight.

---

## Arc 0 — dsv4mini: the mechanism in a synthetic lab (→ preprint V0.2.2)

The closed-world phase: prove a fast-weight bank can carry *rules*, not data.
Full detail in the preprint (Zenodo DOI 21225721); condensed here because its
verdicts gate everything after.

| Test | Question | Verdict | Not established / caveat |
|---|---|---|---|
| teacher-forcing bootstrap | can the ignore-bank fixed point be broken? | ✅ rule transport 0.03→0.99, teacher = pure kick (anneal β→0) | recipe, not a capability claim |
| K=2 keyed routing | can the read route between two stored rules? | ✅ 0.99 (better than K=1) | closed repertoire |
| held-out shifts | is transport rule *learning*? | ❌ 0.011 vs 0.97 train → recognition, reframed | led to the diversity work |
| diversity threshold | what flips memorization → generalization? | ✅ threshold in (25,112] train rules | synthetic family only |
| capacity unlock | what capped acc at 0.25? | ✅ two bottlenecks (graft 103×, S=128 88× held) | — |
| TTT demo (2 acts) | bank vs test-time training? | ✅ bank = 1/138ᵉ cost, neighbor key preserved vs TTT −62 % interference | synthetic tasks |
| switch / rehearsal | does memory policy come for free? | ❌ zero-shot persévération STICK 1.0 → policy is *trained* behavior | foundational for the RL arc |
| think cell (dsv5f/g) | can the bank be a write→read workspace? | 🔶 mechanism proven (1.000 vs ablation 0.000), *necessity* refuted (one-forward shortcut learnable) | claim needs non-commutative composition |
| gate v2c dedup-refresh | supersession primitive? | 🔶 repl 0.27→0.55, no auto-fusion; late retrieval 1.000 | erase primitive absent by design (superposition = feature) |
| family transfer (dsv4y/z) | warm-start into a new family? | 🔶 learns (0.255 ≫ ref) but plateaus + total forgetting of the old family | family distance is real |

## Arc 1 — grafting onto a pretrained host (dead end that set the course)

| Test | Question | Verdict | Not established / caveat |
|---|---|---|---|
| SmolLM graft v1→v10 (2026-07-08) | can the bank be bolted onto a frozen pretrained LM? | ❌ no placement works: consume→drift, bound→starved, between→blow-up | says nothing about the bank itself → from-scratch pivot |

## Arc 2 — dsv6 native: the bank as long-context memory on real data

| Test | Question | Verdict | Not established / caveat |
|---|---|---|---|
| native v1 ragged 47M (FINDINGS 2026-07-09) | does the dual loss train real multi-turn memory on code? | ✅ GAP curve FLAT 1→10 writes | single scale, single source |
| v2 mem_dim 512 | naive width scale-up | ❌ GAP collapse → **Muon √cols trap** (update RMS ∝ shape) | diagnostic, not a memory result; fixed by per-group lr_scale |
| v2b_mix (2026-07-09) | triple validation: lr_scale+decay, 50/50 code+web, batched B=4 | ✅ GAP +0.79/+0.87 to d8 on both sources | — |
| v2c_varlen (2026-07-10) | variable chunk lengths + continued pretrain | ✅ GAP +1.44/+0.82 @400 | anti-positional-shortcut prerequisite for RL |
| scaling 97→135M (2026-07-10) | does the GAP grow with scale? | ✅ code ~+1.07 flat, better at every depth | web data-capped (3 epochs); 2 points ≠ a law; 350M pending |
| cohab probe (135M, zero-shot) | two files in one bank? | ✅ A+B both recoverable 60/82 % | cohabitation ≠ transfer |
| reflect-k probe (zero-shot) | do free "thought" turns help? | ❌ untrained thoughts dilute | motivates trained loop (B5/GRPO), not a dead end |
| invar probe (renaming) | is the gist abstract or surface? | 🔶 ~47 % of GAP survives total renaming | THE scaling axis to watch; divmix re-measures on unseen languages |
| GRPO v1 write gate (FINDINGS 2026-07-10) | is the write decision learnable as a policy? | ✅ 75 % of always-write at half the writes (384-float policy) | write policy only; addressing policy = phase 2 |
| bf16 validation (2026-07-11) | precision cut safe? | ✅ carried identical | reset +1.2 nat = reads carried, not GAP (rule 4 born here) |

## Arc 3 — memory *policies* (user designs A–G, one letter at a time)

| Test | Question | Verdict | Not established / caveat |
|---|---|---|---|
| v2d no-reset (D) (FINDINGS 2026-07-12) | does life-without-resets teach a stale-filter? | 🔶 learns a **boundary inference**, not a filter (distractor-last collapses below reset) | worst-case under reset worse; killed by interleaving |
| v2e interleave (G) (FINDINGS 2026-07-12) | spaced practice → content-based selection? | ✅ MID filter acquired (+0.03) | ❌ blank query still = recency (+2.42) → cued probe insight: bank unread once context exists |
| v2f addr (G2) (FINDINGS 2026-07-13) | is *addressing* trainable by SFT? | ✅ 800 steps create it, 2 seeds × 2 datasets (label-cue −1.29/−1.15) | mechanism, not policy — emission policy = GRPO phase 2 |
| v2g carry | REGISTER across no-reset carry | ✅ prerequisite holds | REGISTER volatile by seed (+0.44/+0.11); s3 in flight measures the rate |
| v2h stack (FINDINGS 2026-07-13 (2)) | do D+G+G2 compose or cannibalize? | ✅ composes at ~zero carried cost, 2 seeds; leak ÷3 | blank-query recency unchanged (owned by GRPO/pages) |
| v2e_long (1600 steps) | is 800 steps saturated? | ✅ not saturated (carried −0.11) | budget signal for 350M |
| mem_dim grid 512/256/128 (s4) | how much does width buy? | 🔶 512→256 costs +0.31, 256→128 ~free | zero-shot grid; trained taper arm (102) in flight |
| longlife 8→4096 writes (FINDINGS 2026-07-13 (2)) | does bank health survive long lives? | ✅ norms flat, erosion ~0.13 nat/decade | **soft refutation published**: ±0.15 criterion exceeded on web (+0.16) |
| capacity n=48 (FINDINGS 2026-07-13 (2)) | is addressed recall content-specific? | 🔶 foreign label −0.56 below reset = register effect; thread-specific ≈ −0.75 (not −1.31) | funding-figure caption must decompose the two |

## Arc 4 — v3 cascade (tensor overflow memory) & the B backlog

| Test | Question | Verdict | Not established / caveat |
|---|---|---|---|
| merge avg32/64 (FINDINGS 2026-07-12 + 07-13 (2)) | is merge-at-read viable at depth? | ✅ plateau at avg64, floor −0.58..−1.27 vs reset | zero-shot merge ≠ trained page read |
| v3_lite ×2 seeds — PAGE VERDICT (2026-07-13) | does reach-back through the page *emerge*? | ❌ 2× RED: page ablation changes nothing (d ≤ 0.008, signs disagree) | **but**: reach-back itself is real (−0.37..−0.72 vs reset, 4/4 cells) via live-bank superposition residue; page costs ~0 on recent → healthy substrate, train it (option 2, v2f recipe) |
| v3_deep (depth 2) (2026-07-13) | is the depth flag neutral when unused? | ✅ carried Δ −0.004 code / +0.032 web vs v3_lite (grid ±0.15) → depth = pure deployment flag, 350M ships depth 4; page verdict replicated a 3rd time (emergence null \|t\|~1.3, reach-back real −1.42 code) | level 2 stayed ~empty by design — neutrality of an *unused* level, says nothing about a filled one |
| B1 trained taper (v2e_md256) | does training close the 512→256 gap? | 🟡 job 102 in flight | md128 arm promotable after |
| B2 reset-cue neutrality (v2e_resetcue) | does *announcing* resets warp writes? (measure-only) | 🟡 job 104 in flight — prediction: NEUTRAL; any drift = publishable safety result | no loss/reward attached, by design (standing warning) |
| B3 cross-modal doc↔body zero-shot (job 105, 2026-07-13) | does a docstring-only write help generate the body? | ✅ **GREEN on v2e**: doc→body +1.17 (\|t\| 14.1), specific (+0.21), only 0.31 from the body-written ceiling; v2f +0.26 just under the 0.3 bar | zero-shot on Python only; asymmetry (addressed model transfers *less*) unexplained → doc-only training mode now codable |
| B4 internal DeltaNet steelman (v2e_delta) | does a gated delta-rule inter-chunk carry match the trained bank? | 🟡 job 106 in flight | 97M internal science; the public commitment stays "at target scale" |
| B5 output→input loop (H) | rehearsal by re-narration | frozen until v3-lite verdict + B2 measured (spec order) | — |

## Arc 5 — data & scale (current)

| Test | Question | Verdict | Not established / caveat |
|---|---|---|---|
| divmix zero-shot (smoke 2026-07-13) | does the v2c bank transfer to unseen languages/domains? | ✅ GAP positive on all 14 sources incl. never-seen C/Rust/JS/SQL (+1.2..+1.7), arXiv grows with depth | **surface-reuse confound open**: invar on unseen languages (in job 107) is the discriminating test — do not claim "abstraction" yet |
| v2e_divmix trained (job 107) | 14-source diversity at constant recipe: anchors hold? new-domain GAP? | 🟡 staged (anchors codeparrot/fineweb kept comparable to v2e) | GREEN ⇒ official 350M mix |
| reach-back SFT (option 2, v3_reach, job 108) | can the page read be *created* like addressing was? | 🟡 queued — targets stratified by eviction age (user reservation: deepest-block targets may not learn → stratum s2 is the sub-verdict; s3 "destroyed" = negative control) | verdict = page probe on final ckpt (early_on vs early_off must separate) |
| 350M validated run | the funded/self-funded scale point | pending full validation arc | plan in passation §2bis |

---

*Maintenance rule: one row per experiment, added when the verdict lands (or as
🟡 when queued). The "not established" column is mandatory — it is where the
next experiment comes from.*
