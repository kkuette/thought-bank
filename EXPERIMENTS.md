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
| v2g carry | REGISTER across no-reset carry | ✅ prerequisite holds | REGISTER volatile by seed — s3 closes the question (FINDINGS 2026-07-13 (5)): {+0.44, +0.11, +0.53} code, usually positive, never load-bearing (d SPECIFIC stable ~+0.8) |
| v2h stack (FINDINGS 2026-07-13 (2)) | do D+G+G2 compose or cannibalize? | ✅ composes at ~zero carried cost, 2 seeds; leak ÷3 | blank-query recency unchanged (owned by GRPO/pages) |
| v2e_long (1600 steps) | is 800 steps saturated? | ✅ not saturated (carried −0.11) | budget signal for 350M |
| mem_dim grid 512/256/128 (s4) | how much does width buy? | 🔶 512→256 costs +0.31, 256→128 ~free | zero-shot grid; superseded by B1 trained taper (102): trained 256 = free |
| longlife 8→4096 writes (FINDINGS 2026-07-13 (2)) | does bank health survive long lives? | ✅ norms flat, erosion ~0.13 nat/decade | **soft refutation published**: ±0.15 criterion exceeded on web (+0.16) |
| capacity n=48 (FINDINGS 2026-07-13 (2)) | is addressed recall content-specific? | 🔶 foreign label −0.56 below reset = register effect; thread-specific ≈ −0.75 (not −1.31) | funding-figure caption must decompose the two |

## Arc 4 — v3 cascade (tensor overflow memory) & the B backlog

| Test | Question | Verdict | Not established / caveat |
|---|---|---|---|
| merge avg32/64 (FINDINGS 2026-07-12 + 07-13 (2)) | is merge-at-read viable at depth? | ✅ plateau at avg64, floor −0.58..−1.27 vs reset | zero-shot merge ≠ trained page read |
| v3_lite ×2 seeds — PAGE VERDICT (2026-07-13) | does reach-back through the page *emerge*? | ❌ 2× RED: page ablation changes nothing (d ≤ 0.008, signs disagree) | **but**: reach-back itself is real (−0.37..−0.72 vs reset, 4/4 cells) via live-bank superposition residue; page costs ~0 on recent → healthy substrate, train it (option 2, v2f recipe) |
| v3_deep (depth 2) (2026-07-13) | is the depth flag neutral when unused? | ✅ carried Δ −0.004 code / +0.032 web vs v3_lite (grid ±0.15) → depth = pure deployment flag, 350M ships depth 4; page verdict replicated a 3rd time (emergence null \|t\|~1.3, reach-back real −1.42 code) | level 2 stayed ~empty by design — neutrality of an *unused* level, says nothing about a filled one |
| B1 trained taper (v2e_md256) (FINDINGS 2026-07-13 (5)) | does training close the 512→256 gap? | ✅ carried Δ −0.014 code / +0.042 web @800 (grid ±0.15) → trained taper 512→256 is FREE; 350M VRAM budgets with 512/256/128 stand | init v2b_md256_s4 vs v2c (equal post-init budget, seed noise uncontrolled); md128 arm promoted (job 112, 🟡) |
| B2 reset-cue neutrality (v2e_resetcue) (FINDINGS 2026-07-13 (6)) | does *announcing* resets warp writes? (measure-only) | ✅ NEUTRAL inside the pre-registered grid: CE unchanged, norm −0.5 % (deflationary), redundancy *down* = no defensive rehearsal; OOD control (v2e, never saw the marker) shows the same drift ⇒ residue is lexical, not policy | task-loss-only baseline at 97M; says nothing about what a retention *reward* would create (that is the standing warning); unfreezes B5 |
| B3 cross-modal doc↔body zero-shot (job 105, 2026-07-13) | does a docstring-only write help generate the body? | ✅ **GREEN on v2e**: doc→body +1.17 (\|t\| 14.1), specific (+0.21), only 0.31 from the body-written ceiling; v2f +0.26 just under the 0.3 bar | zero-shot on Python only; asymmetry (addressed model transfers *less*) unexplained → doc-only training mode now codable |
| B4 internal DeltaNet steelman (v2e_delta) (FINDINGS 2026-07-14) | does a gated delta-rule inter-chunk carry match the trained bank? | ✅ split verdict: delta CARRIES (GAP @800 +0.92 code / +1.22 web, bank order of magnitude) but CANNOT ADDRESS — junk-last cost label-cued +2.07/+1.26 (bank v2f: −0.00/+0.03), cue value recency-bound, xmodal +0.26 under the +0.3 grid (bank +1.17) → the bank's niche is addressing, not carry | 97M internal science; the public commitment stays "at target scale" |
| B5 output→input loop (H) | rehearsal by re-narration | ⚪ UNFROZEN (v3-lite verdict landed + B2 measured) — design next, post-rehearsal verdicts | — |

## Arc 5 — data & scale (current)

| Test | Question | Verdict | Not established / caveat |
|---|---|---|---|
| divmix zero-shot (smoke 2026-07-13) | does the v2c bank transfer to unseen languages/domains? | ✅ GAP positive on all 14 sources incl. never-seen C/Rust/JS/SQL (+1.2..+1.7), arXiv grows with depth | **surface-reuse confound open**: invar on unseen languages (in job 107) is the discriminating test — do not claim "abstraction" yet |
| v2e_divmix trained (job 107) (FINDINGS 2026-07-14) | 14-source diversity at constant recipe: anchors hold? new-domain GAP? | ✅ **GREEN — official 350M mix frozen**: GAP @800 positive on all 14 sources (+1.15 khan … +1.96 sql; anchors codeparrot +1.23 / fineweb +1.41 ≈ v2e), depth-flat d2→d8; invariance closes the surface-reuse confound (reseg ≈ 0 on all 14, swap distance positive everywhere incl. unseen languages +0.67..+1.11, rename +0.30 of a +0.83 ceiling) | khan/openstax invar at n=3/29 (small held pools); abstraction claim = ~2/3 of gist survives rename, on codeparrot only |
| reach-back SFT (option 2, v3_reach, job 108) (FINDINGS 2026-07-14) | can the page read be *created* like addressing was? | ❌ **page dead, 3rd strike**: the behavior trains (label-cue value −0.81/−0.48, reach-back vs reset −0.29/−1.10) but the pre-registered page ablation does not separate (+0.020 code wrong sign / −0.002 web) — all value flows through live-bank superposition even when SFT'd on evicted targets (user's deep-block reservation confirmed) | page-read drops off the 350M critical path; revisit only if the 350M register saturates; cascade stays a free deployment flag |
| capacity curve across eviction (job 109) (FINDINGS 2026-07-13 (5) + 2026-07-14) | addressed recall(N) for N up to 2×max_mem: residents vs evicted, page contribution by ablation | 🔶 curve exists: N=2 recall −1.11 (code, v2h), gentle interference (+0.26 @N=8), residents immune to eviction pressure (\|t\|~0.1); evicted recall ≈ register — v3_lite arm adds the killer control: evicted −0.84/−0.87 vs reset but UNWRITTEN label −0.843 ⇒ address-specificity past eviction ≈ 0; page contributes −0.004 nat (cascade on or off, same profile) | web compressed ~0; figure caption must decompose specific vs register |
| capacity deep / saturation (probe `capacity_deep`, jobs 113-115 en .hold) | à ~8192 writes (une vie cumulative, bins d'âge en octaves) : la page s'allume-t-elle (own−ablated) précisément là où le registre en superposition meurt (own−reset → 0, saturation longlife ~1024) ? | 🟡 staged 2026-07-14 : probe codée + smoke OK sur v3_reach (profil 109 répliqué aux petits W) ; checkpoints ALIGNÉS SUR FILL (formule user vérifiée sur cascade.py : 1re destruction à M + 2·M^depth writes — v3_reach d1=24, rehearsal d2=136, d3=1032, d4=8200 ; rétention totale M + 2·Σ M^k = 9368 @d4 = la spec) — d2/d3 comparés à capacité égale, pas à writes égaux ; jobs .hold dans la queue, à déposer quand 110/111 sortent ; 3 issues informatives (page s'allume = « jamais nécessaire » ; tout meurt = structurel ; registre tient = capacité sans page sous-estimée) | zero-shot, probe only (~4-6 h/ckpt d2, <1 h v3_reach) ; pool held (~8200 code / ~6000 web) plafonne d3 à fill ~8 ; contrôle own−foreign = spécificité d'adresse par bin |
| v350_rehearsal (job 110) | dress rehearsal: the FULL 350M recipe (divmix + D+G+G2 + cascade depth 2 map [0,0,0,0,1,2] + reach + bf16) trained from scratch at 97M — do the SFT-created mechanisms appear under joint training? | 🟡 queued (2000 steps ~26h; guard: NaN before 200 ⇒ rerun fp32 = a bf16 verdict) | mechanisms missing ⇒ 350M needs a curriculum (pretrain → SFT), to know before paying; depth 2 = half the 8-file life demoted, oldest half destroyed |
| v350_rehearsal_d3 (job 111) | controlled twin of 110, ONE variable: cascade_map [0,0,0,1,2,3] — depth 3 = total retention of an 8-file life (8+16+32 = 56 evictions covered) at the price of 3 live-bank layers instead of 4 | 🟡 queued (~26h; predictions: carried d3 ≤ d2 + 0.3 nat; reach s3 drops below its d2 level late) | first FILLED-level test; a d3 loss at 97M is a pessimistic bound (350M has proportionally more live layers) |
| v350 bring-up + batch sweep (Vast.ai, `configs/v350_bringup.yaml` + `vast/`, 2026-07-13, ~$1 total) | the missing sizing numbers: true VRAM/conv of the full stack at 768d, steps/h, max batch | ✅ measured: **4090 24GB OOMs on the FIRST forward** (stack pays its read graphs; old ~14 GB estimate was bare backbone); A100 80GB B=1 = **26.75 GB peak, 42.2 s/step net** → 2000 steps ≈ 23.5 h ≈ $9-21 depending on host; **B>1 impossible under current recipe**: trainer asserts `var_chunk` requires batch=1 (the ×3.7 batched mode was v2b_mix, pre-var_chunk) — batching lever does not exist without code; A100 **40GB** ($0.40/h) fits B=1 with 13 GB margin = cheapest option; checkpoint egress costs money on some hosts ($52/TB vs $4/TB — select on `inet_up_cost`) | throughput measured on A100 PCIe only; multi-GPU wall-clock lever = DDP with B=1/rank (sidesteps the var_chunk assert), unbuilt; 10 steps say nothing about training health (jobs 110/111) |
| v350 phase-1 batch sweep (A100 80GB, `configs/v350_phase1_bringup.yaml`, 2026-07-13, ~$0.5) | does the EXISTING batched path (v2b_mix recipe, fixed chunks, no stack) deliver its ×3.7 at 768d? | ✅ B=1 18.6 s/step 26.8 GB, B=2 11.1 s 39.2 GB, **B=4 5.25 s 59.6 GB = ×3.54** (predictions: VRAM ~60 ✓, gain ×3-3.7 ✓); full stack costs ×2.3 by itself (42.2 vs 18.6 at B=1) → 2-phase curriculum (batched bulk + full-stack continued, the v2b→v2c precedent) cuts a 6000+2000-step 350M from ~$53-82 to ~$18-26 | the curriculum itself is NOT yet validated as science — that is exactly what rehearsals 110/111 arbitrate (from-scratch composition vs curriculum needed); B=4 leaves 20 GB margin, B=8 untested (would need effective-batch-16 revalidation anyway) |
| 350M validated run | the funded/self-funded scale point | pending full validation arc | plan in passation §2bis |

---

*Maintenance rule: one row per experiment, added when the verdict lands (or as
🟡 when queued). The "not established" column is mandatory — it is where the
next experiment comes from.*
