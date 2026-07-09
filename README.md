# Thought Bank

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21225721.svg)](https://doi.org/10.5281/zenodo.21225721)

Research repo exploring **persistent thought memory** for language models. The
active line of work is **`deepseek_v4_mini`**: a small reproduction of the
DeepSeek-V4 architecture fused with a **fast-weight thought bank** — a rolling
memory the model reads as *weights* (a per-slot low-rank MLP applied to the token
stream) and writes to itself, targeting **continual learning at inference without
a backward pass**.

## 📄 Paper

**A Trained Fast-Weight Memory: Continual Rule Binding at Inference
Without Backward** — [PDF](paper/paper.pdf) ·
[DOI 10.5281/zenodo.21225721](https://doi.org/10.5281/zenodo.21225721)
(this version; all versions: [10.5281/zenodo.21222901](https://doi.org/10.5281/zenodo.21222901))

The paper corresponds to tag
[`V0.2.2-preprint`](https://github.com/kkuette/thought-bank/tree/V0.2.2-preprint),
archived at the DOI above; `main` continues to evolve past it.

Three claims, all on a 3.08M-parameter DeepSeek-style trunk with an
8-slot bank, two seeds:

1. **A functional, generalizing memory**: a single 13-token presentation
   installs a *never-trained* rule binding at **0.79–1.00** accuracy on
   unseen queries (chance 0.008), retained past physical slot eviction,
   replaced mid-conversation in one forward pass (old-rule persistence
   0.000).
2. **The only functional adaptation pathway**: on the same conversations,
   test-time training fits its adaptation examples (0.99) and transfers
   **exactly nothing**, at **138×** the cost per update and −62%
   catastrophic interference on a concurrent rule (bank: −14%, by
   eviction); in-window ICL is at chance.
3. **Memory policy is a trained behaviour, not an architectural
   property**: the identical architecture trained on fixed-structure
   conversations perseverates totally on a rule switch, zero-shot
   (old-rule persistence 1.000, unreadable dirty-bank writes);
   randomizing conversation *structure* at training time installs the
   full policy.

Reproduce Tables 1–4 and Figures 3–5 from a fresh clone:
```bash
bash repro/run_all.sh               # 3 training runs (~5 h each, one RTX 3090) + probes + figures
bash repro/run_all.sh --skip-train  # probes + figures on existing checkpoints
```

The driving questions, in the order they were answered:
1. *Is an external memory bank useful at all, and when?* → only when persistent,
   and judged by `content_gap` ([historical findings](#-findings-historical-memory-as-data-era)).
2. *Can a rule cross turn boundaries as a fast weight?* → yes — after a
   teacher-forced bootstrap breaks the ignore-bank fixed point (§5 of the paper).
3. *Does it generalize to never-trained rules?* → **yes, given rule diversity**:
   held 0.79–1.00 at 112 training rules; ≤25 rules → exactly 0.000 (the read
   memorizes; the transition sits in (25, 112]).
4. *Does the memory POLICY (retain / overwrite / write-on-dirty) come with the
   architecture?* → **no — it is a trained behaviour**: zero-shot on a
   fixed-structure model, STICK = 1.000 (total perseveration); trained with
   randomized structure, STICK = 0.000 at every switch position
   ([current findings](#-findings-fast-weight-memory-current)).

> **History:** the project started as a diffusion / 3D-thought-tensor prototype
> (hence the repo's former name). That line was abandoned for the autoregressive
> fast-weight bank; the old code was removed and remains available in git history.

## 🔬 After the paper: the bank on real data

`main` has moved past the synthetic-rule benchmark: the current line trains
from-scratch models (47M-97M) where the bank is the **only** channel carrying a
real document (Python code / web text) across 512-token chunks, measured by a
deferred-continuation loss. Latest results — +0.85 nats of bank advantage, flat
to 10 chunks deep, shown by inference probes to be **file-specific content in a
recency-weighted superposition** — are documented with exact reproduction
commands in **[FINDINGS.md](FINDINGS.md)**.

---

## 🧠 Core idea

![The Thought Bank architecture](paper/figures/fig1_architecture.png)

- **Text stream** does next-token prediction (CSA/HCA attention + MoE, mHC residuals).
- **The bank is read as fast weights, not attended data**: each slot is expanded
  by a learned hypernet into a low-rank MLP layer; the token stream passes
  *through* the stack of slot-layers. What the model wrote becomes part of its
  own forward pass — a rule inferred at turn 0 can be *applied* at turn 20.
- **Write once per turn** (optionally gated `α·p·m`; gate off in the current
  recipe). The bank is FIFO-capped at `max_mem`.
- The bank is the **only cross-turn channel**: each turn is a separate forward,
  so anything older than the current window must travel through the bank.

Full architecture notes (mHC, CSA, HCA, MoE, thought stream) are in the package
README: [`deepseek_v4_mini/README.md`](deepseek_v4_mini/README.md).

---

## 🚀 Quick start

### Environment
```bash
# conda env used in development
conda activate diffusion-thought          # see setup_environment.sh
pip install -r requirements.txt           # torch, transformers, datasets, ...
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```
Target hardware: a single 24 GB GPU (RTX 3090).

### Train
```bash
# language modelling
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/tiny.yaml      # ~19M, TinyStories
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/code.yaml      # code, per-sequence reset
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/code_persist.yaml  # code, PERSISTENT bank

# memory diagnostics (synthetic, no tokenizer)
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/synth_recall.yaml  # addressable recall
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/gist.yaml          # latent-context gist

# the paper's cells (keyed fresh-rule benchmark, S=128)
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/multiturn_rule_k2_inter_s128_dsv4m.yaml        # fixed structure (zero-shot arm)
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/multiturn_rule_k2_inter_s128struct_dsv4w.yaml  # policy cell, seed 42
python -m deepseek_v4_mini.train deepseek_v4_mini/configs/multiturn_rule_k2_inter_s128struct_dsv4w_s43.yaml  # replication, seed 43

# or everything at once (training + probes + figures):
bash repro/run_all.sh
```
> Scripts importing the package need `PYTHONPATH=<repo-root>`.

### Use the model
```python
from deepseek_v4_mini import ThoughtBankLM, ThoughtBankConfig
import torch

cfg   = ThoughtBankConfig.tiny()
model = ThoughtBankLM(cfg)
ids   = torch.randint(0, cfg.vocab_size, (2, 64))

out  = model(ids)                            # first segment: fresh (random-seed) bank
out2 = model(ids, init_mem=out["mem_bank"])  # carry the bank to the next segment
```

To probe a paper checkpoint instead:
```python
cfg   = ThoughtBankConfig.from_yaml("deepseek_v4_mini/configs/multiturn_rule_k2_inter_s128struct_dsv4w.yaml")
model = ThoughtBankLM(cfg)
model.load_state_dict(torch.load("checkpoints/multiturn_rule_k2_inter_s128_dsv4w/step_3000.pt",
                                 map_location="cpu")["model"])
```

---

## 🔬 Measuring whether the memory helps

Two probes run during training and write to `runs/<run_name>/metrics.jsonl`:

| Metric | Meaning |
|---|---|
| `mem_ablation_gap` | CE without the bank − CE with it, on the same tokens (>0 ⇒ helps) — but ablating removes the **whole** cross-modal pathway, not just content |
| `mem_diversity` | std across bank slots (~0 ⇒ slots collapsed = useless) |
| `mem_write_rate` (α) | mean write probability — does the model choose to write? |
| `persist_gap` | (persistent runs) CE with the bank **carried across chunks** of one file vs reset each chunk. **Conflates content and structure** — see below; kept as the legacy headline |
| **`content_gap`** | **the metric to trust**: CE with writes **zeroed** vs real, slot count held identical — the *pure* benefit of what is written into the bank |
| `structure_gap` | `persist_gap − content_gap`: the part explained by slot count + slot positional embeddings, independent of content |

Offline analysis: [`deepseek_v4_mini/eval_memory.py`](deepseek_v4_mini/eval_memory.py)
(PPL with vs without the bank).

---

## 📊 Findings — fast-weight memory (current)

Benchmark: keyed fresh-rule conversations (`multiturn_rule_k2_inter_s128*`) —
each conversation binds K=2 key tokens to fresh shift rules `y=(x+s)%128`
(112 training offsets / 15 held out, never trained), presents each rule once
(13 tokens), then queries **unseen** symbols on later turns; the rule can only
cross turn boundaries through the bank. Chance 0.008; bank ablation is an
exact control and sits at chance everywhere.

| Question | Verdict |
|---|---|
| Can a fresh rule be installed at inference, forward-only? | **Yes** — one 13-token presentation → 0.95–1.00 (train) on unseen queries |
| Does it generalize to never-trained rules? | **Yes** — held 0.79–1.00 across two seeds, *given diversity*: at ≤25 training rules held is exactly 0.000 (the read memorizes); at 112 rules held tracks train. Transition in (25, 112] |
| How does it compare to test-time training? | TTT on the same conversations fits its 12 adaptation pairs (0.99) and transfers **nothing** (chance on unseen queries, all LRs × 1–50 steps); in-window ICL also at chance. The bank is the **only** functional adaptation pathway, at 1/138th the cost per update |
| Can a rule be replaced mid-conversation? | **Yes** — one forward on the dirty bank: 0.95 train / 0.78 held post-switch, old-rule persistence (STICK) 0.000 at every switch position 2–14; the untouched key loses −14% (eviction pressure) where sequential TTT loses −62% (catastrophic interference) |
| Does FIFO eviction kill retention? | **No cliff** (structure-randomized model) — storage is a redundant superposition: the 8 slots carry near-copies of one superposed vector (bank eff. rank ~1.1–1.5/8, ablation gap +4.6 nats), the key-conditioned read disambiguates; evicting a slot removes a copy, not the content |
| Is the memory policy architectural? | **No — it is trained.** The same architecture at matched held competence, trained on *fixed* structure, perseverates totally zero-shot (STICK 1.000; its write head cannot produce a readable code on a non-empty bank, 1-NN 0.05). Randomizing conversation structure (lengths 8–16, ≤2 switches at random positions) installs the full policy |
| What is out of reach? | A never-trained rule *family* (subtraction on the same circle) defeats bank, TTT and ICL equally — the boundary is the meta-training envelope, not the mechanism. And replacement *selectivity* bifurcates across seeds (selective update vs flush-and-rewrite), decided at bootstrap |

**Headline: the bank is a functional, generalizing, forward-only memory — and
what it *does* (keep, overwrite, write-on-dirty, survive eviction) is decided
by the training distribution, not by the architecture.** Training it requires
breaking an ignore-the-bank fixed point (teacher-forced Fourier bootstrap +
mastery-gated curriculum + rule diversity; §5 and App. E of the
[paper](paper/paper.pdf)). Mechanistic evidence and probe scripts:
[`deepseek_v4_mini/analysis/`](deepseek_v4_mini/analysis/README.md).

> Earlier findings on the ≤25-rule *memorizing* regime (K=1 0.948, K=2 keyed
> 0.99, emergent rehearsal, switch STICK=0 on a switch-trained model) were
> re-audited on the generalizing regime; the ones that survive are folded into
> the table above, the historical arc is in the
> [package README](deepseek_v4_mini/README.md).

---

## 📊 Findings (historical, memory-as-data era)

**The memory bank only earns its keep when it is allowed to *persist* across
sequences.** Resetting it every sequence (the default) makes it look useless.

A/B on the **same architecture and code dataset**, at matched steps:

| Setup | `ablation_gap` | `persist_gap` | slot `diversity` |
|---|---|---|---|
| per-sequence reset (`code.yaml`) | ~+0.02 → +0.10 | — | ~0.15 |
| **persistent** (`code_persist.yaml`) | **+1.0 → +1.8** | **≈ +0.24–0.30 (stable)** | **~0.41** |

### ⚠️ But most of `persist_gap` is structure, not content

A control on the persistent checkpoint (step 2000, averaged over 6 files)
**decomposes** `persist_gap` by zeroing the written content while keeping the
slot count identical:

| Component | Value | Share |
|---|---|---|
| `persist_gap` (carried vs reset) | **+0.236** | 100% |
| **`content_gap`** (pure content) | **+0.077** | **33%** |
| `structure_gap` (slot count / positions) | +0.159 | 67% |

- **The written content genuinely helps — but modestly.** `content_gap = +0.077`
  was positive on all 6 files (0.046–0.095, σ=0.018), so the bank content is not
  noise. But it is small.
- **~2/3 of the headline `persist_gap` is a structural artifact**: a carried bank
  has ~`max_mem` positionally-encoded slots, a reset one rebuilds from empty, and
  that difference alone moves later-chunk CE — even with a **zero-content** bank
  (the sparse run with α≈0 still showed `persist_gap ≈ +0.32`). So `persist_gap`
  overstates the memory's content value by ~3×. **Trust `content_gap`.**
- Likewise `ablation_gap` is inflated: ablating the bank removes the *entire*
  cross-modal pathway, not just the content, so its large value (+1.0–1.8) mostly
  reflects "the pathway exists", not "the stored thoughts are useful".

### Other findings

- On short / locally-redeterminable data (TinyStories, dense contexts) the gap
  stays small: when the relevant "gist" fits in the attention window, the bank is
  redundant. Memory pays off for **non-local** context beyond the window.
- The bank is a **gist/summary** memory, not an addressable key→value store: the
  synthetic `associative_recall` task does *not* get solved by it (slots collapse
  to a single direction). Use it to remember *what is going on broadly*, not to
  recall exact values.
- The write-decision α **saturates to 1.0** (always write) without a cost — the
  write/skip "choice" decides nothing. A sparsity budget (`mem_write_cost`,
  `cost · E[-log(1-α)]`) gives writing an opportunity cost; applied from step 0 it
  over-corrects (α→0), so it needs a warmup. Judge selectivity by `content_gap`
  holding with fewer writes, not by `persist_gap`.

### Things that were required to get here
- **Next-token alignment**: the loaders pre-shift targets, so the loss must *not*
  shift again — a fixed double-shift had been training a +2-token objective.
- **Write-head gradient**: `mem_bptt_window ≥ 2`, otherwise the write head never
  receives gradient and the bank is filled by an untrained projection.
- **NaN stability**: `muon_lr ≈ 0.003` and `sinkhorn_iters = 20` (the bigger
  levers); RMSNorm variance in fp32; Sinkhorn with per-matrix max-subtract.

---

## ⚙️ Configs

| File | Dataset / task | Purpose |
|---|---|---|
| `configs/tiny.yaml` | TinyStories (~19M) | fast LM iteration |
| `configs/small.yaml` | TinyStories (~32M) | single RTX 3090 |
| `configs/code.yaml` | codeparrot (Python) | baseline, bank reset per sequence |
| `configs/code_persist.yaml` | codeparrot (Python) | bank **persists** across steps |
| `configs/synth_recall.yaml` | synthetic | addressable key→value recall test |
| `configs/gist.yaml` | synthetic | latent-context (gist) test |
| `configs/multiturn_rule_k2_inter_s128_dsv4m.yaml` | synthetic | **paper**: fixed-structure cell (Table 2, zero-shot arm of Table 4 / Fig 5) |
| `configs/multiturn_rule_k2_inter_s128struct_dsv4w*.yaml` | synthetic | **paper**: policy cells, seeds 42/43 (Tables 1/3, Figs 3–5) |
| `configs/multiturn_rule*.yaml` (others) | synthetic | historical continual-rule family (K=1/K=2, held-out, horizon, switch, joint) — see the [package README](deepseek_v4_mini/README.md) |

Key memory knobs (full list in [`deepseek_v4_mini/README.md`](deepseek_v4_mini/README.md)):

| Parameter | Description |
|---|---|
| `mem_dim`, `max_mem` | thought-vector size and FIFO bank capacity |
| `mem_segment_len` | attention window; smaller ⇒ more reliance on the bank |
| `mem_bptt_window` | TBPTT span; **≥2 required** to train the write head |
| `mem_probe_every` | how often to run the ablation / persistence probes |
| `mem_write_cost` | sparsity budget on α (`cost · E[-log(1-α)]`); 0 = α free to saturate at 1 |
| `data.persist: true` | per-file ordered lanes + carry the bank across steps |

---

## 📁 Repository layout

```
paper/                   ← the paper (paper.pdf, draft.md, figures/)
repro/                   ← end-to-end reproduction of the paper (run_all.sh)
deepseek_v4_mini/        ← active project (fast-weight thought bank)
  model.py  memory.py  attention.py  moe.py  mhc.py  config.py  train.py
  eval_memory.py         ← offline PPL with/without the bank
  analysis/              ← mechanistic diagnostics + campaign results
  configs/               ← tiny, small, code, code_persist, synth_recall, gist,
                           multiturn_rule family (k2, heldout, horizon, switch, joint)
thought_lm_minimal/      ← minimal thought-LM baseline
checkpoints/, runs/      ← training outputs
```

---

## 📚 References
- DeepSeek-V4 (architecture base), DeepSeekMoE (Dai et al., 2024)
- Hyper-Connections (Zhu et al., 2024), Muon optimizer (Jordan et al., 2024)
- Thought-memory baseline: [`thought_lm_minimal/`](thought_lm_minimal/)

## License

MIT — see [LICENSE](LICENSE). The paper (`paper/`) is distributed under the
arXiv.org perpetual, non-exclusive license.
