# Diffusion Thought Tensor

Research repo exploring **persistent thought memory** for language models. The
active line of work is **`deepseek_v4_mini`**: a small reproduction of the
DeepSeek-V4 architecture fused with a **dual-stream thought-memory bank** — a text
stream plus a second stream that maintains a compressed "gist" of what has already
happened and carries it forward.

The driving question: *is an external memory bank actually useful, and when?*
This session answered it (see [Findings](#-findings)).

> **Legacy:** the earlier diffusion / 3D-thought-tensor prototype lives under
> [`diffusion_thought_tensor/`](diffusion_thought_tensor/) and [`docs/old/`](docs/old/).
> It is no longer the focus and is kept only for reference.

---

## 🧠 Core idea

```
        thought stream  [B, M, mem_dim]         (the running "gist" / memory bank)
                 │  cross-modal at every layer
                 ▼
        text stream     [B, T, d_model]         (predicts the next token)
                 │
                 ▼
        gated write  α·p·m   →   append to bank   →   FIFO-evict oldest
                 │
        bank carried to the next segment / sequence
```

- **Text stream** does next-token prediction (CSA/HCA attention + MoE, mHC residuals).
- **Thought stream** is a second small transformer over the memory bank
  `[B, M, mem_dim]`, slot index = temporal position (slot 0 oldest, slot M-1 newest).
- **Gated write**: each segment produces a candidate thought `α·p·m` where the
  scalar `α = sigmoid(write_decision)` is the model's *write/skip* choice and `p`
  is a per-dimension content gate. The bank is FIFO-capped at `max_mem`.
- The bank is the **only cross-segment channel**: the text stream's attention is
  capped to one `mem_segment_len` window, so anything older must travel through
  the bank.

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
```
> Scripts importing the package need `PYTHONPATH=<repo-root>`.

### Use the model
```python
from deepseek_v4_mini import DualModalDeepSeekV4Mini, DeepSeekV4MiniConfig
import torch

cfg   = DeepSeekV4MiniConfig.tiny()
model = DualModalDeepSeekV4Mini(cfg)
ids   = torch.randint(0, cfg.vocab_size, (2, 64))

out  = model(ids)                            # first pass: empty bank
out2 = model(ids, init_mem=out["mem_bank"])  # carry the bank forward
```

---

## 🔬 Measuring whether the memory helps

Two probes run during training and write to `runs/<run_name>/metrics.jsonl`:

| Metric | Meaning |
|---|---|
| `mem_ablation_gap` | CE without the bank − CE with it, on the same tokens (>0 ⇒ helps) |
| `mem_diversity` | std across bank slots (~0 ⇒ slots collapsed = useless) |
| `mem_write_rate` (α) | mean write probability — does the model choose to write? |
| `persist_gap` | (persistent runs) CE with the bank **carried across chunks** of one file vs reset each chunk, on later chunks — the clean cross-sequence verdict |

Offline analysis: [`deepseek_v4_mini/eval_memory.py`](deepseek_v4_mini/eval_memory.py)
(PPL with vs without the bank).

---

## 📊 Findings

**The memory bank only earns its keep when it is allowed to *persist* across
sequences.** Resetting it every sequence (the default) makes it look useless.

A/B on the **same architecture and code dataset**, at matched steps:

| Setup | `ablation_gap` | `persist_gap` | slot `diversity` |
|---|---|---|---|
| per-sequence reset (`code.yaml`) | ~+0.02 → +0.10 | — | ~0.15 |
| **persistent** (`code_persist.yaml`) | **+1.0 → +1.8** | **≈ +0.30 (stable)** | **~0.41** |

- **`persist_gap ≈ +0.30`, stable** across probes — carrying the bank across
  chunks of the same file consistently lowers later-chunk loss. This is the real
  "remember earlier context" use case, and it works.
- On short / locally-redeterminable data (TinyStories, dense contexts) the gap
  stays small: when the relevant "gist" fits in the attention window, the bank is
  redundant. Memory pays off for **non-local** context beyond the window.
- The bank is a **gist/summary** memory, not an addressable key→value store: the
  synthetic `associative_recall` task does *not* get solved by it (slots collapse
  to a single direction). Use it to remember *what is going on broadly*, not to
  recall exact values.

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

Key memory knobs (full list in [`deepseek_v4_mini/README.md`](deepseek_v4_mini/README.md)):

| Parameter | Description |
|---|---|
| `mem_dim`, `max_mem` | thought-vector size and FIFO bank capacity |
| `mem_segment_len` | attention window; smaller ⇒ more reliance on the bank |
| `mem_bptt_window` | TBPTT span; **≥2 required** to train the write head |
| `mem_probe_every` | how often to run the ablation / persistence probes |
| `data.persist: true` | per-file ordered lanes + carry the bank across steps |

---

## 📁 Repository layout

```
deepseek_v4_mini/        ← active project (dual-stream thought memory)
  model.py  memory.py  attention.py  moe.py  mhc.py  config.py  train.py
  eval_memory.py         ← offline PPL with/without the bank
  configs/               ← tiny, small, code, code_persist, synth_recall, gist
thought_lm_minimal/      ← minimal thought-LM baseline
diffusion_thought_tensor/, docs/old/   ← legacy diffusion prototype (reference only)
checkpoints/, runs/      ← training outputs
```

---

## 📚 References
- DeepSeek-V4 (architecture base), DeepSeekMoE (Dai et al., 2024)
- Hyper-Connections (Zhu et al., 2025), Muon optimizer (Jordan et al., 2024)
- Thought-memory baseline: [`thought_lm_minimal/`](thought_lm_minimal/)
