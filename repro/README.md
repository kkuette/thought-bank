# Reproducing the paper

`run_all.sh` reproduces Tables 2–4 and Figures 3–5 of the paper from a
fresh clone:

```bash
bash repro/run_all.sh               # 3 training runs (GPU) + probes + figures
bash repro/run_all.sh --skip-train  # probes + figures on existing checkpoints
```

## What it runs

| stage | artifact | cell | hardware |
|---|---|---|---|
| train `..._s128_dsv4m.yaml` | fixed-structure model (Table 3, Fig 5 zero-shot arm) | seed 42, 4000 steps | ~5 h on one RTX 3090 |
| train `..._s128struct_dsv4w.yaml` | policy model (Tables 2/4, Figs 3–5) | seed 42; paper uses step 3000 | ~5 h |
| train `..._s128struct_dsv4w_s43.yaml` | policy model, replication | seed 43, 4000 steps | ~5 h |
| `analysis/ttt_demo.py` | Table 3 (bank / TTT / ICL / ablate; held, train, subtraction) | dsv4m final | GPU if available, else CPU |
| `analysis/ttt_demo_act2.py` | Table 4 (replacement vs sequential TTT) | dsv4w@3000 | GPU if available, else CPU |
| `analysis/switch_probe_k2.py` (+`--sweep --dump`) | Table 2 switch rows, Figure 5 | all three models | GPU if available, else CPU |
| `analysis/superposition_probe.py` | Figure 4 | both policy seeds | GPU if available, else CPU |
| `paper/figures/make_fig{3,4,5}.py` | Figures 3–5 (png+pdf) | — | CPU |

Outputs land in `repro/out/`. Figures 1–2 are hand-drawn SVG masters
(`paper/figures/fig{1,2}_*.svg`), re-rendered with
`paper/figures/make_fig{1,2}.py` (needs `svglib reportlab pymupdf`).

## Determinism

Data generators are seeded from the configs (seeds 42/43); every probe
sets `torch.manual_seed(0)` and builds its conversations with the CPU
RNG, so the evaluation data is identical on CPU and GPU. The bank's
random seed slots are drawn on the model's device, whose RNG stream
differs between CPU and CUDA — probe numbers therefore shift by a few
tenths of a point across devices (e.g. Table 3 held bank 0.799 CPU vs
0.793 CUDA). Training on GPU is deterministic up to
cuDNN/atomics noise: expect the paper's numbers to within a few points,
with the documented seed-level bifurcation (§9) — the *selectivity of
replacement* is basin-dependent, and a re-run of either seed may land in
either attractor. Checkpoints are saved every 100 steps; the paper's
probes read `dsv4m/final.pt`, `dsv4w/step_3000.pt`, `dsv4w_s43/final.pt`.

## Environment

`setup_environment.sh` creates the conda env (python 3.10, torch, yaml,
matplotlib). Override the interpreter with `PY=... bash repro/run_all.sh`.
