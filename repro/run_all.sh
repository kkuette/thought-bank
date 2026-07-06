#!/usr/bin/env bash
# End-to-end reproduction of the paper's Tables 2-4 and Figures 3-5 from a
# fresh clone. Three training runs (GPU, ~5 h each on one RTX 3090), then
# probes and figures (CPU).
#
# Usage:
#   bash repro/run_all.sh              # everything
#   bash repro/run_all.sh --skip-train # probes + figures on existing ckpts
#
# Requirements: the conda env of setup_environment.sh (torch, yaml,
# matplotlib; svglib+reportlab+pymupdf only for re-rendering Figs 1-2).
# Run from the repository root. Outputs land in repro/out/.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=${PY:-python}
OUT=repro/out
mkdir -p "$OUT"

CFG=deepseek_v4_mini/configs
AN=deepseek_v4_mini/analysis
CKPT_M=checkpoints/multiturn_rule_k2_inter_s128_dsv4m/final.pt
CKPT_W=checkpoints/multiturn_rule_k2_inter_s128_dsv4w/step_3000.pt
CKPT_S43=checkpoints/multiturn_rule_k2_inter_s128_dsv4w_s43/final.pt
CFG_W=$CFG/multiturn_rule_k2_inter_s128struct_dsv4w.yaml
CFG_S43=$CFG/multiturn_rule_k2_inter_s128struct_dsv4w_s43.yaml

# ---------------------------------------------------------------- training
if [[ "${1:-}" != "--skip-train" ]]; then
  echo "=== [1/3] fixed-structure cell (dsv4m, seed 42) — Table 3 + Fig 5 zero-shot arm"
  $PY -m deepseek_v4_mini.train $CFG/multiturn_rule_k2_inter_s128_dsv4m.yaml

  echo "=== [2/3] policy cell (dsv4w, seed 42) — Tables 2/4, Figs 3-5"
  $PY -m deepseek_v4_mini.train $CFG_W

  echo "=== [3/3] policy cell (dsv4w_s43, seed 43) — Tables 2, Figs 3-5"
  $PY -m deepseek_v4_mini.train $CFG_S43
fi

for f in $CKPT_M $CKPT_W $CKPT_S43; do
  [[ -f $f ]] || { echo "missing checkpoint: $f (run without --skip-train)"; exit 1; }
done

# ------------------------------------------------------------------ probes
echo "=== Table 3 — bank vs TTT vs ICL vs ablate (held pool + subtraction), dsv4m"
$PY $AN/ttt_demo.py $CKPT_M          | tee "$OUT/table3_held.txt"
$PY $AN/ttt_demo.py $CKPT_M --train-pool | tee "$OUT/table3_train.txt"
$PY $AN/ttt_demo.py $CKPT_M --sub    | tee "$OUT/table3_subtraction.txt"

echo "=== Table 4 — replacement under concurrent load, bank vs sequential TTT, dsv4w@3000"
$PY $AN/ttt_demo_act2.py $CKPT_W         | tee "$OUT/table4_train.txt"
$PY $AN/ttt_demo_act2.py $CKPT_W --held  | tee "$OUT/table4_held.txt"

echo "=== Table 2 + Fig 5 — switch arms and position sweeps, all three models"
$PY $AN/switch_probe_k2.py $CKPT_M                    | tee "$OUT/switch_arms_dsv4m.txt"
$PY $AN/switch_probe_k2.py $CKPT_W   --cfg $CFG_W     | tee "$OUT/switch_arms_dsv4w.txt"
$PY $AN/switch_probe_k2.py $CKPT_S43 --cfg $CFG_S43   | tee "$OUT/switch_arms_s43.txt"
$PY $AN/switch_probe_k2.py $CKPT_M                  --sweep --dump "$OUT/sweep_dsv4m.json"
$PY $AN/switch_probe_k2.py $CKPT_W   --cfg $CFG_W   --sweep --dump "$OUT/sweep_dsv4w.json"
$PY $AN/switch_probe_k2.py $CKPT_S43 --cfg $CFG_S43 --sweep --dump "$OUT/sweep_dsv4w_s43.json"

echo "=== Fig 4 — superposition probe, both policy seeds"
$PY $AN/superposition_probe.py $CKPT_W   --cfg $CFG_W   --dump "$OUT/fig4_dsv4w.json"
$PY $AN/superposition_probe.py $CKPT_S43 --cfg $CFG_S43 --dump "$OUT/fig4_dsv4w_s43.json"

# ----------------------------------------------------------------- figures
echo "=== Figures 3-5"
$PY paper/figures/make_fig3.py --runs-root runs --out "$OUT/fig3_training_dynamics"
$PY paper/figures/make_fig4.py "$OUT/fig4_dsv4w.json" "$OUT/fig4_dsv4w_s43.json" \
    --out "$OUT/fig4_superposition"
$PY paper/figures/make_fig5.py "$OUT/sweep_dsv4m.json" "$OUT/sweep_dsv4w.json" \
    "$OUT/sweep_dsv4w_s43.json" --out "$OUT/fig5_policy_trained"

echo "done — outputs in $OUT/"
