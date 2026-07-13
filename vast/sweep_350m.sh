#!/usr/bin/env bash
# dsv6 — sweep batch 350M sur A100 80GB (Vast.ai). Suite du bring-up 4090 :
# OOM au 1er forward (pic conv ~23-24 GB, pas ~14 — le stack cascade+reach+
# layer_banks paie ses graphes de read). Ici : B in {1,2,4}, grad_accum 8/B
# (batch EFFECTIF 8 constant = sémantique recette intacte), 10 steps par
# point, pic VRAM + sec/step par point. Verdict = le plus grand B qui tient
# et son débit → devis N×A100.
set -uo pipefail

WORK=/workspace
REPO_URL=${REPO_URL:-https://github.com/kkuette/thought-bank.git}
BRANCH=${BRANCH:-claude/status-check-2fa903}
BASECFG=deepseek_v4_mini/configs/v350_bringup.yaml
SUMMARY=$WORK/sweep_summary.txt

mkdir -p $WORK && cd $WORK
export HF_HOME=$WORK/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -d thought-bank ]; then
  git clone --depth 1 -b "$BRANCH" "$REPO_URL" thought-bank
fi
cd thought-bank
git pull --ff-only || true
pip install -q -r requirements.txt

echo "=== SWEEP 350M $(date -u +%FT%TZ) — $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ===" | tee "$SUMMARY"

for B in 1 2 4; do
  G=$((8 / B))
  CFG=$WORK/bringup_B${B}.yaml
  LOG=$WORK/bringup_B${B}.log
  VRAMLOG=$WORK/vram_B${B}.csv
  sed -e "s/batch_size: 1/batch_size: $B/" \
      -e "s/grad_accum: 8/grad_accum: $G/" \
      -e "s/steps: 40/steps: 10/" \
      -e "s/eval_every: 20/eval_every: 1000/" \
      -e "s#/workspace/checkpoints/v350_bringup#/workspace/ckpt_B${B}#" \
      -e "s#/workspace/runs/v350_bringup#/workspace/runs_B${B}#" \
      "$BASECFG" > "$CFG"

  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "$VRAMLOG" &
  SMI_PID=$!
  T0=$(date +%s)
  python -m deepseek_v4_mini.code_defer_native "$CFG" > "$LOG" 2>&1
  RC=$?
  T1=$(date +%s)
  kill $SMI_PID 2>/dev/null

  PEAK=$(awk 'BEGIN{m=0}{if($1+0>m)m=$1+0}END{print m}' "$VRAMLOG")
  if [ $RC -eq 0 ]; then
    STATUS="OK sec_total=$((T1-T0)) (10 steps, warm-cache des B suivants ; net = timestamps steps 3->10 du log)"
  elif grep -q "OutOfMemoryError" "$LOG"; then
    STATUS="OOM"
  else
    STATUS="FAIL rc=$RC (voir $LOG)"
  fi
  echo "B=$B G=$G : $STATUS | vram_peak_mb=$PEAK" | tee -a "$SUMMARY"
  # OOM à ce B => inutile d'essayer plus grand
  [ "$STATUS" = "OOM" ] && break
done

echo "=== SWEEP DONE ===" | tee -a "$SUMMARY"
cat "$SUMMARY"
