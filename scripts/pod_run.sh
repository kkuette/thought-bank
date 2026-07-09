#!/usr/bin/env bash
# dsv6 pod kit — launch (or auto-resume) a code_defer_native run on a rented GPU.
#
#   bash scripts/pod_run.sh deepseek_v4_mini/configs/code_defer_native_350m.yaml
#
# Preemption-safe: always passes --resume (fresh start if no checkpoint), and
# relaunches after a crash/preemption until final.pt exists (capped retries so a
# deterministic failure doesn't burn money). Tokenized corpus is disk-cached by
# code_data.py (data_cache/) — a restarted pod skips the 10-25 min tokenize pass
# if data_cache/ is on the persistent volume.
#
# Pod checklist: repo + data_cache/ + checkpoints/ on the PERSISTENT volume;
# HF_HOME too if you want the raw dataset cache to survive restarts.
set -uo pipefail
cd "$(dirname "$0")/.."

CFG=${1:?usage: pod_run.sh <config.yaml>}
NAME=$(basename "$CFG" .yaml)
PY=${PYTHON:-python}
MAX_RETRIES=${MAX_RETRIES:-50}

$PY -c "import torch, transformers, datasets, yaml, tensorboard, tqdm" 2>/dev/null || \
    pip install -q torch transformers datasets pyyaml tensorboard tqdm

SAVE_DIR=$($PY -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['training']['save_dir'])" "$CFG")
mkdir -p "runs/$NAME" "$SAVE_DIR"

tries=0
while [ ! -f "$SAVE_DIR/final.pt" ]; do
    tries=$((tries + 1))
    if [ "$tries" -gt "$MAX_RETRIES" ]; then
        echo "[pod_run] $MAX_RETRIES retries exhausted — deterministic failure? bailing." \
            | tee -a "runs/$NAME/train.log"
        exit 1
    fi
    echo "[pod_run] attempt $tries — $(date -u +%FT%TZ)" | tee -a "runs/$NAME/train.log"
    PYTHONUNBUFFERED=1 $PY -m deepseek_v4_mini.code_defer_native "$CFG" --resume \
        2>&1 | tee -a "runs/$NAME/train.log"
    [ -f "$SAVE_DIR/final.pt" ] && break
    echo "[pod_run] exited without final.pt — retry in 30s" | tee -a "runs/$NAME/train.log"
    sleep 30
done
echo "[pod_run] final.pt present — done."
