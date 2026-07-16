#!/usr/bin/env bash
# dsv6 — bring-up 350M sur Vast.ai (mesure débit + pic VRAM, PAS un entraînement).
# À utiliser comme script onstart d'une instance Vast (image pytorch), ou à la
# main via ssh. Sortie finale : bloc "=== BRINGUP RESULT ===" avec steps/h et
# pic VRAM — les 2 nombres du devis 350M.
#
# Prérequis instance : GPU >= 24 GB (4090), disque >= 60 GB, image avec
# CUDA + python3 (ex : pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime).
set -uo pipefail

WORK=/workspace
REPO_URL=${REPO_URL:-https://github.com/kkuette/thought-bank.git}
BRANCH=${BRANCH:-claude/status-check-2fa903}
CFG=deepseek_v4_mini/configs/v350_bringup.yaml
LOG=$WORK/bringup.log
VRAMLOG=$WORK/vram_samples.csv

mkdir -p $WORK && cd $WORK
export HF_HOME=$WORK/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- setup (idempotent : relançable après préemption/retry)
if [ ! -d thought-bank ]; then
  git clone --depth 1 -b "$BRANCH" "$REPO_URL" thought-bank
fi
cd thought-bank
git pull --ff-only || true
pip install -q -r requirements.txt

# --- échantillonneur VRAM (le trainer ne logge pas la mémoire : on échantillonne
# nvidia-smi à 5 s — le pic process serait plus précis mais ceci suffit au devis)
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -l 5 > "$VRAMLOG" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null' EXIT

# --- le run de mesure (40 steps, ~1 éval au milieu)
T0=$(date +%s)
python -m deepseek_v4_mini.code_defer_native "$CFG" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
T1=$(date +%s)
kill $SMI_PID 2>/dev/null

# --- dépouillement
ELAPSED=$((T1 - T0))
PEAK_MB=$(awk -F', ' 'BEGIN{m=0} {if ($2+0>m) m=$2+0} END{print m}' "$VRAMLOG")
STEPS=$(grep -cE "step[ =]" "$LOG" || true)

echo ""
echo "=== BRINGUP RESULT ==="
echo "exit_code:      $RC"
echo "wall_seconds:   $ELAPSED (40 steps cible, download+cache inclus)"
echo "sec_per_step:   ~$((ELAPSED / 40)) (BRUT — soustraire le warm-up des caches :"
echo "                comparer plutôt les timestamps des lignes step 5 -> 40 dans $LOG)"
echo "vram_peak_mb:   $PEAK_MB"
echo "gpu:            $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "======================"
echo "Verdict devis : steps/h = 3600 / sec_per_step_net ; 2000 steps => heures ;"
echo "si > ~60 h de 4090, passer A100 80GB B=4 (cf. FINDINGS scaling)."
