#!/usr/bin/env bash
# Agent de statut d'un nœud de la ferme : écrit toutes les 10 s un JSON
# (GPU, RAM, swap, load) dans $TB_MNT/status/<hostname>.json.
# Le dashboard central (farm_dashboard.py, VM data) agrège ces fichiers —
# ajouter un rig = démarrer cet agent dessus, rien d'autre.
set -u
TB_MNT="${TB_MNT:-/mnt/tb}"
OUT_DIR="$TB_MNT/status"
HOST="$(hostname)"
mkdir -p "$OUT_DIR"

while true; do
  ts=$(date +%s)
  gpus=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
         --format=csv,noheader,nounits 2>/dev/null | \
         awk -F', ' '{printf "%s{\"i\":%s,\"util\":%s,\"vram\":%s,\"vram_tot\":%s,\"w\":%s,\"temp\":%s}", (NR>1?",":""), $1,$2,$3,$4,$5,$6}')
  read -r mt mu <<<"$(free -m | awk '/^Mem:/{print $2, $3}')"
  read -r st su <<<"$(free -m | awk '/^Swap:/{print $2, $3}')"
  load=$(cut -d' ' -f1 /proc/loadavg)
  tmp="$OUT_DIR/.${HOST}.json.tmp"
  printf '{"host":"%s","ts":%s,"load":%s,"mem_mb":[%s,%s],"swap_mb":[%s,%s],"gpus":[%s]}\n' \
    "$HOST" "$ts" "$load" "$mu" "$mt" "$su" "$st" "$gpus" > "$tmp"
  mv -f "$tmp" "$OUT_DIR/$HOST.json"
  sleep 10
done
