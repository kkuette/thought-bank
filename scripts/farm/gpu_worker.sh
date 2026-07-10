#!/usr/bin/env bash
# Worker : une instance par GPU (tb-worker@N). Boucle : prend atomiquement le
# prochain job de $TB_MNT/queue/, l'exécute sur son GPU, range en done/failed.
#
# Un job = un fichier bash "*.job" déposé dans queue/. Il s'exécute avec :
#   CUDA_VISIBLE_DEVICES  déjà fixé (le job voit UN seul GPU, index 0)
#   TB_MNT (/mnt/tb), TB_REPO, WORKER (hostname-gpuN), cwd = TB_REPO
#   le venv actif
# Convention : checkpoints/logs sous $TB_MNT/checkpoints et $TB_MNT/runs.
set -uo pipefail

GPU_ID="${GPU_ID:?}"
TB_MNT="${TB_MNT:-/mnt/tb}"
TB_REPO="${TB_REPO:-/opt/thought-bank}"
TB_VENV="${TB_VENV:-/opt/tb-venv}"
WORKER="$(hostname)-gpu${GPU_ID}"
Q="$TB_MNT/queue"

echo "[$WORKER] demarrage, file: $Q"
while true; do
  job=""
  # tri lexicographique => on peut prioriser en nommant 00_xxx.job, 10_yyy.job
  for f in $(ls "$Q"/*.job 2>/dev/null | sort); do
    claimed="$Q/running/${WORKER}__$(basename "$f")"
    if mv "$f" "$claimed" 2>/dev/null; then   # mv NFS = atomique : un seul worker gagne
      job="$claimed"; break
    fi
  done

  if [ -z "$job" ]; then sleep 20; continue; fi

  name="$(basename "$job" .job)"
  log="$TB_MNT/runs/${name}.workerlog"
  echo "[$WORKER] job: $name" | tee -a "$log"
  (
    export CUDA_VISIBLE_DEVICES="$GPU_ID" TB_MNT TB_REPO WORKER
    source "$TB_VENV/bin/activate"
    cd "$TB_REPO"
    bash "$job"
  ) >> "$log" 2>&1
  rc=$?

  if [ $rc -eq 0 ]; then
    mv "$job" "$Q/done/"
    echo "[$WORKER] OK: $name" | tee -a "$log"
  else
    mv "$job" "$Q/failed/"
    echo "[$WORKER] ECHEC rc=$rc: $name (voir $log)" | tee -a "$log"
    sleep 60   # carte peut-etre en cause : ne pas devorer la file en boucle
  fi
done
