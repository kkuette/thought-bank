#!/usr/bin/env bash
# Boot pod v350_sif_repass — 8xA100, image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel.
# Prérequis : HF_TOKEN dans l'env (passé au create pod). Rejouable (idempotent).
set -euo pipefail
COMMIT=c0132c8
CFG=deepseek_v4_mini/configs/v350_sif_repass.yaml

export DEBIAN_FRONTEND=noninteractive
mkdir -p /workspace/data_cache /workspace/ckpts /workspace/checkpoints /workspace/runs

pip install -q -U "huggingface_hub" transformers datasets pyyaml tensorboard tqdm

# 1. code épinglé
if [ ! -d /workspace/thought-bank/.git ]; then
  git clone --quiet https://github.com/kkuette/thought-bank /workspace/thought-bank
fi
git -C /workspace/thought-bank fetch --all --quiet
git -C /workspace/thought-bank checkout --quiet "$COMMIT"

# 2. cache data 10B (tar HF privé) — no-op si déjà extrait
if [ ! -e /workspace/data_cache/.ok ]; then
  python - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download("kkuette/tb-cache-v350-phase1", "data_cache_v350p1.tar",
                    repo_type="dataset", local_dir="/workspace")
print("tar:", p)
PY
  # le tar peut embarquer un préfixe data_cache/ ou pas : on gère les deux
  if tar -tf /workspace/data_cache_v350p1.tar | head -1 | grep -q "^data_cache/"; then
    tar -xf /workspace/data_cache_v350p1.tar -C /workspace
  else
    tar -xf /workspace/data_cache_v350p1.tar -C /workspace/data_cache
  fi
  touch /workspace/data_cache/.ok
  rm -f /workspace/data_cache_v350p1.tar
fi
ls /workspace/data_cache | head -5

# 3. release phase 1 AVEC optim
if [ ! -f /workspace/ckpts/v350_phase1_release_final.pt ]; then
  python - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download("kkuette/tb-v350-phase1-ckpt", "final.pt",
                    local_dir="/workspace/ckpts")
print("ckpt:", p)
PY
  mv /workspace/ckpts/final.pt /workspace/ckpts/v350_phase1_release_final.pt
fi

# 4. run (preemption-safe : --resume = fresh start si aucun ckpt)
cd /workspace/thought-bank
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 \
  -m deepseek_v4_mini.code_defer_native "$CFG" --resume \
  2>&1 | tee -a /workspace/runs/v350_sif_repass/train.log

# 5. rapatriement : ckpts jalons + final vers HF privé (repo dédié)
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi()
repo = "kkuette/tb-v350-sif-repass"
api.create_repo(repo, private=True, exist_ok=True)
d = "/workspace/checkpoints/v350_sif_repass"
for f in sorted(os.listdir(d)):
    if f.endswith(".pt"):
        print("upload", f)
        api.upload_file(path_or_fileobj=os.path.join(d, f), path_in_repo=f, repo_id=repo)
api.upload_file(path_or_fileobj="/workspace/runs/v350_sif_repass/train.log",
                path_in_repo="train.log", repo_id=repo)
api.super_squash_history(repo)
print("done — https://huggingface.co/" + repo)
PY
