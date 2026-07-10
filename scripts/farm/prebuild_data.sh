#!/usr/bin/env bash
# Data server (server0/Unraid) : pré-tokenise des configs dans le cache partagé,
# via Docker (CPU only, aucun GPU requis). À lancer depuis le terminal Unraid.
#
# Usage : ./prebuild_data.sh deepseek_v4_mini/configs/code_defer_native_v2c_varlen.yaml [...]
#   (chemins de configs relatifs à la racine du repo)
#
# Le HF cache partagé (data/hf_cache, déjà peuplé) sert aux téléchargements :
# les datasets déjà vus par la 3090 ne sont PAS re-téléchargés.
set -euo pipefail

SHARE=/mnt/user/llm_research
[ $# -ge 1 ] || { echo "usage: $0 <config.yaml> [...]"; exit 1; }

docker run --rm \
  -v "$SHARE":/tb \
  -e HF_HOME=/tb/data/hf_cache \
  python:3.11-slim bash -c "
    set -e
    apt-get update -qq && apt-get install -y -qq git > /dev/null
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    pip install -q transformers datasets pyyaml tqdm
    git clone -q --depth 1 https://github.com/kkuette/thought-bank /opt/tb
    cd /opt/tb
    python scripts/farm/prebuild_data.py --cache-dir /tb/data_cache $*
  "
