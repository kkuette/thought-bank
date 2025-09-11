#!/usr/bin/env bash
set -euo pipefail
python -m thought_lm.train "${1:-configs/default.yaml}"

