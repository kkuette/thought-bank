#!/usr/bin/env python3
"""Pré-tokenise les corpus d'une (ou plusieurs) config d'entraînement dans le
cache partagé du NAS — à lancer sur le data server (server0), PAS sur un GPU.

Rejoue EXACTEMENT la construction de données du trainer (mêmes clés de cache
md5 : dataset, split, seq_len, n_files, tokenizer, …) via CodeChunkStream,
en pointant cache_dir vers le share. Ensuite, tout run (3090 ou rig) dont la
config data est identique ET qui utilise le même cache_dir démarre sur un
cache hit au lieu de 10-25 min de tokenisation.

Usage :
  python scripts/farm/prebuild_data.py [--cache-dir /tb/data_cache] cfg1.yaml [cfg2.yaml ...]

Note : les configs d'entraînement destinées à la ferme doivent déclarer
`data.cache_dir: /mnt/tb/data_cache` (le défaut du trainer est un data_cache/
local, qui raterait ce cache partagé).
"""
from __future__ import annotations

import argparse
import os
import sys

# racine du repo : TB_REPO si défini (cas : script lancé depuis la copie NAS,
# hors du repo), sinon déduite du chemin du script (cas : copie dans le repo).
REPO = os.environ.get(
    "TB_REPO",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

import yaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from deepseek_v4_mini.code_data import CodeChunkStream  # noqa: E402


def prebuild(cfg_path: str, cache_dir: str) -> None:
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    # miroir du trainer : <think>/<blank> ajoutés AVANT la construction des données
    # (len(tokenizer) fait partie de la clé de cache md5 — sans ça, cache miss).
    add = [x for x in ("<think>", "<blank>") if x not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    d = raw["data"]
    t = raw.get("training", {}) or {}
    # miroir du bloc `sd` de code_defer_native.py — seules différences : batch
    # forcé à 1 (sans effet sur le cache, évite l'assert defer-pair) et
    # cache_dir imposé.
    sd = dict(seq_len=int(d["seq_len"]), chunks_per_conv=int(d["chunks_per_conv"]),
              batch=1,
              n_files=int(d.get("n_files", 800)),
              dataset=d.get("dataset", "codeparrot/codeparrot-clean-valid"),
              data_dir=d.get("data_dir", ""), stream_cap=int(d.get("stream_cap", 60000)),
              cache_dir=cache_dir,
              content_key=d.get("content_key", "content"),
              config_name=d.get("config_name", ""),
              min_chunks=int(d.get("min_chunks", 1)),
              stream_skip=int(d.get("stream_skip", 0)),
              sources=d.get("sources"),
              var_chunk=d.get("var_chunk"),
              seed=int(t.get("seed", 0)))
    for split in ("train", "held"):
        s = CodeChunkStream(tok, split=split, **sd)
        print(f"[{os.path.basename(cfg_path)}] {split}: {s.n_files} files / "
              f"{s.n_chunk} chunks -> cache {cache_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="+", help="config(s) YAML d'entraînement")
    ap.add_argument("--cache-dir", default="/tb/data_cache",
                    help="cache partagé (défaut : /tb/data_cache, vue conteneur)")
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    for c in args.configs:
        prebuild(c, args.cache_dir)
    print("prebuild: OK", flush=True)


if __name__ == "__main__":
    main()
