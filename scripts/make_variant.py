"""dsv6 pod kit — derive a DATASET-VARIANT config from a base yaml.

    python scripts/make_variant.py deepseek_v4_mini/configs/code_defer_native_350m.yaml \
        bigcode/the-stack-dedup --data-dir data/python --suffix stack_py

Writes deepseek_v4_mini/configs/<base>_<suffix>.yaml with dataset/data_dir swapped
and save_dir/metrics_file suffixed, everything else (the validated recipe) untouched.
One variant = one pod: `bash scripts/pod_run.sh <variant>.yaml`.
"""
import argparse
import os
import sys

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="base config yaml (the validated recipe)")
    ap.add_argument("dataset", help="HF dataset id (streamed, needs a 'content' column)")
    ap.add_argument("--data-dir", default="", help="optional dataset data_dir (subset)")
    ap.add_argument("--config-name", default="", help="HF config name (e.g. fineweb 'sample-10BT' or a CC dump)")
    ap.add_argument("--content-key", default="", help="text column (fineweb: 'text'; default 'content')")
    ap.add_argument("--min-chunks", type=int, default=0, help="keep only docs with >= N chunks (web text: 2)")
    ap.add_argument("--stream-skip", type=int, default=0, help="skip first N docs (per-pod shard offset)")
    ap.add_argument("--suffix", default="", help="name suffix (default: derived from dataset)")
    ap.add_argument("--n-files", type=int, default=0, help="override data.n_files")
    ap.add_argument("--stream-cap", type=int, default=0, help="override data.stream_cap")
    a = ap.parse_args()

    raw = yaml.safe_load(open(a.base))
    suffix = a.suffix or a.dataset.split("/")[-1].replace("-", "_")
    base_name = os.path.basename(a.base)[: -len(".yaml")]

    raw["data"]["dataset"] = a.dataset
    raw["data"]["data_dir"] = a.data_dir
    if a.config_name:
        raw["data"]["config_name"] = a.config_name
    if a.content_key:
        raw["data"]["content_key"] = a.content_key
    if a.min_chunks:
        raw["data"]["min_chunks"] = a.min_chunks
    if a.stream_skip:
        raw["data"]["stream_skip"] = a.stream_skip
        suffix += f"_skip{a.stream_skip}"
    if a.n_files:
        raw["data"]["n_files"] = a.n_files
    if a.stream_cap:
        raw["data"]["stream_cap"] = a.stream_cap
    name = f"{base_name}_{suffix}"
    raw["training"]["save_dir"] = f"checkpoints/{name}"
    raw["training"]["metrics_file"] = f"runs/{name}/metrics.jsonl"

    out = os.path.join(os.path.dirname(a.base), name + ".yaml")
    if os.path.exists(out):
        sys.exit(f"refusing to overwrite existing {out}")
    with open(out, "w") as f:
        f.write(f"# AUTO-DERIVED from {os.path.basename(a.base)} — dataset variant "
                f"'{a.dataset}'{' / ' + a.data_dir if a.data_dir else ''}.\n"
                f"# Recipe untouched; only dataset/save_dir/metrics_file differ.\n")
        yaml.safe_dump(raw, f, sort_keys=False)
    print(f"wrote {out}\n  -> bash scripts/pod_run.sh {out}")


if __name__ == "__main__":
    main()
