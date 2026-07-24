"""VRAM smoke — 386M ROLLOUT (inference only) on an 8 GB farm card.

The farm's 97M-OK / 135M-KO frontier is a TRAINING frontier; the
disaggregated GRPO plan (rl_disagg) only needs INFERENCE on the 3070Ti:
write-policy rollout over a full session + greedy decode of the call turn,
bank + cascade carried. This script measures exactly that path — the same
rollout()/decode_lb the worker runs — at the WORST-CASE session shape of the
config (every chunk at max var_chunk length), with random weights (VRAM does
not care about the values; no checkpoint download on the rig).

Verdict line per dtype: peak VRAM for one GRPO group (G sequential rollouts,
banks of the group resident together, as in the worker) + decode. Green if
fp32 OR bf16 fits well under 8 GB with margin for fragmentation.

  python -m deepseek_v4_mini.vram_smoke_350m \
      deepseek_v4_mini/configs/rl_disagg_350m.yaml

Farm job: scripts/farm/vram_smoke_350m.job (copy into /mnt/tb/queue/).
"""
from __future__ import annotations

import random as _random
import sys
import time

import torch
import yaml

from .cascade import CascadeMemory
from .config import ThoughtBankConfig
from .model import ThoughtBankLM
from .rl_defer_grpo_lives import _lb, rollout
from .rl_disagg import decode_lb
from .rl_lives import mem_fork


def measure(raw: dict, dtype: torch.dtype, device) -> dict:
    r, d, mcfg = raw["rl"], raw["data"], dict(raw["model"])
    torch.manual_seed(0)
    rng = _random.Random(0)
    mcfg["vocab_size"] = int(raw.get("vocab_size", 49154))  # SmolLM2 + 2 spéciaux
    model = ThoughtBankLM(ThoughtBankConfig(**mcfg)).to(device=device,
                                                        dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_par = model.num_params()

    # worst-case session: chunks_per_conv turns, ALL at the max chunk length
    L = int((d.get("var_chunk") or [d["seq_len"]])[-1])
    K = int(d["chunks_per_conv"])
    V = mcfg["vocab_size"]
    chunks = [torch.randint(4, V - 4, (1, L), device=device)
              for _ in range(K - 1)]
    tgt = torch.randint(4, V - 4, (1, int(d.get("defer_len", 16))),
                        device=device)

    G = int(r.get("group_size", 8))
    casc_depth = int(r.get("cascade_depth", 0))
    cmap = r.get("cascade_map") or [0] * int(mcfg["n_layers"])
    max_mem = int(mcfg["max_mem"])
    seed_slots = int(mcfg.get("mem_seed_slots", 0))
    ids = (1, 2)                              # any ids: VRAM only

    p0 = next(model.parameters())
    with torch.no_grad():
        bank = model.thought_stream.seed_bank(1, p0.device, p0.dtype)
    casc = CascadeMemory(casc_depth, max_mem) if casc_depth else None
    # pre-warm the cascade so layer_banks are materialized (worst case)
    if casc is not None:
        with torch.no_grad():
            for _ in range(max_mem * 4):
                casc.push_slot(torch.randn(1, mcfg["mem_dim"], device=device,
                                           dtype=dtype))

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    forks = mem_fork(bank, casc, G)
    outs = [rollout(model, chunks, tgt, 8.0, 0.03, ids, rng, fb, fc, 0,
                    seed_slots, max_mem, cmap) for fb, fc in forks]
    a_open = torch.tensor([[5]], dtype=torch.long, device=device)
    txt_ids = decode_lb(model, a_open, outs[0]["bank"],
                        _lb(outs[0]["casc"], outs[0]["bank"], cmap),
                        int(r.get("max_new", 64)), -1, amp=False)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    del model, outs, forks, chunks
    torch.cuda.empty_cache()
    return {"params": n_par, "peak_gb": peak, "s_group": dt,
            "turns": K - 1, "chunk_len": L, "G": G,
            "decoded": txt_ids.size(1)}


def main(cfg_path: str) -> None:
    assert torch.cuda.is_available(), "VRAM smoke needs a GPU"
    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(device).total_memory / 2**30
    raw = yaml.safe_load(open(cfg_path))
    print(f"device {torch.cuda.get_device_name(device)} ({total:.1f} GB) | "
          f"config {cfg_path}", flush=True)
    for dtype in (torch.float32, torch.bfloat16):
        try:
            m = measure(raw, dtype, device)
            verdict = "OK" if m["peak_gb"] < total * 0.85 else "TIGHT"
            print(f"{str(dtype):>14}: peak {m['peak_gb']:.2f} GB / {total:.1f} GB "
                  f"[{verdict}] | {m['params']:,} params | "
                  f"group G={m['G']} x {m['turns']} turns x {m['chunk_len']} tok "
                  f"+ decode {m['decoded']} in {m['s_group']:.1f}s "
                  f"({m['s_group']/m['G']:.1f}s/rollout)", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{str(dtype):>14}: OOM — does not fit", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "deepseek_v4_mini/configs/rl_disagg_350m.yaml")
