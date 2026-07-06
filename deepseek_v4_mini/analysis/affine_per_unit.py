"""Per-unit stratification of affine rule accuracy — test 1 of the convergence autopsy.

The affine family y = (a·x + s) mod S mixes two very different sub-problems: the
shift +s (a translation — solved since the start of the program) and the unit
multiplication a·x mod S (a permutation with hard cycle structure; modular
multiplication is the slow lane of grokking in Power et al.). A global rule_acc
plateau at ~0.13 ≈ 2/16 is exactly the signature of "solved ~2 easy units
(a=1 shifts, a=S-1 negation), chance on the rest".

This script measures rule_acc (train pool) and rule_acc_held per unit a by
restricting the probe's rule pools to one unit at a time (monkeypatching
train._rule_space, which synthetic_rule_probe resolves at module level). If the
accuracy mass sits on a∈{1, S-1}, the lock is the multiplicative apply circuit
downstream of the bank — not the transport; the diversity result stands and
s256L (many rules, no multiplication) is the right next vehicle.

Usage (CPU by default: the GPU belongs to the training run):
  python -m deepseek_v4_mini.analysis.affine_per_unit \
      deepseek_v4_mini/configs/multiturn_rule_k2_inter_affineL_wr.yaml \
      checkpoints/multiturn_rule_k2_inter_affineL_wr/step_XXXX.pt [--gpu] [--n-conv 48]
"""
from __future__ import annotations

import sys

import torch
import yaml

from deepseek_v4_mini import train as T
from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.model import ThoughtBankLM


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cfg_path, ckpt_path = args[0], args[1]
    use_gpu = "--gpu" in sys.argv and torch.cuda.is_available()
    n_conv = 48
    if "--n-conv" in sys.argv:
        n_conv = int(sys.argv[sys.argv.index("--n-conv") + 1])

    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    model_cfg = ThoughtBankConfig.from_yaml(cfg_path)
    device = torch.device("cuda" if use_gpu else "cpu")

    model = ThoughtBankLM(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {ckpt_path} (step {ckpt['step']}) on {device}")

    d = raw["data"]
    S = int(d.get("n_symbols", 32))
    units, _, train_pool, held_pool, _ = T._rule_space(d)
    assert str(d.get("rule_family")) == "affine", "per-unit stratification is affine-only"

    torch.manual_seed(0)
    probe_kw = dict(
        fused_ce=True, ce_chunk=1024,
        balance_w=float(model_cfg.balance_loss_weight),
        device=device, amp_dtype=torch.bfloat16, use_amp=use_gpu,
        n_conv=n_conv,
    )

    orig_rule_space = T._rule_space
    rows = []
    try:
        for a_i, a in enumerate(units):
            tp = [r for r in train_pool if r // S == a_i]
            hp = [r for r in held_pool if r // S == a_i]

            def patched(dd, tp=tp, hp=hp):
                u, n, _, _, ap = orig_rule_space(dd)
                return u, n, tp, hp, ap

            T._rule_space = patched
            res = T.synthetic_rule_probe(model, raw, **probe_kw)
            rows.append((a, res["rule_acc"], res.get("rule_acc_held", float("nan")),
                         len(tp), len(hp)))
            print(f"  a={a:3d}  train_acc={res['rule_acc']:.3f}  "
                  f"held_acc={res.get('rule_acc_held', float('nan')):.3f}  "
                  f"(pools {len(tp)}/{len(hp)})")
    finally:
        T._rule_space = orig_rule_space

    chance = 1.0 / S
    easy = {1, S - 1}
    acc_easy = sum(r[1] for r in rows if r[0] in easy) / len(easy)
    hard = [r for r in rows if r[0] not in easy]
    acc_hard = sum(r[1] for r in hard) / len(hard)
    print(f"\nchance={chance:.3f}")
    print(f"easy units (a=1, a={S-1}): mean train_acc={acc_easy:.3f}")
    print(f"hard units ({len(hard)}):        mean train_acc={acc_hard:.3f}")
    n_above = sum(1 for r in rows if r[1] > 2 * chance)
    print(f"units above 2x chance: {n_above}/{len(rows)}")
    print("verdict: mass on easy units => lock is the multiplicative apply circuit "
          "(downstream of the bank); flat across units => lock is elsewhere "
          "(transport/capacity/steps)")


if __name__ == "__main__":
    main()
