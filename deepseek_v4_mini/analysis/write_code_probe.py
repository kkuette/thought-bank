"""Linear probe of the WRITTEN code — localizes the 'gap without decision' failure.

s256L v2 reduced CE via the bank (gap +2.0) but never produced decisions (acc at
chance). Two candidate locks: (a) the WRITE never encodes which of the 255 rules
was presented (identification fails at the source), or (b) the write encodes s
but the READ cannot apply it. This probe answers (a) directly, offline, on CPU:

  present rule s -> capture the written slot w0 = bank[:, -1]  (n_rep times)
  1) consistency: intra-rule vs inter-rule cosine of w0
  2) ridge probe w0 -> (cos, sin)(2*pi*s/S): median |error| in SYMBOL steps,
     acc@±1 and ±4 symbols — fitted on train rules, tested on unseen reps AND on
     HELD rules (does the write place unseen rules on the manifold, as it did at
     25 shifts?)
  3) 1-NN rule identification of a fresh presentation against per-rule mean codes

If (2) is tight: the write is innocent — the lock is read/apply, and read-side
interventions (SwiGLU, Fourier teacher) are aimed right. If (2) is at chance:
identification dies at the write and no read recipe can work.

Usage (CPU, safe to run alongside a GPU training):
  python -m deepseek_v4_mini.analysis.write_code_probe \
      deepseek_v4_mini/configs/multiturn_rule_k2_inter_s256L.yaml \
      checkpoints/multiturn_rule_k2_inter_s256L/step_2000.pt
"""
from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as Fn
import yaml

from deepseek_v4_mini import train as T
from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.model import ThoughtBankLM


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cfg_path, ckpt_path = args[0], args[1]
    n_rep = 8

    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    d = raw["data"]
    model_cfg = ThoughtBankConfig.from_yaml(cfg_path)
    device = torch.device("cpu")
    model = ThoughtBankLM(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {ckpt_path} (step {ckpt['step']}) on cpu")

    S = int(d.get("n_symbols", 32))
    m = int(d.get("n_examples", 6))
    K = int(d.get("n_contexts", 1))
    D = int(model_cfg.mem_dim)
    SYM, KEY = 3, 3 + S
    _units, _n, train_pool, held_pool, apply_ = T._rule_space(d)
    assert str(d.get("rule_family", "shift")) == "shift", "angle probe assumes rid = s (shift)"
    torch.manual_seed(0)

    @torch.no_grad()
    def codes_for(rules):
        out = torch.zeros(len(rules), n_rep, D)
        for rep in range(n_rep):
            for i0 in range(0, len(rules), 64):
                chunk = rules[i0:i0 + 64]
                rows = []
                for s in chunk:
                    perm = torch.randperm(S).tolist()[:m]
                    row = [KEY + 0] if K > 1 else []
                    for xi in perm:
                        row += [SYM + xi, SYM + apply_(s, xi)]
                    rows.append(row)
                x = torch.tensor(rows, dtype=torch.long)
                o = model(x, init_mem=None, compute_logits=False, pad_mask=(x != 0))
                out[i0:i0 + len(chunk), rep] = o["mem_bank"][:, -1].float()
        return out

    tr = codes_for(train_pool)    # [Nt, R, D]
    hd = codes_for(held_pool)     # [Nh, R, D]
    print(f"codes: train {tuple(tr.shape)}  held {tuple(hd.shape)}")

    # 1) consistency
    def consistency(c):
        cn = Fn.normalize(c, dim=-1)
        iu = torch.triu_indices(n_rep, n_rep, 1)
        intra = torch.einsum("nrd,nqd->nrq", cn, cn)[:, iu[0], iu[1]].mean().item()
        mu = Fn.normalize(c.mean(1), dim=-1)
        g = mu @ mu.T
        inter = g[~torch.eye(len(mu), dtype=torch.bool)].mean().item()
        return intra, inter

    it_, in_ = consistency(tr)
    print(f"[1] cosine intra-rule {it_:.3f} vs inter-rule {in_:.3f}  "
          f"(identifiable code needs intra >> inter)")

    # 2) ridge probe -> circle
    def angles(rules):
        th = torch.tensor([2 * math.pi * r / S for r in rules])
        return torch.stack([th.cos(), th.sin()], 1)

    fit_reps = n_rep - 2
    X = tr[:, :fit_reps].reshape(-1, D)
    Y = angles(train_pool).repeat_interleave(fit_reps, 0)
    lam = 1e-2 * (X.T @ X).diagonal().mean()
    W = torch.linalg.solve(X.T @ X + lam * torch.eye(D), X.T @ Y)

    def eval_probe(c, rules, name):
        P = c.reshape(-1, D) @ W
        th_hat = torch.atan2(P[:, 1], P[:, 0])
        th = torch.tensor([2 * math.pi * r / S for r in rules]).repeat_interleave(c.size(1))
        err = (th_hat - th + math.pi) % (2 * math.pi) - math.pi
        sym = err.abs() * S / (2 * math.pi)
        print(f"[2] {name:18s} median|err| {sym.median().item():6.2f} symbols   "
              f"acc@±1 {(sym <= 1).float().mean().item():.3f}   "
              f"acc@±4 {(sym <= 4).float().mean().item():.3f}   (chance@±1 ≈ {3 / S:.3f})")

    eval_probe(tr[:, fit_reps:], train_pool, "train rules (new reps)")
    eval_probe(hd, held_pool, "HELD rules")

    # 3) 1-NN identification
    mu = Fn.normalize(tr.mean(1), dim=-1)                      # [Nt, D] per-rule anchors
    q = Fn.normalize(tr[:, fit_reps:].reshape(-1, D), dim=-1)  # fresh presentations
    nn_idx = (q @ mu.T).argmax(1)
    truth = torch.arange(len(train_pool)).repeat_interleave(2)
    print(f"[3] 1-NN rule identification (255-way): "
          f"{(nn_idx == truth).float().mean().item():.3f}  (chance {1 / len(train_pool):.4f})")


if __name__ == "__main__":
    main()
