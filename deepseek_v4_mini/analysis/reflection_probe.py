"""REFLECTION ZERO-SHOT PROBE — can the bank be used as a scratchpad?

Re-creation of the 2026-07-06 probe (run then on dsv4w@3000): the bank does
STORAGE (bind/retain/replace); reflection = ITERATIVE computation through it
(write an intermediate, read it back, transform). Zero-shot decomposition,
K=2 native format (rules A->key0, B->key1 presented once, held pool):

  sanity   [k0,x]        -> f_A(x)        the trained behaviour (ceiling)
  doubled  [k0,k0,x]     -> f_A(f_A(x))?  or is the doubled key ignored (f_A(x))?
  two-key  [k0,k1,x]     -> f_B(f_A(x)) / f_A(f_B(x)) / f_A(x) / f_B(x)?
  external chain: query [k0,x] -> y_hat; then [k0,y_hat] (bank carried,
  fresh forward) -> f_A(f_A(x)). All loop pieces exist externally; only the
  internal trigger is missing (prior run: 0.961 external vs 0.008 internal).

Arms: dsv4w_s43@4000 (FIFO, gate off) and dsv5b final (merge semantics) —
merge gives unlimited retention at 8 slots, the scratchpad property the
reflection cell will lean on.

Usage: PYTHONPATH=. python deepseek_v4_mini/analysis/reflection_probe.py
"""
import sys
import torch, yaml

sys.path.insert(0, ".")
from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.model import ThoughtBankLM
from deepseek_v4_mini.train import _rule_space

torch.manual_seed(11)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG = "deepseek_v4_mini/configs/multiturn_rule_k2_inter_s128struct_dsv4w_s43.yaml"
S, m, SYM_OFF = 128, 6, 3
KEY_OFF = SYM_OFF + S
N_CONV = 64

raw = yaml.safe_load(open(CFG))
_u, _n, TRAIN, HELD, _apply = _rule_space(raw["data"])
POOL = torch.tensor(HELD)

CONVS = []
for _ in range(N_CONV):
    a = int(POOL[int(torch.randint(0, len(POOL), (1,)))])
    while True:
        b = int(POOL[int(torch.randint(0, len(POOL), (1,)))])
        if b != a:
            break
    perm = torch.randperm(S).tolist()
    CONVS.append((a, b, perm[:m], perm[m:2 * m], perm[2 * m:]))

def pres_rows(k):
    rows = []
    for a, b, exa, exb, _ in CONVS:
        s, ex = (a, exa) if k == 0 else (b, exb)
        row = [KEY_OFF + k]
        for xi in ex:
            row += [SYM_OFF + xi, SYM_OFF + _apply(s, xi)]
        rows.append(row)
    return torch.tensor(rows)

@torch.no_grad()
def install(model):
    mem = None
    for k in (0, 1):
        mem = model(pres_rows(k).to(DEV), init_mem=mem, compute_logits=False)["mem_bank"]
    return mem

@torch.no_grad()
def acc(model, mem, rows, targets):
    out = model(rows.to(DEV), init_mem=mem, compute_logits=True)
    pred = out["logits"][:, -1].argmax(-1).cpu()
    return {name: float((pred == t).float().mean()) for name, t in targets.items()}, out["mem_bank"], pred

ARMS = [
    ("s43 FIFO",     "checkpoints/multiturn_rule_k2_inter_s128_dsv4w_s43/step_4000.pt",   False),
    ("dsv5b merge",  "checkpoints/multiturn_rule_k2_inter_s128structmerge_dsv5b/final.pt", True),
]

for name, ckpt, merge in ARMS:
    cfg = ThoughtBankConfig.from_yaml(CFG)
    cfg.mem_write_gate = False
    cfg.mem_write_gate_merge = merge
    model = ThoughtBankLM(cfg)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["model"]); model.eval(); model.to(DEV)
    print(f"\n== {name} (step {sd['step']}) | chance {1/S:.3f}")

    qs = torch.tensor([c[4][0] for c in CONVS])                 # unseen query symbol
    A  = torch.tensor([c[0] for c in CONVS]); B = torch.tensor([c[1] for c in CONVS])
    fA   = torch.tensor([_apply(int(a), int(q)) for a, q in zip(A, qs)])
    fAA  = torch.tensor([_apply(int(a), int(f)) for a, f in zip(A, fA)])
    fB   = torch.tensor([_apply(int(b), int(q)) for b, q in zip(B, qs)])
    fBA  = torch.tensor([_apply(int(b), int(f)) for b, f in zip(B, fA)])   # f_B∘f_A
    fAB  = torch.tensor([_apply(int(a), int(f)) for a, f in zip(A, fB)])   # f_A∘f_B
    sym  = lambda t: SYM_OFF + t

    mem0 = install(model)
    r = lambda cols: torch.stack(cols, dim=1)
    K0 = torch.full_like(qs, KEY_OFF); K1 = torch.full_like(qs, KEY_OFF + 1)

    a1, _, _ = acc(model, mem0, r([K0, sym(qs)]), {"f_A": sym(fA)})
    print(f"  sanity [k0,x]        : f_A {a1['f_A']:.3f}")

    a2, _, _ = acc(model, mem0, r([K0, K0, sym(qs)]), {"f_A(f_A)": sym(fAA), "f_A (ignore)": sym(fA)})
    print(f"  doubled [k0,k0,x]    : f_A(f_A) {a2['f_A(f_A)']:.3f}  |  f_A (clé ignorée) {a2['f_A (ignore)']:.3f}")

    a3, _, _ = acc(model, mem0, r([K0, K1, sym(qs)]),
                   {"f_B(f_A)": sym(fBA), "f_A(f_B)": sym(fAB), "f_A": sym(fA), "f_B": sym(fB)})
    print(f"  two-key [k0,k1,x]    : f_B∘f_A {a3['f_B(f_A)']:.3f}  f_A∘f_B {a3['f_A(f_B)']:.3f}  "
          f"f_A {a3['f_A']:.3f}  f_B {a3['f_B']:.3f}")

    # external chain: y_hat = model([k0,x]); then [k0,y_hat] with carried bank
    a4, mem1, pred1 = acc(model, mem0, r([K0, sym(qs)]), {"f_A": sym(fA)})
    a5, _, _ = acc(model, mem1, r([K0, pred1]), {"f_A(f_A)": sym(fAA)})
    print(f"  external chain       : step1 f_A {a4['f_A']:.3f} -> step2 f_A(f_A) {a5['f_A(f_A)']:.3f}")
