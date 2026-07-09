"""HOP VERDICT PROBE — did the model learn to think through the bank?

Run on dsv5c checkpoints (merge semantics). K=2 held rules installed as
usual, then on unseen queries:

  1. emission      : [k,k,x] -> does argmax say THINK? (the learned policy)
  2. protocol      : grant [THINK] with carried bank -> acc vs f(f(x))
  3. free run      : grant ONLY if THINK was emitted; overall acc + split
  4. bank ablation : the think segment runs on the bank from BEFORE the hop
                     forward (rules present, intermediate write removed) —
                     if acc holds, the scratch lives in the residual stream,
                     not the bank, and the claim dies. CAUSAL control.
  5. n=3 (jackpot) : [k,k,k,x], grant while THINK is emitted (max 3) ->
                     acc vs f^3(x) + how many thinks it asks for. Never
                     trained; even partial >chance = length generalization
                     of latent deliberation.
  6. sanity        : plain [k,x] must NOT emit THINK (no think-spam) and
                     keep f_A acc.

Usage: PYTHONPATH=. python deepseek_v4_mini/analysis/hop_probe.py [ckpt]
"""
import sys
import torch, yaml

sys.path.insert(0, ".")
from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.model import ThoughtBankLM
from deepseek_v4_mini.train import _rule_space

torch.manual_seed(23)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else
        "checkpoints/multiturn_rule_k2_inter_s128hop_dsv5c/final.pt")
CFG = "deepseek_v4_mini/configs/multiturn_rule_k2_inter_s128hop_dsv5c.yaml"
if "--cfg" in sys.argv:
    CFG = sys.argv[sys.argv.index("--cfg") + 1]

S, m, SYM_OFF, THINK, HOPTOK = 128, 6, 3, 1, 2
KEY_OFF = SYM_OFF + S
# --op: probe the [k,<hop>,k,x] operator format (dsv5d+); default = the bare
# doubled-key format [k,k,x] that dsv5c was trained on.
OPFMT = "--op" in sys.argv
N_CONV = 64

raw = yaml.safe_load(open(CFG))
cfg = ThoughtBankConfig.from_yaml(CFG)
model = ThoughtBankLM(cfg)
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd["model"]); model.eval(); model.to(DEV)
print(f"loaded step {sd['step']} | merge={getattr(cfg, 'mem_write_gate_merge', False)} "
      f"| held pool | chance {1/S:.3f}")

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
def fwd(rows, mem):
    out = model(rows.to(DEV), init_mem=mem, compute_logits=True)
    return out["logits"][:, -1].argmax(-1).cpu(), out["mem_bank"]

@torch.no_grad()
def install():
    mem = None
    for k in (0, 1):
        mem = model(pres_rows(k).to(DEV), init_mem=mem, compute_logits=False)["mem_bank"]
    return mem

A  = torch.tensor([c[0] for c in CONVS])
qs = torch.tensor([c[4][0] for c in CONVS])
K0 = torch.full_like(qs, KEY_OFF)
sym = lambda t: SYM_OFF + t
f1 = torch.tensor([_apply(int(a), int(q)) for a, q in zip(A, qs)])
f2 = torch.tensor([_apply(int(a), int(v)) for a, v in zip(A, f1)])
f3 = torch.tensor([_apply(int(a), int(v)) for a, v in zip(A, f2)])

mem0 = install()

# 6. sanity: [k0,x] answers f_A directly, no think-spam
pred, _ = fwd(torch.stack([K0, sym(qs)], 1), mem0)
print(f"sanity [k0,x]     : f_A {float((pred == sym(f1)).float().mean()):.3f}"
      f"  THINK-spam {float((pred == THINK).float().mean()):.3f}")

# 1+2+3. hop: emission, then granted protocol on emitted lanes
HOP = torch.full_like(qs, HOPTOK)
hop2 = ([K0, HOP, K0, sym(qs)] if OPFMT else [K0, K0, sym(qs)])
hop3 = ([K0, HOP, K0, HOP, K0, sym(qs)] if OPFMT else [K0, K0, K0, sym(qs)])
pred1, mem1 = fwd(torch.stack(hop2, 1), mem0)
emit = pred1 == THINK
print(f"hop {'[k,<hop>,k,x]' if OPFMT else '[k,k,x]':16s}: émission THINK {float(emit.float().mean()):.3f}"
      f"  | réponse directe f²(x) {float((pred1 == sym(f2)).float().mean()):.3f} (contrôle no-think : doit rester à chance)")

if OPFMT:
    # multi-output think: X=[THINK, mid] teacher-forced, Y=[mid, final].
    TH1 = torch.full((N_CONV, 1), THINK)
    mid_pred, _ = fwd(TH1, mem1)                 # position 0: decode mid from bank
    acc_mid = float((mid_pred == sym(f1)).float().mean())
    fin_tf, _ = fwd(torch.cat([TH1, sym(f1).unsqueeze(1)], 1), mem1)   # true mid in-window
    acc_fin_tf = float((fin_tf == sym(f2)).float().mean())
    fin_ar, _ = fwd(torch.cat([TH1, mid_pred.unsqueeze(1)], 1), mem1)  # free chain
    acc_ar = float((fin_ar == sym(f2)).float().mean())
    print(f"think chaîné      : mid décodé (banque) {acc_mid:.3f}  | final (mid forcé) {acc_fin_tf:.3f}"
          f"  | chaîne libre f²(x) {acc_ar:.3f}")
    # causal control: decode mid from the PRE-hop bank (intermediate removed)
    mid_pre, _ = fwd(TH1, mem0)
    print(f"ablation banque   : mid décodé avec banque PRÉ-hop {float((mid_pre == sym(f1)).float().mean()):.3f}"
          f"  (doit s'effondrer si le brouillon vit dans la banque)")
    # n=3, never trained: [k,HOP,k,HOP,k,x] then a length-3 autoregressive chain
    p3, mem3 = fwd(torch.stack(hop3, 1), mem0)
    m1, _ = fwd(TH1, mem3)
    m2, _ = fwd(torch.cat([TH1, m1.unsqueeze(1)], 1), mem3)
    fin3, _ = fwd(torch.cat([TH1, m1.unsqueeze(1), m2.unsqueeze(1)], 1), mem3)
    print(f"n=3 op-format     : émission {float((p3 == THINK).float().mean()):.3f}"
          f"  mid1 {float((m1 == sym(f1)).float().mean()):.3f}"
          f"  mid2 {float((m2 == sym(f2)).float().mean()):.3f}"
          f"  f³(x) {float((fin3 == sym(f3)).float().mean()):.3f}")
else:
    pred2, _ = fwd(torch.full((N_CONV, 1), THINK), mem1)
    acc_all  = float((pred2 == sym(f2)).float().mean())
    acc_emit = (float((pred2[emit] == sym(f2[emit])).float().mean()) if emit.any() else float("nan"))
    print(f"                    acc f²(x) après think : toutes {acc_all:.3f}  émises {acc_emit:.3f}")
    pred2c, _ = fwd(torch.full((N_CONV, 1), THINK), mem0)
    print(f"ablation banque   : acc f²(x) avec banque PRÉ-hop {float((pred2c == sym(f2)).float().mean()):.3f}"
          f"  (doit s'effondrer si le brouillon vit dans la banque)")
    pred, memc = fwd(torch.stack(hop3, 1), mem0)
    thinks_used = torch.zeros(N_CONV, dtype=torch.long)
    final = pred.clone()
    for step in range(3):
        active = final == THINK
        if not bool(active.any()):
            break
        thinks_used[active] += 1
        p, memc = fwd(torch.full((N_CONV, 1), THINK), memc)
        final = torch.where(active, p, final)
    acc3 = float((final == sym(f3)).float().mean())
    hist = torch.bincount(thinks_used, minlength=4).tolist()
    print(f"n=3 [k,k,k,x]     : acc f³(x) {acc3:.3f}  | thinks demandés histo 0/1/2/3 : {hist}")
