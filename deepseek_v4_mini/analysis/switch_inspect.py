"""Mirror of rehearsal_inspect: what happens to s1 in the writes after the switch?

Runs full switch conversations (present s1, 12 queries, present s2 mid-conv,
12 queries) on the switch checkpoint and tracks, per turn, the cosine of each
write to the s1 presentation write vs the s2 presentation write of the SAME
conversation.

Usage: PYTHONPATH=. python deepseek_v4_mini/analysis/switch_inspect.py [ckpt]

RESULT (2026-07-03, step_1100.pt of the switch run, rule_acc 0.795, STICK=0.000):
raw cosines are a TRAP here — query writes stay ~0.50-correlated with w_s1 and
slightly ANTI-correlated with w_s2 (-0.04 vs +0.37 cross-conv baseline) both
before and after the switch, because writes are bank-conditioned objects. The
canonical-space diagnostic (see analysis snippet in the session notes) resolves
it: in THIS model query writes carry NO rule identity at all (identifiability
0.03 = chance, vs ~0.48 in the horizon model) — 12-turn phases never lose the
presentation slot, so no rehearsal is needed or learned; dirty-bank
presentations remain canonically identifiable (0.56 as s2). Forgetting is a
RECENCY OVERRIDE in the read: at q13-q15 s1's code is still physically in the
bank yet never used. Memory policy is task-adaptive/minimal.
"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from deepseek_v4_mini.config import DeepSeekV4MiniConfig
from deepseek_v4_mini.model import DualModalDeepSeekV4Mini

torch.manual_seed(0)
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/multiturn_rule_switch/step_1100.pt"
CFG  = "deepseek_v4_mini/configs/multiturn_rule_switch.yaml"
S, m, SYM_OFF, SW, TURNS = 32, 6, 3, 12, 24
N = 96                                   # conversations (batched lanes)

cfg = DeepSeekV4MiniConfig.from_yaml(CFG)
model = DualModalDeepSeekV4Mini(cfg)
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd["model"]); model.eval()
print(f"loaded {CKPT} (step {sd['step']})")

@torch.no_grad()
def run():
    s1 = torch.randint(1, S, (N,))
    s2 = torch.randint(1, S, (N,))
    while bool((s2 == s1).any()):
        cl = s2 == s1
        s2[cl] = torch.randint(1, S, (int(cl.sum()),))
    pools, press = [], []
    for s_k in (s1, s2):
        pres = torch.zeros(N, 2 * m, dtype=torch.long)
        un = []
        for b in range(N):
            perm = torch.randperm(S).tolist()
            for j, xi in enumerate(perm[:m]):
                pres[b, 2 * j] = SYM_OFF + xi
                pres[b, 2 * j + 1] = SYM_OFF + (xi + int(s_k[b])) % S
            un.append(perm[m:])
        press.append(pres); pools.append(un)

    out = model(press[0], init_mem=None, compute_logits=False)
    mem = out["mem_bank"]
    w_s1 = F.normalize(mem[:, -1], dim=1)            # s1 presentation write
    writes, accs, w_s2 = [], [], None
    for t in range(TURNS):
        if t == SW:
            out = model(press[1], init_mem=mem, compute_logits=False)
            mem = out["mem_bank"]
            w_s2 = F.normalize(mem[:, -1], dim=1)    # s2 presentation write
        ph, idx = (1, t - SW) if t >= SW else (0, t)
        s_k = (s2 if ph else s1)
        xq = torch.tensor([[SYM_OFF + pools[ph][b][idx % len(pools[ph][b])]] for b in range(N)])
        out = model(xq, init_mem=mem, compute_logits=True)
        mem = out["mem_bank"]
        writes.append(F.normalize(mem[:, -1], dim=1))
        y = torch.tensor([SYM_OFF + (pools[ph][b][idx % len(pools[ph][b])] + int(s_k[b])) % S
                          for b in range(N)])
        accs.append(float((out["logits"][:, -1].argmax(-1) == y).float().mean()))
    return w_s1, w_s2, writes, accs, s1, s2

w_s1, w_s2, writes, accs, s1, s2 = run()
# cross-rule baseline: mean |cos| between presentation writes of DIFFERENT rules
base = float((w_s1 @ w_s2.T).masked_select(~torch.eye(N, dtype=torch.bool)).mean())
print(f"\ncross-conv baseline sim: {base:+.3f}   sim(w_s1, w_s2) same conv: "
      f"{float((w_s1 * w_s2).sum(1).mean()):+.3f}")
print(f"{'turn':>4} {'sim_s1':>7} {'sim_s2':>7} {'acc':>5}")
for t, w in enumerate(writes):
    a = float((w * w_s1).sum(1).mean())
    b = float((w * w_s2).sum(1).mean()) if w_s2 is not None else float("nan")
    mark = " <- SWITCH" if t == SW else ""
    print(f" q{t+1:<3} {a:7.3f} {b:7.3f} {accs[t]:5.2f}{mark}")
