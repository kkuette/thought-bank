"""Rehearsal inspection: do query-turn writes re-encode the rule?

Loads the horizon final.pt, runs full 24-turn conversations, captures the write
(newly appended slot) at every turn, and asks:

  1. cos-sim(write_t, presentation write of the SAME conversation) per turn —
     is the model literally rewriting the rule code (copy) or something else?
  2. rule identifiability: nearest mean-presentation-code over all shifts —
     what fraction of turn-t writes point to the RIGHT rule?
  3. cross-rule margin: sim to own rule mean vs best other rule mean.

If (2) stays high across turns, the rehearsal loop re-encodes the rule in its
query-turn writes (the info survives slot eviction through them). If writes are
near-orthogonal to the rule code yet accuracy holds late, the encoding is
distributed/rotated rather than a copy.

Usage: PYTHONPATH=. python deepseek_v4_mini/analysis/rehearsal_inspect.py [ckpt]

RESULT (2026-07-03, horizon final.pt @1500, rule_acc plateau ~0.48): query-turn
writes are NOISY PARTIAL COPIES — sim_pres ~0.51, identifiability ~0.48 (chance
0.125; the presentation write itself is only 0.64). Post-eviction (q17+, writes
built from prior query-writes only) sim drops to ~0.41 and ident to ~0.35 while
accuracy HOLDS ~0.50: the read integrates redundant copies across slots. This is
the mechanism behind the "no FIFO cliff" result, and the noisy-copy cloud is the
likely cause of the 0.48 plateau (vs 0.95 at 9-turn maintenance).
"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.model import ThoughtBankLM

torch.manual_seed(0)
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/multiturn_rule_horizon/final.pt"
CFG  = "deepseek_v4_mini/configs/multiturn_rule_horizon.yaml"
S, m, SYM_OFF = 32, 6, 3
TURNS = 24
SHIFTS = list(range(1, 32, 4))          # 8 probe shifts
R = 16                                  # conversations per shift

cfg = ThoughtBankConfig.from_yaml(CFG)
model = ThoughtBankLM(cfg)
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd["model"]); model.eval()
print(f"loaded {CKPT} (step {sd['step']})")

@torch.no_grad()
def conversation_writes(s: int, n: int):
    """Run n full conversations of rule s; return writes [n, 1+TURNS, mem_dim]
    (presentation write + one write per query turn) and per-turn correctness."""
    pres = torch.zeros(n, 2 * m, dtype=torch.long)
    unseen = []
    for b in range(n):
        perm = torch.randperm(S).tolist()
        for j, xi in enumerate(perm[:m]):
            pres[b, 2 * j] = SYM_OFF + xi
            pres[b, 2 * j + 1] = SYM_OFF + (xi + s) % S
        unseen.append(perm[m:])
    out = model(pres, init_mem=None, compute_logits=False)
    mem = out["mem_bank"]
    writes = [mem[:, -1]]
    corr = []
    for t in range(TURNS):
        xq = torch.tensor([[SYM_OFF + unseen[b][t % len(unseen[b])]] for b in range(n)])
        out = model(xq, init_mem=mem, compute_logits=True)
        mem = out["mem_bank"]
        writes.append(mem[:, -1])
        y = torch.tensor([SYM_OFF + ((unseen[b][t % len(unseen[b])]) + s) % S for b in range(n)])
        corr.append((out["logits"][:, -1].argmax(-1) == y).float().mean().item())
    return torch.stack(writes, dim=1), corr        # [n, 1+TURNS, d]

W = {s: conversation_writes(s, R) for s in SHIFTS}
means = {s: F.normalize(W[s][0][:, 0].mean(0), dim=0) for s in SHIFTS}  # mean pres code
M = torch.stack([means[s] for s in SHIFTS])                              # [8, d]

print(f"\nper-turn stats over shifts {SHIFTS} ({R} convs each):")
print(f"{'turn':>4} {'sim_pres':>9} {'ident':>6} {'margin':>7} {'acc':>5}")
for t in range(1 + TURNS):
    sims, ids, margins, accs = [], [], [], []
    for si, s in enumerate(SHIFTS):
        w = F.normalize(W[s][0][:, t], dim=1)                # [R, d]
        p = F.normalize(W[s][0][:, 0], dim=1)                # own presentation write
        sims.append(float((w * p).sum(1).mean()))
        sim_all = w @ M.T                                    # [R, 8]
        ids.append(float((sim_all.argmax(1) == si).float().mean()))
        own = sim_all[:, si]
        oth = sim_all.clone(); oth[:, si] = -2
        margins.append(float((own - oth.max(1).values).mean()))
        if t > 0:
            accs.append(W[s][1][t - 1])
    label = "pres" if t == 0 else f"q{t}"
    acc_s = f"{sum(accs)/len(accs):5.2f}" if t > 0 else "    -"
    print(f"{label:>4} {sum(sims)/len(sims):9.3f} {sum(ids)/len(ids):6.2f} "
          f"{sum(margins)/len(margins):7.3f} {acc_s}")
