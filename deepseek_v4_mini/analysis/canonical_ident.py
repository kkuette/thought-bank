"""Canonical-space identifiability: what rule (if any) does each write encode?

Raw cosines between writes are misleading (writes are bank-conditioned objects
— see switch_inspect.py). The robust measure: build CANONICAL rule codes (mean
presentation write per shift, fresh banks), then classify any write by its
nearest canonical mean. Chance = 1/31.

Applied to the switch checkpoint it separates the two memory regimes:

  - s1 query write (q12, pre-switch)  as s1: 0.03  -> NO rehearsal in this model
  - s2 DIRTY-bank presentation write  as s2: 0.56  -> presentations stay canonical
  - post-switch query writes          as s2: 0.05-0.06, as s1: 0.02-0.03

vs the HORIZON model where query writes identify as their rule at ~0.48
(rehearsal). Memory policy is task-adaptive: rehearsal only under eviction
pressure (phases longer than the FIFO survival window of 15 query turns).

Usage: PYTHONPATH=. python deepseek_v4_mini/analysis/canonical_ident.py [ckpt] [cfg]
"""
import sys, yaml, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.model import ThoughtBankLM

torch.manual_seed(0)
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/multiturn_rule_switch/step_1100.pt"
CFG  = sys.argv[2] if len(sys.argv) > 2 else "deepseek_v4_mini/configs/multiturn_rule_switch.yaml"
S, m, SYM_OFF, N = 32, 6, 3, 64

cfg = ThoughtBankConfig.from_yaml(CFG)
_d    = yaml.safe_load(open(CFG)).get("data", {})
SW    = int(_d.get("switch_at", 12))
TURNS = int(_d.get("turns_per_conv", 24))
model = ThoughtBankLM(cfg)
model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"]); model.eval()
print(f"loaded {CKPT}")

@torch.no_grad()
def pres_batch(s_vec, mem):
    n = s_vec.size(0)
    X = torch.zeros(n, 2 * m, dtype=torch.long); un = []
    for b in range(n):
        perm = torch.randperm(S).tolist()
        for j, xi in enumerate(perm[:m]):
            X[b, 2 * j] = SYM_OFF + xi
            X[b, 2 * j + 1] = SYM_OFF + (xi + int(s_vec[b])) % S
        un.append(perm[m:])
    out = model(X, init_mem=mem, compute_logits=False)
    return out["mem_bank"], un

# canonical codes: fresh-bank presentation, R reps per shift
R = 16
means = []
with torch.no_grad():
    for s in range(1, S):
        mem, _ = pres_batch(torch.full((R,), s), None)
        means.append(F.normalize(mem[:, -1].mean(0), dim=0))
M = torch.stack(means)   # [31, d]

def ident(w, s_vec):
    """fraction of rows whose nearest canonical mean is their own rule"""
    sim = F.normalize(w, dim=1) @ M.T
    return float((sim.argmax(1) == (s_vec - 1)).float().mean())

with torch.no_grad():
    s1 = torch.randint(1, S, (N,)); s2 = torch.randint(1, S, (N,))
    while bool((s2 == s1).any()):
        cl = s2 == s1; s2[cl] = torch.randint(1, S, (int(cl.sum()),))
    mem, un1 = pres_batch(s1, None)
    for t in range(SW):                      # queries on s1
        xq = torch.tensor([[SYM_OFF + un1[b][t % len(un1[b])]] for b in range(N)])
        mem = model(xq, init_mem=mem, compute_logits=False)["mem_bank"]
    pre_q = mem[:, -1]
    mem, un2 = pres_batch(s2, mem)           # DIRTY-bank presentation of s2
    w2_dirty = mem[:, -1]
    post_writes = []
    for t in range(TURNS - SW):
        xq = torch.tensor([[SYM_OFF + un2[b][t % len(un2[b])]] for b in range(N)])
        mem = model(xq, init_mem=mem, compute_logits=False)["mem_bank"]
        post_writes.append(mem[:, -1])

print(f"identifiability vs canonical codes (chance {1/(S-1):.3f}):")
print(f"  s1 query write (q{SW}, pre-switch)   as s1: {ident(pre_q, s1):.2f}   as s2: {ident(pre_q, s2):.2f}")
print(f"  s2 DIRTY presentation write        as s2: {ident(w2_dirty, s2):.2f}   as s1: {ident(w2_dirty, s1):.2f}")
for i in (0, (TURNS - SW) // 2, TURNS - SW - 1):
    w = post_writes[i]
    print(f"  post-switch query write q{SW+i+1:<2}       as s2: {ident(w, s2):.2f}   as s1: {ident(w, s1):.2f}")
