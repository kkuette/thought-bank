"""Joint-model inspection: per-turn accuracy + canonical write identity across
the switch. Phase 1 = 24 queries on s1 (presentation evicted ~q16 -> rehearsal
pressure), then dirty-bank presentation of s2, then 16 queries on s2.

For each turn: acc, write identifiability as s1 / as s2 (nearest canonical mean
over all 31 shifts, chance 0.032), sim to own s1 presentation write.

Usage: PYTHONPATH=. python deepseek_v4_mini/analysis/joint_inspect.py [ckpt]

RESULT (2026-07-03, step_1200 @ plateau 0.747/0.743, STICK 0.02): NO per-turn
cliff anywhere — acc 0.6-0.8 uniform across all 40 turns, including q17-q24
(post-eviction) and post-switch. But unlike the horizon model (noisy copies,
ident ~0.48), query writes here carry NO canonical rule identity (0.00-0.05 =
chance) and are ANTI-correlated with the presentation write (sim -0.2/-0.3) —
yet the rule survives eviction. Rehearsal exists functionally but in a COVERT
distributed code, off the presentation manifold. Presentations stay canonical
(ident 0.77-0.78 even dirty-bank). Interpretation: switch pressure forces a
code-space separation — presentations on the canonical manifold, maintenance
traffic elsewhere — so the recency override can tell "new rule" from "old
copies". Bonus: 24-turn maintenance at 0.74 vs 0.48 for the horizon model —
training to retain-then-replace yields a BETTER maintenance code than training
to retain alone. Caveat for all diagnostics: canonical-nearest-mean
identifiability only detects presentation-style codes; ident-at-chance does NOT
prove absence of information (a trained linear decoder is the next measure).
"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.model import ThoughtBankLM

torch.manual_seed(0)
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/multiturn_rule_joint/step_1200.pt"
CFG  = "deepseek_v4_mini/configs/multiturn_rule_joint.yaml"
S, m, SYM_OFF = 32, 6, 3
SW, TURNS = 24, 40          # queries on s1, total query turns
N = 64

cfg = ThoughtBankConfig.from_yaml(CFG)
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

# canonical means (fresh-bank presentations)
R = 16
means = []
with torch.no_grad():
    for s in range(1, S):
        mem, _ = pres_batch(torch.full((R,), s), None)
        means.append(F.normalize(mem[:, -1].mean(0), dim=0))
M = torch.stack(means)

def ident(w, s_vec):
    sim = F.normalize(w, dim=1) @ M.T
    return float((sim.argmax(1) == (s_vec - 1)).float().mean())

with torch.no_grad():
    s1 = torch.randint(1, S, (N,)); s2 = torch.randint(1, S, (N,))
    while bool((s2 == s1).any()):
        cl = s2 == s1; s2[cl] = torch.randint(1, S, (int(cl.sum()),))
    mem, un1 = pres_batch(s1, None)
    w_pres1 = F.normalize(mem[:, -1], dim=1)
    print(f"pres s1 write ident: as s1 {ident(mem[:, -1], s1):.2f}")
    print(f"\n{'turn':>5} {'acc':>5} {'id_s1':>6} {'id_s2':>6} {'sim_p1':>7}")
    rows = []
    def step(xq, y, mem, phase):
        out = model(xq, init_mem=mem, compute_logits=True)
        mem = out["mem_bank"]; w = mem[:, -1]
        acc = float((out["logits"][:, -1].argmax(-1) == y).float().mean())
        simp = float((F.normalize(w, dim=1) * w_pres1).sum(1).mean())
        return mem, (acc, ident(w, s1), ident(w, s2), simp)
    for t in range(SW):
        xq = torch.tensor([[SYM_OFF + un1[b][t % len(un1[b])]] for b in range(N)])
        y = torch.tensor([SYM_OFF + ((un1[b][t % len(un1[b])]) + int(s1[b])) % S for b in range(N)])
        mem, r = step(xq, y, mem, 1)
        print(f"  q{t+1:<3} {r[0]:5.2f} {r[1]:6.2f} {r[2]:6.2f} {r[3]:7.3f}")
    mem, un2 = pres_batch(s2, mem)
    print(f" pres2   -   {ident(mem[:, -1], s1):6.2f} {ident(mem[:, -1], s2):6.2f} "
          f"{float((F.normalize(mem[:, -1], dim=1) * w_pres1).sum(1).mean()):7.3f}")
    for t in range(TURNS - SW):
        xq = torch.tensor([[SYM_OFF + un2[b][t % len(un2[b])]] for b in range(N)])
        y = torch.tensor([SYM_OFF + ((un2[b][t % len(un2[b])]) + int(s2[b])) % S for b in range(N)])
        mem, r = step(xq, y, mem, 2)
        print(f"  q{SW+t+1:<3} {r[0]:5.2f} {r[1]:6.2f} {r[2]:6.2f} {r[3]:7.3f}")
