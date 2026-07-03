"""
CODE GEOMETRY: are the written rule codes w0(s) a structured manifold or a lookup?

Loads a multiturn_rule checkpoint, presents each shift s in 1..31 (fresh
conversations, presentation turn only, pure written code — no teacher), collects
the written slot w0, and asks:

  1. cos-sim structure across s — is similarity a function of |s - s'| (circular
     geometry: s is modular!) or flat (unstructured repertoire)?
  2. effective rank of the mean-code cloud (31 x mem_dim) — rank ~3 with circular
     structure = compositional; rank >> 3 flat = 31 memorized clusters.
  3. where do HELD codes (25..31) land — inside the trained manifold (write
     generalizes, read is the blocker) or off-manifold (write is the blocker)?
  4. within-s vs between-s similarity: do codes cluster by rule at all?

Usage: PYTHONPATH=. python deepseek_v4_mini/analysis/code_geometry.py [ckpt] [cfg]
CPU-friendly: model is ~M params, 31*R conversations, presentation turn only.

RESULT (2026-07-02, K=2 contiguous-holdout run, train s in 1..24, steps 800/1200):
the write head builds a CIRCULAR manifold — cos-sim decreases smoothly and
monotonically with circular distance (0.98 @d=1 -> 0.09 @d>=13), effective rank
3.4/32; HELD shifts (25..31) are written ON-manifold at the correct circular
position (off-span residual 0.02-0.05, wraparound s=31 -> nearest s=1 correct).
Yet rule_HELD stays at chance: the READ, not the write, is the generalization
blocker (it decodes only the discrete trained code points).
"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from deepseek_v4_mini.config import DeepSeekV4MiniConfig
from deepseek_v4_mini.model import DualModalDeepSeekV4Mini

torch.manual_seed(0)
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/multiturn_rule/final.pt"
CFG  = sys.argv[2] if len(sys.argv) > 2 else "deepseek_v4_mini/configs/multiturn_rule_k2_heldout.yaml"
S, m, SYM_OFF = 32, 6, 3
KEY_OFF = SYM_OFF + S
S_MAX = 24                      # trained pool = 1..24, held = 25..31
R = 32                          # conversations sampled per shift

cfg = DeepSeekV4MiniConfig.from_yaml(CFG)
model = DualModalDeepSeekV4Mini(cfg)
sd = torch.load(CKPT, map_location="cpu")["model"]
model.load_state_dict(sd)
model.eval()
print(f"loaded {CKPT} (step {torch.load(CKPT, map_location='cpu')['step']})")

@torch.no_grad()
def write_codes(s: int, n: int) -> torch.Tensor:
    """n presentation turns of rule s (key 0) -> written slot w0 [n, mem_dim]."""
    X = torch.zeros(n, 1 + 2 * m, dtype=torch.long)
    X[:, 0] = KEY_OFF + 0
    for b in range(n):
        perm = torch.randperm(S).tolist()[:m]
        for j, xi in enumerate(perm):
            X[b, 1 + 2 * j] = SYM_OFF + xi
            X[b, 2 + 2 * j] = SYM_OFF + (xi + s) % S
    out = model(X, init_mem=None, compute_logits=False)
    return out["mem_bank"][:, -1].float()          # the just-written slot

codes = {s: write_codes(s, R) for s in range(1, S)}
mean  = torch.stack([F.normalize(codes[s].mean(0), dim=0) for s in range(1, S)])  # [31, d]

# 1-2. similarity matrix + effective rank of the mean-code cloud
sim = mean @ mean.T                                # [31, 31]
X = mean - mean.mean(0, keepdim=True)
ev = torch.linalg.svdvals(X)**2
erank = float(torch.exp(-(ev/ev.sum() * torch.log(ev/ev.sum() + 1e-12)).sum()))
print(f"\neffective rank of mean codes (31 shifts): {erank:.2f} / {mean.size(1)}")

# circular structure: sim(s, s') vs circular distance d = min(|s-s'|, 32-|s-s'|)
byd = {}
for i in range(31):
    for j in range(i + 1, 31):
        d = min(abs(i - j), S - abs(i - j))
        byd.setdefault(d, []).append(float(sim[i, j]))
print("cos-sim by circular distance d(s,s'):")
for d in sorted(byd):
    v = torch.tensor(byd[d])
    print(f"  d={d:2d}  mean={v.mean():+.3f}  (n={len(v)})")

# 4. within-s coherence vs between-s similarity
within = torch.tensor([
    float((F.normalize(codes[s], dim=1) @ F.normalize(codes[s], dim=1).T)
          .masked_select(~torch.eye(R, dtype=torch.bool)).mean())
    for s in range(1, S)])
off = sim.masked_select(~torch.eye(31, dtype=torch.bool))
print(f"\nwithin-s cos-sim : {within.mean():.3f}  (per-rule cluster tightness)")
print(f"between-s cos-sim: {off.mean():.3f} ± {off.std():.3f}  (cluster separation)")

# 3. held codes vs trained manifold: distance to trained span + nearest trained code
tr = mean[:S_MAX]                                   # trained means (s=1..24)
Q, _ = torch.linalg.qr(tr.T)                        # basis of trained span
for s in range(S_MAX + 1, S):
    c = mean[s - 1]
    resid = float((c - Q @ (Q.T @ c)).norm())       # off-span residual (0 = in-span)
    near  = sim[s - 1, :S_MAX]
    print(f"held s={s}: off-span resid={resid:.3f}  nearest trained s={int(near.argmax())+1} "
          f"(cos {float(near.max()):.3f})  within-s coherence={float(within[s-1]):.3f}")
