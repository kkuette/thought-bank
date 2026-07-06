"""HEADLINE DEMO — fresh-rule installation at inference: fast-weight bank vs TTT.

The program's central claim: continual learning at inference WITHOUT backward.
This harness measures it head-to-head on the same conversations, same model:

  bank   (ours) : rule presented ONCE, forward-only -> written to the bank ->
                  applied to unseen queries across turns. Cost: 1 forward of a
                  13-token segment per rule.
  ttt           : bank ablated; a per-conversation CLONE of the model gets the
                  same 6 example pairs in query format ([key, x] -> y) and takes
                  N optimizer steps (AdamW, lr swept); then answers the same
                  queries with a fresh bank. Cost: N x (forward+backward).
  icl           : example pairs IN the query window ([key, x0,y0..x5,y5, xq]) —
                  the in-window ICL reference (this position IS trained: it is
                  exactly the presentation-turn supervision).
  ablate        : no presentation, no adaptation — chance floor.

Rules are drawn from the HELD pool (never seen in training = genuinely fresh
laws), K=2 keyed rules per conversation (the model's native format — TTT must
install BOTH concurrently, like the bank does). Same conversations across arms.

Cost accounting: FLOPs proxy = 2 * params * tokens per forward (x3 for a
training step), plus wall-clock. The headline number is the cost RATIO at
matched accuracy: how many gradient steps TTT needs to reach the bank's
accuracy, and what that costs relative to one forward.

--sub (the FAIRNESS arm, user 2026-07-05): rules become y = (s - x) mod S —
subtraction, i.e. affine a = -1 = 127. The MINIMAL fresh family: same circle,
same geometry, reversed direction, NEVER trained. Here TTT is allowed to leave
the meta-learned family (gradient updates weights); the bank is not (forward
only). This measures the claim boundary honestly: within-family the bank wins
on cost; out-of-family gradient keeps an edge the bank structurally lacks
(family-transfer arc dsv4n/o/q closed negative).

Usage: PYTHONPATH=. python deepseek_v4_mini/analysis/ttt_demo.py [ckpt] [--train-pool] [--sub]
CPU-friendly (tiny model); safe to run alongside a GPU training.
"""
import copy, math, sys, time
import torch, torch.nn.functional as F, yaml

sys.path.insert(0, ".")
from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.model import ThoughtBankLM
from deepseek_v4_mini.train import _rule_space

torch.manual_seed(0)
CKPT = (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else
        "/home/kkuette/thought-bank/.claude/worktrees/interesting-clarke-c68f5c/"
        "checkpoints/multiturn_rule_k2_inter_s128_dsv4m/final.pt")
CFG  = "deepseek_v4_mini/configs/multiturn_rule_k2_inter_s128_dsv4m.yaml"
if "--cfg" in sys.argv:
    CFG = sys.argv[sys.argv.index("--cfg") + 1]
USE_TRAIN = "--train-pool" in sys.argv
NO_TTT    = "--no-ttt" in sys.argv
S, m, K, SYM_OFF = 128, 6, 2, 3
KEY_OFF = SYM_OFF + S
N_CONV, TURNS = 64, 8
TTT_LRS   = (3e-4, 1e-3, 3e-3)
TTT_EVALS = (1, 2, 5, 10, 20, 50)            # cumulative step counts

raw = yaml.safe_load(open(CFG))
cfg = ThoughtBankConfig.from_yaml(CFG)
model = ThoughtBankLM(cfg)
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd["model"]); model.eval()
P = sum(p.numel() for p in model.parameters())
print(f"loaded step {sd['step']} | params {P/1e6:.2f}M | pool = {'TRAIN' if USE_TRAIN else 'HELD (fresh laws)'}")

_units, _n, TRAIN, HELD, _apply = _rule_space(raw["data"])
POOL = torch.tensor(TRAIN if USE_TRAIN else HELD)
if "--sub" in sys.argv:
    # fresh-FAMILY arm: subtraction (a = -1), never trained; any s is fresh
    _apply = lambda rid, x: (rid - x) % S
    POOL = torch.arange(1, S)
    print("family: SUBTRACTION y=(s-x)%S — fresh family, out of the meta-learned one")

# ── shared conversations ─────────────────────────────────────────────────────
def make_convs():
    convs = []
    for _ in range(N_CONV):
        ctxs = []
        for k in range(K):
            while True:
                s = int(POOL[int(torch.randint(0, len(POOL), (1,)))])
                if not ctxs or s != ctxs[0][0]:
                    break
            perm = torch.randperm(S).tolist()
            ctxs.append((s, perm[:m], perm[m:]))
        convs.append(ctxs)
    return convs

CONVS = make_convs()

def pres_rows(k):
    rows = []
    for ctxs in CONVS:
        s, ex, _ = ctxs[k]
        row = [KEY_OFF + k]
        for xi in ex:
            row += [SYM_OFF + xi, SYM_OFF + _apply(s, xi)]
        rows.append(row)
    return torch.tensor(rows)

def query_batch(k, idx):
    rows, ys = [], []
    for ctxs in CONVS:
        s, _, unseen = ctxs[k]
        q = unseen[idx % len(unseen)]
        rows.append([KEY_OFF + k, SYM_OFF + q])
        ys.append(SYM_OFF + _apply(s, q))
    return torch.tensor(rows), torch.tensor(ys)

@torch.no_grad()
def eval_queries(mdl, mem0=None, carry=False):
    """TURNS query turns; carry=True threads the bank across turns (bank arm)."""
    mem, hits, tot = mem0, 0, 0
    for t in range(TURNS):
        k, idx = t % K, t // K
        xq, y = query_batch(k, idx)
        out = mdl(xq, init_mem=mem if carry else mem0, compute_logits=True)
        if carry:
            mem = out["mem_bank"]
        hits += int((out["logits"][:, -1].argmax(-1) == y).sum()); tot += y.numel()
    return hits / tot

fwd_flops = lambda tokens: 2 * P * tokens                    # per-lane proxy

# ── arm: bank (ours, forward-only) ───────────────────────────────────────────
t0 = time.perf_counter()
with torch.no_grad():
    mem = None
    for k in range(K):
        mem = model(pres_rows(k), init_mem=mem, compute_logits=False)["mem_bank"]
    acc_bank = eval_queries(model, mem, carry=True)
t_bank = (time.perf_counter() - t0) / N_CONV
cost_bank = fwd_flops(K * (1 + 2 * m))                       # 2 x 13-token forwards
print(f"\nbank    acc={acc_bank:.3f}   install cost/conv: {K}x{1+2*m}-token fwd "
      f"= {cost_bank/1e6:.1f} MFLOPs, {t_bank*1e3:.0f} ms (incl. queries)")

# ── arm: ablate + icl ────────────────────────────────────────────────────────
with torch.no_grad():
    acc_abl = eval_queries(model, None, carry=False)
    hits = tot = 0
    for t in range(TURNS):
        k, idx = t % K, t // K
        xq, y = query_batch(k, idx)
        rows = torch.cat([pres_rows(k), xq[:, 1:]], dim=1)   # pairs + query in-window
        out = model(rows, init_mem=None, compute_logits=True)
        hits += int((out["logits"][:, -1].argmax(-1) == y).sum()); tot += y.numel()
    acc_icl = hits / tot
print(f"ablate  acc={acc_abl:.3f}   (chance = {1/S:.3f})")
print(f"icl     acc={acc_icl:.3f}   (pairs in-window at every query)")

# ── arm: TTT (per-conversation gradient adaptation, bank ablated) ────────────
# 12 supervised examples per conversation (6 pairs x 2 keys) in query format.
def ttt_examples(ci):
    rows, ys = [], []
    for k in range(K):
        s, ex, _ = CONVS[ci][k]
        for xi in ex:
            rows.append([KEY_OFF + k, SYM_OFF + xi])
            ys.append(SYM_OFF + _apply(s, xi))
    return torch.tensor(rows), torch.tensor(ys)

def ttt_eval(mdl, ci):
    hits = tot = 0
    for t in range(TURNS):
        k, idx = t % K, t // K
        s, _, unseen = CONVS[ci][k]
        q = unseen[idx % len(unseen)]
        xq = torch.tensor([[KEY_OFF + k, SYM_OFF + q]])
        out = mdl(xq, init_mem=None, compute_logits=True)
        hits += int(out["logits"][0, -1].argmax(-1) == SYM_OFF + _apply(s, q)); tot += 1
    return hits, tot

if NO_TTT:
    sys.exit(0)
print(f"\nttt (bank ablated, AdamW on {K*m} pairs/conv, per-conv clone):")
best = {n: (0.0, None) for n in TTT_EVALS}
for lr in TTT_LRS:
    accs = {n: 0 for n in TTT_EVALS}; tots = {n: 0 for n in TTT_EVALS}
    fit_h = fit_t = 0                      # does TTT at least FIT its 12 pairs?
    loss0 = lossN = 0.0
    t0 = time.perf_counter()
    for ci in range(N_CONV):
        mdl = copy.deepcopy(model)
        opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.0)
        xr, yr = ttt_examples(ci)
        for step in range(1, max(TTT_EVALS) + 1):
            out = mdl(xr, init_mem=None, compute_logits=True)
            loss = F.cross_entropy(out["logits"][:, -1], yr)
            if step == 1:
                loss0 += float(loss)
            opt.zero_grad(); loss.backward(); opt.step()
            if step in accs:
                with torch.no_grad():
                    h, tt = ttt_eval(mdl, ci)
                accs[step] += h; tots[step] += tt
        lossN += float(loss)
        with torch.no_grad():
            out = mdl(xr, init_mem=None, compute_logits=True)
            fit_h += int((out["logits"][:, -1].argmax(-1) == yr).sum()); fit_t += yr.numel()
    dt = time.perf_counter() - t0
    line = f"  lr={lr:g}: " + "  ".join(
        f"acc@{n}={accs[n]/tots[n]:.3f}" for n in TTT_EVALS)
    print(line + f"   ({dt/N_CONV*1e3:.0f} ms/conv total)")
    print(f"           fit: pair-loss {loss0/N_CONV:.2f} -> {lossN/N_CONV:.2f}, "
          f"pair-acc@{max(TTT_EVALS)} = {fit_h/fit_t:.3f} (memorize-vs-generalize check)")
    for n in TTT_EVALS:
        if accs[n] / tots[n] > best[n][0]:
            best[n] = (accs[n] / tots[n], lr)

# ── headline table ───────────────────────────────────────────────────────────
step_cost = 3 * fwd_flops(K * m * 2)                         # fwd+bwd on 12 2-token rows
print("\n== headline ==")
print(f"bank: {acc_bank:.3f} for {cost_bank/1e6:.1f} MFLOPs (one forward pass, no backward)")
reached = None
for n in TTT_EVALS:
    a, lr = best[n]
    ratio = n * step_cost / cost_bank
    flag = ""
    if reached is None and a >= acc_bank:
        reached, flag = n, "  <- matches bank"
    print(f"ttt@{n:>3} (best lr={lr:g}): {a:.3f} for {n*step_cost/1e6:.1f} MFLOPs "
          f"({ratio:.1f}x bank){flag}")
if reached is None:
    print(f"TTT never reaches the bank's {acc_bank:.3f} within {max(TTT_EVALS)} steps.")
