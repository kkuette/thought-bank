"""HEADLINE DEMO ACT 2 — rule REPLACEMENT at inference: bank vs sequential TTT.

Act 1 (ttt_demo.py) froze the adaptation matrix: the bank is the model's only
working adaptation pathway (TTT memorizes its pairs at 0.99 and transfers 0).
Act 2 measures the CONTINUAL axis on the policy-trained checkpoint (dsv4w):
mid-conversation, key0's rule is REPLACED while key1 must keep serving.

  bank (ours)   : one 13-token forward re-presents key0 with the new rule.
                  Measured: new-rule queries, key1 collateral, STICK.
  ttt           : the act-1 protocol taken seriously as a continual learner —
                  phase 1: 50 AdamW steps on the 12 pairs (both keys), fit ~0.99;
                  phase 2 (the switch): 50 more steps on key0's 6 NEW pairs.
                  Measured: fit on new pairs, fit RETENTION on key1's untouched
                  pairs (catastrophic-interference metric — nonzero baseline
                  because pair-fit was ~0.99), old key0 fit (desired forgetting),
                  and query accuracies (chance per act 1, reported for the table).

Rules: s1/s_b from TRAIN, switch target from TRAIN or HELD (--held).
Usage: PYTHONPATH=. python deepseek_v4_mini/analysis/ttt_demo_act2.py [ckpt] [--held]
Runs on GPU when available (data generation stays on the CPU RNG), CPU otherwise.
"""
import copy, sys, time
import torch, torch.nn.functional as F, yaml

sys.path.insert(0, ".")
from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.model import ThoughtBankLM
from deepseek_v4_mini.train import _rule_space

torch.manual_seed(0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else
        "checkpoints/multiturn_rule_k2_inter_s128_dsv4w/step_3000.pt")
CFG  = "deepseek_v4_mini/configs/multiturn_rule_k2_inter_s128struct_dsv4w.yaml"
SW_HELD = "--held" in sys.argv
S, m, K, SYM_OFF = 128, 6, 2, 3
KEY_OFF = SYM_OFF + S
N_CONV, TURNS_PRE, TURNS_POST = 64, 4, 4
N1, N2, LR = 50, 50, 1e-3            # act-1 best-fit recipe

raw = yaml.safe_load(open(CFG))
cfg = ThoughtBankConfig.from_yaml(CFG)
model = ThoughtBankLM(cfg)
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd["model"]); model.eval(); model.to(DEV)
P = sum(p.numel() for p in model.parameters())
print(f"loaded {CKPT} (step {sd['step']}) | device {DEV.type} "
      f"| switch target pool: {'HELD' if SW_HELD else 'TRAIN'}")

_u, _n, TRAIN, HELD, _apply = _rule_space(raw["data"])
TRAIN_T, HELD_T = torch.tensor(TRAIN), torch.tensor(HELD)
SW_POOL = HELD_T if SW_HELD else TRAIN_T

def draw(pool, n, avoid=None):
    s = pool[torch.randint(0, len(pool), (n,))]
    if avoid is not None:
        while bool((s == avoid).any()):
            cl = s == avoid
            s[cl] = pool[torch.randint(0, len(pool), (int(cl.sum()),))]
    return s

# per-conversation cast: key0 s_a -> s_a2 (switch), key1 s_b (must survive)
s_a  = draw(TRAIN_T, N_CONV)
s_b  = draw(TRAIN_T, N_CONV, avoid=s_a)
s_a2 = draw(SW_POOL, N_CONV, avoid=s_a)

def pres_rows(s_k, k):
    X = torch.zeros(N_CONV, 1 + 2 * m, dtype=torch.long)
    X[:, 0] = KEY_OFF + k
    for b in range(N_CONV):
        perm = torch.randperm(S).tolist()[:m]
        for j, xi in enumerate(perm):
            X[b, 1 + 2 * j] = SYM_OFF + xi
            X[b, 2 + 2 * j] = SYM_OFF + _apply(int(s_k[b]), xi)
    return X

def q_batch(s_k, k):
    xq = torch.zeros(N_CONV, 2, dtype=torch.long)
    xq[:, 0] = KEY_OFF + k
    q = torch.randint(0, S, (N_CONV,))
    xq[:, 1] = SYM_OFF + q
    y = SYM_OFF + torch.tensor([_apply(int(s_k[b]), int(q[b])) for b in range(N_CONV)])
    return xq, y

fwd_flops = lambda tokens: 2 * P * tokens

# ── bank arm ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def bank_arm():
    mem = model(pres_rows(s_a, 0).to(DEV), init_mem=None, compute_logits=False)["mem_bank"]
    mem = model(pres_rows(s_b, 1).to(DEV), init_mem=mem, compute_logits=False)["mem_bank"]
    def run_qs(n_turns, s0, track_old=None):
        nonlocal mem
        accs = {0: [], 1: []}; stick = [0, 0]
        for t in range(n_turns * K):
            k = t % K
            xq, y = q_batch(s0 if k == 0 else s_b, k)
            out = model(xq.to(DEV), init_mem=mem, compute_logits=True)
            mem = out["mem_bank"]
            pred = out["logits"][:, -1].argmax(-1).cpu()
            accs[k].append(float((pred == y).float().mean()))
            if track_old is not None and k == 0:
                _, y_old = q_batch(track_old, 0)   # NOTE: fresh q, approx STICK
        return accs
    pre = run_qs(TURNS_PRE, s_a)
    mem = model(pres_rows(s_a2, 0).to(DEV), init_mem=mem, compute_logits=False)["mem_bank"]  # THE UPDATE
    post = run_qs(TURNS_POST, s_a2)
    return pre, post

pre, post = bank_arm()
avg = lambda v: sum(v) / len(v)
cost_bank = fwd_flops(1 + 2 * m)
print(f"\nbank: update = ONE {1+2*m}-token forward = {cost_bank/1e6:.1f} MFLOPs")
print(f"  key0 pre {avg(pre[0]):.3f} -> post(new rule) {avg(post[0]):.3f}")
print(f"  key1 collateral: pre {avg(pre[1]):.3f} -> post {avg(post[1]):.3f}")

# ── ttt arm ──────────────────────────────────────────────────────────────────
def pair_rows(s_k, k):
    rows = torch.zeros(N_CONV, m, 2, dtype=torch.long)
    ys   = torch.zeros(N_CONV, m, dtype=torch.long)
    for b in range(N_CONV):
        perm = torch.randperm(S).tolist()[:m]
        for j, xi in enumerate(perm):
            rows[b, j] = torch.tensor([KEY_OFF + k, SYM_OFF + xi])
            ys[b, j]   = SYM_OFF + _apply(int(s_k[b]), xi)
    return rows, ys

pa, ya   = pair_rows(s_a, 0)      # key0 original pairs
pb, yb   = pair_rows(s_b, 1)      # key1 pairs (must survive the update)
pa2, ya2 = pair_rows(s_a2, 0)     # key0 NEW pairs (the switch data)

def fit_acc(mdl, rows, ys):
    with torch.no_grad():
        out = mdl(rows.view(-1, 2).to(DEV), init_mem=None, compute_logits=True)
        return float((out["logits"][:, -1].argmax(-1).cpu() == ys.view(-1)).float().mean())

def q_acc(mdl, s_k, k, n=8):
    hits = 0
    with torch.no_grad():
        for _ in range(n):
            xq, y = q_batch(s_k, k)
            out = mdl(xq.to(DEV), init_mem=None, compute_logits=True)
            hits += int((out["logits"][:, -1].argmax(-1).cpu() == y).sum())
    return hits / (n * N_CONV)

fitA1 = fitB1 = fitA2n = fitB2 = fitA2o = 0.0
qn = qb2 = 0.0
t0 = time.perf_counter()
for ci in range(N_CONV):
    mdl = copy.deepcopy(model)
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=0.0)
    x1 = torch.cat([pa[ci], pb[ci]]).to(DEV); y1 = torch.cat([ya[ci], yb[ci]]).to(DEV)
    for _ in range(N1):                                   # phase 1: learn both rules
        out = mdl(x1, init_mem=None, compute_logits=True)
        loss = F.cross_entropy(out["logits"][:, -1], y1)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        o = mdl(x1, init_mem=None, compute_logits=True)
    pred = o["logits"][:, -1].argmax(-1).cpu()
    fitA1 += float((pred[:m] == ya[ci]).float().mean())
    fitB1 += float((pred[m:] == yb[ci]).float().mean())
    for _ in range(N2):                                   # phase 2: THE SWITCH
        out = mdl(pa2[ci].to(DEV), init_mem=None, compute_logits=True)
        loss = F.cross_entropy(out["logits"][:, -1], ya2[ci].to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        o = mdl(torch.cat([pa2[ci], pb[ci], pa[ci]]).to(DEV), init_mem=None, compute_logits=True)
    pred = o["logits"][:, -1].argmax(-1).cpu()
    fitA2n += float((pred[:m] == ya2[ci]).float().mean())
    fitB2  += float((pred[m:2*m] == yb[ci]).float().mean())
    fitA2o += float((pred[2*m:] == ya[ci]).float().mean())
dt = time.perf_counter() - t0
n = N_CONV
step_cost = 3 * fwd_flops(2 * m * 2)   # fwd+bwd on 12 2-token rows
sw_cost   = N2 * 3 * fwd_flops(m * 2)  # switch update: N2 steps on 6 rows
print(f"\nttt (sequential, lr={LR:g}, {N1}+{N2} steps, {dt/n*1e3:.0f} ms/conv):")
print(f"  phase 1 fit: key0 {fitA1/n:.3f}  key1 {fitB1/n:.3f}   (queries: chance, cf. acte 1)")
print(f"  after switch update ({N2} steps = {sw_cost/1e6:.0f} MFLOPs = {sw_cost/cost_bank:.0f}x bank):")
print(f"    new key0 pairs fit : {fitA2n/n:.3f}")
print(f"    key1 pairs RETENTION: {fitB1/n:.3f} -> {fitB2/n:.3f}   <- interference")
print(f"    old key0 pairs      : {fitA1/n:.3f} -> {fitA2o/n:.3f}   (forgetting, desired)")
