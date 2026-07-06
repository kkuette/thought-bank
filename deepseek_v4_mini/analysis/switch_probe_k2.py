"""Zero-shot SWITCH probe on the GENERALIZING model (dsv4m final.pt, K=2).

Re-audit of the memorizing-regime claims (switch STICK=0 / recency-override,
canonical ident) on a model with held ~ rule_acc and a shared Fourier-drifted
code map. K=2 adaptation of switch_inspect/canonical_ident:

  conversation = present key0(s1), present key1(s3), queries alternating keys;
  at query turn SW, RE-present key0 with s2 != s1 (bank carried); queries go on.

Measured per arm:
  - per-turn accuracy for both keys (pre/post switch)
  - STICK: post-switch key0 answers matching the OLD rule s1
  - key1 (untouched) accuracy across the switch = collateral interference
  - bank-slot identity vs an EMPIRICAL clean-code dictionary (1-NN, not ridge:
    post-anneal drift) — is s1's code still physically in the bank post-switch?
Arms:
  sanity   : no switch, 8 turns (must replicate rule_acc ~1.0 / held ~0.85)
  sw_train : switch@4, 8 turns, s2 from TRAIN pool
  sw_held  : switch@4, 8 turns, s2 from HELD pool (fresh-rule install zero-shot)
  sw_long  : switch@8, 16 turns, s2 train (stretch: interaction with horizon)

Usage: PYTHONPATH=. python deepseek_v4_mini/analysis/switch_probe_k2.py [ckpt]
       [--cfg <yaml>] [--sweep]   (--sweep: switch-position invariance sweep)
       [--dump <out.json>]        (per-turn accuracy + old-rule-match arrays)
"""
import json, sys, torch, torch.nn.functional as F

WT = "."
sys.path.insert(0, WT)
from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.model import ThoughtBankLM
from deepseek_v4_mini.train import _rule_space

torch.manual_seed(0)
CKPT = (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else
        "checkpoints/multiturn_rule_k2_inter_s128_dsv4m/final.pt")
CFG  = "deepseek_v4_mini/configs/multiturn_rule_k2_inter_s128_dsv4m.yaml"
if "--cfg" in sys.argv:
    CFG = sys.argv[sys.argv.index("--cfg") + 1]
S, m, K, SYM_OFF = 128, 6, 2, 3
KEY_OFF = SYM_OFF + S
N = 128                                  # conversations (batched lanes)

import yaml
raw = yaml.safe_load(open(CFG))
cfg = ThoughtBankConfig.from_yaml(CFG)
model = ThoughtBankLM(cfg)
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd["model"]); model.eval()
print(f"loaded {CKPT.split('/')[-2]}/{CKPT.split('/')[-1]} (step {sd['step']})")

_units, _n, TRAIN, HELD, _apply = _rule_space(raw["data"])
TRAIN_T, HELD_T = torch.tensor(TRAIN), torch.tensor(HELD)
print(f"pools: {len(TRAIN)} train / {len(HELD)} held")

def draw(pool, n, avoid=None):
    s = pool[torch.randint(0, len(pool), (n,))]
    if avoid is not None:
        while bool((s == avoid).any()):
            cl = s == avoid
            s[cl] = pool[torch.randint(0, len(pool), (int(cl.sum()),))]
    return s

def present_seg(s_k, k, ex_k):
    X = torch.zeros(N, 1 + 2 * m, dtype=torch.long)
    X[:, 0] = KEY_OFF + k
    for b in range(N):
        for j, xi in enumerate(ex_k[b]):
            X[b, 1 + 2 * j] = SYM_OFF + xi
            X[b, 2 + 2 * j] = SYM_OFF + _apply(int(s_k[b]), xi)
    return X

def fresh_pools():
    ex, un = [], []
    for b in range(N):
        perm = torch.randperm(S).tolist()
        ex.append(perm[:m]); un.append(perm[m:])
    return ex, un

@torch.no_grad()
def clean_codes(rids):
    """Empirical clean-code dictionary: fresh-bank presentation write per rule."""
    codes = torch.zeros(len(rids), cfg.mem_dim)
    B = 64
    for i0 in range(0, len(rids), B):
        chunk = rids[i0:i0 + B]
        X = torch.zeros(len(chunk), 1 + 2 * m, dtype=torch.long)
        X[:, 0] = KEY_OFF + 0
        for r, rid in enumerate(chunk):
            perm = torch.randperm(S).tolist()[:m]
            for j, xi in enumerate(perm):
                X[r, 1 + 2 * j] = SYM_OFF + xi
                X[r, 2 + 2 * j] = SYM_OFF + _apply(int(rid), xi)
        out = model(X, init_mem=None, compute_logits=False)
        codes[i0:i0 + len(chunk)] = out["mem_bank"][:, -1]
    return F.normalize(codes, dim=1)

ALL_RIDS = TRAIN + HELD
DICT = clean_codes(ALL_RIDS)                       # [R, D] unit-norm
RID_IDX = {rid: i for i, rid in enumerate(ALL_RIDS)}

def nn_ident(bank, target_rid):
    """Fraction of lanes whose bank contains >=1 slot 1-NN-matching target_rid."""
    sl = F.normalize(bank.float(), dim=2)          # [N, slots, D]
    sim = torch.einsum("nsd,rd->nsr", sl, DICT)    # [N, slots, R]
    nn = sim.argmax(-1)                            # [N, slots]
    hit = torch.zeros(bank.size(0), dtype=torch.bool)
    for b in range(bank.size(0)):
        hit[b] = bool((nn[b] == RID_IDX[int(target_rid[b])]).any())
    return float(hit.float().mean())

@torch.no_grad()
def run(turns, sw, s2_pool, s1_pool=None):
    """One arm. sw=0 -> no switch. Returns per-turn acc per key + STICK + idents."""
    s = torch.zeros(N, K, dtype=torch.long)
    s[:, 0] = draw(s1_pool if s1_pool is not None else TRAIN_T, N)
    s[:, 1] = draw(TRAIN_T, N, avoid=s[:, 0])
    ex, un = {}, {}
    for k in range(K):
        ex[k], un[k] = fresh_pools()
    mem = None
    for k in range(K):
        out = model(present_seg(s[:, k], k, ex[k]), init_mem=mem, compute_logits=False)
        mem = out["mem_bank"]
    s1 = s[:, 0].clone()
    acc = {0: [], 1: []}; old_acc = {0: [], 1: []}; stick_n = stick_hit = 0
    ident_s1_pre = ident_s1_post = ident_s2_post = None
    q_cnt = [0] * K
    for t in range(turns):
        if sw and t == sw:
            ident_s1_pre = nn_ident(mem, s1)               # s1 in bank just before switch
            s[:, 0] = draw(s2_pool, N, avoid=s[:, 0])      # switch key0 -> s2
            ex[0], un[0] = fresh_pools()
            q_cnt[0] = 0
            out = model(present_seg(s[:, 0], 0, ex[0]), init_mem=mem, compute_logits=False)
            mem = out["mem_bank"]
        k = t % K
        xq = torch.zeros(N, 2, dtype=torch.long)
        xq[:, 0] = KEY_OFF + k
        q = torch.tensor([un[k][b][q_cnt[k] % len(un[k][b])] for b in range(N)])
        q_cnt[k] += 1
        xq[:, 1] = SYM_OFF + q
        out = model(xq, init_mem=mem, compute_logits=True)
        mem = out["mem_bank"]
        pred = out["logits"][:, -1].argmax(-1)
        y = SYM_OFF + torch.tensor([_apply(int(s[b, k]), int(q[b])) for b in range(N)])
        acc[k].append(float((pred == y).float().mean()))
        if k == 0:                                         # per-turn old-rule match
            y_old = SYM_OFF + torch.tensor([_apply(int(s1[b]), int(q[b])) for b in range(N)])
            old_acc[0].append(float((pred == y_old).float().mean()))
        if sw and t >= sw and k == 0:                      # STICK: old-rule answers
            y_old = SYM_OFF + torch.tensor([_apply(int(s1[b]), int(q[b])) for b in range(N)])
            stick_n += N; stick_hit += int((pred == y_old).sum())
    if sw:
        ident_s1_post = nn_ident(mem, s1)                  # s1 still in bank at end?
        ident_s2_post = nn_ident(mem, s[:, 0])
    return acc, (stick_hit / stick_n if stick_n else None), ident_s1_pre, ident_s1_post, ident_s2_post, old_acc

def fmt(a): return " ".join(f"{v:.2f}" for v in a)

ARMS = {
    "sanity  ": (8, 0, None, None),
    "sanity_h": (8, 0, None, HELD_T),
    "sw_train": (8, 4, TRAIN_T, None),
    "sw_held ": (8, 4, HELD_T, None),
    "sw_long ": (16, 8, TRAIN_T, None),
}
if "--sweep" in sys.argv:   # switch-POSITION invariance: same 16-turn conv, sw moves
    ARMS = {f"sw@{p:<2}   ": (16, p, TRAIN_T, None) for p in (2, 4, 6, 8, 10, 12, 14)}
dump = {}
for name, (turns, sw, pool, s1p) in ARMS.items():
    acc, stick, i1pre, i1post, i2post, old_acc = run(turns, sw, pool, s1p)
    k0, k1 = acc[0], acc[1]
    line = f"{name} key0[{fmt(k0)}] key1[{fmt(k1)}]"
    if sw:
        n_pre = (sw + 1) // 2
        pre0, post0 = k0[:n_pre], k0[n_pre:]
        line += (f"\n         pre0={sum(pre0)/len(pre0):.3f} post0={sum(post0)/len(post0):.3f}"
                 f" STICK={stick:.3f} | key1 avg={sum(k1)/len(k1):.3f}"
                 f" | bank1NN s1: pre-sw {i1pre:.2f} end {i1post:.2f} ; s2 end {i2post:.2f}")
    print(line)
    dump[name.strip()] = {
        "turns": turns, "sw": sw, "key0": k0, "key1": k1,
        "key0_old_rule": old_acc[0], "stick": stick,
        "ident_s1_pre": i1pre, "ident_s1_end": i1post, "ident_s2_end": i2post,
    }
if "--dump" in sys.argv:
    out = sys.argv[sys.argv.index("--dump") + 1]
    json.dump({"ckpt": CKPT, "arms": dump}, open(out, "w"), indent=1)
    print("dumped ->", out)
