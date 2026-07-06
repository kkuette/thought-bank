"""Superposition mechanism probe (paper Fig. 4), K=2 policy checkpoint.

Three measurements on the same checkpoint:
  1. end-of-conversation bank geometry: mean slot-slot cosine matrix and
     entropy effective rank of the 8 slots (expected ~1: all slots carry
     near-copies of one superposed vector);
  2. rule-code geometry: one clean-bank presentation write per rule, over
     the full pool -> effective rank of the code cloud and the inter-rule
     cosine distribution (expected rank >> 1: rules stay separable);
  3. per-turn write redundancy across a switch conversation: max cosine of
     each new write against the resident bank (expected ~1 for steady-state
     rehearsal writes, a dip at the switch write = genuinely novel content).

Usage: PYTHONPATH=. python deepseek_v4_mini/analysis/superposition_probe.py \
           <ckpt> --cfg <yaml> [--dump out.json]
"""
import json, sys, torch, torch.nn.functional as F

sys.path.insert(0, ".")
from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.model import ThoughtBankLM
from deepseek_v4_mini.train import _rule_space, _effective_rank

torch.manual_seed(0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = sys.argv[1]
CFG = sys.argv[sys.argv.index("--cfg") + 1] if "--cfg" in sys.argv else \
    "deepseek_v4_mini/configs/multiturn_rule_k2_inter_s128struct_dsv4w.yaml"
S, m, K, SYM_OFF = 128, 6, 2, 3
KEY_OFF = SYM_OFF + S
N = 128

import yaml
raw = yaml.safe_load(open(CFG))
cfg = ThoughtBankConfig.from_yaml(CFG)
model = ThoughtBankLM(cfg)
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd["model"]); model.eval(); model.to(DEV)
print(f"loaded {CKPT} (step {sd['step']}) | device {DEV.type}")

_units, _n, TRAIN, HELD, _apply = _rule_space(raw["data"])
TRAIN_T = torch.tensor(TRAIN)

def draw(pool, n, avoid=None):
    s = pool[torch.randint(0, len(pool), (n,))]
    if avoid is not None:
        while bool((s == avoid).any()):
            cl = s == avoid
            s[cl] = pool[torch.randint(0, len(pool), (int(cl.sum()),))]
    return s

def present_seg(s_k, k, n=N):
    X = torch.zeros(n, 1 + 2 * m, dtype=torch.long)
    X[:, 0] = KEY_OFF + k
    for b in range(n):
        perm = torch.randperm(S).tolist()[:m]
        for j, xi in enumerate(perm):
            X[b, 1 + 2 * j] = SYM_OFF + xi
            X[b, 2 + 2 * j] = SYM_OFF + _apply(int(s_k[b]), xi)
    return X

def query_seg(k, n=N):
    X = torch.zeros(n, 2, dtype=torch.long)
    X[:, 0] = KEY_OFF + k
    X[:, 1] = SYM_OFF + torch.randint(0, S, (n,))
    return X

def write_redundancy(prev_mem, new_mem):
    """Max cosine of the newly appended slot vs the previous bank content."""
    if prev_mem is None or prev_mem.size(1) == 0:
        return None
    new = F.normalize(new_mem[:, -1].float(), dim=1)          # [N, D]
    old = F.normalize(prev_mem.float(), dim=2)                # [N, M, D]
    cos = torch.einsum("nd,nmd->nm", new, old)
    return float(cos.amax(dim=1).mean())

out = {}

# --- 1. end-of-conversation bank geometry --------------------------------
with torch.no_grad():
    s = torch.zeros(N, K, dtype=torch.long)
    s[:, 0] = draw(TRAIN_T, N)
    s[:, 1] = draw(TRAIN_T, N, avoid=s[:, 0])
    mem = None
    for k in range(K):
        mem = model(present_seg(s[:, k], k).to(DEV), init_mem=mem, compute_logits=False)["mem_bank"]
    for t in range(8):
        mem = model(query_seg(t % K).to(DEV), init_mem=mem, compute_logits=False)["mem_bank"]
    sl = F.normalize(mem.float(), dim=2)                      # [N, M, D]
    sim = torch.einsum("nad,nbd->nab", sl, sl).mean(0)        # [M, M]
    ranks = [_effective_rank(mem[b:b + 1]) for b in range(N)]
    out["bank_slot_sim"] = sim.tolist()
    out["bank_eff_rank_mean"] = sum(ranks) / len(ranks)
    print(f"bank eff_rank (mean over {N} conv): {out['bank_eff_rank_mean']:.2f} / {mem.size(1)}")

# --- 2. rule-code geometry ------------------------------------------------
with torch.no_grad():
    rids = TRAIN + HELD
    codes = torch.zeros(len(rids), cfg.mem_dim)
    B = 64
    for i0 in range(0, len(rids), B):
        chunk = torch.tensor(rids[i0:i0 + B])
        bank = model(present_seg(chunk, 0, n=len(chunk)).to(DEV), init_mem=None,
                     compute_logits=False)["mem_bank"]
        codes[i0:i0 + len(chunk)] = bank[:, -1].cpu()
    out["rule_eff_rank"] = _effective_rank(codes.unsqueeze(0))
    out["rule_codes"] = codes.tolist()
    out["rule_ids"] = rids
    cn = F.normalize(codes, dim=1)
    cs = cn @ cn.t()
    iu = torch.triu_indices(len(rids), len(rids), offset=1)
    inter = cs[iu[0], iu[1]]
    out["inter_rule_cos"] = inter.tolist()
    out["inter_rule_cos_mean"] = float(inter.mean())
    out["n_rules"] = len(rids)
    out["held_idx_start"] = len(TRAIN)
    print(f"rule-code cloud: eff_rank {out['rule_eff_rank']:.1f} / {cfg.mem_dim}, "
          f"inter-rule cos {out['inter_rule_cos_mean']:.2f}")

# --- 3. per-write redundancy across a switch -----------------------------
with torch.no_grad():
    s = torch.zeros(N, K, dtype=torch.long)
    s[:, 0] = draw(TRAIN_T, N)
    s[:, 1] = draw(TRAIN_T, N, avoid=s[:, 0])
    mem = None
    redund, labels = [], []
    for k in range(K):
        new = model(present_seg(s[:, k], k).to(DEV), init_mem=mem, compute_logits=False)["mem_bank"]
        r = write_redundancy(mem, new)
        if r is not None:
            redund.append(r); labels.append(f"present k{k}")
        mem = new
    for t in range(16):
        if t == 8:                                            # switch key0
            s[:, 0] = draw(TRAIN_T, N, avoid=s[:, 0])
            new = model(present_seg(s[:, 0], 0).to(DEV), init_mem=mem, compute_logits=False)["mem_bank"]
            redund.append(write_redundancy(mem, new)); labels.append("SWITCH k0")
            mem = new
        new = model(query_seg(t % K).to(DEV), init_mem=mem, compute_logits=False)["mem_bank"]
        redund.append(write_redundancy(mem, new)); labels.append(f"query t{t}")
        mem = new
    out["write_redundancy"] = redund
    out["write_labels"] = labels
    print("write redundancy:", " ".join(f"{l}={r:.2f}" for l, r in zip(labels, redund)))

if "--dump" in sys.argv:
    path = sys.argv[sys.argv.index("--dump") + 1]
    json.dump(out, open(path, "w"))
    print("dumped ->", path)
