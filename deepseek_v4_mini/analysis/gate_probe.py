"""GATE v2 INFERENCE PROBE — screen write-gate variants on a trained gate-OFF ckpt.

Budget rationale (user, 2026-07-07): a from-scratch struct run costs ~4h30 on
the 3090 and the gate discriminant is late (dsv4u: bootstrap boost,
consolidation cost). This probe screens the variants at INFERENCE on
dsv4w_s43@4000 in minutes: the gate only touches the write path, so flipping
the config flag on the loaded model measures the MECHANICAL effect of each
gate on a circuit whose read was trained on full-norm ungated codes. It
cannot say what a model TRAINED with the gate would learn — it kills
dead-on-arrival variants and picks the one worth the single from-scratch run.

Arms (same conversations, same rules, same seeds):
  off      : baseline write (what the model was trained with)
  novelty  : v2a — write scaled by g = clamp(1 - max_cos(m, bank), 0, 1)
  delta    : v2b — write m - max(<m, s*>, 0) * s*  (subtract stored component)

Measures per arm:
  1. installation  : K=2 held rules presented once, queried over 16 turns
                     (carry) -> acc per turn; early (0-7) vs late (8-15)
                     halves separate "read survives the gate" from
                     "rehearsal carry starved" (FIFO evicts presentation
                     writes ~turn 6; late acc rides on query-write copies).
  2. replacement   : key 0 re-presented with a FRESH rule mid-conversation,
                     8 more query turns -> does the gate pass the overwrite
                     (new code ~ novel) and preserve the neighbour key?
  3. telemetry     : E[gate] (novelty: g; delta: |m'|/|m|) and E[max_cos]
                     per phase -> does the gate BITE on real conversations
                     (redund ~0.5 expected on query writes)?

Usage: PYTHONPATH=. python deepseek_v4_mini/analysis/gate_probe.py [ckpt] [--cfg path]
"""
import sys
import torch, yaml

sys.path.insert(0, ".")
from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.model import ThoughtBankLM
from deepseek_v4_mini.train import _rule_space

torch.manual_seed(0)
DEV  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else
        "checkpoints/multiturn_rule_k2_inter_s128_dsv4w_s43/step_4000.pt")
CFG  = "deepseek_v4_mini/configs/multiturn_rule_k2_inter_s128struct_dsv4w_s43.yaml"
if "--cfg" in sys.argv:
    CFG = sys.argv[sys.argv.index("--cfg") + 1]

S, m, K, SYM_OFF = 128, 6, 2, 3
KEY_OFF = SYM_OFF + S
N_CONV, TURNS, TURNS_POST = 64, 16, 8

raw = yaml.safe_load(open(CFG))
cfg = ThoughtBankConfig.from_yaml(CFG)
model = ThoughtBankLM(cfg)
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd["model"]); model.eval(); model.to(DEV)
print(f"loaded step {sd['step']} | device {DEV.type} | held pool (fresh rules)")

_units, _n, TRAIN, HELD, _apply = _rule_space(raw["data"])
POOL = torch.tensor(HELD)

# ── shared conversations: K=2 held rules + a replacement rule for key 0 ──────
def draw(exclude):
    while True:
        s = int(POOL[int(torch.randint(0, len(POOL), (1,)))])
        if s not in exclude:
            return s

CONVS = []
for _ in range(N_CONV):
    s0 = draw(()); s1 = draw((s0,)); s0b = draw((s0, s1))   # s0b replaces s0 later
    perm  = torch.randperm(S).tolist()
    permb = torch.randperm(S).tolist()
    CONVS.append({"rules": (s0, s1, s0b),
                  "ex":    (perm[:m], perm[m:2 * m], permb[:m]),
                  "pool":  perm[2 * m:]})

def pres_rows(which):        # which: 0, 1 (initial keys) or 2 (replacement on key 0)
    key = 0 if which in (0, 2) else 1
    rows = []
    for c in CONVS:
        row = [KEY_OFF + key]
        for xi in c["ex"][which]:
            row += [SYM_OFF + xi, SYM_OFF + _apply(c["rules"][which], xi)]
        rows.append(row)
    return torch.tensor(rows)

def query_batch(k, idx, post=False):
    rows, ys = [], []
    for c in CONVS:
        q = c["pool"][idx % len(c["pool"])]
        rid = c["rules"][2] if (post and k == 0) else c["rules"][k]
        rows.append([KEY_OFF + k, SYM_OFF + q])
        ys.append(SYM_OFF + _apply(rid, q))
    return torch.tensor(rows), torch.tensor(ys)

ARMS = {
    "off":     {},
    "novelty": {"mem_write_gate_novelty": True},
    "delta":   {"mem_write_gate_delta": True},
    "merge":   {"mem_write_gate_merge": True},   # v2c: dedup-refresh, no attenuation
}
FLAGS = ("mem_write_gate_novelty", "mem_write_gate_delta", "mem_write_gate_merge")

@torch.no_grad()
def run_arm(name, flags):
    for f in FLAGS: setattr(cfg, f, False)
    for f, v in flags.items(): setattr(cfg, f, v)
    stats = {"pres": [], "query": []}          # (gate, redund) per phase

    def fwd(rows, mem, phase, logits=False):
        out = model(rows.to(DEV), init_mem=mem, compute_logits=logits)
        if name != "off" and out.get("write_alpha") is not None:
            stats[phase].append((float(out["write_alpha"]),
                                 float(out["write_redundancy"])))
        else:
            stats[phase].append((1.0, float(out["write_redundancy"])))
        return out

    mem = None
    for k in (0, 1):                                        # install both rules
        mem = fwd(pres_rows(k), mem, "pres")["mem_bank"]

    acc_turn = []
    for t in range(TURNS):                                  # phase 1: retention
        k, idx = t % K, t // K
        xq, y = query_batch(k, idx)
        out = fwd(xq, mem, "query", logits=True)
        mem = out["mem_bank"]
        acc_turn.append(float((out["logits"][:, -1].argmax(-1).cpu() == y).float().mean()))

    mem = fwd(pres_rows(2), mem, "pres")["mem_bank"]        # phase 2: replace key 0
    hits = {0: [], 1: []}
    for t in range(TURNS_POST):
        k, idx = t % K, TURNS // K + t // K
        xq, y = query_batch(k, idx, post=True)
        out = fwd(xq, mem, "query", logits=True)
        mem = out["mem_bank"]
        hits[k].append(float((out["logits"][:, -1].argmax(-1).cpu() == y).float().mean()))

    early = sum(acc_turn[:8]) / 8; late = sum(acc_turn[8:]) / 8
    repl  = sum(hits[0]) / len(hits[0]); pres = sum(hits[1]) / len(hits[1])
    gq  = sum(g for g, _ in stats["query"]) / len(stats["query"])
    gp  = sum(g for g, _ in stats["pres"])  / len(stats["pres"])
    rq  = sum(r for _, r in stats["query"]) / len(stats["query"])
    print(f"{name:8s} early={early:.3f}  late={late:.3f}  repl(k0)={repl:.3f}  "
          f"pres(k1)={pres:.3f} | gate: pres={gp:.3f} query={gq:.3f}  "
          f"redund(query)={rq:.3f}")
    print(f"         per-turn: " + " ".join(f"{a:.2f}" for a in acc_turn))
    return early, late, repl, pres

print(f"\nchance = {1/S:.3f} | max_mem={cfg.max_mem}, FIFO evicts presentation "
      f"writes ~turn 6 -> late half rides on query-write rehearsal\n")
for name, flags in ARMS.items():
    run_arm(name, flags)
for f in FLAGS: setattr(cfg, f, False)
