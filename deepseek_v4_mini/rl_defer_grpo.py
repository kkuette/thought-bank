"""dsv6 RL phase — GRPO on the WRITE DECISION (when to write, not what).

The pretrained model writes one gist per forward UNCONDITIONALLY (model.py step 5);
<think> is just an input token. The harness turns this into a policy:

  action a_i ∈ {write, skip} at each chunk boundary
    write -> the bank returned by the turn's forward (new gist, FIFO) is carried;
    skip  -> it is DISCARDED (the chunk is forgotten: nothing crosses turns but
             the bank, so skip = real information loss, not a no-op).

  π(write | state) = the model's own next-token P(<think>) at the last content
  position of the chunk (temperature-scaled). RL shapes the LM's <think> head
  directly, so the artifact stays a pure LM: at inference, "emits <think>" =
  "writes". No separate policy head, no value network.

  reward (terminal, verifiable by construction — RLVR):
      r = -CE(deferred continuation of the final target from the final bank)
          - lambda_write * n_writes
  The GRPO group = G action-samplings of the SAME conversation; advantage =
  (r - mean)/std within the group. lambda_write prices a write: the marginal
  GAP per write is known from the depth curve (~+0.1 nat/write incremental),
  lambda must sit BELOW it so informative writes stay profitable.

DAPO-era fixes wired from day 1 (the 2026 post-training lessons):
  * dynamic sampling — groups with ~zero reward std carry no gradient; drop and
    resample them, and REPORT the dropped fraction (a silent effective-batch
    collapse is the failure mode);
  * asymmetric clip (clip_high > clip_low, "clip-higher") so low-probability
    actions can gain mass — our good behavior is "write rarely but well";
  * KL to the frozen reference ON THE BOUNDARY BERNOULLI ONLY: the chunk tokens
    are teacher-forced (never sampled), so the LM content distribution has no
    ratio to constrain — only the write policy does.

Scope note: pass 2 replays each turn from the STORED (detached) bank, so the
policy gradient flows into "when to write" (boundary logits given the state),
NOT into the write contents. Training WHAT to write is a later phase.

REWARD DESIGN RULE (see the standing note in FINDINGS.md): never make the
survival of specific memories intrinsically rewarded. Every reward term here is
task-grounded (final-continuation CE, a per-write cost); none rewards retaining
particular contents. Keep it that way when extending this harness: rewarding
retention itself creates instrumental pressure to resist resets/rollbacks of
the bank (dsv4 already showed covert rehearsal emerging from eviction pressure
alone — state-preserving behavior needs no "self", only an incentive).

Anti reward-hacking instrumentation: per-eval, the positional correlation of
writes (a policy re-discovering "write every ~N tokens" without reading content
shows up as corr -> 1) + policy reward vs the always-write / never-write forced
baselines (the emergent-behavior headline: selective > always at equal budget).

  python -m deepseek_v4_mini.rl_defer_grpo deepseek_v4_mini/configs/rl_defer_grpo_97m.yaml
"""
from __future__ import annotations

import copy
import json
import math
import os
import statistics as st
import sys
import time

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer

from .config import ThoughtBankConfig
from .model import ThoughtBankLM
from .code_data import CodeChunkStream


# ── policy primitives ────────────────────────────────────────────────────────

def boundary_step(model, x, bank, think_id, temp):
    """One turn: forward chunk x (+<think> appended) on the carried bank.
    Returns (logp_write, logp_skip, new_bank, p_write): the Bernoulli in log
    space from the last CONTENT position (the one predicting <think>), and the
    bank as returned by the forward (carried only if the action says write)."""
    xt = torch.cat([x, torch.full((1, 1), think_id, dtype=torch.long, device=x.device)], 1)
    o = model(xt, init_mem=bank)
    lg = o["logits"].float()[0, x.size(1) - 1] / temp
    logp = F.log_softmax(lg, dim=-1)
    p_w = logp[think_id].exp().clamp(1e-6, 1.0 - 1e-6)
    return torch.log(p_w), torch.log1p(-p_w), o["mem_bank"], p_w


def defer_ce(model, bank, tgt, blank_id):
    """CE of the deferred continuation: DL <blank> tokens on the bank alone."""
    di = torch.full((1, tgt.size(1)), blank_id, dtype=torch.long, device=tgt.device)
    lg = model(di, init_mem=bank)["logits"].float()
    return F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1))


# ── rollouts (pass 1, no grad) ───────────────────────────────────────────────
#
# seed_bank note: with init_mem=None the model draws RANDOM seed slots per
# forward (by design), which would inject reward noise UNCORRELATED with the
# actions — enough std (~0.01 nat) to defeat dynamic sampling and blur the
# advantages. The harness therefore materializes the seed bank ONCE per
# conversation and shares it across the G rollouts and both forced baselines:
# same state distribution as pretraining (fresh seeds per conv), but within a
# group the reward differences are actions-only.

def conv_seed_bank(model):
    p = next(model.parameters())
    with torch.no_grad():
        return model.thought_stream.seed_bank(1, p.device, p.dtype)


@torch.no_grad()
def forced_reward(model, chunks, tgt, write_all, lam, ids, sbank):
    """Baseline trajectory: every action forced to write (True) or skip (False)."""
    think_id, blank_id = ids
    bank = sbank
    if write_all:
        for x in chunks:
            _, _, bank, _ = boundary_step(model, x, bank, think_id, 1.0)
    ce = float(defer_ce(model, bank, tgt, blank_id))
    return -ce - lam * (len(chunks) if write_all else 0), ce


@torch.no_grad()
def rollout(model, chunks, tgt, temp, lam, ids, rng, sbank):
    """One sampled trajectory. Returns dict with per-turn records for pass 2:
    action, old logp (behavior policy), the INPUT bank of the turn (detached —
    pass 2 replays each turn from this exact state), plus reward pieces."""
    think_id, blank_id = ids
    bank = sbank
    recs = []
    for x in chunks:
        lw, ls, new_bank, p_w = boundary_step(model, x, bank, think_id, temp)
        a = 1 if rng.random() < float(p_w) else 0
        recs.append({"x": x, "a": a,
                     "logp_old": float(lw if a else ls),
                     "bank_in": bank.detach(),
                     "p": float(p_w)})
        if a:
            bank = new_bank.detach()
    n_w = sum(r["a"] for r in recs)
    ce = float(defer_ce(model, bank, tgt, blank_id))
    return {"recs": recs, "ce": ce, "n_writes": n_w, "reward": -ce - lam * n_w}


# ── GRPO update (pass 2, with grad) ──────────────────────────────────────────

def grpo_backward(model, ref, group, advs, temp, clip_lo, clip_hi, kl_coef,
                  ids, scale):
    """Replay each rollout turn-by-turn from the stored banks, accumulate the
    clipped policy-gradient loss + Bernoulli KL to ref, and .backward().
    Per-turn replay from detached banks => no cross-turn graph (memory-light);
    identical numerics to pass 1 because pass 1 detached banks the same way."""
    think_id, _ = ids
    tot_loss = tot_kl = 0.0; n_act = 0
    for ro, A in zip(group, advs):
        loss = torch.zeros((), device=next(model.parameters()).device)
        for r in ro["recs"]:
            lw, ls, _, p_w = boundary_step(model, r["x"], r["bank_in"], think_id, temp)
            logp_new = lw if r["a"] else ls
            ratio = (logp_new - r["logp_old"]).exp()
            surr = torch.min(ratio * A,
                             ratio.clamp(1.0 - clip_lo, 1.0 + clip_hi) * A)
            with torch.no_grad():
                _, _, _, p_ref = boundary_step(ref, r["x"], r["bank_in"], think_id, temp)
            kl = (p_w * (p_w / p_ref).log()
                  + (1 - p_w) * ((1 - p_w) / (1 - p_ref)).log())
            loss = loss + (-surr + kl_coef * kl)
            tot_kl += float(kl.detach()); n_act += 1
        (loss * scale).backward()
        tot_loss += float(loss.detach())
    return tot_loss, tot_kl / max(n_act, 1)


# ── data plumbing ────────────────────────────────────────────────────────────

def next_rl_conv(stream, defer_len, device, min_turns=2):
    """A conversation for RL: segs[:-1] = writable turns, the LAST seg's opening
    defer_len tokens = the terminal target. Resamples until deep enough."""
    while True:
        segs = stream.next_conv()
        if len(segs) >= min_turns + 1 and segs[-1]["input_ids"].size(1) >= defer_len:
            chunks = [s["input_ids"].to(device) for s in segs[:-1]]
            tgt = segs[-1]["input_ids"][:, :defer_len].to(device)
            return chunks, tgt


def pos_write_corr(rollouts):
    """|corr(boundary token-position, write)| across a batch of rollouts — the
    positional-ritual detector (1.0 = writes are a pure function of position)."""
    xs, ys = [], []
    for ro in rollouts:
        pos = 0
        for r in ro["recs"]:
            pos += r["x"].size(1)
            xs.append(float(pos)); ys.append(float(r["a"]))
    if len(xs) < 8 or len(set(ys)) < 2 or len(set(xs)) < 2:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = sum((a - mx) ** 2 for a in xs); vy = sum((b - my) ** 2 for b in ys)
    return abs(cov / max((vx * vy) ** 0.5, 1e-9))


# ── main ─────────────────────────────────────────────────────────────────────

def main(cfg_path: str) -> None:
    raw = yaml.safe_load(open(cfg_path)); r = raw["rl"]; d = raw["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(r.get("seed", 0)))
    import random as _random
    rng = _random.Random(int(r.get("seed", 0)) + 17)

    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    add = [x for x in ("<think>", "<blank>") if x not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    ids = (tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("<blank>"))

    mcfg = dict(raw["model"]); mcfg["vocab_size"] = len(tok)
    model = ThoughtBankLM(ThoughtBankConfig(**mcfg)).to(device)
    ck = torch.load(r["init_from"], map_location="cpu")
    model.load_state_dict(ck["model"])
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    print(f"GRPO policy {model.num_params():,} params <- {r['init_from']} "
          f"(step {ck.get('step', '?')}) | ref frozen | device {device}", flush=True)

    L, K = int(d["seq_len"]), int(d["chunks_per_conv"])
    defer_len = int(d.get("defer_len", 16))
    sd = dict(seq_len=L, chunks_per_conv=K, batch=1,
              n_files=int(d.get("n_files", 800)),
              dataset=d.get("dataset", "codeparrot/codeparrot-clean-valid"),
              data_dir=d.get("data_dir", ""), stream_cap=int(d.get("stream_cap", 60000)),
              cache_dir=d.get("cache_dir", "data_cache"),
              content_key=d.get("content_key", "content"),
              config_name=d.get("config_name", ""),
              min_chunks=int(d.get("min_chunks", 1)),
              stream_skip=int(d.get("stream_skip", 0)),
              sources=d.get("sources"), var_chunk=d.get("var_chunk"),
              seed=int(r.get("seed", 0)))
    train_stream = CodeChunkStream(tok, split="train", **sd)
    eval_stream = CodeChunkStream(tok, split="held", **sd)

    G = int(r.get("group_size", 8)); gps = int(r.get("groups_per_step", 4))
    steps = int(r["steps"]); temp = float(r.get("temp", 1.0))
    lam = float(r.get("lambda_write", 0.03))
    clip_lo = float(r.get("clip_low", 0.2)); clip_hi = float(r.get("clip_high", 0.28))
    kl_coef = float(r.get("kl_coef", 1.0e-3))
    min_std = float(r.get("min_reward_std", 1.0e-4))
    max_rs = int(r.get("max_resample", 4))
    opt = torch.optim.AdamW(model.parameters(), lr=float(r.get("lr", 5.0e-6)),
                            weight_decay=0.0)
    save_dir = r.get("save_dir", "checkpoints/rl_defer_grpo")
    os.makedirs(save_dir, exist_ok=True)
    mfile = r.get("metrics_file")
    if mfile:
        os.makedirs(os.path.dirname(mfile), exist_ok=True)
    print(f"GRPO: G {G} x groups/step {gps} | steps {steps} | temp {temp} | "
          f"lambda_write {lam} | clip [{clip_lo},{clip_hi}] (asym) | kl {kl_coef} | "
          f"var_chunk {d.get('var_chunk')}", flush=True)

    model.train(); t0 = time.time()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        m = {"reward": [], "p": [], "writes": [], "turns": [], "ce": [],
             "dropped": 0, "kl": [], "loss": []}
        rolls_flat = []
        for _ in range(gps):
            group = advs = None
            for _try in range(max_rs + 1):                       # dynamic sampling
                chunks, tgt = next_rl_conv(train_stream, defer_len, device)
                sbank = conv_seed_bank(model)                    # shared: actions-only variance
                cand = [rollout(model, chunks, tgt, temp, lam, ids, rng, sbank)
                        for _ in range(G)]
                rs = [c["reward"] for c in cand]
                mu = st.mean(rs); sd_r = st.pstdev(rs)
                if sd_r >= min_std:
                    group = cand
                    advs = [(x - mu) / (sd_r + 1e-6) for x in rs]
                    break
                m["dropped"] += 1
            if group is None:
                continue                                          # all resamples degenerate
            lo, kl = grpo_backward(model, ref, group, advs, temp, clip_lo, clip_hi,
                                   kl_coef, ids, scale=1.0 / (gps * G))
            rolls_flat += group
            m["loss"].append(lo); m["kl"].append(kl)
            m["reward"] += rs; m["ce"] += [c["ce"] for c in group]
            m["writes"] += [c["n_writes"] for c in group]
            m["turns"] += [len(c["recs"]) for c in group]
            m["p"] += [rec["p"] for c in group for rec in c["recs"]]
        if not m["reward"]:
            print(f"step {step:4d}  ALL groups degenerate ({m['dropped']} drops) — "
                  f"no update. If persistent: raise temp or check the varlen ckpt.",
                  flush=True)
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(r.get("grad_clip", 1.0)))
        opt.step()
        if step % int(r.get("log_every", 1)) == 0:
            wr = sum(m["writes"]) / max(sum(m["turns"]), 1)
            line = {"step": step, "reward": st.mean(m["reward"]), "ce": st.mean(m["ce"]),
                    "p_write": st.mean(m["p"]), "write_rate": wr,
                    "kl": st.mean(m["kl"]) if m["kl"] else 0.0,
                    "dropped_groups": m["dropped"],
                    "pos_corr": pos_write_corr(rolls_flat)}
            print(f"step {step:4d}  r {line['reward']:+.3f}  ce {line['ce']:.3f}  "
                  f"p(w) {line['p_write']:.3f}  write% {wr:.2f}  kl {line['kl']:.2e}  "
                  f"drop {m['dropped']}  poscorr {line['pos_corr']:.2f}  "
                  f"{(time.time()-t0)/step:.1f}s/step", flush=True)
            if mfile:
                with open(mfile, "a") as fh:
                    fh.write(json.dumps(line) + "\n")

        if step % int(r.get("eval_every", 25)) == 0 or step == steps:
            model.eval()
            ev = {"pol": [], "alw": [], "nev": [], "w": [], "t": []}
            with torch.no_grad():
                for _ in range(int(r.get("eval_convs", 16))):
                    chunks, tgt = next_rl_conv(eval_stream, defer_len, device)
                    sbank = conv_seed_bank(model)       # shared across the 3 arms
                    ro = rollout(model, chunks, tgt, 0.0 + temp, lam, ids,
                                 _random.Random(0), sbank)  # fixed rng: comparable evals
                    ra, _ = forced_reward(model, chunks, tgt, True, lam, ids, sbank)
                    rn, _ = forced_reward(model, chunks, tgt, False, lam, ids, sbank)
                    ev["pol"].append(ro["reward"]); ev["alw"].append(ra)
                    ev["nev"].append(rn); ev["w"].append(ro["n_writes"])
                    ev["t"].append(len(ro["recs"]))
            print(f"  eval@{step}: policy {st.mean(ev['pol']):+.3f} | "
                  f"always {st.mean(ev['alw']):+.3f} | never {st.mean(ev['nev']):+.3f} | "
                  f"writes {sum(ev['w'])}/{sum(ev['t'])}  "
                  f"(selective>always = the emergent-behavior headline)", flush=True)
            model.train()

        if step % int(r.get("save_every", 100)) == 0 or step == steps:
            torch.save({"model": model.state_dict(), "cfg": mcfg, "step": step},
                       os.path.join(save_dir, "final.pt" if step == steps
                                    else f"step{step}.pt"))

    print("done.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
