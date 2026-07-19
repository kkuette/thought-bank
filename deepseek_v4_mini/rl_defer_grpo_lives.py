"""dsv6 GRPO on CONTINUOUS LIVES — phase-2 groundwork over rl_defer_grpo v1.

v1 trained the write policy on isolated conversations (fresh seed bank each
group). Here the unit is a LIFE: the bank (+ cascade) is carried across
episodes drawn from a weighted MIX of environments and never reset between
them (design session 2026-07-13: episode = thread, no-reset = D, mix = G —
the SFT arc showed the mix is what prevents the boundary shortcut).

Loop, per step and per group:
  1. pick a life (round-robin), sample an episode from the EnvMixer;
  2. FORK the life's carried state G times (rl_lives.mem_fork) — the group
     baseline is only meaningful if all G rollouts start from the SAME bank
     (seed_bank trap, generalized);
  3. G rollouts (write/skip Bernoulli at each boundary, v1 policy), reward
     from the episode's env (dense -CE default; verifiers rubric hook later);
  4. GRPO update (asym clip + Bernoulli KL to ref + dynamic sampling, v1);
  5. commit ONE rollout's final state back into the life — chosen UNIFORMLY
     at random, never argmax-reward: selecting lucky banks would be a covert
     retention pressure (a distribution shift toward states that scored well,
     exactly what the standing note says not to build).

Checkpoints carry model/opt AND LivesState (banks, cascades, per-env counts,
mixer+stream rng): a resume that reset the banks would only ever train on
life-beginnings. Guard: --resume restores everything or starts fresh loudly.

Cascade: optional (training.cascade_depth in the model's own config was for
SFT; here rl.cascade_depth/rl.cascade_map). Rollout replays store per-turn
layer_banks so pass 2 sees the exact pass-1 state. Eviction capture follows
the trainer: slot 0 of a FULL bank descends on write; the first
mem_seed_slots evictions are seeds and do not descend (per-life counter,
forked with the group).

  python -m deepseek_v4_mini.rl_defer_grpo_lives deepseek_v4_mini/configs/rl_lives_97m.yaml
"""
from __future__ import annotations

import copy
import json
import math
import os
import random as _random
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
from .cascade import CascadeMemory
from .rl_lives import EnvMixer, EnvSpec, Life, LivesState, mem_fork
from .rl_defer_grpo import pos_write_corr


# ── policy primitives (cascade-aware variants of v1) ────────────────────────

def boundary_step(model, x, bank, think_id, temp, lb=None):
    """One turn on the carried bank (+ optional cascade layer_banks)."""
    xt = torch.cat([x, torch.full((1, 1), think_id, dtype=torch.long, device=x.device)], 1)
    o = model(xt, init_mem=bank, layer_banks=lb)
    lg = o["logits"].float()[0, x.size(1) - 1]
    logp = F.log_softmax(lg, dim=-1)
    p1 = logp[think_id].exp().clamp(1e-6, 1.0 - 1e-6)
    p_w = torch.sigmoid(torch.logit(p1) / temp).clamp(1e-6, 1.0 - 1e-6)  # Bernoulli temp (v1)
    return torch.log(p_w), torch.log1p(-p_w), o["mem_bank"], p_w


def defer_ce(model, bank, tgt, blank_id, lb=None):
    di = torch.full((1, tgt.size(1)), blank_id, dtype=torch.long, device=tgt.device)
    lg = model(di, init_mem=bank, layer_banks=lb)["logits"].float()
    return F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1))


def _lb(casc, bank, cmap):
    return casc.layer_banks(bank, cmap) if casc is not None else None


@torch.no_grad()
def rollout(model, chunks, tgt, temp, lam, ids, rng, bank, casc, n_evict,
            seed_slots, max_mem, cmap):
    """One sampled trajectory from a FORKED life state. Mutates its own copies
    only. Stores per-turn detached (bank_in, lb_in) for exact pass-2 replay."""
    think_id, blank_id = ids
    recs = []
    for x in chunks:
        lb = _lb(casc, bank, cmap)
        lw, ls, new_bank, p_w = boundary_step(model, x, bank, think_id, temp, lb)
        a = 1 if rng.random() < float(p_w) else 0
        recs.append({"x": x, "a": a, "logp_old": float(lw if a else ls),
                     "bank_in": bank.detach(),
                     "lb_in": None if lb is None else [None if t is None else t.detach()
                                                       for t in lb],
                     "p": float(p_w)})
        if a:
            if casc is not None and bank.size(1) >= max_mem:
                n_evict += 1
                if n_evict > seed_slots:
                    casc.push_slot(bank[:, 0].detach())
            bank = new_bank.detach()
    n_w = sum(r["a"] for r in recs)
    ce = float(defer_ce(model, bank, tgt, blank_id, _lb(casc, bank, cmap)))
    return {"recs": recs, "ce": ce, "n_writes": n_w,
            "bank": bank, "casc": casc, "n_evict": n_evict}


@torch.no_grad()
def forced_reward(model, chunks, tgt, write_all, lam, ids, bank, casc, n_evict,
                  seed_slots, max_mem, cmap):
    think_id, blank_id = ids
    if write_all:
        for x in chunks:
            lb = _lb(casc, bank, cmap)
            _, _, nb, _ = boundary_step(model, x, bank, think_id, 1.0, lb)
            if casc is not None and bank.size(1) >= max_mem:
                n_evict += 1
                if n_evict > seed_slots:
                    casc.push_slot(bank[:, 0].detach())
            bank = nb.detach()
    ce = float(defer_ce(model, bank, tgt, blank_id, _lb(casc, bank, cmap)))
    return -ce - lam * (len(chunks) if write_all else 0), ce


def grpo_backward(model, ref, group, advs, temp, clip_lo, clip_hi, kl_coef,
                  ids, scale):
    """v1 update, replaying with the stored per-turn layer_banks."""
    think_id, _ = ids
    tot_loss = tot_kl = 0.0
    n_act = 0
    for ro, A in zip(group, advs):
        loss = torch.zeros((), device=next(model.parameters()).device)
        for r in ro["recs"]:
            lw, ls, _, p_w = boundary_step(model, r["x"], r["bank_in"], think_id,
                                           temp, r["lb_in"])
            logp_new = lw if r["a"] else ls
            ratio = (logp_new - r["logp_old"]).exp()
            surr = torch.min(ratio * A,
                             ratio.clamp(1.0 - clip_lo, 1.0 + clip_hi) * A)
            with torch.no_grad():
                _, _, _, p_ref = boundary_step(ref, r["x"], r["bank_in"], think_id,
                                               temp, r["lb_in"])
            kl = (p_w * (p_w / p_ref).log()
                  + (1 - p_w) * ((1 - p_w) / (1 - p_ref)).log())
            loss = loss + (-surr + kl_coef * kl)
            tot_kl += float(kl.detach())
            n_act += 1
        (loss * scale).backward()
        tot_loss += float(loss.detach())
    return tot_loss, tot_kl / max(n_act, 1)


# ── main ─────────────────────────────────────────────────────────────────────

def main(cfg_path: str) -> None:
    raw = yaml.safe_load(open(cfg_path))
    r, d = raw["rl"], raw["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(r.get("seed", 0)))
    rng = _random.Random(int(r.get("seed", 0)) + 17)

    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    add = [x for x in ("<think>", "<blank>") if x not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    ids = (tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("<blank>"))

    mcfg = dict(raw["model"])
    mcfg["vocab_size"] = len(tok)
    model = ThoughtBankLM(ThoughtBankConfig(**mcfg)).to(device)
    ck = torch.load(r["init_from"], map_location="cpu")
    model.load_state_dict(ck["model"])
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    scope = r.get("train_scope", "think_row")     # v1 post-mortem default
    if scope == "think_row":
        for p in model.parameters():
            p.requires_grad_(False)
        W = model.lm_head.weight
        W.requires_grad_(True)
        _mask = torch.zeros_like(W)
        _mask[ids[0]] = 1.0
        W.register_hook(lambda g: g * _mask)
        opt_params = [W]
    else:
        opt_params = list(model.parameters())

    # environments: one stream per env (own rng via seed offset), weighted mix
    L, K = int(d["seq_len"]), int(d["chunks_per_conv"])
    defer_len = int(d.get("defer_len", 16))
    base_sd = dict(seq_len=L, chunks_per_conv=K, batch=1,
                   cache_dir=d.get("cache_dir", "data_cache"),
                   var_chunk=d.get("var_chunk"))
    envs = []
    for i, e in enumerate(d["envs"]):
        sd_e = dict(base_sd)
        sd_e.update(n_files=int(e.get("n_files", 800)),
                    dataset=e["dataset"], data_dir=e.get("data_dir", ""),
                    stream_cap=int(e.get("stream_cap", 60000)),
                    content_key=e.get("content_key", "content"),
                    config_name=e.get("config_name", ""),
                    min_chunks=int(e.get("min_chunks", 2)),
                    seed=int(r.get("seed", 0)) + 31 * i)
        envs.append(EnvSpec(e["name"], CodeChunkStream(tok, split="train", **sd_e),
                            weight=float(e.get("weight", 1.0))))
    mixer = EnvMixer(envs, seed=int(r.get("seed", 0)) + 977)
    eval_env = envs[0]                            # anchors: first env, held split
    sd_h = dict(base_sd)
    sd_h.update(n_files=int(d["envs"][0].get("n_files", 800)),
                dataset=d["envs"][0]["dataset"],
                data_dir=d["envs"][0].get("data_dir", ""),
                stream_cap=int(d["envs"][0].get("stream_cap", 60000)),
                content_key=d["envs"][0].get("content_key", "content"),
                config_name=d["envs"][0].get("config_name", ""),
                min_chunks=int(d["envs"][0].get("min_chunks", 2)),
                seed=int(r.get("seed", 0)) + 5555)
    eval_stream = CodeChunkStream(tok, split="held", **sd_h)

    # lives: seed bank materialized once per life; cascade optional
    n_lives = int(r.get("n_lives", 4))
    casc_depth = int(r.get("cascade_depth", 0))
    cmap = r.get("cascade_map") or [0] * int(mcfg["n_layers"])
    max_mem = int(mcfg["max_mem"])
    seed_slots = int(mcfg.get("mem_seed_slots", 0))
    p0 = next(model.parameters())

    def fresh_life(i):
        with torch.no_grad():
            b = model.thought_stream.seed_bank(1, p0.device, p0.dtype)
        lf = Life(i, b, CascadeMemory(casc_depth, max_mem) if casc_depth else None)
        lf.n_evict = 0
        return lf

    lives = LivesState([fresh_life(i) for i in range(n_lives)], mixer)

    G = int(r.get("group_size", 8))
    gps = int(r.get("groups_per_step", 4))
    steps = int(r["steps"])
    temp = float(r.get("temp", 1.0))
    lam_default = float(r.get("lambda_write", 0.03))
    lam_env = {e["name"]: float(e.get("lambda_write", lam_default)) for e in d["envs"]}
    clip_lo = float(r.get("clip_low", 0.2))
    clip_hi = float(r.get("clip_high", 0.28))
    kl_coef = float(r.get("kl_coef", 1.0e-3))
    min_std = float(r.get("min_reward_std", 1.0e-4))
    max_rs = int(r.get("max_resample", 4))
    max_epi = int(r.get("max_episodes_per_life", 0))   # 0 = unbounded
    opt = torch.optim.AdamW(opt_params, lr=float(r.get("lr", 5.0e-6)), weight_decay=0.0)
    save_dir = r.get("save_dir", "checkpoints/rl_lives")
    os.makedirs(save_dir, exist_ok=True)
    mfile = r.get("metrics_file")
    if mfile:
        os.makedirs(os.path.dirname(mfile), exist_ok=True)

    start_step = 0
    ck_path = os.path.join(save_dir, "last.pt")
    if "--resume" in sys.argv and os.path.exists(ck_path):
        rk = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(rk["model"])
        opt.load_state_dict(rk["opt"])
        lives.load_state_dict(rk["lives"], device=p0.device, dtype=p0.dtype)
        for lf, s in zip(lives.lives, rk["lives"]["lives"]):
            lf.n_evict = s.get("n_evict", 0)
        start_step = rk["step"]
        print(f"resume: step {start_step}, lives restored "
              f"({[lf.n_episodes for lf in lives.lives]} episodes)", flush=True)
    elif "--resume" in sys.argv:
        print("resume: no checkpoint, fresh lives (loud on purpose)", flush=True)

    print(f"GRPO-lives {model.num_params():,} params <- {r['init_from']} | scope {scope} | "
          f"lives {n_lives} | envs {[e.name for e in envs]} w={[e.weight for e in envs]} | "
          f"cascade depth {casc_depth} map {cmap if casc_depth else '-'} | "
          f"G {G} x {gps}/step | steps {steps} | temp {temp} | lam {lam_env}", flush=True)

    model.train()
    t0 = time.time()
    li = 0
    for step in range(start_step + 1, steps + 1):
        opt.zero_grad(set_to_none=True)
        m = {"reward": [], "p": [], "writes": [], "turns": [], "ce": [],
             "dropped": 0, "kl": [], "loss": [], "env": {}}
        rolls_flat = []
        for _ in range(gps):
            life = lives.lives[li % n_lives]
            li += 1
            if max_epi and life.n_episodes >= max_epi:        # natural death: fresh life
                lives.lives[life.id] = life = fresh_life(life.id)
            group = advs = env_name = None
            for _try in range(max_rs + 1):                    # dynamic sampling (v1)
                env_name, chunks, tgt = mixer.next_episode(defer_len, device)
                lam = lam_env[env_name]
                forks = mem_fork(life.bank, life.casc, G)
                cand = [rollout(model, chunks, tgt, temp, lam, ids, rng,
                                fb, fc, life.n_evict, seed_slots, max_mem, cmap)
                        for fb, fc in forks]
                rs = [-c["ce"] - lam * c["n_writes"] for c in cand]
                mu, sd_r = st.mean(rs), st.pstdev(rs)
                if sd_r >= min_std:
                    group = cand
                    advs = [(x - mu) / (sd_r + 1e-6) for x in rs]
                    break
                m["dropped"] += 1
            if group is None:
                continue
            lo, kl = grpo_backward(model, ref, group, advs, temp, clip_lo, clip_hi,
                                   kl_coef, ids, scale=1.0 / (gps * G))
            # commit: uniform choice, never argmax (covert retention pressure)
            keep = group[rng.randrange(G)]
            life.bank, life.casc, life.n_evict = keep["bank"], keep["casc"], keep["n_evict"]
            life.advance(keep["bank"], env_name)
            rolls_flat += group
            m["loss"].append(lo)
            m["kl"].append(kl)
            m["reward"] += rs
            m["ce"] += [c["ce"] for c in group]
            m["writes"] += [c["n_writes"] for c in group]
            m["turns"] += [len(c["recs"]) for c in group]
            m["p"] += [rec["p"] for c in group for rec in c["recs"]]
            m["env"][env_name] = m["env"].get(env_name, 0) + 1
        if not m["reward"]:
            print(f"step {step:4d}  ALL groups degenerate ({m['dropped']} drops)", flush=True)
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(r.get("grad_clip", 1.0)))
        opt.step()

        if step % int(r.get("log_every", 1)) == 0:
            wr = sum(m["writes"]) / max(sum(m["turns"]), 1)
            epis = [lf.n_episodes for lf in lives.lives]
            line = {"step": step, "reward": st.mean(m["reward"]), "ce": st.mean(m["ce"]),
                    "p_write": st.mean(m["p"]), "write_rate": wr,
                    "kl": st.mean(m["kl"]) if m["kl"] else 0.0,
                    "dropped_groups": m["dropped"], "pos_corr": pos_write_corr(rolls_flat),
                    "life_episodes": epis, "env_mix": m["env"]}
            print(f"step {step:4d}  r {line['reward']:+.3f}  ce {line['ce']:.3f}  "
                  f"p(w) {line['p_write']:.3f}  write% {wr:.2f}  kl {line['kl']:.2e}  "
                  f"drop {m['dropped']}  poscorr {line['pos_corr']:.2f}  "
                  f"lives {epis}  {m['env']}  {(time.time()-t0)/max(step-start_step,1):.1f}s/step",
                  flush=True)
            if mfile:
                with open(mfile, "a") as fh:
                    fh.write(json.dumps(line) + "\n")

        if step % int(r.get("eval_every", 25)) == 0 or step == steps:
            # anchors on FRESH state (comparable across steps: no life leakage)
            model.eval()
            ev = {"pol": [], "alw": [], "nev": [], "w": [], "t": []}
            with torch.no_grad():
                for _ in range(int(r.get("eval_convs", 16))):
                    while True:
                        segs = eval_stream.next_conv()
                        if len(segs) >= 3 and segs[-1]["input_ids"].size(1) >= defer_len:
                            break
                    chunks = [s["input_ids"].to(device) for s in segs[:-1]]
                    tgt = segs[-1]["input_ids"][:, :defer_len].to(device)
                    sb = model.thought_stream.seed_bank(1, p0.device, p0.dtype)
                    ec = CascadeMemory(casc_depth, max_mem) if casc_depth else None
                    args = (0, seed_slots, max_mem, cmap)
                    forks = mem_fork(sb, ec, 3)
                    ro = rollout(model, chunks, tgt, temp, lam_default, ids,
                                 _random.Random(0), *forks[0], *args)
                    ra, _ = forced_reward(model, chunks, tgt, True, lam_default, ids,
                                          *forks[1], *args)
                    rn, _ = forced_reward(model, chunks, tgt, False, lam_default, ids,
                                          *forks[2], *args)
                    ev["pol"].append(-ro["ce"] - lam_default * ro["n_writes"])
                    ev["alw"].append(ra)
                    ev["nev"].append(rn)
                    ev["w"].append(ro["n_writes"])
                    ev["t"].append(len(ro["recs"]))
            print(f"  eval@{step}: policy {st.mean(ev['pol']):+.3f} | "
                  f"always {st.mean(ev['alw']):+.3f} | never {st.mean(ev['nev']):+.3f} | "
                  f"writes {sum(ev['w'])}/{sum(ev['t'])}", flush=True)
            model.train()

        if step % int(r.get("save_every", 100)) == 0 or step == steps:
            sd_l = lives.state_dict()
            for ls_, lf in zip(sd_l["lives"], lives.lives):   # per-life eviction counter
                ls_["n_evict"] = getattr(lf, "n_evict", 0)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "lives": sd_l, "cfg": mcfg, "step": step}, ck_path)
            if step == steps:
                torch.save({"model": model.state_dict(), "cfg": mcfg, "step": step},
                           os.path.join(save_dir, "final.pt"))

    print("done.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
