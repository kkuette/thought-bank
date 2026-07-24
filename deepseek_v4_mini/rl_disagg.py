"""dsv6 disaggregated GRPO — rollouts on the 3070Ti farm, updates on the 3090.

Why: GRPO's bottleneck is rollout generation, and the farm's 97M/135M VRAM
frontier is a TRAINING frontier (optimizer + activations). Inference of the
386M fits a 3070Ti with room to spare — so the 6 farm cards generate while
the 3090 (or a pod) grades and updates. Weights ride the NFS share; one
iteration of staleness is tolerated (the stored logp_old makes the update
properly off-policy through the existing clipped ratio — nothing new to add).

Split of rl_defer_grpo_lives across the share (root = rl.disagg.root):

  WORKER (one per farm GPU, `WORKER` env = its id)
    owns a PARTITION of the lives (seeds offset by worker id — lives never
    migrate, so the no-reset invariant holds per worker without locks);
    samples episodes from its own mixer, forks the carried bank G ways,
    rolls out the write policy (rl_defer_grpo_lives.rollout verbatim),
    computes REWARDS in place: dense -ce - lam*n_writes by default, or the
    verifiable rubric for tool envs (decode the call turn from the final
    bank, rl_rewards.grade_calls x think_economy, n_think = n_writes: think
    turns ARE bank writes); dynamic-resamples degenerate groups; commits one
    rollout uniformly (never argmax — covert retention pressure); ships the
    group (actions, logp_old, per-turn bank_in/lb_in) to rollouts/incoming.

  LEARNER (single consumer)
    takes groups whose weights_step >= current - max_lag (older = stale/),
    normalizes rewards into advantages, replays grpo_backward (ratio vs the
    SHIPPED logp_old = the off-policy correction), steps, publishes weights
    (atomic tmp+rename, LATEST pointer). ref for the KL = the init
    checkpoint, never the moving weights.

  PROBES (worker-side, log-only, never shipped for training)
    every xdom_every groups: same eval episode rolled from the life's OWN
    bank vs a FOREIGN life's bank (the banque-xdom adversity probe, memory
    dsv6-grpo-recence-feature) + always/never anchors on the own bank.

Files: weights/step_%06d.pt + LATEST | rollouts/{incoming,stale}/ |
w{N}_lives.pt (auto-resumed) | STOP kills everyone politely.

  learner:  python -m deepseek_v4_mini.rl_disagg learner <cfg.yaml>
  worker:   python -m deepseek_v4_mini.rl_disagg worker  <cfg.yaml> [--worker N]

CPU self-test (tiny model, stub envs, in-process worker+learner):
  python -m deepseek_v4_mini.rl_disagg
"""
from __future__ import annotations

import copy
import json
import os
import random as _random
import statistics as st
import sys
import time
import uuid

import torch
import torch.nn.functional as F

from .cascade import CascadeMemory
from .rl_defer_grpo import pos_write_corr
from .rl_defer_grpo_lives import (_lb, boundary_step, defer_ce, forced_reward,
                                  grpo_backward, rollout)
from .rl_lives import EnvMixer, EnvSpec, Life, LivesState, mem_fork
from .rl_rewards import make_exec_reward, make_tool_reward


# ── shared-FS primitives ─────────────────────────────────────────────────────

def _atomic_save(obj, path: str) -> None:
    tmp = f"{path}.tmp.{uuid.uuid4().hex[:8]}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write(text: str, path: str) -> None:
    tmp = f"{path}.tmp.{uuid.uuid4().hex[:8]}"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


class WeightHub:
    """Publish/fetch model weights through the share. LATEST is a pointer
    file (atomic replace): readers never see a half-written checkpoint."""

    def __init__(self, root: str, keep: int = 3):
        self.dir = os.path.join(root, "weights")
        self.keep = int(keep)
        os.makedirs(self.dir, exist_ok=True)

    def publish(self, model_sd: dict, step: int) -> None:
        name = f"step_{step:06d}.pt"
        _atomic_save({"model": {k: v.cpu() for k, v in model_sd.items()},
                      "step": step}, os.path.join(self.dir, name))
        _atomic_write(name, os.path.join(self.dir, "LATEST"))
        pts = sorted(p for p in os.listdir(self.dir)
                     if p.startswith("step_") and p.endswith(".pt"))
        for p in pts[:-self.keep]:
            try:
                os.remove(os.path.join(self.dir, p))
            except OSError:
                pass

    def latest_step(self):
        try:
            name = open(os.path.join(self.dir, "LATEST")).read().strip()
            return int(name[len("step_"):-len(".pt")])
        except (OSError, ValueError):
            return None

    def fetch(self, known_step):
        """(state_dict, step) if newer than known_step, else None."""
        s = self.latest_step()
        if s is None or (known_step is not None and s <= known_step):
            return None
        path = os.path.join(self.dir, f"step_{s:06d}.pt")
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError):
            return None                        # pruned or racing — next poll
        return ck["model"], ck["step"]


class RolloutStore:
    """Group files through the share. Workers write atomically to incoming/;
    the single learner claims by os.replace into claimed/ (rename is the
    lock), loads, unlinks. Groups older than min_step land in stale/."""

    def __init__(self, root: str):
        self.inc = os.path.join(root, "rollouts", "incoming")
        self.clm = os.path.join(root, "rollouts", "claimed")
        self.stl = os.path.join(root, "rollouts", "stale")
        for d in (self.inc, self.clm, self.stl):
            os.makedirs(d, exist_ok=True)

    def put(self, group: dict, weights_step: int, worker: int) -> None:
        name = f"w{worker:02d}_s{weights_step:06d}_{uuid.uuid4().hex[:8]}.pt"
        _atomic_save(group, os.path.join(self.inc, name))

    def pending(self) -> int:
        return len([p for p in os.listdir(self.inc) if p.endswith(".pt")])

    def take(self, n: int, min_step: int):
        """Up to n groups, oldest first; stale ones moved aside, counted."""
        got, n_stale = [], 0
        names = sorted((p for p in os.listdir(self.inc) if p.endswith(".pt")),
                       key=lambda p: os.path.getmtime(os.path.join(self.inc, p)))
        for name in names:
            if len(got) >= n:
                break
            ws = int(name.split("_s")[1].split("_")[0])
            src = os.path.join(self.inc, name)
            if ws < min_step:
                os.replace(src, os.path.join(self.stl, name))
                n_stale += 1
                continue
            dst = os.path.join(self.clm, name)
            try:
                os.replace(src, dst)
            except OSError:
                continue                       # raced (should not happen: 1 learner)
            try:
                got.append(torch.load(dst, map_location="cpu",
                                      weights_only=False))
            finally:
                try:
                    os.remove(dst)
                except OSError:
                    pass
        return got, n_stale


# ── group (de)hydration ──────────────────────────────────────────────────────

def group_to_cpu(chunks, rollouts, env_name, weights_step, worker) -> dict:
    """Ship exactly what grpo_backward replays. x is NOT stored per rec —
    rollout() emits one rec per chunk in order, so position rebuilds it."""
    cpu = lambda t: t.detach().cpu()
    return {
        "env": env_name, "weights_step": int(weights_step),
        "worker": int(worker),
        "chunks": [cpu(x) for x in chunks],
        "rollouts": [{
            "reward": float(ro["reward"]), "ce": float(ro["ce"]),
            "n_writes": int(ro["n_writes"]),
            "recs": [{"a": r["a"], "logp_old": r["logp_old"], "p": r["p"],
                      "bank_in": cpu(r["bank_in"]),
                      "lb_in": None if r["lb_in"] is None else
                      [None if t is None else cpu(t) for t in r["lb_in"]]}
                     for r in ro["recs"]],
        } for ro in rollouts],
    }


def group_to_device(g: dict, device, dtype):
    """Rollout dicts in grpo_backward's shape, tensors on the learner."""
    mv = lambda t: t.to(device=device, dtype=dtype)
    chunks = [c.to(device) for c in g["chunks"]]
    out = []
    for ro in g["rollouts"]:
        recs = []
        for i, r in enumerate(ro["recs"]):
            recs.append({"x": chunks[i], "a": r["a"],
                         "logp_old": r["logp_old"], "p": r["p"],
                         "bank_in": mv(r["bank_in"]),
                         "lb_in": None if r["lb_in"] is None else
                         [None if t is None else mv(t) for t in r["lb_in"]]})
        out.append({"recs": recs, "reward": ro["reward"], "ce": ro["ce"],
                    "n_writes": ro["n_writes"]})
    return out


# ── env construction (worker side) ───────────────────────────────────────────

def build_envs(d: dict, r: dict, tok, seed: int):
    """EnvSpecs from the config's data.envs. kind: code (CodeChunkStream,
    dense -ce), tool (ToolSessionStream + verifiable rubric), exec
    (CodeExecStream + sandboxed unit tests), sota (SotaSessionStream, dense).
    Chat-kind streams return conv DICTS — sample_episode below normalizes
    both shapes."""
    from .code_data import CodeChunkStream
    envs = []
    for i, e in enumerate(d["envs"]):
        kind = e.get("kind", "code")
        w = float(e.get("weight", 1.0))
        if kind == "code":
            sd_e = dict(seq_len=int(d["seq_len"]),
                        chunks_per_conv=int(d["chunks_per_conv"]), batch=1,
                        cache_dir=d.get("cache_dir", "data_cache"),
                        var_chunk=d.get("var_chunk"),
                        n_files=int(e.get("n_files", 800)),
                        dataset=e["dataset"], data_dir=e.get("data_dir", ""),
                        stream_cap=int(e.get("stream_cap", 60000)),
                        content_key=e.get("content_key", "content"),
                        config_name=e.get("config_name", ""),
                        min_chunks=int(e.get("min_chunks", 2)),
                        seed=seed + 31 * i)
            stream = CodeChunkStream(tok, split="train", **sd_e)
            envs.append(EnvSpec(e["name"], stream, weight=w))
        elif kind == "tool":
            from .tool_env_data import ToolSessionStream
            stream = ToolSessionStream(tok, seed=seed + 31 * i,
                                       **(e.get("gen") or {}))
            fn = make_tool_reward(int(r.get("think_nmax", 8)),
                                  float(r.get("think_floor", 0.4)))
            envs.append(EnvSpec(e["name"], stream, weight=w, reward_fn=fn))
        elif kind == "exec":
            from .code_exec_data import CodeExecStream
            stream = CodeExecStream(tok, seed=seed + 31 * i,
                                    **(e.get("gen") or {}))
            fn = make_exec_reward(int(r.get("think_nmax", 8)),
                                  float(r.get("think_floor", 0.4)),
                                  float(e.get("exec_timeout", 6.0)))
            envs.append(EnvSpec(e["name"], stream, weight=w, reward_fn=fn))
        elif kind == "sota":
            from .sota_session_data import SotaSessionStream
            stream = SotaSessionStream(tok, seed=seed + 31 * i,
                                       **(e.get("gen") or {}))
            envs.append(EnvSpec(e["name"], stream, weight=w))
        else:
            raise ValueError(f"unknown env kind {kind!r}")
    return envs


def sample_episode(mixer: EnvMixer, defer_len: int, device, rng=None):
    """Weighted env choice + episode extraction for BOTH stream shapes.
    Returns (env, chunks, tgt, info). Uses mixer.rng so mixer.state_dict()
    keeps full sampling determinism."""
    name = mixer.rng.choices(mixer._names, weights=mixer._weights, k=1)[0]
    env = mixer.envs[name]
    while True:
        got = env.stream.next_conv()
        segs, info = (got["segs"], dict(got.get("info", {}))) \
            if isinstance(got, dict) else (got, {})
        if len(segs) >= 3 and segs[-1]["input_ids"].size(1) >= 1:
            tgt = segs[-1]["input_ids"][:, :defer_len]
            if tgt.size(1) < 1:
                continue
            chunks = [s["input_ids"].to(device) for s in segs[:-1]]
            return env, chunks, tgt.to(device), info


# ── rubric decode (tool envs) ────────────────────────────────────────────────

@torch.no_grad()
def decode_lb(model, prefix, bank, lb, max_new, stop_id, amp):
    """code_defer_native._greedy + layer_banks: the call turn is decoded from
    the exact carried state the rollout ended in (reads only)."""
    out = prefix
    for _ in range(max_new):
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=amp and out.is_cuda):
            o = model(out, init_mem=bank, layer_banks=lb)
        nt = o["logits"][:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, nt], dim=1)
        if int(nt) == stop_id:
            break
    return out[:, prefix.size(1):]


# ── worker ───────────────────────────────────────────────────────────────────

class Worker:
    def __init__(self, raw: dict, worker_id: int, *, tok=None, model=None,
                 envs=None, device=None):
        r, d = raw["rl"], raw["data"]
        self.r, self.d = r, d
        self.wid = int(worker_id)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        seed = int(r.get("seed", 0)) + 1000 * self.wid
        torch.manual_seed(seed)
        self.rng = _random.Random(seed + 17)
        dg = r["disagg"]
        self.root = dg["root"]
        self.hub = WeightHub(self.root, keep=int(dg.get("keep_weights", 3)))
        self.store = RolloutStore(self.root)
        self.max_pending = int(dg.get("max_pending", 24))
        self.poll_s = float(dg.get("poll_s", 2.0))

        self.tok = tok
        if self.tok is None:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
            add = [x for x in ("<think>", "<blank>")
                   if x not in self.tok.get_vocab()]
            if add:
                self.tok.add_special_tokens({"additional_special_tokens": add})
        self.ids = (self.tok.convert_tokens_to_ids("<think>"),
                    self.tok.convert_tokens_to_ids("<blank>"))

        if model is None:
            from .config import ThoughtBankConfig
            from .model import ThoughtBankLM
            mcfg = dict(raw["model"])
            mcfg["vocab_size"] = len(self.tok)
            model = ThoughtBankLM(ThoughtBankConfig(**mcfg)).to(self.device)
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mcfg = raw["model"]

        self.envs = envs if envs is not None else build_envs(d, r, self.tok, seed)
        self.mixer = EnvMixer(self.envs, seed=seed + 977)
        self.defer_len = int(d.get("defer_len", 16))

        self.G = int(r.get("group_size", 8))
        self.temp = float(r.get("temp", 1.0))
        lam_default = float(r.get("lambda_write", 0.03))
        self.lam = {e["name"]: float(e.get("lambda_write", lam_default))
                    for e in d["envs"]}
        self.min_std = float(r.get("min_reward_std", 1.0e-4))
        self.max_rs = int(r.get("max_resample", 4))
        self.casc_depth = int(r.get("cascade_depth", 0))
        self.cmap = r.get("cascade_map") or [0] * int(self.mcfg["n_layers"])
        self.max_mem = int(self.mcfg["max_mem"])
        self.seed_slots = int(self.mcfg.get("mem_seed_slots", 0))
        self.max_new = int(r.get("max_new", 64))
        # per-env decode budget (code turns need more than call turns)
        self.max_new_env = {e["name"]: int(e.get("max_new", self.max_new))
                            for e in d["envs"]}
        self.amp = bool(r.get("amp", True))
        stop = "<|im_end|>"
        self.stop_id = (self.tok.convert_tokens_to_ids(stop)
                        if stop in self.tok.get_vocab() else -1)
        from .math_school_data import A_OPEN
        a_ids = self.tok(A_OPEN, add_special_tokens=False)["input_ids"]
        self.a_open = torch.tensor(a_ids, dtype=torch.long,
                                   device=self.device).unsqueeze(0)

        n_lives = int(r.get("n_lives_per_worker", 2))
        p0 = next(self.model.parameters())
        self._p0 = p0
        self.lives = LivesState([self._fresh_life(i) for i in range(n_lives)],
                                self.mixer)
        self.lives_path = os.path.join(self.root, f"w{self.wid:02d}_lives.pt")
        if os.path.exists(self.lives_path):
            lk = torch.load(self.lives_path, map_location="cpu",
                            weights_only=False)
            self.lives.load_state_dict(lk, device=p0.device, dtype=p0.dtype)
            for lf, s in zip(self.lives.lives, lk["lives"]):
                lf.n_evict = s.get("n_evict", 0)
            print(f"worker {self.wid}: lives resumed "
                  f"({[lf.n_episodes for lf in self.lives.lives]} episodes)",
                  flush=True)
        self.wstep = None
        self.li = 0
        self.n_groups = 0
        self.mfile = os.path.join(self.root, f"worker{self.wid:02d}_metrics.jsonl")

    def _fresh_life(self, i):
        with torch.no_grad():
            b = self.model.thought_stream.seed_bank(1, self._p0.device,
                                                    self._p0.dtype)
        lf = Life(i, b, CascadeMemory(self.casc_depth, self.max_mem)
                  if self.casc_depth else None)
        lf.n_evict = 0
        return lf

    # ── weights ──────────────────────────────────────────────────────────────
    def refresh(self) -> bool:
        got = self.hub.fetch(self.wstep)
        if got is None:
            return False
        sd, s = got
        self.model.load_state_dict(sd)
        self.model.eval()
        self.wstep = s
        return True

    def wait_weights(self):
        while self.wstep is None:
            if self.refresh():
                print(f"worker {self.wid}: weights step {self.wstep}", flush=True)
                return
            time.sleep(self.poll_s)

    # ── reward ───────────────────────────────────────────────────────────────
    def _reward(self, env, ro, lam, info) -> float:
        if env.reward_fn is None:
            return -ro["ce"] - lam * ro["n_writes"]
        lb = _lb(ro["casc"], ro["bank"], self.cmap)
        max_new = self.max_new_env.get(env.name, self.max_new)
        txt = self.tok.decode(decode_lb(self.model, self.a_open, ro["bank"],
                                        lb, max_new, self.stop_id,
                                        self.amp)[0].tolist())
        # rubric payload: the LAST episode's gold, whichever family (tool
        # envs read gold_calls, exec envs read tests)
        return env.reward(ro["ce"], {
            "text": txt, "n_think": ro["n_writes"],
            "gold_calls": (info.get("gold_calls") or [[]])[-1],
            "tests": (info.get("tests") or [[]])[-1]})

    # ── one group ────────────────────────────────────────────────────────────
    def one_group(self):
        life = self.lives.lives[self.li % len(self.lives.lives)]
        self.li += 1
        max_epi = int(self.r.get("max_episodes_per_life", 0))
        if max_epi and life.n_episodes >= max_epi:
            self.lives.lives[life.id] = life = self._fresh_life(life.id)
        for _try in range(self.max_rs + 1):
            env, chunks, tgt, info = sample_episode(self.mixer, self.defer_len,
                                                    self.device)
            lam = self.lam[env.name]
            forks = mem_fork(life.bank, life.casc, self.G)
            cand = [rollout(self.model, chunks, tgt, self.temp, lam, self.ids,
                            self.rng, fb, fc, life.n_evict, self.seed_slots,
                            self.max_mem, self.cmap) for fb, fc in forks]
            for c in cand:
                c["reward"] = self._reward(env, c, lam, info)
            rs = [c["reward"] for c in cand]
            if st.pstdev(rs) >= self.min_std:
                keep = cand[self.rng.randrange(self.G)]
                life.bank, life.casc = keep["bank"], keep["casc"]
                life.n_evict = keep["n_evict"]
                life.advance(keep["bank"], env.name)
                self.store.put(group_to_cpu(chunks, cand, env.name,
                                            self.wstep, self.wid),
                               self.wstep, self.wid)
                self.n_groups += 1
                return {"env": env.name, "reward": st.mean(rs),
                        "ce": st.mean([c["ce"] for c in cand]),
                        "writes": sum(c["n_writes"] for c in cand),
                        "turns": sum(len(c["recs"]) for c in cand),
                        "tries": _try}
        return None                            # degenerate after resamples

    # ── probes (log-only) ────────────────────────────────────────────────────
    @torch.no_grad()
    def xdom_probe(self):
        """Same episode, own vs foreign bank + always/never anchors."""
        env, chunks, tgt, info = sample_episode(self.mixer, self.defer_len,
                                                self.device)
        lam = self.lam[env.name]
        own = self.lives.lives[0]
        other = self.lives.lives[1 % len(self.lives.lives)]
        out = {}
        for tag, src in (("own", own), ("xdom", other)):
            (fb, fc), = mem_fork(src.bank, src.casc, 1)
            ro = rollout(self.model, chunks, tgt, self.temp, lam, self.ids,
                         _random.Random(0), fb, fc, src.n_evict,
                         self.seed_slots, self.max_mem, self.cmap)
            ro["reward"] = self._reward(env, ro, lam, info)
            out[f"r_{tag}"] = ro["reward"]
        args = (own.n_evict, self.seed_slots, self.max_mem, self.cmap)
        forks = mem_fork(own.bank, own.casc, 2)
        out["r_always"], _ = forced_reward(self.model, chunks, tgt, True, lam,
                                           self.ids, *forks[0], *args)
        out["r_never"], _ = forced_reward(self.model, chunks, tgt, False, lam,
                                          self.ids, *forks[1], *args)
        out["env"] = env.name
        return out

    def save_lives(self):
        sd = self.lives.state_dict()
        for ls_, lf in zip(sd["lives"], self.lives.lives):
            ls_["n_evict"] = getattr(lf, "n_evict", 0)
        _atomic_save(sd, self.lives_path)

    # ── loop ─────────────────────────────────────────────────────────────────
    def run(self):
        self.wait_weights()
        dg = self.r["disagg"]
        max_groups = int(dg.get("max_groups", 0))
        lives_every = int(dg.get("lives_save_every", 20))
        xdom_every = int(dg.get("xdom_every", 50))
        t0 = time.time()
        while not os.path.exists(os.path.join(self.root, "STOP")):
            if max_groups and self.n_groups >= max_groups:
                break
            if self.store.pending() >= self.max_pending:
                time.sleep(self.poll_s)        # learner is behind — don't flood
                self.refresh()
                continue
            self.refresh()
            line = self.one_group()
            if line is None:
                print(f"worker {self.wid}: degenerate group (all resamples)",
                      flush=True)
                continue
            line.update(n=self.n_groups, wstep=self.wstep,
                        s_per_group=(time.time() - t0) / max(self.n_groups, 1))
            with open(self.mfile, "a") as fh:
                fh.write(json.dumps(line) + "\n")
            if self.n_groups % lives_every == 0:
                self.save_lives()
            if xdom_every and self.n_groups % xdom_every == 0:
                probe = self.xdom_probe()
                probe["n"] = self.n_groups
                probe["probe"] = "xdom"
                with open(self.mfile, "a") as fh:
                    fh.write(json.dumps(probe) + "\n")
        self.save_lives()
        print(f"worker {self.wid}: done ({self.n_groups} groups)", flush=True)


# ── learner ──────────────────────────────────────────────────────────────────

class Learner:
    def __init__(self, raw: dict, *, tok_len=None, model=None, device=None):
        r = raw["rl"]
        self.r = r
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(int(r.get("seed", 0)))
        dg = r["disagg"]
        self.root = dg["root"]
        os.makedirs(self.root, exist_ok=True)
        self.hub = WeightHub(self.root, keep=int(dg.get("keep_weights", 3)))
        self.store = RolloutStore(self.root)
        self.publish_every = int(dg.get("publish_every", 1))
        self.max_lag = int(dg.get("max_lag", 2))
        self.poll_s = float(dg.get("poll_s", 2.0))

        if model is None:
            from transformers import AutoTokenizer
            from .config import ThoughtBankConfig
            from .model import ThoughtBankLM
            if tok_len is None:
                tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
                add = [x for x in ("<think>", "<blank>")
                       if x not in tok.get_vocab()]
                if add:
                    tok.add_special_tokens(
                        {"additional_special_tokens": add})
                tok_len = len(tok)
                self._think_id = tok.convert_tokens_to_ids("<think>")
            mcfg = dict(raw["model"])
            mcfg["vocab_size"] = tok_len
            model = ThoughtBankLM(ThoughtBankConfig(**mcfg)).to(self.device)
            ck = torch.load(r["init_from"], map_location="cpu")
            model.load_state_dict(ck["model"])
        self.model = model
        if not hasattr(self, "_think_id"):
            self._think_id = int(r.get("think_id", 0)) or 0
        self.ids = (self._think_id, -1)        # blank unused in the update
        self.ref = copy.deepcopy(self.model).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)

        scope = r.get("train_scope", "think_row")
        if scope == "think_row":
            for p in self.model.parameters():
                p.requires_grad_(False)
            W = self.model.lm_head.weight
            W.requires_grad_(True)
            mask = torch.zeros_like(W)
            mask[self.ids[0]] = 1.0
            W.register_hook(lambda g: g * mask)
            opt_params = [W]
        else:
            opt_params = list(self.model.parameters())
        self.opt = torch.optim.AdamW(opt_params, lr=float(r.get("lr", 5.0e-6)),
                                     weight_decay=0.0)
        self.temp = float(r.get("temp", 1.0))
        self.clip_lo = float(r.get("clip_low", 0.2))
        self.clip_hi = float(r.get("clip_high", 0.28))
        self.kl_coef = float(r.get("kl_coef", 1.0e-3))
        self.gps = int(r.get("groups_per_step", 4))
        self.G = int(r.get("group_size", 8))
        self.step = 0
        self.mfile = os.path.join(self.root, "learner_metrics.jsonl")
        self.ck_path = os.path.join(self.root, "learner_last.pt")
        if os.path.exists(self.ck_path):
            ck = torch.load(self.ck_path, map_location="cpu",
                            weights_only=False)
            self.model.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["opt"])
            self.step = ck["step"]
            print(f"learner: resumed step {self.step}", flush=True)
        self.hub.publish(self.model.state_dict(), self.step)

    def step_once(self, groups) -> dict:
        """One GRPO update from consumed groups (advantages in-group)."""
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        p0 = next(self.model.parameters())
        m = {"reward": [], "ce": [], "writes": 0, "turns": 0, "loss": [],
             "kl": [], "env": {}}
        rolls_flat = []
        for g in groups:
            rolls = group_to_device(g, self.device, p0.dtype)
            rs = [ro["reward"] for ro in rolls]
            mu, sd_r = st.mean(rs), st.pstdev(rs)
            advs = [(x - mu) / (sd_r + 1e-6) for x in rs]
            lo, kl = grpo_backward(self.model, self.ref, rolls, advs,
                                   self.temp, self.clip_lo, self.clip_hi,
                                   self.kl_coef, self.ids,
                                   scale=1.0 / (len(groups) * len(rolls)))
            rolls_flat += rolls
            m["loss"].append(lo)
            m["kl"].append(kl)
            m["reward"] += rs
            m["ce"] += [ro["ce"] for ro in rolls]
            m["writes"] += sum(ro["n_writes"] for ro in rolls)
            m["turns"] += sum(len(ro["recs"]) for ro in rolls)
            m["env"][g["env"]] = m["env"].get(g["env"], 0) + 1
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       float(self.r.get("grad_clip", 1.0)))
        self.opt.step()
        self.step += 1
        if self.step % self.publish_every == 0:
            self.hub.publish(self.model.state_dict(), self.step)
        return {"step": self.step, "reward": st.mean(m["reward"]),
                "ce": st.mean(m["ce"]),
                "write_rate": m["writes"] / max(m["turns"], 1),
                "kl": st.mean(m["kl"]), "loss": sum(m["loss"]),
                "pos_corr": pos_write_corr(rolls_flat),
                "groups": len(groups), "env_mix": m["env"]}

    def run(self):
        r = self.r
        steps = int(r["steps"])
        save_every = int(r.get("save_every", 50))
        t0 = time.time()
        n_stale_tot = 0
        while self.step < steps:
            if os.path.exists(os.path.join(self.root, "STOP")):
                break
            groups, n_stale = self.store.take(self.gps,
                                              self.step - self.max_lag)
            n_stale_tot += n_stale
            if not groups:
                time.sleep(self.poll_s)
                continue
            line = self.step_once(groups)
            line["stale"] = n_stale_tot
            line["s_per_step"] = (time.time() - t0) / max(self.step, 1)
            print(f"step {line['step']:4d}  r {line['reward']:+.3f}  "
                  f"ce {line['ce']:.3f}  write% {line['write_rate']:.2f}  "
                  f"kl {line['kl']:.2e}  groups {line['groups']}  "
                  f"stale {n_stale_tot}  {line['env_mix']}  "
                  f"{line['s_per_step']:.1f}s/step", flush=True)
            with open(self.mfile, "a") as fh:
                fh.write(json.dumps(line) + "\n")
            if self.step % save_every == 0 or self.step >= steps:
                _atomic_save({"model": self.model.state_dict(),
                              "opt": self.opt.state_dict(),
                              "step": self.step}, self.ck_path)
        _atomic_save({"model": self.model.state_dict(),
                      "opt": self.opt.state_dict(), "step": self.step},
                     self.ck_path)
        print(f"learner: done at step {self.step}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv):
    import yaml
    role, cfg_path = argv[0], argv[1]
    raw = yaml.safe_load(open(cfg_path))
    if role == "learner":
        Learner(raw).run()
    elif role == "worker":
        if "--worker" in argv:
            wid = int(argv[argv.index("--worker") + 1])
        else:
            # farm convention: WORKER = "<hostname>-gpuN" (gpu_worker.sh) —
            # the trailing digits are the per-rig GPU index
            import re
            m = re.search(r"(\d+)$", os.environ.get("WORKER", "0"))
            wid = int(m.group(1)) if m else 0
        Worker(raw, wid).run()
    else:
        raise SystemExit(f"role {role!r} not in (learner, worker)")


# ── CPU self-test (tiny model, stub envs, in-process) ────────────────────────

def _self_test():
    import shutil
    import tempfile
    from .config import ThoughtBankConfig
    from .model import ThoughtBankLM

    root = tempfile.mkdtemp(prefix="rl_disagg_")
    torch.manual_seed(0)
    V, THINK, BLANK, IM_END = 96, 1, 2, 3
    cfg = ThoughtBankConfig(vocab_size=V, d_model=32, n_layers=2, n_heads=2,
                            d_head=8, max_seq_len=128, n_hc=2,
                            sinkhorn_iters=5, csa_m=4, hca_m=8, top_k_csa=2,
                            n_win=4, d_latent_q=16, n_groups=1, n_experts=2,
                            n_shared=1, top_k_experts=1, d_ff=64,
                            mem_dim=16, max_mem=4, mem_seed_slots=2,
                            use_dual_stream=True)
    model = ThoughtBankLM(cfg)

    class _Tok:                                # decode: call / code / garbage
        def __init__(self, rng):
            self._r = rng

        def decode(self, ids):
            return self._r.choice(
                ['{"name": "fn_0", "arguments": {"x": 0}}',
                 "```python\ndef add(a, b):\n    return a + b\n```",
                 "not a call"])

        def get_vocab(self):
            return {"<think>": THINK, "<blank>": BLANK, "<|im_end|>": IM_END}

        def convert_tokens_to_ids(self, t):
            return self.get_vocab().get(t, 0)

        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [5]}

    class StubStream:
        """Chat-shaped conv dicts with gold_calls (tool bridge)."""

        def __init__(self, seed):
            self.rng = _random.Random(seed)

        def next_conv(self):
            n = self.rng.randint(3, 5)
            segs = [{"input_ids": torch.randint(4, V, (1, 12))}
                    for _ in range(n)]
            return {"kind": "toolcall", "segs": segs,
                    "info": {"gold_calls":
                             [[{"name": "fn_0", "arguments": {"x": 0}}]]}}

    class ExecStub:
        """Chat-shaped conv dicts with tests (exec bridge, REAL sandbox)."""

        def __init__(self, seed):
            self.rng = _random.Random(seed)

        def next_conv(self):
            n = self.rng.randint(3, 5)
            segs = [{"input_ids": torch.randint(4, V, (1, 12))}
                    for _ in range(n)]
            return {"kind": "codeexec", "segs": segs,
                    "info": {"tests": [["assert add(1, 2) == 3"]]}}

    class CodeStub:
        def __init__(self, seed):
            self.rng = _random.Random(seed)

        def next_conv(self):
            return [{"input_ids": torch.randint(4, V, (1, 12))}
                    for _ in range(self.rng.randint(3, 5))]

    raw = {"model": {"n_layers": 2, "max_mem": 4, "mem_seed_slots": 2},
           "data": {"defer_len": 8,
                    "envs": [{"name": "code", "lambda_write": 0.03},
                             {"name": "tools", "lambda_write": 0.0},
                             {"name": "exec", "lambda_write": 0.0,
                              "max_new": 4}]},
           "rl": {"seed": 0, "steps": 3, "group_size": 4,
                  # temp 8: a random tiny model has p(think) ~ 1/V => every
                  # group degenerates (no writes anywhere); flattening the
                  # Bernoulli to ~0.5 exercises the dense-reward path too
                  "groups_per_step": 2, "lr": 1e-4, "temp": 8.0,
                  "min_reward_std": 1e-6, "max_resample": 8,
                  "n_lives_per_worker": 2, "max_new": 4, "amp": False,
                  "think_nmax": 8,
                  "disagg": {"root": root, "publish_every": 1, "max_lag": 2,
                             "poll_s": 0.01, "xdom_every": 0}}}

    # 1. hub round-trip + prune
    hub = WeightHub(root, keep=2)
    for s in (0, 1, 2):
        hub.publish(model.state_dict(), s)
    assert hub.latest_step() == 2
    assert hub.fetch(2) is None and hub.fetch(1)[1] == 2
    assert len([p for p in os.listdir(os.path.join(root, "weights"))
                if p.endswith(".pt")]) == 2   # pruned to keep

    # 2. learner init (publishes step 0 over the pruned hub)
    learner = Learner(raw, model=copy.deepcopy(model),
                      device=torch.device("cpu"))
    assert hub.latest_step() == 0

    # 3. worker: envs injected, produces groups against published weights
    tok = _Tok(_random.Random(3))
    envs = [EnvSpec("code", CodeStub(1), weight=1.0),
            EnvSpec("tools", StubStream(2), weight=1.0,
                    reward_fn=make_tool_reward(8)),
            EnvSpec("exec", ExecStub(4), weight=1.0,
                    reward_fn=make_exec_reward(8))]
    w = Worker(raw, 0, tok=tok, model=copy.deepcopy(model), envs=envs,
               device=torch.device("cpu"))
    w.ids = (THINK, BLANK)
    w.stop_id, w.max_new = IM_END, 4
    w.a_open = torch.tensor([[5]], dtype=torch.long)
    w.wait_weights()
    assert w.wstep == 0
    lines = []
    while len(lines) < 9 or \
            {ln["env"] for ln in lines} != {"code", "tools", "exec"}:
        got = w.one_group()
        if got:
            lines.append(got)
    assert w.store.pending() == len(lines)
    envs_seen = {ln["env"] for ln in lines}    # all three reward paths
    assert lines[0]["turns"] > 0

    # 4. staleness: a group tagged far behind gets quarantined
    g_old = torch.load(os.path.join(w.store.inc,
                                    sorted(os.listdir(w.store.inc))[0]),
                       map_location="cpu", weights_only=False)
    w.store.put(g_old, weights_step=-10, worker=9)

    # 5. learner consumes (quarantining the stale one), steps, republishes
    groups, n_stale = learner.store.take(len(lines) + 2, -5)
    assert n_stale == 1 and len(groups) == len(lines)
    line = learner.step_once(groups)
    assert line["step"] == 1 and line["groups"] == len(lines)
    assert hub.latest_step() == 1
    assert w.refresh() and w.wstep == 1

    # 6. rewards sane: rubric rewards (tools AND exec) within [0, 1]
    for g in groups:
        if g["env"] in ("tools", "exec"):
            assert all(0.0 <= ro["reward"] <= 1.0 for ro in g["rollouts"])

    # 7. lives persisted + xdom probe returns the four figures
    w.save_lives()
    assert os.path.exists(w.lives_path)
    probe = w.xdom_probe()
    assert {"r_own", "r_xdom", "r_always", "r_never"} <= set(probe)

    shutil.rmtree(root)
    print(f"rl_disagg self-test: OK (hub, store+staleness, worker groups "
          f"[{', '.join(sorted(envs_seen))}], learner step+republish, "
          f"refresh, lives, xdom probe)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        _self_test()
