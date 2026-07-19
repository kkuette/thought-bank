"""dsv6 RL phase 2 groundwork — continuous LIVES across mixed environments.

Design (user, 2026-07-13): episodes come from a weighted MIX of environments
and the bank is NEVER reset between episodes. This rebuilds, at the policy
level, the exact structure the SFT arc validated at the data level:
  episode = file/thread, no-reset between envs = idea D, mixing = idea G.
The SFT history says the mix is not a bonus but a REQUIREMENT: no-reset alone
taught a boundary shortcut ("last gist off-topic => previous context is
disposable"); interleaving killed it. Blocked env schedules would resurrect it.

This module provides the three state primitives that make continuous lives
trainable with GRPO, plus the orchestrator:

  1. mem_snapshot / mem_restore / mem_fork — pure-state capture of the memory
     (live bank + cascade levels + counters). Snapshots are DETACHED CLONES,
     never recomputed from chunks (bf16 replay would drift from the original).
     fork() is the GRPO group primitive: the G rollouts of a group must start
     from the SAME carried bank (the seed_bank trap, generalized — different
     histories inside a group confound the advantage with the policy signal).

  2. EnvMixer — weighted episode sampling over named environments. Each env
     wraps a stream (anything with .next_conv() and .rng) and, later, a scorer
     (the verifiers rubric hook: reward_fn(completion/ce, info) -> float).
     v1 reward stays dense (-CE of the deferred continuation) — at 97M,
     verifiable rubrics risk flat rewards (whole group fails => no gradient);
     rubrics take over at scale, through the same hook.

  3. LivesState — N parallel lives, each carrying its memory snapshot across
     episodes, with full state_dict()/load_state_dict(): mem snapshots, per-env
     episode counts, mixer AND stream rng states. An RL checkpoint that resumes
     with reset banks silently re-biases the policy toward fresh-bank behavior
     (it only ever samples life-beginnings); resume must restore the lives.

REWARD DESIGN RULE (standing note, FINDINGS.md): every reward stays
task-grounded. Cross-episode continuity is free: retrieving a fact planted
three episodes ago IS use, not retention — no term here rewards keeping
particular contents alive, and none should ever.

CPU self-test:  python -m deepseek_v4_mini.rl_lives
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

import torch

from .cascade import CascadeMemory


# ── memory state primitives ──────────────────────────────────────────────────

def mem_snapshot(bank: torch.Tensor, casc: Optional[CascadeMemory] = None,
                 to_cpu: bool = False) -> dict:
    """Pure-state capture: detached clones of the live bank and every cascade
    tensor + counters. to_cpu=True for disk checkpoints (fork keeps device)."""
    mv = (lambda t: t.detach().clone().cpu()) if to_cpu else (lambda t: t.detach().clone())
    snap = {"bank": mv(bank), "casc": None}
    if casc is not None:
        snap["casc"] = {
            "depth": casc.depth, "M": casc.M,
            "n_pushed": casc.n_pushed, "n_destroyed": casc.n_destroyed,
            "lv": {k: {"p0": [mv(u) for u in L["p0"]],
                       "p1": None if L["p1"] is None else mv(L["p1"])}
                   for k, L in casc.lv.items()},
        }
    return snap


def mem_restore(snap: dict, device=None, dtype=None):
    """Rebuild (bank, casc) from a snapshot. Returns fresh tensors — restoring
    twice gives two independent states."""
    mv = lambda t: t.clone().to(device=device or t.device, dtype=dtype or t.dtype)
    bank = mv(snap["bank"])
    casc = None
    if snap["casc"] is not None:
        c = snap["casc"]
        casc = CascadeMemory(c["depth"], c["M"])
        casc.n_pushed, casc.n_destroyed = c["n_pushed"], c["n_destroyed"]
        for k, L in c["lv"].items():
            casc.lv[int(k)] = {"p0": [mv(u) for u in L["p0"]],
                               "p1": None if L["p1"] is None else mv(L["p1"])}
    return bank, casc


def mem_fork(bank: torch.Tensor, casc: Optional[CascadeMemory], n: int):
    """GRPO group primitive: n independent (bank, casc) copies of the carried
    state, all identical, all detached. High-frequency: stays on device."""
    snap = mem_snapshot(bank, casc)
    return [mem_restore(snap) for _ in range(n)]


# ── environments & mixer ─────────────────────────────────────────────────────

class EnvSpec:
    """One environment: a named stream + optional scorer.

    stream    anything exposing .next_conv() -> list[seg] and .rng (Random)
    reward_fn hook for verifiers-style rubrics: (ce, info: dict) -> float.
              None => dense default, reward = -ce (the v1 signal). The rubric
              path plugs here WITHOUT touching the orchestrator: score the
              rollout text, ignore ce.
    """

    def __init__(self, name: str, stream, weight: float = 1.0,
                 reward_fn: Optional[Callable[[float, dict], float]] = None):
        self.name, self.stream, self.weight, self.reward_fn = name, stream, weight, reward_fn

    def reward(self, ce: float, info: dict) -> float:
        return -ce if self.reward_fn is None else float(self.reward_fn(ce, info))


class EnvMixer:
    """Weighted episode sampling over environments. One episode = one conv of
    the underlying stream (chunks + terminal deferred target), tagged with its
    env. λ_write and the emission-rate guard are calibrated PER ENV by the
    caller (different rubric scales must not let the write policy over-invest
    generous envs — GRPO's in-group baseline absorbs most of it, the cost
    terms do not)."""

    def __init__(self, envs: List[EnvSpec], seed: int = 0):
        assert envs, "EnvMixer needs at least one environment"
        self.envs = {e.name: e for e in envs}
        self._names = [e.name for e in envs]
        self._weights = [float(e.weight) for e in envs]
        self.rng = random.Random(seed)

    def next_episode(self, defer_len: int, device, min_turns: int = 2):
        """Sample env by weight, then a conversation deep enough for RL:
        segs[:-1] = writable turns, last seg's opening = terminal target.
        Returns (env_name, chunks, tgt)."""
        name = self.rng.choices(self._names, weights=self._weights, k=1)[0]
        stream = self.envs[name].stream
        while True:
            segs = stream.next_conv()
            if len(segs) >= min_turns + 1 and segs[-1]["input_ids"].size(1) >= defer_len:
                chunks = [s["input_ids"].to(device) for s in segs[:-1]]
                tgt = segs[-1]["input_ids"][:, :defer_len].to(device)
                return name, chunks, tgt

    # rng of the mixer AND of every stream — full sampling determinism on resume
    def state_dict(self) -> dict:
        return {"mixer_rng": self.rng.getstate(),
                "stream_rng": {n: e.stream.rng.getstate() for n, e in self.envs.items()}}

    def load_state_dict(self, sd: dict) -> None:
        self.rng.setstate(_as_rng_state(sd["mixer_rng"]))
        for n, st in sd["stream_rng"].items():
            self.envs[n].stream.rng.setstate(_as_rng_state(st))


def _as_rng_state(st):
    """random.setstate needs (int, tuple[int...], float|None); pickle keeps
    tuples but a JSON detour would listify them — normalize defensively."""
    return (st[0], tuple(st[1]), st[2])


# ── lives ────────────────────────────────────────────────────────────────────

class Life:
    """One continuous life: memory carried across episodes, never reset.
    Holds the CURRENT state as tensors (hot path) and snapshots on demand."""

    def __init__(self, life_id: int, bank: torch.Tensor,
                 casc: Optional[CascadeMemory] = None):
        self.id = life_id
        self.bank = bank
        self.casc = casc
        self.n_episodes = 0
        self.env_counts: Dict[str, int] = {}

    def advance(self, bank: torch.Tensor, env_name: str) -> None:
        """Commit the post-episode state (the CHOSEN rollout's bank)."""
        self.bank = bank.detach()
        self.n_episodes += 1
        self.env_counts[env_name] = self.env_counts.get(env_name, 0) + 1

    def fork_group(self, g: int):
        """G identical (bank, casc) copies for one GRPO group."""
        return mem_fork(self.bank, self.casc, g)

    def state_dict(self) -> dict:
        return {"id": self.id, "n_episodes": self.n_episodes,
                "env_counts": dict(self.env_counts),
                "mem": mem_snapshot(self.bank, self.casc, to_cpu=True)}


class LivesState:
    """N parallel lives + the mixer: everything an RL checkpoint must carry
    beyond model/optimizer so a resumed run CONTINUES the same lives."""

    def __init__(self, lives: List[Life], mixer: EnvMixer):
        self.lives = lives
        self.mixer = mixer

    def state_dict(self) -> dict:
        return {"lives": [lf.state_dict() for lf in self.lives],
                "mixer": self.mixer.state_dict()}

    def load_state_dict(self, sd: dict, device=None, dtype=None) -> None:
        assert len(sd["lives"]) == len(self.lives), \
            f"checkpoint has {len(sd['lives'])} lives, orchestrator {len(self.lives)}"
        for lf, s in zip(self.lives, sd["lives"]):
            lf.id = s["id"]
            lf.n_episodes = s["n_episodes"]
            lf.env_counts = dict(s["env_counts"])
            lf.bank, lf.casc = mem_restore(s["mem"], device=device, dtype=dtype)
        self.mixer.load_state_dict(sd["mixer"])


# ── CPU self-test (hermetic: stub streams, no model, no GPU) ─────────────────

def _self_test() -> None:
    torch.manual_seed(0)
    B, M, D = 1, 4, 8

    # 1. snapshot/restore round-trip, live bank only
    bank = torch.randn(B, M, D)
    s = mem_snapshot(bank)
    b2, c2 = mem_restore(s)
    assert torch.equal(bank, b2) and c2 is None
    b2 += 1.0
    assert not torch.equal(bank, b2), "restore must be independent"

    # 2. with cascade: state equality via read() and stats(), independence
    casc = CascadeMemory(depth=2, max_mem=M)
    for _ in range(11):                       # fills lv1 p1 + partial p0
        casc.push_slot(torch.randn(B, D))
    s = mem_snapshot(bank, casc)
    b3, c3 = mem_restore(s)
    assert c3.stats() == casc.stats()
    assert torch.equal(c3.read(1), casc.read(1))
    c3.push_slot(torch.randn(B, D))
    assert c3.stats() != casc.stats(), "restored cascade must be independent"

    # 3. fork: g identical, mutating one leaves the others intact
    forks = mem_fork(bank, casc, 3)
    for fb, fc in forks:
        assert torch.equal(fb, bank) and fc.stats() == casc.stats()
    forks[0][0].zero_()
    forks[0][1].push_slot(torch.randn(B, D))
    assert torch.equal(forks[1][0], bank)
    assert forks[1][1].stats() == casc.stats()

    # 4. disk round-trip (to_cpu) — the checkpoint path
    import io
    buf = io.BytesIO()
    torch.save(mem_snapshot(bank, casc, to_cpu=True), buf)
    buf.seek(0)
    b4, c4 = mem_restore(torch.load(buf, weights_only=False))
    assert torch.equal(b4, bank) and c4.stats() == casc.stats()
    assert torch.equal(c4.read(2) if c4.read(2) is not None else torch.zeros(1),
                       casc.read(2) if casc.read(2) is not None else torch.zeros(1))

    # 5. mixer: weighted sampling + full rng determinism through state_dict
    class StubStream:
        def __init__(self, seed, tag):
            self.rng = random.Random(seed)
            self.tag = tag
        def next_conv(self):
            n = self.rng.randint(3, 5)
            return [{"input_ids": torch.full((1, 16), self.tag + i)} for i in range(n)]

    def mk():
        return EnvMixer([EnvSpec("code", StubStream(1, 100), weight=0.7),
                         EnvSpec("math", StubStream(2, 200), weight=0.3)], seed=7)

    mx = mk()
    for _ in range(5):
        mx.next_episode(8, "cpu")
    sd = mx.state_dict()
    # torch round-trip (tuples -> lists) then resume
    buf = io.BytesIO(); torch.save(sd, buf); buf.seek(0)
    sd2 = torch.load(buf, weights_only=False)
    a = [mx.next_episode(8, "cpu") for _ in range(6)]
    my = mk()
    for _ in range(5):
        my.next_episode(8, "cpu")            # advance naturally...
    my.load_state_dict(sd2)                   # ...then overwrite with saved state
    b = [my.next_episode(8, "cpu") for _ in range(6)]
    for (na, ca, ta), (nb, cb, tb) in zip(a, b):
        assert na == nb and len(ca) == len(cb)
        assert all(torch.equal(x, y) for x, y in zip(ca, cb)) and torch.equal(ta, tb)

    # 6. lives: advance + full checkpoint round-trip
    lives = [Life(i, torch.randn(B, M, D), CascadeMemory(2, M)) for i in range(2)]
    lives[0].casc.push_slot(torch.randn(B, D))
    lives[0].advance(torch.randn(B, M, D), "code")
    st8 = LivesState(lives, mx)
    buf = io.BytesIO(); torch.save(st8.state_dict(), buf); buf.seek(0)
    fresh = LivesState([Life(i, torch.zeros(B, M, D), CascadeMemory(2, M))
                        for i in range(2)], mk())
    fresh.load_state_dict(torch.load(buf, weights_only=False))
    assert torch.equal(fresh.lives[0].bank, lives[0].bank)
    assert fresh.lives[0].env_counts == {"code": 1}
    assert fresh.lives[0].casc.stats() == lives[0].casc.stats()
    g = fresh.lives[0].fork_group(4)
    assert len(g) == 4 and all(torch.equal(fb, lives[0].bank) for fb, _ in g)

    print("rl_lives self-test: OK (snapshot/restore/fork, disk round-trip, "
          "mixer determinism, lives checkpoint)")


if __name__ == "__main__":
    _self_test()
