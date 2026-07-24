"""ChatMixStream — weighted mix of chat streams behind the ONE-stream contract.

Motivation (smoke disagg 97M, 2026-07-24): the tools GRPO env cold-starts at
grade 0 (whole groups degenerate, dynamic sampling discards them — lesson 5i
observed live). The bootstrap is SFT-side: the sota SFT must contain a share
of tool-call sessions so grade > 0 exists before the ratchet's first RL pass.
code_defer_native wires exactly ONE chat stream (next_conv + rng get/set/seed
+ grade_conv); this wrapper mixes N streams behind that same contract, so the
trainer only learns one new name ("chat_mix").

  gen:
    streams:
    - stream: sota_session        # sota_session | tool_session | persona |
      weight: 0.8                 #   math_school
      gen: {...}                  # kwargs of that stream
    - stream: tool_session
      weight: 0.2
      gen: {...}

Checkpoint determinism: the trainer saves/restores chat_stream.rng state. Here
.rng is a composite (picker + every sub-stream rng): getstate() bundles all,
setstate() restores all, seed() reseeds all — a resumed run CONTINUES the
same mixed sequence. grade_conv dispatches to the stream that PRODUCED the
conv (tagged in info), so per-kind graders keep their own semantics.

CPU self-test (hermetic):  python -m deepseek_v4_mini.chat_mix
"""
from __future__ import annotations

import random

_TAG = "chat_mix_rng_v1"


class _MixRng:
    """Composite rng honoring the trainer's rng contract (getstate/setstate/
    seed/random) across the picker and every sub-stream."""

    def __init__(self, pick: random.Random, subs):
        self._pick = pick
        self._subs = subs                      # list of streams (own .rng each)

    def getstate(self):
        return (_TAG, self._pick.getstate(),
                [s.rng.getstate() for s in self._subs])

    def setstate(self, st):
        assert isinstance(st, tuple) and st[0] == _TAG, \
            "rng state is not a chat_mix bundle (resume from a mono-stream " \
            "checkpoint: start fresh instead)"
        self._pick.setstate(st[1])
        for s, sub_st in zip(self._subs, st[2]):
            s.rng.setstate(sub_st)

    def seed(self, n):
        self._pick.seed(n)
        for i, s in enumerate(self._subs):
            s.rng.seed(n + 101 * (i + 1))

    def random(self):
        return self._pick.random()


def _build(name: str, tok, seed: int, gen: dict):
    if name == "sota_session":
        from .sota_session_data import SotaSessionStream as C
    elif name == "tool_session":
        from .tool_env_data import ToolSessionStream as C
    elif name == "persona":
        from .persona_chat_data import PersonaChatStream as C
    elif name == "math_school":
        from .math_school_data import MathSchoolStream as C
    else:
        raise ValueError(f"chat_mix: unknown sub-stream {name!r}")
    return C(tok, seed=seed, **(gen or {}))


class ChatMixStream:
    def __init__(self, tok, *, seed: int = 0, streams: list = None, _subs=None):
        assert streams or _subs, "chat_mix needs a streams: list"
        if _subs is not None:                  # test injection
            self.subs = [s for s, _ in _subs]
            self._weights = [w for _, w in _subs]
        else:
            self.subs = [_build(e["stream"], tok, seed + 101 * (i + 1),
                                e.get("gen") or {})
                         for i, e in enumerate(streams)]
            self._weights = [float(e.get("weight", 1.0)) for e in streams]
        self.rng = _MixRng(random.Random(seed), self.subs)
        self._idx = list(range(len(self.subs)))

    def next_conv(self) -> dict:
        i = self.rng._pick.choices(self._idx, weights=self._weights, k=1)[0]
        conv = self.subs[i].next_conv()
        conv.setdefault("info", {})["_mix_src"] = i
        return conv

    def grade_conv(self, conv: dict, texts) -> float:
        src = self.subs[conv.get("info", {}).get("_mix_src", 0)]
        return src.grade_conv(conv, texts)


# ── self-test (hermetic: stub sub-streams) ───────────────────────────────────
if __name__ == "__main__":
    class Stub:
        def __init__(self, tag, seed):
            self.tag = tag
            self.rng = random.Random(seed)

        def next_conv(self):
            return {"kind": self.tag, "segs": [self.rng.random()],
                    "info": {}}

        def grade_conv(self, conv, texts):
            return {"a": 0.25, "b": 0.75}[self.tag]

    def mk():
        return ChatMixStream(None, seed=7,
                             _subs=[(Stub("a", 1), 0.7), (Stub("b", 2), 0.3)])

    st = ChatMixStream(None, seed=7,
                       _subs=[(Stub("a", 1), 0.7), (Stub("b", 2), 0.3)])
    kinds = {"a": 0, "b": 0}
    for _ in range(300):
        c = st.next_conv()
        kinds[c["kind"]] += 1
        # grade dispatch follows the producer
        assert st.grade_conv(c, []) == {"a": 0.25, "b": 0.75}[c["kind"]]
    assert kinds["a"] > kinds["b"] > 20, kinds

    # checkpoint round-trip: same mixed sequence after restore
    sd = st.rng.getstate()
    a = [st.next_conv() for _ in range(20)]
    st2 = mk()
    for _ in range(11):                        # advance differently...
        st2.next_conv()
    st2.rng.setstate(sd)                       # ...then restore
    b = [st2.next_conv() for _ in range(20)]
    assert [(x["kind"], x["segs"]) for x in a] == \
           [(x["kind"], x["segs"]) for x in b]

    # mono-stream state refused loudly
    try:
        st.rng.setstate(random.Random(0).getstate())
        raise SystemExit("should have refused")
    except AssertionError:
        pass

    # seed() reseeds everything deterministically
    st.rng.seed(42)
    s1 = [st.next_conv()["segs"] for _ in range(5)]
    st.rng.seed(42)
    assert s1 == [st.next_conv()["segs"] for _ in range(5)]

    print("chat_mix self-test: OK (mix, grade dispatch, rng bundle "
          "round-trip, refus mono-stream, seed)")
