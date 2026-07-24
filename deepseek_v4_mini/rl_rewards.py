"""dsv6 phase-2 rewards — verifiable tool-call grading + think economy.

Two pure ingredients for the GRPO ratchet on tool use (design 2026-07-23,
MAI-style SFT/RL cliquet; memory: xlam/glaive = cheap verifiable grader):

  1. TOOL-CALL GRADING (verifiable, no judge model). The gold of an episode
     is a list of calls {"name": str, "arguments": dict}. The rollout text is
     parsed for JSON call objects (fenced or bare); score per call = name gate
     (wrong function => 0) x argument F1 (exact match on normalized values —
     numbers compared as floats, strings stripped). Hallucinated arguments
     lower precision, missing ones lower recall: the F1 IS the anti-slop term,
     no extra penalty needed. Multiple calls: greedy 1-1 matching by best
     score (order-free — the model may emit calls in any order).

  2. THINK ECONOMY (decision 2026-07-23). <think> turns are bank WRITES (the
     H loop realized): each one costs a forward and a slot, so left free the
     policy pads. Shaping:
       n_think >  n_max          -> reward 0 (hard timeout, no partial credit)
       n_think <= n_max, success -> success x eff, eff linear 1 -> floor
     The floor (default 0.4, NEVER -> 0) keeps a successful-but-verbose
     rollout strictly better than a failed terse one: the standing reward rule
     says every term stays task-grounded — efficiency modulates success, it
     never becomes a reason to skip the task.

Everything here is text/floats in, float out: usable identically by the farm
rollout workers (rl_disagg), the learner's regrade path, and offline SFT
filtering of RL traces (the distill step of the ratchet).

CPU self-test:  python -m deepseek_v4_mini.rl_rewards
"""
from __future__ import annotations

import json
import re
from typing import Callable, List, Optional


# ── JSON call extraction ─────────────────────────────────────────────────────

_FENCE = re.compile(r"```(?:json|tool)?\s*(.*?)```", re.S)


def _balanced_spans(text: str):
    """Top-level {...} / [...] spans, string-aware (no regex for nesting)."""
    spans, depth, start, in_str, esc = [], 0, -1, False, False
    opener = closer = ""
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch in "{[":
            if depth == 0:
                start, opener = i, ch
                closer = "}" if ch == "{" else "]"
            if ch == opener:
                depth += 1
        elif ch in "}]":
            if depth > 0 and ch == closer:
                depth -= 1
                if depth == 0:
                    spans.append(text[start:i + 1])
    return spans


def extract_calls(text: str) -> List[dict]:
    """Every parseable call object in the text, fenced blocks first (a fenced
    call also appears bare — dedup by identity of the parsed object)."""
    raw = []
    for m in _FENCE.finditer(text):
        raw += _balanced_spans(m.group(1))
    raw += _balanced_spans(text)
    calls, seen = [], set()
    for s in raw:
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            continue
        objs = obj if isinstance(obj, list) else [obj]
        for o in objs:
            if not (isinstance(o, dict) and isinstance(o.get("name"), str)):
                continue
            args = o.get("arguments", o.get("parameters", {}))
            if isinstance(args, str):           # glaive nests args as a string
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    continue
            if not isinstance(args, dict):
                continue
            key = json.dumps({"name": o["name"], "arguments": args},
                             sort_keys=True)
            if key not in seen:
                seen.add(key)
                calls.append({"name": o["name"], "arguments": args})
    return calls


# ── grading ──────────────────────────────────────────────────────────────────

def _norm(v):
    """Comparable form: numbers as floats, strings stripped, containers deep."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        try:                                   # "2" vs 2 (LLMs stringify)
            return float(s)
        except ValueError:
            return s
    if isinstance(v, list):
        return tuple(_norm(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _norm(x)) for k, x in v.items()))
    return v


def grade_call(pred: dict, gold: dict) -> float:
    """Name gate x argument F1. Both arg dicts empty => 1.0 (name alone)."""
    if pred.get("name") != gold.get("name"):
        return 0.0
    pa = {k: _norm(v) for k, v in (pred.get("arguments") or {}).items()}
    ga = {k: _norm(v) for k, v in (gold.get("arguments") or {}).items()}
    if not pa and not ga:
        return 1.0
    hit = sum(1 for k, v in ga.items() if k in pa and pa[k] == v)
    if hit == 0:
        return 0.0
    p, r = hit / len(pa) if pa else 0.0, hit / len(ga)
    return 2 * p * r / (p + r)


def grade_calls(pred_text: str, golds: List[dict]) -> float:
    """Mean over gold calls of their best greedy 1-1 match among predicted
    calls. Extra predicted calls beyond the matching dilute nothing here —
    their args already paid inside grade_call; call-count discipline is the
    think-economy term's job, not the grader's."""
    if not golds:
        return 1.0
    preds = extract_calls(pred_text)
    if not preds:
        return 0.0
    pairs = sorted(((grade_call(p, g), pi, gi)
                    for pi, p in enumerate(preds)
                    for gi, g in enumerate(golds)), reverse=True)
    used_p, used_g, tot = set(), set(), 0.0
    for s, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        tot += s
    return tot / len(golds)


# ── code extraction (exec envs) ──────────────────────────────────────────────

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    """Candidate solution from rollout text: first fenced block wins (python
    fences first, then any fence with a def/class/import inside); bare text
    that LOOKS like code (starts a def/class/import somewhere) is the
    fallback. Returns "" when nothing code-like is found."""
    fences = _CODE_FENCE.findall(text)
    for f in fences:
        if re.search(r"^\s*(def |class |import |from )", f, re.M):
            return f.strip()
    if fences:
        return fences[0].strip()
    m = re.search(r"^(?:def |class |import |from )", text, re.M)
    return text[m.start():].strip() if m else ""


# ── think economy ────────────────────────────────────────────────────────────

def think_economy(success: float, n_think: int, n_max: int,
                  floor: float = 0.4) -> float:
    """success x eff, eff linear 1 -> floor over [0, n_max]; 0 past n_max.
    floor > 0 by design: efficiency modulates success, never erases it."""
    assert 0.0 < floor <= 1.0 and n_max > 0
    if n_think > n_max:
        return 0.0
    eff = floor + (1.0 - floor) * (1.0 - n_think / n_max)
    return float(success) * eff


def make_tool_reward(n_max: int, floor: float = 0.4
                     ) -> Callable[[Optional[float], dict], float]:
    """EnvSpec.reward_fn for tool envs: grades the rollout text against the
    episode's gold calls, then applies think economy. Ignores ce (the dense
    fallback) — this is the rubric path of the rl_lives hook.

    info: {"text": str, "gold_calls": [..], "n_think": int}"""
    def fn(ce, info):
        s = grade_calls(info["text"], info["gold_calls"])
        return think_economy(s, int(info.get("n_think", 0)), n_max, floor)
    return fn


def make_exec_reward(n_max: int, floor: float = 0.4, timeout: float = 6.0
                     ) -> Callable[[Optional[float], dict], float]:
    """EnvSpec.reward_fn for code-exec envs: extract the solution from the
    rollout text, run the episode's unit tests in the sandbox, success =
    fraction passed, then the same think economy as tool envs.

    info: {"text": str, "tests": [assert-str, ...], "n_think": int}"""
    from .exec_sandbox import pass_frac

    def fn(ce, info):
        s = pass_frac(extract_code(info["text"]), info["tests"],
                      timeout=timeout)
        return think_economy(s, int(info.get("n_think", 0)), n_max, floor)
    return fn


# ── self-test (hermetic) ─────────────────────────────────────────────────────

def _self_test() -> None:
    g = {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}

    # extraction: fenced, bare, list, glaive string-args, junk around
    t = 'sure!\n```json\n{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}\n```\ndone'
    assert extract_calls(t) == [g]
    t2 = 'call: {"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\", \\"days\\": 3}"} ok'
    assert extract_calls(t2) == [g]
    t3 = '[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"x": 1}}]'
    assert [c["name"] for c in extract_calls(t3)] == ["a", "b"]
    assert extract_calls("no json here { broken") == []
    assert extract_calls('{"not_a_call": 1}') == []

    # grading: exact, normalization, partial, wrong name, hallucinated args
    assert grade_call(g, g) == 1.0
    assert grade_call({"name": "get_weather",
                       "arguments": {"city": " Paris ", "days": "3"}}, g) == 1.0
    p_half = grade_call({"name": "get_weather",
                         "arguments": {"city": "Paris"}}, g)
    assert 0.5 < p_half < 1.0                  # recall 1/2, precision 1
    assert grade_call({"name": "get_wether", "arguments": g["arguments"]}, g) == 0.0
    p_extra = grade_call({"name": "get_weather",
                          "arguments": {"city": "Paris", "days": 3,
                                        "units": "C"}}, g)
    assert p_half < 1.0 and p_extra < 1.0      # both imperfect...
    assert p_extra > p_half                    # ...but hallucination < omission? no: 2 hits
    assert grade_calls(t, [g]) == 1.0
    assert grade_calls("nothing", [g]) == 0.0
    # order-free multi-call matching
    two = ('{"name": "b", "arguments": {"x": 1}} then '
           '{"name": "a", "arguments": {}}')
    assert grade_calls(two, [{"name": "a", "arguments": {}},
                             {"name": "b", "arguments": {"x": 1}}]) == 1.0
    assert grade_calls("", []) == 1.0          # no gold => control episode

    # think economy: linear, floor at n_max, hard 0 past it, success-gated
    assert think_economy(1.0, 0, 8) == 1.0
    assert abs(think_economy(1.0, 8, 8) - 0.4) < 1e-9
    assert think_economy(1.0, 9, 8) == 0.0
    assert think_economy(0.0, 2, 8) == 0.0
    assert think_economy(0.5, 4, 8) == 0.5 * (0.4 + 0.6 * 0.5)
    # monotone in n_think on [0, n_max]
    vals = [think_economy(1.0, n, 8) for n in range(9)]
    assert all(a > b for a, b in zip(vals, vals[1:]))

    # EnvSpec hook end-to-end
    fn = make_tool_reward(n_max=8)
    r = fn(None, {"text": t, "gold_calls": [g], "n_think": 4})
    assert abs(r - think_economy(1.0, 4, 8)) < 1e-9

    # code extraction: fenced python, generic fence with code, bare, none
    body = "def add(a, b):\n    return a + b"
    assert extract_code(f"Sure!\n```python\n{body}\n```\ndone") == body
    assert extract_code(f"```\n{body}\n```") == body
    assert extract_code(f"chat chat\n{body}\nmore chat").startswith("def add")
    assert extract_code("no code at all") == ""
    # prefers the fence that actually contains code
    two_f = f"```\njust text\n```\n```python\n{body}\n```"
    assert extract_code(two_f) == body

    # exec reward end-to-end (real sandbox)
    fx = make_exec_reward(n_max=8)
    txt = f"Here you go:\n```python\n{body}\n```"
    ts = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
    assert abs(fx(None, {"text": txt, "tests": ts, "n_think": 2})
               - think_economy(1.0, 2, 8)) < 1e-9
    half = fx(None, {"text": txt, "tests": ts + ["assert add(1, 1) == 3"] * 2,
                     "n_think": 0})
    assert abs(half - 0.5) < 1e-9              # 2/4 tests, eff=1 at n_think 0
    assert fx(None, {"text": "no code", "tests": ts, "n_think": 0}) == 0.0
    assert fx(None, {"text": txt, "tests": ts, "n_think": 9}) == 0.0

    print("rl_rewards self-test: OK (extraction, grading, matching, economy, code+exec)")


if __name__ == "__main__":
    _self_test()
