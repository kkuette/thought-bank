"""Code-execution sessions — problems stated early, implemented late, graded
by RUNNING the code (phase 2 — the exec twin of tool_env_data).

Why this shape: served as-is, a code-instruct dataset puts the problem right
above the solution — a context task, zero memory pressure. Here one session
packs 2..4 problems: each problem is DECLARED in its own early turn (P1., P2.,
...), then the implementation requests arrive in a DIFFERENT order, whole
declarations and implementations apart. By the time "Now implement P2" lands,
the spec of P2 is out of the local window: writing correct code = reading the
spec back from the bank. Same lengthen-and-distance principle as
tool_env_data, with the STRONGEST verifiable grader there is — the Python
interpreter (exec_sandbox: reward = fraction of unit tests passed, no judge).

Episode pool: nvidia/OpenCodeInstruct (CC-BY-4.0 — compatible open-core;
open-r1/PrimeIntellect pools have NO declared license, excluded like
APIGen-MT). Kept rows: average_test_score == 1, parseable unit_tests, code
extractable from the fenced output, token budgets met, and the gold VERIFIED
in the sandbox at pool build (grade(gold) == 1.0 by construction — the same
guarantee the tool pool gets for free from its JSON gold).

Interface = the chat-stream contract of code_defer_native (.next_conv() +
.rng + .grade_conv). info.tests (per graded turn, query order) is the RL
bridge: rl_disagg decodes the implementation turn from the carried bank and
rewards it with rl_rewards.make_exec_reward — same sessions for SFT and GRPO.

Hermetic self-test (stub tokenizer, real sandbox, no downloads):
  python -m deepseek_v4_mini.code_exec_data
Real smoke (streams OpenCodeInstruct, prints one session):
  python -m deepseek_v4_mini.code_exec_data <yaml with tokenizer: + exec.gen>
"""
from __future__ import annotations

import json
import sys

from .exec_sandbox import pass_frac, run_tests
from .persona_chat_data import PersonaChatStream
from .rl_rewards import extract_code

_ASK = ("Now implement {label}. Reply with a single ```python``` code block "
        "containing the full solution.")


def parse_oci_row(row: dict, max_tests: int = 8):
    """{"problem", "code", "tests"} or None — nvidia/OpenCodeInstruct layout:
    input = problem statement, output = fenced solution, unit_tests =
    JSON-string list of assert snippets, average_test_score = "0".."1"."""
    try:
        if float(row.get("average_test_score") or 0) < 1.0:
            return None
        tests = [t.strip() for t in json.loads(row["unit_tests"])
                 if isinstance(t, str) and t.strip()]
    except (KeyError, TypeError, ValueError):
        return None
    code = extract_code(row.get("output") or "")
    problem = (row.get("input") or "").strip()
    if not (problem and code and len(tests) >= 2):
        return None
    return {"problem": problem, "code": code, "tests": tests[:max_tests]}


class CodeExecStream(PersonaChatStream):
    """Packs verified coding problems into spec-far sessions.

    Reuses the persona plumbing (_seg/_user/_assistant: ChatML, loss masks,
    surp_w/SIF) exactly like ToolSessionStream. A pool item is
    {"problem": str, "code": str, "tests": [str], "gold": str}."""

    def __init__(self, tok, *,
                 dataset: str = "nvidia/OpenCodeInstruct",
                 split: str = "train",
                 problem_cap: int = 4000,       # verified problems kept
                 max_problem_tok: int = 256,
                 max_code_tok: int = 224,
                 max_tests: int = 8,
                 probs_per_session: tuple = (2, 4),
                 verify_gold: bool = True,      # sandbox-check at pool build
                 real_cache_dir: str = None,
                 surprisal_ref: str = None, surprisal_device: str = "cpu",
                 surprisal_alpha: float = 2.0, surprisal_mode: str = "nll",
                 sif_a: float = 1e-4, seed: int = 0, _pool=None) -> None:
        super().__init__(tok, real_filler=None,
                         surprisal_ref=surprisal_ref,
                         surprisal_device=surprisal_device,
                         surprisal_alpha=surprisal_alpha,
                         surprisal_mode=surprisal_mode,
                         sif_a=sif_a, seed=seed)
        self.probs_per_session = tuple(int(v) for v in probs_per_session)
        if _pool is not None:
            self.pool = _pool
        else:
            self.pool = self._load_pool(
                dataset, split, int(problem_cap), int(max_problem_tok),
                int(max_code_tok), int(max_tests), bool(verify_gold),
                real_cache_dir)
        assert self.pool, "code-exec problem pool is empty"

    # ── data ─────────────────────────────────────────────────────────────────
    def _n_tok(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def _load_pool(self, name, split, cap, max_prob, max_code, max_tests,
                   verify, cache_dir):
        import hashlib, os
        import torch
        key = hashlib.sha256(
            f"{name}|{split}|{cap}|{max_prob}|{max_code}|{max_tests}|{verify}"
            .encode()).hexdigest()[:16]
        path = (os.path.join(cache_dir, f"exec_probs_{key}.pt")
                if cache_dir else None)
        if path and os.path.exists(path):
            pool = torch.load(path)
            print(f"exec sessions: cache hit {path} — {len(pool)} problems")
            return pool
        from datasets import load_dataset
        rows = load_dataset(name, split=split, streaming=True)
        pool, n_seen, n_bad_gold = [], 0, 0
        for row in rows:
            n_seen += 1
            it = parse_oci_row(row, max_tests)
            if it is None:
                continue
            if (self._n_tok(it["problem"]) > max_prob
                    or self._n_tok(it["code"]) > max_code):
                continue
            if verify:                         # the grader's own guarantee:
                p, n = run_tests(it["code"], it["tests"])
                if p < n:                      # gold must grade 1.0
                    n_bad_gold += 1
                    continue
            it["gold"] = f"```python\n{it['code']}\n```"
            pool.append(it)
            if len(pool) >= cap:
                break
        print(f"exec sessions: {name}[{split}] — {len(pool)} problems kept "
              f"/ {n_seen} seen ({n_bad_gold} gold failed own tests)")
        if path:
            torch.save(pool, path)
        return pool

    # ── grading (verifiable — the interpreter, no judge) ─────────────────────
    @staticmethod
    def grade_conv(conv: dict, texts: list) -> float:
        tests = conv["info"]["tests"]
        if not tests:
            return 1.0
        tail = texts[-len(tests):]
        return sum(pass_frac(extract_code(t or ""), ts)
                   for t, ts in zip(tail, tests)) / len(tests)

    # ── session assembly ─────────────────────────────────────────────────────
    def next_conv(self) -> dict:
        n = self.rng.randint(*self.probs_per_session)
        probs = [self.pool[i]
                 for i in self.rng.sample(range(len(self.pool)), n)]
        # declarations first, one turn each (chunk-friendly; each is a write
        # opportunity), labels in declaration order
        segs = [self._user(f"P{i + 1}. {p['problem']}")
                for i, p in enumerate(probs)]
        # implementation requests in an order DIFFERENT from declaration
        order = list(range(n))
        for _ in range(4):
            self.rng.shuffle(order)
            if n == 1 or order != list(range(n)):
                break
        truths, tests, ages = [], [], []
        for k in order:
            p = probs[k]
            segs.append(self._user(_ASK.format(label=f"P{k + 1}")))
            ages.append(len(segs) - 1 - k)     # writes since P{k+1}'s turn
            segs.append(self._assistant(p["gold"]))
            truths.append(p["gold"])
            tests.append(p["tests"])
        return {"kind": "codeexec", "segs": segs,
                "info": {"truths": truths, "tests": tests,
                         "ages": ages, "n_probs": n}}


# ── smoke ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:                      # real tokenizer + real dataset
        import yaml
        from transformers import AutoTokenizer
        raw = yaml.safe_load(open(sys.argv[1]))
        tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
        gen = ((raw.get("exec", {}) or {}).get("gen", {}) or {})
        st = CodeExecStream(tok, **gen)
        n_segs, ages = [], []
        for _ in range(50):
            c = st.next_conv()
            n_segs.append(len(c["segs"]))
            ages += c["info"]["ages"]
        print(f"segs/session min {min(n_segs)} med "
              f"{sorted(n_segs)[len(n_segs)//2]} max {max(n_segs)} | "
              f"ages {sorted(ages)[:5]}..{sorted(ages)[-5:]}")
        c = st.next_conv()
        for s in c["segs"][:3] + c["segs"][-2:]:
            print(repr(tok.decode(s["input_ids"][0].tolist())[:140]))
        pad = [""] * (len(c["segs"]) - len(c["info"]["truths"]))
        print("grade(gold) =", st.grade_conv(c, pad + c["info"]["truths"]))
        sys.exit(0)

    # hermetic: stub tokenizer + synthetic pool, REAL sandbox grading
    class _Tok:
        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [ord(ch) % 251 for ch in s]}

    row = {"input": "Write add(a, b) returning the sum.",
           "output": "```python\ndef add(a, b):\n    return a + b\n```",
           "unit_tests": json.dumps(["assert add(1, 2) == 3",
                                     "assert add(0, 0) == 0"]),
           "average_test_score": "1"}
    it = parse_oci_row(row)
    assert it and it["code"].startswith("def add") and len(it["tests"]) == 2
    assert parse_oci_row({**row, "average_test_score": "0.9"}) is None
    assert parse_oci_row({**row, "output": "no code"}) is None
    assert parse_oci_row({**row, "unit_tests": "[]"}) is None

    pool = []
    for i in range(12):
        code = f"def fn_{i}(x):\n    return x + {i}"
        pool.append({"problem": f"Write fn_{i}(x) returning x plus {i}.",
                     "code": code,
                     "tests": [f"assert fn_{i}(0) == {i}",
                               f"assert fn_{i}(10) == {10 + i}"],
                     "gold": f"```python\n{code}\n```"})
    st = CodeExecStream(_Tok(), _pool=pool, surprisal_mode="sif", seed=0)
    saw_multi = False
    for _ in range(30):
        c = st.next_conv()
        info = c["info"]
        n = info["n_probs"]
        assert len(info["truths"]) == len(info["tests"]) == n
        assert len(c["segs"]) == 3 * n and all(a > 0 for a in info["ages"])
        for s in c["segs"]:
            assert "role" in s and s["surp_w"].shape == s["input_ids"].shape
        # declarations never supervised; implementation turns are
        assert float(c["segs"][0]["loss_mask"].sum()) == 0.0
        assert float(c["segs"][-1]["loss_mask"].sum()) > 0.0
        pad = [""] * (len(c["segs"]) - n)
        assert st.grade_conv(c, pad + info["truths"]) == 1.0
        assert st.grade_conv(c, pad + ["nope"] * n) == 0.0
        if n >= 2:
            saw_multi = True
            half = info["truths"][:-1] + ["def broken(:"]
            g = st.grade_conv(c, pad + half)
            assert abs(g - (n - 1) / n) < 1e-9
    assert saw_multi
    print("code_exec_data self-test: OK (parse, assembly, ages, sandbox "
          "grading)")
