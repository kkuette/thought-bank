"""Sandboxed execution of model-written Python against unit tests.

The code-exec env's verifiable grader (phase 2 — same family as the tool-call
grader in rl_rewards, but the checker is the Python interpreter itself): run
the candidate solution, then each test (assert line) in its own try/except,
reward = fraction passed. Everything runs in a THROWAWAY subprocess:

  * `python -I` (isolated: no site, no cwd on path, env-var imports ignored);
  * rlimits set in the child pre-exec: CPU seconds, address space, file size,
    core dumps off — an infinite loop dies on RLIMIT_CPU, a memory bomb on
    RLIMIT_AS, and the wall-clock timeout backstops anything sleeping;
  * cwd = a fresh temp dir, deleted after; payload rides stdin as JSON (no
    shell quoting surface), verdicts come back as one JSON line on stdout.

Not a security boundary against an adversary (network is not blocked — the
rollout workers run as an unprivileged user on the rigs and the code comes
from OUR 350M policy, not from strangers); it IS a reliable crash/hang/hog
boundary, which is what grading needs.

Cost: one subprocess per grading call (~50-150 ms interpreter start + tests).
At G=8 rollouts/group that is well under the decode cost of the call turn.

CPU self-test:  python -m deepseek_v4_mini.exec_sandbox
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

# Child harness: read {"code", "tests"} from stdin, exec the code once, then
# each test against a shallow copy of the namespace; print [bool,...].
_HARNESS = r"""
import json, sys
payload = json.load(sys.stdin)
ns = {}
try:
    exec(compile(payload["code"], "<solution>", "exec"), ns)
except BaseException:
    print(json.dumps([False] * len(payload["tests"])))
    raise SystemExit(0)
res = []
for t in payload["tests"]:
    try:
        exec(compile(t, "<test>", "exec"), dict(ns))
        res.append(True)
    except BaseException:
        res.append(False)
print(json.dumps(res))
"""


def _limits(cpu_s: int, mem_mb: int):
    def fn():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
        resource.setrlimit(resource.RLIMIT_AS,
                           (mem_mb << 20, mem_mb << 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return fn


def run_tests(code: str, tests: list, *, timeout: float = 6.0,
              cpu_s: int = 5, mem_mb: int = 512) -> tuple:
    """(n_pass, n_total). Timeout / crash / unparseable output => 0 passed."""
    n = len(tests)
    if not code or not code.strip() or n == 0:
        return 0, n
    with tempfile.TemporaryDirectory(prefix="exec_sbx_") as cwd:
        try:
            out = subprocess.run(
                [sys.executable, "-I", "-c", _HARNESS],
                input=json.dumps({"code": code, "tests": list(tests)}),
                capture_output=True, text=True, cwd=cwd, timeout=timeout,
                preexec_fn=_limits(cpu_s, mem_mb),
                env={"PATH": os.defpath})
        except (subprocess.TimeoutExpired, OSError):
            return 0, n
    try:
        res = json.loads(out.stdout.strip().splitlines()[-1])
        assert isinstance(res, list) and len(res) == n
    except (ValueError, AssertionError, IndexError):
        return 0, n
    return sum(1 for r in res if r is True), n


def pass_frac(code: str, tests: list, **kw) -> float:
    p, n = run_tests(code, tests, **kw)
    return p / n if n else 0.0


# ── self-test ────────────────────────────────────────────────────────────────

def _self_test() -> None:
    ok = "def add(a, b):\n    return a + b"
    ts = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]
    assert run_tests(ok, ts) == (2, 2)
    # one failing test
    assert run_tests(ok, ts + ["assert add(1, 1) == 3"]) == (2, 3)
    # solution that crashes at import time
    assert run_tests("raise RuntimeError('boom')", ts) == (0, 2)
    # syntax error
    assert run_tests("def add(a b): pass", ts) == (0, 2)
    # empty code / no tests
    assert run_tests("", ts) == (0, 2)
    assert run_tests(ok, []) == (0, 0)
    # infinite loop -> killed by rlimit/timeout, graded 0
    assert run_tests("while True:\n    pass",
                     ["assert True"], timeout=4.0, cpu_s=1) == (0, 1)
    # memory hog -> RLIMIT_AS, graded 0
    assert run_tests("x = 'a' * (1 << 33)", ["assert True"],
                     mem_mb=128) == (0, 1)
    # namespace copy is SHALLOW by design: rebinding in a test doesn't leak,
    # mutation of module-level objects does (dataset tests are independent
    # asserts on pure functions — this is the cheap, sufficient contract)
    code = "acc = []\ndef push(v):\n    acc.append(v)\n    return len(acc)"
    assert run_tests(code, ["assert push(1) == 1", "assert push(9) == 2",
                            "push = None", "assert push(0) == 3"]) == (4, 4)
    assert pass_frac(ok, ts) == 1.0
    print("exec_sandbox self-test: OK (pass/fail, crash, loop, memory, empty)")


if __name__ == "__main__":
    _self_test()
