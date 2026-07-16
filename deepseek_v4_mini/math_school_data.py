"""Math school — synthetic curriculum conversations (phase 2, GRPO groundwork).

Design (user, 2026-07-16, memory dsv6-grpo-m2-integre): a human-style school —
graded difficulty stages, each with verifiable tests whose grade is the future
GRPO reward. ONE generator, THREE uses: SFT traces (canonical derivations are
known, so supervision is mechanical), GRPO environment (the graders below are
the reward_fn), ablation probe (grade with live vs ablated bank).

Three conversation kinds:
  * drill      — bare calculation on the stage's bricks (a+b=c), Q/A turns.
                 Pure weight-based skill: the minority contrast that calibrates
                 lambda_write (MAI trick 1 needs both regimes).
  * derivation — "solve step by step, ONE operation per turn". The running
                 value must traverse turns, so with enough interleaving the
                 partials leave the context and only the bank can carry them
                 (working memory, not an archive). Grading rule validated by
                 the user: chained-invariant check, restating a line pays
                 nothing (each step must consume the NEXT term from the
                 CURRENT running value).
  * lesson     — closed-book exam: an INVENTED operation (a?b = p*a+q*b+r,
                 randomized per conversation, cannot be in the weights) is
                 defined + exemplified, filler drills push the lesson out of
                 context, then exam questions use it. The grade rewards the
                 note-taking, never the retention itself (task-grounded).

Difficulty stages: digits grow in octaves (1,2,4,8 — uint32 is the CEILING,
not the starting distribution), then term count. All values are kept >= 0 and
< 2**32 by construction. Stage advancement (pass-rate window, MAI trick 4)
belongs to the GRPO orchestrator, not here: the stream just samples stages by
weight, so the controller only has to reweight.

Segments match chat_defer_data exactly: {"input_ids" [1,T], "loss_mask" [1,T],
"attention_mask", "role", "write"} — loss only on assistant answers + closing
<|im_end|>; template/instruction tokens masked. Interface matches rl_lives
EnvSpec streams (.next_conv() + .rng).

Hermetic self-test (stub tokenizer, no downloads):
  python -m deepseek_v4_mini.math_school_data
Real-tokenizer smoke (decode one conv per kind + stats):
  python -m deepseek_v4_mini.math_school_data deepseek_v4_mini/configs/farm/v3_reach.yaml
"""
from __future__ import annotations

import random
import re
import sys

import torch

# ── ChatML pieces (same as chat_defer_data) ──────────────────────────────────
U_OPEN = "<|im_start|>user\n"
A_OPEN = "<|im_start|>assistant\n"
CLOSE = "<|im_end|>\n"

UINT32 = 2 ** 32

# digits in octaves, term count grows once digits are established
STAGES = [
    dict(digits=1, terms=2, ops="+"),
    dict(digits=1, terms=2, ops="+-"),
    dict(digits=2, terms=2, ops="+-"),
    dict(digits=2, terms=3, ops="+-"),
    dict(digits=4, terms=3, ops="+-"),
    dict(digits=4, terms=5, ops="+-"),
    dict(digits=8, terms=5, ops="+-"),
]

DRILL_INSTR = [
    "Compute:",
    "Calculate:",
    "What is the result?",
]
DERIV_INSTR = [
    "Solve step by step, one operation per turn:",
    "Compute this step by step. Give exactly one operation per turn:",
]
CONT_INSTR = ["Continue.", "Next step.", "Go on."]
LESSON_TMPL = ('Lesson: a new operation "{sym}" is defined as:\n'
               "a {sym} b = {formula}\n"
               "Examples:\n{examples}")
EXAM_INSTR = [
    "Using the operation {sym} from the lesson: {x} {sym} {y} =",
    "Apply the {sym} operation defined earlier: {x} {sym} {y} =",
]
OP_SYMBOLS = ["⊕", "⊖", "⊗", "◇", "∘", "##", "%%", "<>"]


# ── graders (the future GRPO reward_fn; tested here against the generator) ───
_STEP_RE = re.compile(r"(\d+)\s*([+-])\s*(\d+)\s*=\s*(\d+)")
_INT_RE = re.compile(r"-?\d+")


def check_step(line: str, running: int, term: int, op: str):
    """One derivation turn: must be `running op term = v` with v exact.
    Returns the new running value, or None if the step is invalid. Restating
    an old line fails here (its lhs no longer matches the running value), so
    duplication pays nothing — user rule 2026-07-16."""
    m = _STEP_RE.search(line)
    if m is None:
        return None
    a, o, b, c = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
    if a != running or o != op or b != term:
        return None
    v = a + b if o == "+" else a - b
    return v if v == c else None


def grade_derivation(turns: list[str], terms: list[int], ops: list[str]) -> float:
    """Fraction of chained-valid steps; the final answer counts as one extra
    test (progress criterion: a stalled/duplicated turn advances nothing)."""
    running, k = terms[0], 0
    for turn, term, op in zip(turns, terms[1:], ops):
        v = check_step(turn, running, term, op)
        if v is None:
            break
        running, k = v, k + 1
    n_steps = len(terms) - 1
    final_ok = 0.0
    if k == n_steps:
        want = _fold(terms, ops)
        m = _INT_RE.findall(turns[n_steps - 1])
        final_ok = float(bool(m) and int(m[-1]) == want)
    return (k + final_ok) / (n_steps + 1)


def grade_exam(answers: list[str], truths: list[int]) -> float:
    """Fraction of exam turns whose LAST integer equals the truth."""
    ok = 0
    for txt, want in zip(answers, truths):
        m = _INT_RE.findall(txt)
        ok += bool(m) and int(m[-1]) == want
    return ok / max(1, len(truths))


def _fold(terms: list[int], ops: list[str]) -> int:
    v = terms[0]
    for t, o in zip(terms[1:], ops):
        v = v + t if o == "+" else v - t
    return v


def _render(terms: list[int], ops: list[str]) -> str:
    out = [str(terms[0])]
    for t, o in zip(terms[1:], ops):
        out.append(f"{o} {t}")
    return " ".join(out)


# ── the stream ────────────────────────────────────────────────────────────────

class MathSchoolStream:
    def __init__(self, tok, *, p_drill: float = 0.30, p_derivation: float = 0.40,
                 stages: list[dict] = STAGES, stage_weights: list[float] = None,
                 drill_qs: tuple = (1, 4), lesson_examples: tuple = (2, 4),
                 filler_turns: tuple = (2, 6), exam_qs: tuple = (2, 4),
                 seed: int = 0) -> None:
        self.tok = tok
        self.rng = random.Random(seed)
        self.p_drill = float(p_drill)
        self.p_derivation = float(p_derivation)
        self.stages = list(stages)
        self.stage_weights = (list(stage_weights) if stage_weights
                              else [1.0] * len(self.stages))
        assert len(self.stage_weights) == len(self.stages)
        self.drill_qs = tuple(int(v) for v in drill_qs)
        self.lesson_examples = tuple(int(v) for v in lesson_examples)
        self.filler_turns = tuple(int(v) for v in filler_turns)
        self.exam_qs = tuple(int(v) for v in exam_qs)
        self._enc = {}

    # ── token plumbing (identical to chat_defer_data) ─────────────────────────
    def _ids(self, s: str) -> torch.Tensor:
        if s not in self._enc:
            self._enc[s] = torch.tensor(
                self.tok(s, add_special_tokens=False)["input_ids"], dtype=torch.long)
        return self._enc[s]

    def _seg(self, pieces: list[tuple[str, bool]], role: str) -> dict:
        ids = torch.cat([self._ids(p) for p, _ in pieces])
        mask = torch.cat([torch.full((self._ids(p).numel(),), float(sup))
                          for p, sup in pieces])
        return {"input_ids": ids.unsqueeze(0), "loss_mask": mask.unsqueeze(0),
                "attention_mask": torch.ones(1, ids.numel(), dtype=torch.long),
                "role": role, "write": True}

    def _user(self, text: str) -> dict:
        return self._seg([(U_OPEN, False), (text + "\n", False), (CLOSE, False)],
                         "user")

    def _assistant(self, text: str) -> dict:
        return self._seg([(A_OPEN, False), (text, True), ("\n", True),
                          (CLOSE, True)], "assistant")

    # ── number sampling (>= 0, < 2**32 by construction) ───────────────────────
    def _num(self, digits: int) -> int:
        if digits <= 1:
            return self.rng.randint(0, 9)
        return self.rng.randint(10 ** (digits - 1), 10 ** digits - 1)

    def _next_term(self, running: int, digits: int, ops: str):
        """Pick (op, term) keeping the running value in [0, 2**32)."""
        for d in range(digits, 0, -1):
            op = self.rng.choice(ops)
            t = self._num(d)
            if op == "-" and t > running:
                op = "+"
            if op == "+" and running + t >= UINT32:
                continue
            return op, t
        return "-", 0

    def _pick_stage(self) -> int:
        return self.rng.choices(range(len(self.stages)),
                                weights=self.stage_weights, k=1)[0]

    def _expr(self, st: dict):
        """terms/ops for one exercise of stage st."""
        terms = [self._num(st["digits"])]
        ops = []
        for _ in range(st["terms"] - 1):
            op, t = self._next_term(_fold(terms, ops), st["digits"], st["ops"])
            ops.append(op)
            terms.append(t)
        return terms, ops

    # ── conversations ─────────────────────────────────────────────────────────
    def _drill_pair(self, st: dict):
        terms, ops = self._expr(st)
        expr = _render(terms, ops)
        ans = _fold(terms, ops)
        u = self._user(f"{self.rng.choice(DRILL_INSTR)}\n{expr} =")
        a = self._assistant(str(ans))
        return [u, a], ans

    def _conv_drill(self) -> dict:
        si = self._pick_stage()
        st = self.stages[si]
        segs, answers = [], []
        for _ in range(self.rng.randint(*self.drill_qs)):
            pair, ans = self._drill_pair(st)
            segs += pair
            answers.append(ans)
        return {"kind": "drill", "segs": segs, "stage": si,
                "info": {"answers": answers}}

    def _conv_derivation(self) -> dict:
        # derivations need >= 2 steps to exercise the running value
        cand = [i for i, s in enumerate(self.stages) if s["terms"] >= 3]
        if not cand:
            si = self._pick_stage()
        else:
            w = [self.stage_weights[i] for i in cand]
            if sum(w) <= 0:                   # controller weights exclude all
                w = [1.0] * len(cand)         # eligible stages: uniform fallback
            si = self.rng.choices(cand, weights=w, k=1)[0]
        st = self.stages[si]
        terms, ops = self._expr(st)
        segs = [self._user(f"{self.rng.choice(DERIV_INSTR)}\n{_render(terms, ops)}")]
        running, canon = terms[0], []
        for k, (t, op) in enumerate(zip(terms[1:], ops)):
            v = running + t if op == "+" else running - t
            line = f"{running} {op} {t} = {v}"
            if k == len(ops) - 1:
                line += f"\nFinal answer: {v}"
            canon.append(line)
            segs.append(self._assistant(line))
            if k < len(ops) - 1:
                segs.append(self._user(self.rng.choice(CONT_INSTR)))
            running = v
        return {"kind": "derivation", "segs": segs, "stage": si,
                "info": {"terms": terms, "ops": ops, "steps": canon,
                         "final": running}}

    def _invented_op(self):
        """a?b = p*a + q*b + r, never plain addition (p=q=1, r=0 excluded)."""
        while True:
            p, q, r = self.rng.randint(1, 3), self.rng.randint(1, 3), \
                self.rng.randint(0, 9)
            if (p, q, r) != (1, 1, 0):
                break
        parts = [("a" if p == 1 else f"{p}*a"), ("b" if q == 1 else f"{q}*b")]
        if r:
            parts.append(str(r))
        return p, q, r, " + ".join(parts)

    def _conv_lesson(self) -> dict:
        si = self._pick_stage()
        st = self.stages[si]
        sym = self.rng.choice(OP_SYMBOLS)
        p, q, r, formula = self._invented_op()
        f = lambda a, b: p * a + q * b + r
        d = min(st["digits"], 4)              # exam operands stay readable
        ex = []
        for _ in range(self.rng.randint(*self.lesson_examples)):
            a, b = self._num(d), self._num(d)
            ex.append(f"{a} {sym} {b} = {f(a, b)}")
        segs = [self._user(LESSON_TMPL.format(sym=sym, formula=formula,
                                              examples="\n".join(ex)))]
        for _ in range(self.rng.randint(*self.filler_turns)):   # push lesson out
            pair, _ = self._drill_pair(st)
            segs += pair
        truths, qs = [], []
        for _ in range(self.rng.randint(*self.exam_qs)):
            a, b = self._num(d), self._num(d)
            qs.append((a, b))
            truths.append(f(a, b))
            segs.append(self._user(self.rng.choice(EXAM_INSTR)
                                   .format(sym=sym, x=a, y=b)))
            segs.append(self._assistant(str(f(a, b))))
        return {"kind": "lesson", "segs": segs, "stage": si,
                "info": {"sym": sym, "pqr": (p, q, r), "exam": qs,
                         "truths": truths}}

    def next_conv(self) -> dict:
        r = self.rng.random()
        if r < self.p_drill:
            return self._conv_drill()
        if r < self.p_drill + self.p_derivation:
            return self._conv_derivation()
        return self._conv_lesson()


# ── hermetic self-test (stub tokenizer: 1 char = 1 token) ────────────────────

class _StubTok:
    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in s]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _assistant_texts(tok, conv):
    out = []
    for s in conv["segs"]:
        if s["role"] != "assistant":
            continue
        ids = s["input_ids"][0]
        keep = s["loss_mask"][0] > 0
        out.append(tok.decode(ids[keep].tolist()).replace(CLOSE, ""))
    return out


def _self_test() -> None:
    tok = _StubTok()
    ms = MathSchoolStream(tok, seed=0)
    from collections import Counter
    kinds, stage_hist = Counter(), Counter()
    n_deriv = n_lesson = 0

    for _ in range(400):
        c = ms.next_conv()
        kinds[c["kind"]] += 1
        stage_hist[c["stage"]] += 1

        # masks: user segs fully masked; assistant loss starts after A_OPEN
        for s in c["segs"]:
            if s["role"] == "user":
                assert s["loss_mask"].sum() == 0
            else:
                n_open = len(A_OPEN)
                assert s["loss_mask"][0, :n_open].sum() == 0
                assert s["loss_mask"][0, n_open:].min() == 1

        if c["kind"] == "drill":
            for v in c["info"]["answers"]:
                assert 0 <= v < UINT32
        elif c["kind"] == "derivation":
            n_deriv += 1
            info = c["info"]
            assert all(0 <= t < UINT32 for t in info["terms"])
            assert 0 <= info["final"] < UINT32
            # canonical trace must grade 1.0 …
            turns = _assistant_texts(tok, c)
            g = grade_derivation(turns, info["terms"], info["ops"])
            assert g == 1.0, f"canonical derivation graded {g}"
            # … duplicating a line must NOT pay (user rule)
            dup = [turns[0]] + turns[:-1]
            assert grade_derivation(dup, info["terms"], info["ops"]) < 1.0
            # … and a wrong final answer loses exactly the final test
            bad = turns[:-1] + [turns[-1].replace(
                f"Final answer: {info['final']}", "Final answer: -1")]
            gb = grade_derivation(bad, info["terms"], info["ops"])
            assert 0.0 < gb < 1.0
        else:
            n_lesson += 1
            info = c["info"]
            p, q, r = info["pqr"]
            for (a, b), t in zip(info["exam"], info["truths"]):
                assert t == p * a + q * b + r and 0 <= t < UINT32
            answers = _assistant_texts(tok, c)[-len(info["truths"]):]
            assert grade_exam(answers, info["truths"]) == 1.0
            assert grade_exam(["-5"] * len(info["truths"]),
                              info["truths"]) == 0.0
            # the lesson turn must precede filler drills and the exam
            assert c["segs"][0]["role"] == "user"
            assert len(c["segs"]) >= 1 + 2 * 2 + 2  # lesson + >=2 fillers + >=1 exam

    assert n_deriv > 50 and n_lesson > 50
    assert set(stage_hist) == set(range(len(STAGES)))
    # stage reweighting (the curriculum controller knob) actually biases sampling
    # (derivations fall back to eligible terms>=3 stages when weights exclude all)
    ms2 = MathSchoolStream(tok, stage_weights=[1, 0, 0, 0, 0, 0, 0], seed=1)
    for _ in range(50):
        c = ms2.next_conv()
        assert c["stage"] == 0 or c["kind"] == "derivation"
    print(f"math_school self-test: OK ({dict(kinds)}, stages {dict(sorted(stage_hist.items()))})")


# ── real-tokenizer smoke ──────────────────────────────────────────────────────

def _show(tok, conv, max_seg_chars=160):
    print(f"\n===== kind={conv['kind']} stage={conv['stage']} "
          f"segs={len(conv['segs'])} =====")
    for s in conv["segs"]:
        txt = tok.decode(s["input_ids"][0].tolist())
        sup = int(s["loss_mask"].sum().item())
        head = txt[:max_seg_chars].replace("\n", "\\n")
        print(f"  [{s['role']:9s} T={s['input_ids'].numel():4d} loss_on={sup:3d}] "
              f"{head}{'…' if len(txt) > max_seg_chars else ''}")


def main(cfg_path: str) -> None:
    import yaml
    from transformers import AutoTokenizer
    name = (yaml.safe_load(open(cfg_path))["tokenizer"]
            if cfg_path.endswith((".yaml", ".yml")) else cfg_path)
    tok = AutoTokenizer.from_pretrained(name)
    ms = MathSchoolStream(tok, seed=0)

    for want in ("drill", "derivation", "lesson"):
        for _ in range(200):
            c = ms.next_conv()
            if c["kind"] == want:
                _show(tok, c)
                break

    from collections import Counter
    kinds, toks = Counter(), []
    for _ in range(300):
        c = ms.next_conv()
        kinds[c["kind"]] += 1
        toks.append(sum(s["input_ids"].numel() for s in c["segs"]))
    print(f"\nmix over 300 convs: {dict(kinds)}")
    print(f"tokens/conv: min {min(toks)} med {sorted(toks)[len(toks)//2]} "
          f"max {max(toks)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        _self_test()
