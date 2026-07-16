"""Math school — synthetic curriculum conversations (phase 2, GRPO groundwork).

Design (user, 2026-07-16, memory dsv6-grpo-m2-integre): a human-style school —
graded difficulty stages, each with verifiable tests whose grade is the future
GRPO reward. ONE generator, THREE uses: SFT traces (canonical derivations are
known, so supervision is mechanical), GRPO environment (the graders below are
the reward_fn), ablation probe (grade with live vs ablated bank).

Five conversation kinds:
  * drill      — bare calculation on the stage's bricks (a+b=c, tables), Q/A
                 turns. Pure weight-based skill: the minority contrast that
                 calibrates lambda_write (MAI trick 1 needs both regimes).
  * derivation — "solve step by step, ONE operation per turn". The running
                 value must traverse turns, so with enough interleaving the
                 partials leave the context and only the bank can carry them
                 (working memory, not an archive). Grading rule validated by
                 the user: chained-invariant check, restating a line pays
                 nothing (each step must consume the NEXT term from the
                 CURRENT running value).
  * equation   — the user's original idea (2026-07-16): solve for x, one
                 transformation per turn. Grading = EQUIVALENCE invariant:
                 substitute the known solution into each turn's equation and
                 check both sides, plus a strict progress criterion (fewer
                 operators each turn — restating the same equation pays
                 nothing).
  * bindings   — working memory in its purest form: `Let a = 47.` turns, then
                 filler drills push the bindings out of context, then queries
                 over the bound names. Occasional REBINDING (`Now a = 12.`)
                 exercises supersession (recall selects, no erase primitive).
  * lesson     — closed-book exam: an INVENTED operation (a?b = p*a+q*b+r,
                 randomized per conversation, cannot be in the weights) is
                 defined + exemplified, filler drills push the lesson out of
                 context, then exam questions use it. Sometimes a SECOND op is
                 defined IN TERMS OF the first (chained lesson): the exam then
                 requires composing two in-life acquisitions. The grade rewards
                 the note-taking, never the retention itself (task-grounded).

Difficulty stages: digits grow in octaves (1,2,4,8 — uint32 is the CEILING,
not the starting distribution), then term count, then multiplication. All
values are kept >= 0 and < 2**32 by construction. Stage advancement
(pass-rate window, MAI trick 4) belongs to the GRPO orchestrator, not here:
the stream just samples stages by weight, so the controller only has to
reweight. Not covered yet (candidates for later marches): division
(quotient/remainder format), word problems, intra-conversation interleaving
(the EnvMixer already interleaves at the episode level).

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

# digits in octaves, then term count, then multiplication
STAGES = [
    dict(digits=1, terms=2, ops="+"),
    dict(digits=1, terms=2, ops="+-"),
    dict(digits=2, terms=2, ops="+-"),
    dict(digits=2, terms=3, ops="+-"),
    dict(digits=2, terms=2, ops="*"),      # tables, extended
    dict(digits=2, terms=3, ops="+-*"),
    dict(digits=4, terms=3, ops="+-"),
    dict(digits=4, terms=4, ops="+-*"),
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
EQN_INSTR = [
    "Solve for x, one transformation per turn:",
    "Find x. Show one step per turn:",
]
CONT_INSTR = ["Continue.", "Next step.", "Go on."]
LESSON_TMPL = ('Lesson: a new operation "{sym}" is defined as:\n'
               "a {sym} b = {formula}\n"
               "Examples:\n{examples}")
LESSON2_TMPL = ('Second lesson: another operation "{sym2}" is defined using '
                '"{sym}":\na {sym2} b = {formula}\n'
                "Examples:\n{examples}")
EXAM_INSTR = [
    "Using the operation {sym} from the lesson: {x} {sym} {y} =",
    "Apply the {sym} operation defined earlier: {x} {sym} {y} =",
]
BIND_TMPL = ["Let {n} = {v}.", "Define {n} = {v}."]
REBIND_TMPL = ["Now {n} = {v}.", "Update: {n} = {v}."]
BIND_QUERY = ["What is {q}?", "Compute {q}."]
OP_SYMBOLS = ["⊕", "⊖", "⊗", "◇", "∘", "##", "%%", "<>"]
VAR_NAMES = list("abcdmnpq")               # 'x' reserved for equations


# ── expression evaluator (equation grader backbone) ──────────────────────────
_TOK_RE = re.compile(r"\d+|[x()+\-*]")


def eval_expr(s: str, x: int):
    """Evaluate a +/-/* expression with parentheses and variable x.
    Returns None on any syntax error. Grammar:
      additive := term (('+'|'-') term)* ; term := factor ('*' factor)* ;
      factor   := number | 'x' | '(' additive ')'
    """
    toks = _TOK_RE.findall(s)
    if "".join(toks) != re.sub(r"\s+", "", s):
        return None                        # stray characters
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        t = peek()
        pos[0] += 1
        return t

    def factor():
        t = take()
        if t == "(":
            v = additive()
            if v is None or take() != ")":
                return None
            return v
        if t == "x":
            return x
        if t is not None and t.isdigit():
            return int(t)
        return None

    def term():
        v = factor()
        while v is not None and peek() == "*":
            take()
            f = factor()
            v = None if f is None else v * f
        return v

    def additive():
        v = term()
        while v is not None and peek() in ("+", "-"):
            o = take()
            t = term()
            if t is None:
                return None
            v = v + t if o == "+" else v - t
        return v

    v = additive()
    return v if pos[0] == len(toks) else None


def _n_ops(s: str) -> int:
    return sum(1 for t in _TOK_RE.findall(s) if t in "+-*")


# ── graders (the future GRPO reward_fn; tested here against the generator) ───
_STEP_RE = re.compile(r"(\d+)\s*([+*-])\s*(\d+)\s*=\s*(\d+)")
_INT_RE = re.compile(r"-?\d+")


def _apply(a: int, op: str, b: int) -> int:
    return a + b if op == "+" else a - b if op == "-" else a * b


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
    v = _apply(a, o, b)
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


def grade_equation(turns: list[str], sol: int, n_steps: int) -> float:
    """Equivalence invariant (user rule 1b, 2026-07-16): each turn must be an
    equation `E = R` that (a) still holds under the KNOWN solution
    (eval(E, sol) == R — numeric equivalence check) and (b) strictly
    PROGRESSES (fewer operators than the previous turn — restating the same
    line pays nothing). Ends on the atomic `x = sol`; final answer counts as
    one extra test, mirroring grade_derivation."""
    k, prev_ops = 0, None
    solved = False
    for turn in turns[:n_steps]:
        line = turn.strip().splitlines()[0] if turn.strip() else ""
        if line.count("=") != 1:
            break
        lhs, rhs = (p.strip() for p in line.split("="))
        m = _INT_RE.fullmatch(rhs)
        if m is None or eval_expr(lhs, sol) != int(rhs):
            break
        n = _n_ops(lhs)
        if prev_ops is not None and n >= prev_ops:
            break                          # no progress => no pay
        prev_ops, k = n, k + 1
        if lhs == "x" and int(rhs) == sol:
            solved = True
            break
    final_ok = 0.0
    if solved and k == n_steps:
        m = _INT_RE.findall(turns[n_steps - 1])
        final_ok = float(bool(m) and int(m[-1]) == sol)
    return (k + final_ok) / (n_steps + 1)


def grade_exam(answers: list[str], truths: list[int]) -> float:
    """Fraction of exam turns whose LAST integer equals the truth."""
    ok = 0
    for txt, want in zip(answers, truths):
        m = _INT_RE.findall(txt)
        ok += bool(m) and int(m[-1]) == want
    return ok / max(1, len(truths))


def grade_conv(conv: dict, texts: list[str]) -> float:
    """Grade decoded assistant texts (one per assistant seg, in order) against
    conv['info']. This is the eval/GRPO entry point: the trainer decodes, the
    generator grades."""
    info, kind = conv["info"], conv["kind"]
    if kind == "derivation":
        return grade_derivation(texts, info["terms"], info["ops"])
    if kind == "equation":
        return grade_equation(texts, info["sol"], info["n_steps"])
    if kind == "drill":
        return grade_exam(texts, info["answers"])
    # bindings / lesson: graded turns are the LAST len(truths) assistant turns
    return grade_exam(texts[-len(info["truths"]):], info["truths"])


def _fold(terms: list[int], ops: list[str]) -> int:
    v = terms[0]
    for t, o in zip(terms[1:], ops):
        v = _apply(v, o, t)
    return v


def _render(terms: list[int], ops: list[str]) -> str:
    out = [str(terms[0])]
    for t, o in zip(terms[1:], ops):
        out.append(f"{o} {t}")
    return " ".join(out)


# ── the stream ────────────────────────────────────────────────────────────────

class MathSchoolStream:
    def __init__(self, tok, *, p_drill: float = 0.20, p_derivation: float = 0.30,
                 p_equation: float = 0.20, p_bindings: float = 0.10,
                 stages: list[dict] = STAGES, stage_weights: list[float] = None,
                 drill_qs: tuple = (1, 4), lesson_examples: tuple = (2, 4),
                 filler_turns: tuple = (2, 6), exam_qs: tuple = (2, 4),
                 p_chain: float = 0.35, p_rebind: float = 0.30,
                 seed: int = 0) -> None:
        self.tok = tok
        self.rng = random.Random(seed)
        self.p_drill = float(p_drill)
        self.p_derivation = float(p_derivation)
        self.p_equation = float(p_equation)
        self.p_bindings = float(p_bindings)
        self.stages = list(stages)
        self.stage_weights = (list(stage_weights) if stage_weights
                              else [1.0] * len(self.stages))
        assert len(self.stage_weights) == len(self.stages)
        self.drill_qs = tuple(int(v) for v in drill_qs)
        self.lesson_examples = tuple(int(v) for v in lesson_examples)
        self.filler_turns = tuple(int(v) for v in filler_turns)
        self.exam_qs = tuple(int(v) for v in exam_qs)
        self.p_chain = float(p_chain)
        self.p_rebind = float(p_rebind)
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
            if op == "*":
                t = self.rng.randint(2, 9 if d == 1 else min(99, 10 ** d - 1))
                if running * t < UINT32:
                    return op, t
                continue
            t = self._num(d)
            if op == "-" and t > running:
                op = "+"
            if op == "+" and running + t >= UINT32:
                continue
            return op, t
        return "-", 0

    def _pick_stage(self, need=None) -> int:
        """Stage by controller weight; `need` filters eligible stages (uniform
        fallback when the controller weights exclude all of them)."""
        cand = [i for i, s in enumerate(self.stages) if need is None or need(s)]
        if not cand:
            cand = list(range(len(self.stages)))
        w = [self.stage_weights[i] for i in cand]
        if sum(w) <= 0:
            w = [1.0] * len(cand)
        return self.rng.choices(cand, weights=w, k=1)[0]

    def _expr(self, st: dict):
        """terms/ops for one exercise of stage st."""
        terms = [self._num(st["digits"])]
        ops = []
        for _ in range(st["terms"] - 1):
            op, t = self._next_term(_fold(terms, ops), st["digits"], st["ops"])
            ops.append(op)
            terms.append(t)
        return terms, ops

    # ── drill ─────────────────────────────────────────────────────────────────
    def _drill_pair(self, st: dict):
        terms, ops = self._expr(st)
        ans = _fold(terms, ops)
        u = self._user(f"{self.rng.choice(DRILL_INSTR)}\n{_render(terms, ops)} =")
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

    # ── derivation ────────────────────────────────────────────────────────────
    def _conv_derivation(self) -> dict:
        # derivations need >= 2 steps to exercise the running value
        si = self._pick_stage(need=lambda s: s["terms"] >= 3)
        st = self.stages[si]
        terms, ops = self._expr(st)
        segs = [self._user(f"{self.rng.choice(DERIV_INSTR)}\n{_render(terms, ops)}")]
        running, canon = terms[0], []
        for k, (t, op) in enumerate(zip(terms[1:], ops)):
            v = _apply(running, op, t)
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

    # ── equation (solve for x, one transformation per turn) ──────────────────
    def _build_equation(self, st: dict):
        """Wrap x in 1..3 invertible layers with integer-safe values.
        Returns (expr strings innermost->outermost, wraps, solution, rhs)."""
        d = min(st["digits"], 4)
        x = self._num(d)
        n_wraps = max(1, min(3, st["terms"] - 1))
        exprs, wraps = ["x"], []
        v = x
        for _ in range(n_wraps):
            for _ in range(8):             # rejection-sample a safe wrap
                kind = self.rng.choice("+-*")
                if kind == "*":
                    m = self.rng.randint(2, 9)
                    if v * m >= UINT32:
                        continue
                    inner = exprs[-1]
                    s = f"{m}*{inner}" if inner == "x" else f"{m}*({inner})"
                    exprs.append(s)
                    wraps.append(("*", m))
                    v *= m
                    break
                c = self._num(self.rng.randint(1, d))
                if kind == "-" and c > v:
                    kind = "+"
                if kind == "+" and v + c >= UINT32:
                    continue
                exprs.append(f"{exprs[-1]} {kind} {c}")
                wraps.append((kind, c))
                v = v + c if kind == "+" else v - c
                break
        return exprs, wraps, x, v

    def _conv_equation(self) -> dict:
        si = self._pick_stage(need=lambda s: s["terms"] >= 3)
        st = self.stages[si]
        exprs, wraps, x, rhs = self._build_equation(st)
        segs = [self._user(f"{self.rng.choice(EQN_INSTR)}\n{exprs[-1]} = {rhs}")]
        canon, v = [], rhs
        for k in range(len(wraps) - 1, -1, -1):   # unwrap outermost first
            op, c = wraps[k]
            v = v - c if op == "+" else v + c if op == "-" else v // c
            line = f"{exprs[k]} = {v}"
            if k == 0:
                line += f"\nFinal answer: {v}"
            canon.append(line)
            segs.append(self._assistant(line))
            if k > 0:
                segs.append(self._user(self.rng.choice(CONT_INSTR)))
        return {"kind": "equation", "segs": segs, "stage": si,
                "info": {"equation": f"{exprs[-1]} = {rhs}", "sol": x,
                         "n_steps": len(wraps), "steps": canon}}

    # ── bindings (Let a = 47. … What is a + b?) ───────────────────────────────
    def _conv_bindings(self) -> dict:
        si = self._pick_stage()
        st = self.stages[si]
        d = min(st["digits"], 4)
        names = self.rng.sample(VAR_NAMES, self.rng.randint(2, 4))
        env = {}
        segs = []
        for n in names:
            env[n] = self._num(d)
            segs.append(self._user(self.rng.choice(BIND_TMPL)
                                   .format(n=n, v=env[n])))
        rebound = None
        if self.rng.random() < self.p_rebind:
            rebound = self.rng.choice(names)
            env[rebound] = self._num(d)
            segs.append(self._user(self.rng.choice(REBIND_TMPL)
                                   .format(n=rebound, v=env[rebound])))
        for _ in range(self.rng.randint(*self.filler_turns)):
            pair, _ = self._drill_pair(st)
            segs += pair
        queries, truths = [], []
        for _ in range(self.rng.randint(1, 2)):
            a, b = self.rng.sample(names, 2)
            op = self.rng.choice("+-")
            if op == "-" and env[a] < env[b]:
                a, b = b, a
            q = f"{a} {op} {b}"
            queries.append(q)
            truths.append(_apply(env[a], op, env[b]))
            segs.append(self._user(self.rng.choice(BIND_QUERY).format(q=q)))
            segs.append(self._assistant(str(truths[-1])))
        return {"kind": "bindings", "segs": segs, "stage": si,
                "info": {"env": env, "rebound": rebound,
                         "queries": queries, "truths": truths}}

    # ── lesson (invented op, optional chained second op, closed-book exam) ───
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
        f1 = lambda a, b: p * a + q * b + r
        d = min(st["digits"], 4)              # exam operands stay readable
        ex = []
        for _ in range(self.rng.randint(*self.lesson_examples)):
            a, b = self._num(d), self._num(d)
            ex.append(f"{a} {sym} {b} = {f1(a, b)}")
        segs = [self._user(LESSON_TMPL.format(sym=sym, formula=formula,
                                              examples="\n".join(ex)))]

        # chained second lesson: an op defined IN TERMS OF the first — the
        # exam then requires composing two in-life acquisitions
        sym2 = f2 = None
        chained = self.rng.random() < self.p_chain
        if chained:
            sym2 = self.rng.choice([s for s in OP_SYMBOLS if s != sym])
            if self.rng.random() < 0.5:
                k = self.rng.randint(1, 9)
                f2 = lambda a, b: f1(a, b) + k
                formula2 = f"(a {sym} b) + {k}"
            else:
                f2 = lambda a, b: f1(b, a)
                formula2 = f"b {sym} a"
            ex2 = []
            for _ in range(self.rng.randint(*self.lesson_examples)):
                a, b = self._num(d), self._num(d)
                ex2.append(f"{a} {sym2} {b} = {f2(a, b)}")
            segs.append(self._user(LESSON2_TMPL.format(
                sym=sym, sym2=sym2, formula=formula2,
                examples="\n".join(ex2))))

        for _ in range(self.rng.randint(*self.filler_turns)):   # push lesson out
            pair, _ = self._drill_pair(st)
            segs += pair
        truths, qs = [], []
        for _ in range(self.rng.randint(*self.exam_qs)):
            a, b = self._num(d), self._num(d)
            use2 = chained and self.rng.random() < 0.6
            s, fn = (sym2, f2) if use2 else (sym, f1)
            qs.append((a, s, b))
            truths.append(fn(a, b))
            segs.append(self._user(self.rng.choice(EXAM_INSTR)
                                   .format(sym=s, x=a, y=b)))
            segs.append(self._assistant(str(fn(a, b))))
        return {"kind": "lesson", "segs": segs, "stage": si,
                "info": {"sym": sym, "pqr": (p, q, r), "sym2": sym2,
                         "chained": chained, "exam": qs, "truths": truths}}

    def next_conv(self) -> dict:
        r = self.rng.random()
        if r < self.p_drill:
            return self._conv_drill()
        r -= self.p_drill
        if r < self.p_derivation:
            return self._conv_derivation()
        r -= self.p_derivation
        if r < self.p_equation:
            return self._conv_equation()
        r -= self.p_equation
        if r < self.p_bindings:
            return self._conv_bindings()
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
    # evaluator sanity first (the equation grader stands on it)
    assert eval_expr("2*(3*x + 4) - 5", 7) == 2 * (3 * 7 + 4) - 5
    assert eval_expr("x", 3) == 3 and eval_expr("41", 0) == 41
    assert eval_expr("2*(3*x", 1) is None and eval_expr("x ? 3", 1) is None

    tok = _StubTok()
    ms = MathSchoolStream(tok, seed=0)
    from collections import Counter
    kinds, stage_hist = Counter(), Counter()

    for _ in range(600):
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

        # grade_conv routes canonical texts to a perfect grade for EVERY kind
        assert grade_conv(c, _assistant_texts(tok, c)) == 1.0

        info = c["info"]
        if c["kind"] == "drill":
            for v in info["answers"]:
                assert 0 <= v < UINT32
        elif c["kind"] == "derivation":
            assert all(0 <= t < UINT32 for t in info["terms"])
            assert 0 <= info["final"] < UINT32
            turns = _assistant_texts(tok, c)
            g = grade_derivation(turns, info["terms"], info["ops"])
            assert g == 1.0, f"canonical derivation graded {g}"
            dup = [turns[0]] + turns[:-1]      # duplication must NOT pay
            assert grade_derivation(dup, info["terms"], info["ops"]) < 1.0
            bad = turns[:-1] + [turns[-1].replace(
                f"Final answer: {info['final']}", "Final answer: -1")]
            gb = grade_derivation(bad, info["terms"], info["ops"])
            assert 0.0 < gb < 1.0
        elif c["kind"] == "equation":
            turns = _assistant_texts(tok, c)
            g = grade_equation(turns, info["sol"], info["n_steps"])
            assert g == 1.0, f"canonical equation graded {g}: {info}"
            # restating the previous equation = no progress = no pay
            dup = [turns[0]] + turns
            assert grade_equation(dup, info["sol"], info["n_steps"]) < 1.0
            # an equation that breaks equivalence under the solution fails
            wrong = [t.replace("=", "= 1 + ", 1) for t in turns]
            assert grade_equation(wrong, info["sol"], info["n_steps"]) < 1.0
        elif c["kind"] == "bindings":
            env = info["env"]
            answers = _assistant_texts(tok, c)[-len(info["truths"]):]
            assert grade_exam(answers, info["truths"]) == 1.0
            for q, t in zip(info["queries"], info["truths"]):
                a, op, b = q.split()
                assert _apply(env[a], op, env[b]) == t and 0 <= t < UINT32
        else:                                  # lesson
            p, q, r = info["pqr"]
            f1 = lambda a, b: p * a + q * b + r
            for (a, s, b), t in zip(info["exam"], info["truths"]):
                if s == info["sym"]:
                    assert t == f1(a, b)
                assert 0 <= t < UINT32
            answers = _assistant_texts(tok, c)[-len(info["truths"]):]
            assert grade_exam(answers, info["truths"]) == 1.0
            assert grade_exam(["-5"] * len(info["truths"]),
                              info["truths"]) == 0.0
            assert c["segs"][0]["role"] == "user"

    for k in ("drill", "derivation", "equation", "bindings", "lesson"):
        assert kinds[k] > 30, f"kind {k} under-sampled: {kinds}"
    assert set(stage_hist) == set(range(len(STAGES)))
    assert any(c == "lesson" for c in kinds)   # chained lessons covered above
    # stage reweighting (the curriculum controller knob) biases sampling;
    # derivation/equation fall back to eligible terms>=3 stages
    ms2 = MathSchoolStream(tok, stage_weights=[1] + [0] * (len(STAGES) - 1),
                           seed=1)
    for _ in range(50):
        c = ms2.next_conv()
        assert c["stage"] == 0 or c["kind"] in ("derivation", "equation")
    print(f"math_school self-test: OK ({dict(kinds)}, "
          f"stages {dict(sorted(stage_hist.items()))})")


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

    for want in ("drill", "derivation", "equation", "bindings", "lesson"):
        for _ in range(400):
            c = ms.next_conv()
            if c["kind"] == want:
                _show(tok, c)
                break

    from collections import Counter
    kinds, toks = Counter(), []
    for _ in range(500):
        c = ms.next_conv()
        kinds[c["kind"]] += 1
        toks.append(sum(s["input_ids"].numel() for s in c["segs"]))
    print(f"\nmix over 500 convs: {dict(kinds)}")
    print(f"tokens/conv: min {min(toks)} med {sorted(toks)[len(toks)//2]} "
          f"max {max(toks)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        _self_test()
