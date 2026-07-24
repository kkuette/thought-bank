"""Tool-use sessions — xlam/glaive reassembled so the SCHEMA is out of reach
(phase 2, SFT/RL ratchet on tool use — memory dsv6-grpo-m2-integre 07-23).

Why reassemble at all: served as-is, xlam/glaive put the tool schema in the
system turn right above the call — a context task, zero memory pressure. Here
one session packs 2..5 tool episodes and declares ALL their schemas in a
single opening turn; the queries then arrive in a DIFFERENT order, whole
episodes apart. By the time a query lands, its schema is out of the local
window (real turns in between): producing the right call = reading the schema
back from the bank. Same lengthen-and-distance principle as sota_session
(pivot 2026-07-24), with a VERIFIABLE grader on top (rl_rewards.grade_calls
— name gate x args F1, no judge).

Episode pools:
  * xlam  (Salesforce/xlam-function-calling-60k): columns query / tools /
          answers, all JSON strings. Direct.
  * glaive (glaiveai/glaive-function-calling-v2): schemas in `system`, one
          "USER: ... ASSISTANT: <functioncall> {...}" pair mined from `chat`.

Interface = the chat-stream contract of code_defer_native (.next_conv() +
.rng + .grade_conv), segs carry role/loss_mask/surp_w like sota_session; ALL
assistant turns are graded calls (truths = their canonical JSON), ages =
writes from the schema turn to each call. info.gold_calls (parsed dicts, per
graded turn) is the RL bridge: rl_disagg decodes the call turn and rewards it
with rl_rewards.make_tool_reward — same episodes for SFT and for GRPO, which
is what makes the ratchet's distill step a filter, not a new pipeline.

Hermetic self-test (stub tokenizer, no downloads):
  python -m deepseek_v4_mini.tool_env_data
Real smoke (downloads, prints one session):
  python -m deepseek_v4_mini.tool_env_data <yaml with tokenizer: + tools.gen>
"""
from __future__ import annotations

import json
import re
import sys

from .persona_chat_data import PersonaChatStream
from .rl_rewards import _balanced_spans, extract_calls, grade_calls


# ── glaive chat mining ───────────────────────────────────────────────────────

_G_USER = re.compile(r"USER:\s*(.*?)(?=ASSISTANT:|$)", re.S)
_G_CALL = re.compile(r"<functioncall>\s*(.*?)(?:<\|endoftext\|>|FUNCTION RESPONSE|USER:|$)", re.S)
_G_ARGS = re.compile(r"(\"arguments\"\s*:\s*)'(.*)'", re.S)


def parse_glaive_row(system: str, chat: str):
    """(schemas, query, calls) or None. First USER turn + first functioncall;
    rows without a call (pure chat) are skipped — this env is about calls.
    Glaive idiom: arguments nested as a SINGLE-quoted string containing JSON
    ('{"tz": ...}') — rewritten to a proper JSON string before parsing."""
    schemas = [s for s in _balanced_spans(system) if '"name"' in s]
    mu, mc = _G_USER.search(chat), _G_CALL.search(chat)
    if not (schemas and mu and mc):
        return None
    span = _G_ARGS.sub(lambda m: m.group(1) + json.dumps(m.group(2)),
                       mc.group(1))
    calls = extract_calls(span)
    query = mu.group(1).strip()
    if not (calls and query):
        return None
    return schemas, query, calls


# ── stream ───────────────────────────────────────────────────────────────────

class ToolSessionStream(PersonaChatStream):
    """Packs tool episodes into schema-far sessions with a verifiable grader.

    Reuses the persona plumbing (_seg/_user/_assistant: ChatML, loss masks,
    surp_w/SIF) exactly like SotaSessionStream. An episode in the pool is
    {"schemas": [str], "query": str, "calls": [ {name, arguments} ]}."""

    def __init__(self, tok, *,
                 datasets=("Salesforce/xlam-function-calling-60k",),
                 split: str = "train",
                 episode_cap: int = 6000,       # episodes kept per dataset
                 max_turn_tok: int = 192,       # query/call token budget
                 max_schema_tok: int = 256,     # per-episode schema budget
                 eps_per_session: tuple = (2, 5),
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
        self.eps_per_session = tuple(int(v) for v in eps_per_session)
        if _pool is not None:
            self.pool = _pool
        else:
            self.pool = []
            for name in ([datasets] if isinstance(datasets, str) else datasets):
                self.pool += self._load_pool(
                    name, split, int(episode_cap), int(max_turn_tok),
                    int(max_schema_tok), real_cache_dir)
        assert self.pool, "tool episode pool is empty"

    # ── data ─────────────────────────────────────────────────────────────────
    def _n_tok(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def _load_pool(self, name, split, cap, max_tok, max_schema, cache_dir):
        import hashlib, os
        import torch
        key = hashlib.sha256(
            f"{name}|{split}|{cap}|{max_tok}|{max_schema}".encode()) \
            .hexdigest()[:16]
        path = (os.path.join(cache_dir, f"tool_eps_{key}.pt")
                if cache_dir else None)
        if path and os.path.exists(path):
            pool = torch.load(path)
            print(f"tool sessions: cache hit {path} — {len(pool)} episodes")
            return pool
        from datasets import load_dataset
        rows = load_dataset(name, split=split, streaming=True)
        is_glaive = "glaive" in name.lower()
        pool = []
        for row in rows:
            if is_glaive:
                got = parse_glaive_row(row.get("system") or "",
                                       row.get("chat") or "")
                if got is None:
                    continue
                schemas, query, calls = got
            else:                              # xlam layout: JSON-string columns
                try:
                    tools = json.loads(row["tools"])
                    calls = json.loads(row["answers"])
                except (KeyError, json.JSONDecodeError, TypeError):
                    continue
                query = (row.get("query") or "").strip()
                schemas = [json.dumps(t) for t in tools
                           if isinstance(t, dict) and t.get("name")]
                calls = [{"name": c["name"],
                          "arguments": c.get("arguments", {}) or {}}
                         for c in calls
                         if isinstance(c, dict) and c.get("name")]
            if not (schemas and query and calls):
                continue
            gold = json.dumps(calls if len(calls) > 1 else calls[0])
            if (self._n_tok(query) > max_tok or self._n_tok(gold) > max_tok
                    or self._n_tok("\n".join(schemas)) > max_schema):
                continue
            pool.append({"schemas": schemas, "query": query, "calls": calls,
                         "gold": gold})
            if len(pool) >= cap:
                break
        print(f"tool sessions: {name}[{split}] — {len(pool)} episodes "
              f"(schema <= {max_schema} tok, turn <= {max_tok} tok)")
        if path:
            torch.save(pool, path)
        return pool

    # ── grading (verifiable — rl_rewards, no judge) ──────────────────────────
    @staticmethod
    def grade_conv(conv: dict, texts: list[str]) -> float:
        golds = conv["info"]["gold_calls"]
        if not golds:
            return 1.0
        tail = texts[-len(golds):]
        return sum(grade_calls(t, g) for t, g in zip(tail, golds)) / len(golds)

    # ── session assembly ─────────────────────────────────────────────────────
    def next_conv(self) -> dict:
        n_eps = self.rng.randint(*self.eps_per_session)
        eps = [self.pool[i]
               for i in self.rng.sample(range(len(self.pool)), n_eps)]
        # opening turn: every schema of the session, shuffled — the model
        # cannot know which episode a schema belongs to until its query lands
        schemas = [s for e in eps for s in e["schemas"]]
        self.rng.shuffle(schemas)
        segs = [self._user("Tools available this session:\n"
                           + "\n".join(schemas))]
        # queries in an order DIFFERENT from the schema declaration (schema of
        # the last-declared episode may come first): sample a non-identity
        # permutation when possible
        order = list(range(n_eps))
        for _ in range(4):
            self.rng.shuffle(order)
            if n_eps == 1 or order != list(range(n_eps)):
                break
        truths, gold_calls, ages = [], [], []
        for k in order:
            e = eps[k]
            segs.append(self._user(e["query"]))
            ages.append(len(segs))             # writes schema turn -> call
            segs.append(self._assistant(e["gold"]))
            truths.append(e["gold"])
            gold_calls.append(e["calls"])
        return {"kind": "toolcall", "segs": segs,
                "info": {"truths": truths, "gold_calls": gold_calls,
                         "queries": [eps[k]["query"] for k in order],
                         "ages": ages, "n_eps": n_eps}}


# ── smoke ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:                      # real tokenizer + real datasets
        import yaml
        from transformers import AutoTokenizer
        raw = yaml.safe_load(open(sys.argv[1]))
        tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
        gen = ((raw.get("tools", {}) or {}).get("gen", {}) or {})
        st = ToolSessionStream(tok, **gen)
        n_segs, ages = [], []
        for _ in range(50):
            c = st.next_conv()
            n_segs.append(len(c["segs"]))
            ages += c["info"]["ages"]
        print(f"segs/session min {min(n_segs)} med "
              f"{sorted(n_segs)[len(n_segs)//2]} max {max(n_segs)} | "
              f"ages {sorted(ages)[:5]}..{sorted(ages)[-5:]}")
        c = st.next_conv()
        for s in c["segs"][:4] + c["segs"][-2:]:
            print(repr(tok.decode(s["input_ids"][0].tolist())[:140]))
        # grader sanity on gold: decoding the truths must grade 1.0
        pad = [""] * (len(c["segs"]) - len(c["info"]["truths"]))
        print("grade(gold) =", st.grade_conv(c, pad + c["info"]["truths"]))
        sys.exit(0)

    # hermetic: stub tokenizer + synthetic pool + glaive parser unit
    class _Tok:
        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [ord(ch) % 251 for ch in s]}

    g_sys = ('You may use: {"name": "get_time", "parameters": '
             '{"tz": "string"}} if needed.')
    g_chat = ('USER: what time is it in Tokyo? ASSISTANT: <functioncall> '
              '{"name": "get_time", "arguments": \'{"tz": "Asia/Tokyo"}\'} '
              '<|endoftext|> FUNCTION RESPONSE: {"time": "12:00"} '
              'ASSISTANT: It is noon. <|endoftext|>')
    got = parse_glaive_row(g_sys, g_chat)
    assert got is not None
    _, q, calls = got
    assert q.startswith("what time") and calls == [
        {"name": "get_time", "arguments": {"tz": "Asia/Tokyo"}}], (q, calls)
    assert parse_glaive_row(g_sys, "USER: hi ASSISTANT: hello!") is None

    pool = [{"schemas": [json.dumps({"name": f"fn_{i}",
                                     "parameters": {"x": "int"}})],
             "query": f"please run fn_{i} with x={i}",
             "calls": [{"name": f"fn_{i}", "arguments": {"x": i}}],
             "gold": json.dumps({"name": f"fn_{i}", "arguments": {"x": i}})}
            for i in range(30)]
    st = ToolSessionStream(_Tok(), _pool=pool, surprisal_mode="sif", seed=0)
    for _ in range(100):
        c = st.next_conv()
        info = c["info"]
        assert len(info["truths"]) == len(info["gold_calls"]) == info["n_eps"]
        assert all(a > 0 for a in info["ages"])
        for s in c["segs"]:
            assert "role" in s and s["surp_w"].shape == s["input_ids"].shape
        # schema turn is never supervised; call turns are
        assert float(c["segs"][0]["loss_mask"].sum()) == 0.0
        assert float(c["segs"][-1]["loss_mask"].sum()) > 0.0
        # grader: gold answers = 1.0, garbage = 0.0, one-of-two = 0.5
        pad = [""] * (len(c["segs"]) - len(info["truths"]))
        assert st.grade_conv(c, pad + info["truths"]) == 1.0
        assert st.grade_conv(c, pad + ["nope"] * len(info["truths"])) == 0.0
        if info["n_eps"] >= 2:
            half = info["truths"][:-1] + ["nope"]
            g = st.grade_conv(c, pad + half)
            assert abs(g - (info["n_eps"] - 1) / info["n_eps"]) < 1e-9
    print("tool_env_data self-test: OK (glaive parse, assembly, ages, grader)")
