"""SOTA sessions — real instruction data reassembled into long conversations
(phase 2 SFT, data-SOTA family — pivot user 2026-07-24, persona arc closed).

Replaces the persona toy with REAL conversations (smol-smoltalk by default):
the closed value/form repertoire of persona let the model pay the grade with a
pool prior (lookup 6 forms x 8-15 values, insensitive to the session). Here
both repertoires are open by construction.

Two kinds:
  * session — 2..6 real convs packed back-to-back into ONE bank life. The
              bank carries across topic changes: eviction + cross-topic
              distractor pressure on real text, register learned on real
              answers. Plain SFT (CE on assistant turns).
  * requote — a packed session whose FINAL user turn re-asks, verbatim, a
              question from an EARLY conv; the supervised target is the
              original assistant answer. The source exchange is out of the
              context window by construction (whole convs in between): the
              only bridge is the bank. Open-repertoire twin of persona
              recall; age (writes between source answer and decode) is
              measured in the same units as persona.

Interface matches the `chat:` block of code_defer_native (.next_conv() +
.rng + .grade_conv), segs carry surp_w for the SIF teacher when enabled.
Grading: token-F1 of the decoded reply vs the original answer (report-only
figure — Δnll live/ablated stays the sensitive probe).

Hermetic self-test (stub tokenizer, no downloads):
  python -m deepseek_v4_mini.sota_session_data
Real smoke (downloads/caches the dataset, decodes one conv per kind):
  python -m deepseek_v4_mini.sota_session_data <any yaml with tokenizer:>
"""
from __future__ import annotations

import re
import sys

import torch

from .math_school_data import U_OPEN, A_OPEN, CLOSE
from .persona_chat_data import PersonaChatStream

_WORD = re.compile(r"[a-z0-9']+")


def _tok_f1(hyp: str, ref: str) -> float:
    """Bag-of-words F1 — order-free overlap, robust to paraphrase padding."""
    h, r = _WORD.findall(hyp.lower()), _WORD.findall(ref.lower())
    if not h or not r:
        return 0.0
    from collections import Counter
    inter = sum((Counter(h) & Counter(r)).values())
    if inter == 0:
        return 0.0
    p, rc = inter / len(h), inter / len(r)
    return 2 * p * rc / (p + rc)


def grade_conv(conv: dict, texts: list[str]) -> float:
    """Mean token-F1 of the graded turns (last len(truths)) vs the original
    answers. 1.0 for kinds without truths (session = control)."""
    truths = conv["info"]["truths"]
    if not truths:
        return 1.0
    tail = texts[-len(truths):]
    return sum(_tok_f1(t, g) for t, g in zip(tail, truths)) / len(truths)


class SotaSessionStream(PersonaChatStream):
    """Packs real convs into long bank lives; requote = open-repertoire recall.

    Reuses the persona plumbing wholesale: _seg/_user/_assistant (ChatML +
    loss masks + surp_w), SIF unigram table (built on 300 of OUR sessions via
    the inherited _sif_table -> this next_conv), disk caching pattern."""

    def __init__(self, tok, *,
                 dataset="HuggingFaceTB/smol-smoltalk",  # str OU liste : les
                 # pools se concatènent (conv_cap PAR dataset) — Tulu 3 etc.
                 # partagent le format messages, aucun adaptateur requis
                 split: str = "train",
                 conv_cap: int = 8000,          # convs kept in the pool
                 max_turn_tok: int = 192,        # per-turn token budget
                 max_conv_msgs: int = 10,        # turns per source conv
                 convs_per_session: tuple = (2, 6),
                 p_requote: float = 0.5,
                 n_requote: tuple = (1, 2),
                 requote_weight: float = 1.0,
                 real_cache_dir: str = None,
                 surprisal_ref: str = None, surprisal_device: str = "cpu",
                 surprisal_alpha: float = 2.0, surprisal_mode: str = "nll",
                 sif_a: float = 1e-4, seed: int = 0, _pool=None) -> None:
        # parent __init__ builds rng/tokenizer/SIF state; no persona filler
        super().__init__(tok, real_filler=None,
                         surprisal_ref=surprisal_ref,
                         surprisal_device=surprisal_device,
                         surprisal_alpha=surprisal_alpha,
                         surprisal_mode=surprisal_mode,
                         sif_a=sif_a, seed=seed)
        self.convs_per_session = tuple(int(v) for v in convs_per_session)
        self.p_requote = float(p_requote)
        self.n_requote = tuple(int(v) for v in n_requote)
        self.requote_weight = float(requote_weight)
        if _pool is not None:
            self.pool = _pool
        else:
            self.pool = []
            for name in ([dataset] if isinstance(dataset, str) else dataset):
                self.pool += self._load_convs(
                    name, split, int(conv_cap), int(max_turn_tok),
                    int(max_conv_msgs), real_cache_dir)
        assert self.pool, "sota session pool is empty"

    # ── data ─────────────────────────────────────────────────────────────────
    def _load_convs(self, name, split, cap, max_tok, max_msgs, cache_dir):
        """Full user/assistant convs, disk-cached as lists of (role, text)."""
        import hashlib, os
        key = hashlib.sha256(
            f"{name}|{split}|{cap}|{max_tok}|{max_msgs}".encode()) \
            .hexdigest()[:16]
        path = (os.path.join(cache_dir, f"sota_convs_{key}.pt")
                if cache_dir else None)
        if path and os.path.exists(path):
            pool = torch.load(path)
            print(f"sota sessions: cache hit {path} — {len(pool)} convs")
            return pool
        from datasets import load_dataset
        rows = load_dataset(name, split=split, streaming=True)
        pool = []
        for row in rows:
            msgs = row.get("messages") or []
            if not (2 <= len(msgs) <= max_msgs):
                continue
            conv, ok = [], True
            for i, m in enumerate(msgs):
                want = "user" if i % 2 == 0 else "assistant"
                text = (m.get("content") or "").strip()
                if m.get("role") != want or not text or \
                        len(self.tok(text, add_special_tokens=False)
                            ["input_ids"]) > max_tok:
                    ok = False
                    break
                conv.append((want, text))
            if ok:
                pool.append(conv)
                if len(pool) >= cap:
                    break
        print(f"sota sessions: {name}[{split}] — {len(pool)} convs "
              f"(<= {max_msgs} msgs, <= {max_tok} tok/turn)")
        if path:
            torch.save(pool, path)
        return pool

    grade_conv = staticmethod(grade_conv)

    def _assistant_requote(self, text: str) -> dict:
        """Réponse requote : loss_mask modulé par les poids SIF — généralise le
        value_weight ×4 persona (pression native run 7, l'ingrédient qui
        survit au retrait du teacher) à une cible SANS span valeur : les
        tokens informatifs (sif ~1 = l'équivalent des valeurs rares persona)
        montent vers w, les templates (sif ~0) restent à 1. mask = 1+(w-1)·sif,
        échafaudage ChatML inchangé (sif nul dessus)."""
        seg = self._assistant(text)
        w = self.requote_weight
        if w == 1.0 or "surp_w" not in seg:
            return seg
        seg["loss_mask"] = seg["loss_mask"] \
            * (1.0 + (w - 1.0) * seg["surp_w"].clamp(0.0, 1.0))
        return seg

    # ── conv assembly ────────────────────────────────────────────────────────
    def _conv_session(self, requote: bool) -> dict:
        n_convs = self.rng.randint(*self.convs_per_session)
        picks = self.rng.sample(range(len(self.pool)), n_convs)
        segs, pair_index = [], []          # (conv_no, q_text, a_text, a_seg)
        for c_no, pi in enumerate(picks):
            for i, (role, text) in enumerate(self.pool[pi]):
                if role == "user":
                    segs.append(self._user(text))
                else:
                    segs.append(self._assistant(text))
                    # the exchange (question just before, this answer) is a
                    # requote candidate when it opens the conv (self-contained)
                    if i == 1:
                        pair_index.append((c_no, self.pool[pi][0][1], text,
                                           len(segs) - 1))
        queries, truths, ages = [], [], []
        if requote and pair_index:
            # re-ask exchanges from EARLY convs only (never the last conv:
            # distance is the point), oldest-first decode order like persona
            early = [p for p in pair_index if p[0] < n_convs - 1]
            take = self.rng.sample(
                early, min(len(early), self.rng.randint(*self.n_requote)))
            for _, q, a, a_seg in sorted(take, key=lambda p: p[3]):
                segs.append(self._user(q))
                queries.append(q)
                truths.append(a)
                ages.append(len(segs) - a_seg)   # writes source answer -> decode
                segs.append(self._assistant_requote(a))
        return {"kind": "requote" if truths else "session", "segs": segs,
                "info": {"truths": truths, "queries": queries, "ages": ages,
                         "n_convs": n_convs}}

    def next_conv(self) -> dict:
        return self._conv_session(self.rng.random() < self.p_requote)


# ── smoke ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:                    # real tokenizer + real dataset
        import yaml
        from transformers import AutoTokenizer
        raw = yaml.safe_load(open(sys.argv[1]))
        tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
        gen = (raw.get("chat", {}) or {}).get("gen", {}) or {}
        gen = {k: v for k, v in gen.items()
               if k not in ("real_filler",)}
        st = SotaSessionStream(tok, **gen)
        stats = {"session": 0, "requote": 0}
        ages, n_segs = [], []
        for _ in range(50):
            c = st.next_conv()
            stats[c["kind"]] += 1
            ages += c["info"]["ages"]
            n_segs.append(len(c["segs"]))
        print(f"kinds {stats} | segs/session min {min(n_segs)} "
              f"med {sorted(n_segs)[len(n_segs)//2]} max {max(n_segs)} | "
              f"ages {sorted(ages)[:5]}..{sorted(ages)[-5:]}")
        for want in ("session", "requote"):
            while True:
                c = st.next_conv()
                if c["kind"] == want:
                    break
            if want == "requote":
                m = c["segs"][-1]["loss_mask"]
                print(f"requote loss_mask: min {float(m.min()):.2f} "
                      f"max {float(m.max()):.2f} mean {float(m.mean()):.2f}")
            print(f"\n===== {want} ({len(c['segs'])} segs, "
                  f"ages {c['info']['ages']}) =====")
            for s in c["segs"][:6] + c["segs"][-4:]:
                print(repr(tok.decode(s["input_ids"][0].tolist())[:120]))
        sys.exit(0)

    # hermetic: stub tokenizer, synthetic pool
    class _Tok:
        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [ord(c) % 251 for c in s]}

        def __init__(self):
            pass
    pool = [[("user", f"question {i} alpha?"),
             ("assistant", f"answer {i} bravo."),
             ("user", f"followup {i}?"), ("assistant", f"more {i}.")]
            for i in range(40)]
    st = SotaSessionStream(_Tok(), _pool=pool, surprisal_mode="sif",
                           requote_weight=4.0, seed=0)
    kinds = {"session": 0, "requote": 0}
    for _ in range(200):
        c = st.next_conv()
        kinds[c["kind"]] += 1
        for s in c["segs"]:
            assert s["input_ids"].shape == s["loss_mask"].shape[:1] + \
                (s["loss_mask"].shape[1],)
            assert s["surp_w"].shape == s["input_ids"].shape
        if c["kind"] == "requote":
            assert c["info"]["truths"] and c["info"]["ages"]
            assert all(a > 0 for a in c["info"]["ages"])
            # stub tok = 37 types, tout est fréquent => sif ~0 partout ; on ne
            # vérifie que la mécanique (>1 strict) — l'amplitude réelle (~4 sur
            # tokens rares) se lit au smoke real-data (mask max affiché)
            m = c["segs"][-1]["loss_mask"]
            assert float(m.max()) > 1.0 and float(m.min()) >= 0.0, \
                (float(m.min()), float(m.max()))
            g = grade_conv(c, ["xxx"] * 9 + [c["info"]["truths"][-1]])
            assert g > 0.4, g
    assert kinds["requote"] > 40 and kinds["session"] > 40, kinds
    print(f"self-test OK — kinds {kinds}")
