"""Chat-templated conversations over CodeChunkStream (phase 2 SFT — marche 1).

Design (user, 2026-07-14): drop the reserved-address-token mechanism; addressing
becomes ordinary instruction-following inside a standard ChatML template. The
tokens already exist in the SmolLM2 tokenizer (<|im_start|>/<|im_end|>), so this
is the exact format any downstream use will have.

Mapping onto the native defer machinery — two granularities, decoupled:
  * WRITE stays per chunk (1 gist/chunk, mechanism untouched): each emitted
    segment = one forward = one bank write, exactly like CodeChunkStream convs.
  * TURN is a dialogue unit holding 1..k chunks: a big file is ingested over
    several consecutive <user> turns (user addition 2026-07-14) — when the file
    spans enough turns its opening is out of context BY CONSTRUCTION, so the
    answer can only come from the read (anti surface-shortcut control).

Three conversation kinds (recency = feature, memory dsv6-grpo-recence-feature:
train only the DEVIATION from the recency default):
  * vanilla   — plain untemplated chunks, loss everywhere (protects the carried
                next-token circuit; the MAI mid-training-mix lesson).
  * recency   — user turns ingest a window, assistant continues the LAST chunk.
                No instruction most of the time (the majority statistics ARE the
                default); occasionally a bare "continue" instruction for
                instruction diversity.
  * reachback — same ingestion, then an instruction cues an EARLIER chunk
                (quote of its opening, optionally the synthetic file label —
                v2f recipe) and the assistant continues THAT chunk. Target age
                stratified in octaves (OPTION 2). Instruction sits on the USER
                turn or as an ASSISTANT prefill (user addition 2026-07-14:
                loss-masked when given; supervising its emission is marche 2,
                the bridge to <think> re-narration).

Segments: {"input_ids" [1,T], "loss_mask" [1,T], "role", "write"} — loss only on
assistant answer tokens (+ the closing <|im_end|> so the model learns to stop);
template/instruction/prefill tokens are masked. Vanilla segs have loss everywhere.

Smoke:
  python -m deepseek_v4_mini.chat_defer_data deepseek_v4_mini/configs/farm/v3_reach.yaml
"""
from __future__ import annotations

import random
import sys

import torch

from .code_data import CodeChunkStream

# ── ChatML pieces (tokenized once in __init__) ──────────────────────────────
U_OPEN = "<|im_start|>user\n"
A_OPEN = "<|im_start|>assistant\n"
CLOSE = "<|im_end|>\n"

RECENCY_INSTR = [
    "Continue.",
    "Keep going from where the file left off.",
    "Continue the file.",
]
REACH_USER_INSTR = [
    "Go back to the earlier part that begins with:\n{q}\nContinue it from there.",
    "Recall the section starting with:\n{q}\nWrite what comes next.",
    "Earlier you saw a part beginning:\n{q}\nContinue that part.",
]
REACH_USER_INSTR_LABEL = [
    "In file {label}, find the part beginning:\n{q}\nContinue that part.",
    "From {label}, go back to the section starting:\n{q}\nand continue it.",
]
REACH_ASST_PREFILL = "(recalling the part that begins: {q})\n"


def _file_label_str(f, n_digits: int = 6) -> str:
    """Same arithmetic hash as code_data.file_label_ids, rendered as a string."""
    ts = f[0][:16].tolist()
    h = sum((i + 1) * int(t) for i, t in enumerate(ts)) % 10 ** n_digits
    return f"<<FILE:{h:0{n_digits}d}>>"


class ChatDeferStream:
    def __init__(self, stream: CodeChunkStream, *, p_vanilla: float = 0.30,
                 p_reachback: float = 0.25, chunks_per_turn: tuple = (1, 3),
                 quote_len: int = 12, answer_len: int = 64,
                 p_recency_instr: float = 0.15, p_user_instr: float = 0.7,
                 p_label_cue: float = 0.4, seed: int = 0) -> None:
        self.s = stream
        self.tok = stream.tok
        self.rng = random.Random(seed)
        self.p_vanilla = float(p_vanilla)
        self.p_reachback = float(p_reachback)
        self.cpt = tuple(int(v) for v in chunks_per_turn)
        self.quote_len = int(quote_len)
        self.answer_len = int(answer_len)
        self.p_recency_instr = float(p_recency_instr)
        self.p_user_instr = float(p_user_instr)
        self.p_label_cue = float(p_label_cue)
        self._enc = {}                        # template-string -> 1-D LongTensor

    # ── token plumbing ───────────────────────────────────────────────────────
    def _ids(self, s: str) -> torch.Tensor:
        if s not in self._enc:
            self._enc[s] = torch.tensor(
                self.tok(s, add_special_tokens=False)["input_ids"], dtype=torch.long)
        return self._enc[s]

    def _seg(self, pieces: list[tuple[torch.Tensor, bool]], role: str,
             write: bool = True) -> dict:
        """One forward/write unit from (ids, supervised) pieces."""
        ids = torch.cat([p for p, _ in pieces])
        mask = torch.cat([torch.full_like(p, float(sup), dtype=torch.float)
                          for p, sup in pieces])
        return {"input_ids": ids.unsqueeze(0), "loss_mask": mask.unsqueeze(0),
                "attention_mask": torch.ones(1, ids.numel(), dtype=torch.long),
                "role": role, "write": write}

    # ── ingestion: window chunks spread over 1..k-chunk user turns ───────────
    def _ingest_segs(self, chunks: list[torch.Tensor]) -> list[dict]:
        segs, i = [], 0
        u_open, close = self._ids(U_OPEN), self._ids(CLOSE)
        while i < len(chunks):
            n = min(self.rng.randint(*self.cpt), len(chunks) - i)
            for j in range(n):                # each chunk stays its own write seg
                pieces = []
                if j == 0:
                    pieces.append((u_open, False))
                pieces.append((chunks[i + j], False))
                if j == n - 1:
                    pieces.append((close, False))
                segs.append(self._seg(pieces, "user"))
            i += n
        return segs

    def _assistant_seg(self, answer: torch.Tensor, prefill: str = "") -> dict:
        pieces = [(self._ids(A_OPEN), False)]
        if prefill:
            pieces.append((self._ids(prefill), False))
        pieces.append((answer, True))
        pieces.append((self._ids(CLOSE), True))   # supervised stop
        return self._seg(pieces, "assistant")

    def _quote(self, chunk: torch.Tensor) -> str:
        return self.tok.decode(chunk[:self.quote_len].tolist())

    # ── conversations ────────────────────────────────────────────────────────
    def _window(self, need_successor: bool):
        """File + [st, st+m) chunk window (next_conv sampling rule); when
        need_successor, chunk st+m must exist (it is the recency answer)."""
        for _ in range(64):
            f = self.s._pick_file()
            fx = self.s._reslice(f) if self.s.var_chunk else f
            nc = len(fx)
            lim = nc - (1 if need_successor else 0)
            if lim < 2:
                continue
            m = self.rng.randint(2, min(self.s.K, lim))
            st = self.rng.randrange(0, lim - m + 1)
            return f, fx, st, m
        raise RuntimeError("no eligible file for a chat conversation")

    def _conv_recency(self) -> dict:
        f, fx, st, m = self._window(need_successor=True)
        segs = self._ingest_segs(fx[st:st + m])
        if self.rng.random() < self.p_recency_instr:
            instr = self.rng.choice(RECENCY_INSTR)
            segs.append(self._seg([(self._ids(U_OPEN), False),
                                   (self._ids(instr + "\n"), False),
                                   (self._ids(CLOSE), False)], "user"))
        segs.append(self._assistant_seg(fx[st + m][:self.answer_len]))
        return {"kind": "recency", "segs": segs, "age": 1}

    def _conv_reachback(self) -> dict:
        f, fx, st, m = self._window(need_successor=False)
        # target age (writes back from the last ingested chunk to the target's
        # SUCCESSOR = the answer chunk) sampled uniformly over octave bins so
        # old strata are as frequent as fresh ones (OPTION 2 stratification).
        ages = list(range(1, m))              # answer chunk index = st+m-1-age
        bins: dict[int, list[int]] = {}
        for a in ages:
            bins.setdefault(a.bit_length() - 1, []).append(a)
        age = self.rng.choice(bins[self.rng.choice(list(bins))])
        j = st + m - 1 - age                  # target chunk (its opening = cue)
        quote = self._quote(fx[j])
        segs = self._ingest_segs(fx[st:st + m])
        if self.rng.random() < self.p_user_instr:
            pool = (REACH_USER_INSTR_LABEL if self.rng.random() < self.p_label_cue
                    else REACH_USER_INSTR)
            instr = self.rng.choice(pool).format(q=quote, label=_file_label_str(f))
            segs.append(self._seg([(self._ids(U_OPEN), False),
                                   (self._ids(instr + "\n"), False),
                                   (self._ids(CLOSE), False)], "user"))
            prefill = ""
        else:                                 # assistant-side directive (prefill)
            prefill = REACH_ASST_PREFILL.format(q=quote)
        segs.append(self._assistant_seg(fx[j][self.quote_len:
                                              self.quote_len + self.answer_len],
                                        prefill=prefill))
        return {"kind": "reachback", "segs": segs, "age": age}

    def _conv_vanilla(self) -> dict:
        segs = []
        for s in self.s.next_conv():
            ids = s["input_ids"]
            segs.append({"input_ids": ids, "loss_mask": torch.ones_like(ids).float(),
                         "attention_mask": torch.ones_like(ids),
                         "role": "raw", "write": True})
        return {"kind": "vanilla", "segs": segs, "age": 0}

    def next_conv(self) -> dict:
        r = self.rng.random()
        if r < self.p_vanilla:
            return self._conv_vanilla()
        if r < self.p_vanilla + self.p_reachback:
            return self._conv_reachback()
        return self._conv_recency()


# ── smoke: decode one conversation per kind + mix/age stats ─────────────────
def _show(tok, conv, max_seg_chars=200):
    print(f"\n===== kind={conv['kind']} age={conv['age']} "
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
    raw = yaml.safe_load(open(cfg_path)); d = dict(raw["data"])
    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    add = [x for x in ("<think>", "<blank>") if x not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    d.pop("defer_len", None); d.pop("batch_size", None)
    stream = CodeChunkStream(tok, split="held", **d)
    cds = ChatDeferStream(stream, seed=0)

    for want in ("vanilla", "recency", "reachback"):
        for _ in range(200):
            c = cds.next_conv()
            if c["kind"] == want:
                _show(tok, c); break

    from collections import Counter
    kinds, ages, toks = Counter(), Counter(), []
    for _ in range(300):
        c = cds.next_conv()
        kinds[c["kind"]] += 1
        toks.append(sum(s["input_ids"].numel() for s in c["segs"]))
        if c["kind"] == "reachback":
            ages[1 << (c["age"].bit_length() - 1)] += 1
    print(f"\nmix over 300 convs: {dict(kinds)}")
    print(f"reachback age octaves (lower edge): {dict(sorted(ages.items()))}")
    print(f"tokens/conv: min {min(toks)} med {sorted(toks)[len(toks)//2]} "
          f"max {max(toks)}")


if __name__ == "__main__":
    main(sys.argv[1])
