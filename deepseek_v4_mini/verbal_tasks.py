"""Verbalized bank tasks for the SmolLM graft (real-data phase 1).

Natural-language transposition of multiturn_rule (design: memory
dsv4mini-real-data-sft-smollm): a conversation is a sequence of SEGMENTS,
one segment = one forward = one write; the bank is the ONLY bridge between
segments (no KV cache is carried), so "remember what was said N turns ago"
cannot be solved in-context — exactly the synthetic task's structure.

  presentation : "User: From now on, when I say {key}, reply {val}.
                  Assistant: Understood."
  query        : "User: {key}?  Assistant: {val}"       ← the metric position
  distractor   : canned small talk (writes noise into the bank between
                  presentation and query = the delayed-restitution pressure)
  switch       : re-presentation of a known key with a NEW value — the
                  supersession pressure (merge-gate territory), off by default

Keys are fresh PSEUDO-WORDS every conversation (no prior, novel by
construction — the diversity lesson: every rule is one-shot). Values are
common words that tokenize to ONE token with a leading space, so the answer
is a single position: acc is exact, chance = 1/len(value pool). A val-mod
split reserves every Nth value word for held-out eval (decode
generalization), mirroring heldout_rule_mod.

Loss: assistant tokens only (-100 elsewhere). Acks and distractor replies
ARE supervised — they keep the chat format alive (mini-replay); the metric
reads only the query answer position.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch

# ── Pseudo-word keys ─────────────────────────────────────────────────────────
_CONS, _VOW = "bdfgklmnprstvz", "aeiou"


def pseudo_word(rng: random.Random, n_syll: int = 3) -> str:
    return "".join(rng.choice(_CONS) + rng.choice(_VOW) for _ in range(n_syll))


# ── Single-token value pool (filtered against the tokenizer at init) ────────
_VALUE_CANDIDATES = """
tiger lion horse eagle shark whale snake mouse camel zebra panda otter
red blue green black white yellow purple orange pink brown silver gold
apple bread sugar honey lemon grape peach mango olive onion rice corn
table chair house stone river cloud storm frost flame ember shadow light
north south east west spring summer winter autumn morning evening night
copper iron steel glass paper cotton velvet marble amber coral ivory jade
""".split()

# ── Phrasing diversity (v3) ─────────────────────────────────────────────────
# Multiple surface forms for the SAME task: the read must decode the rule, not
# recognize a template. Query replies stay "the bare value word" so the answer
# is always the last token (metric position preserved).
_PRESENT_TEMPLATES = [
    ("From now on, when I say {key}, reply {val}.", "Understood."),
    ("New rule: {key} means {val}.", "Got it."),
    ("Remember this: if I say {key}, you answer {val}.", "Okay, I'll remember."),
    ("Let's define {key} as {val}.", "Noted."),
    ("Whenever you see {key}, respond with {val}.", "Will do."),
    ("The code word {key} stands for {val}.", "Understood."),
]
_QUERY_TEMPLATES = [
    "{key}?",
    "What does {key} mean?",
    "Quick check: {key}?",
    "Tell me: {key}?",
    "And {key}?",
]

_DISTRACTORS = [
    ("How are you today?", "Doing great, thanks for asking!"),
    ("What can you help me with?", "I can answer questions and remember instructions."),
    ("Tell me something interesting.", "Octopuses have three hearts."),
    ("Do you like music?", "I enjoy talking about it, yes."),
    ("What's the weather like?", "I can't see outside, but I hope it's sunny."),
    ("Say something nice.", "You're doing a great job today."),
    ("Are you paying attention?", "Yes, I'm following along."),
    ("What time is it?", "I don't have a clock, sorry."),
    ("Can you count to three?", "One, two, three."),
    ("Thanks for your help.", "You're welcome!"),
]


@dataclass
class VerbalTaskConfig:
    batch_size: int = 8
    n_pairs: int = 2            # rules per conversation (K analogue)
    turns: int = 8              # post-presentation segments
    distractor_p: float = 0.35  # per-turn probability of a distractor turn
    switch_p: float = 0.0       # per-turn probability of a re-presentation w/ new value
    heldout_val_mod: int = 8    # value words with idx % mod == 0 → eval-only
    max_seg_len: int = 40       # tokens per segment (pad/assert)
    seed: int = 0
    # distractor injection: "canned" = 10 fixed QA pairs; "ultrachat" = real
    # dialogue turns streamed from HuggingFaceH4/ultrachat_200k — the memory
    # pressure stays controlled (we know where the keys are) but the noise
    # between presentation and query is real text in all its variety, and the
    # supervised replies double as anti-forgetting replay.
    distractor_source: str = "canned"
    distractor_max_len: int = 96   # token cap for real turns (truncated cleanly)
    ultrachat_split: str = "train_sft"
    # ── diversity knobs (v3) — defaults reproduce v1/v2 exactly ─────────────
    # value_source "vocab" = every clean single-token lowercase word in the
    # tokenizer vocab (thousands) instead of the 60-word hand list — the
    # dsv4m diversity lesson: a closed repertoire trains RECOGNITION of the
    # codes, an open pool forces a general decode (held split stays mod-based).
    value_source: str = "list"      # "list" | "vocab"
    value_min_len: int = 3          # vocab filter: word length bounds
    value_max_len: int = 10
    phrasing_diversity: bool = False  # sample presentation/query templates
    key_syll_min: int = 3           # pseudo-word key length range (syllables)
    key_syll_max: int = 3
    n_pairs_max: int = 0            # >0: per-conv K ~ U[n_pairs, n_pairs_max]
    value_pool_cap: int = 0         # >0: keep only the first N vocab values
                                    # (by token id) before the train/held split
                                    # — a tractable pool for the decode


class UltraChatTurns:
    """Infinite stream of (user, assistant) text pairs from ultrachat_200k,
    token-truncated to fit a segment budget. Deterministic per seed."""

    def __init__(self, tokenizer, cfg: VerbalTaskConfig) -> None:
        from datasets import load_dataset          # lazy: only when injected
        self.ds  = load_dataset("HuggingFaceH4/ultrachat_200k", split=cfg.ultrachat_split)
        self.tok = tokenizer
        self.cap = int(cfg.distractor_max_len)
        self.rng = random.Random(cfg.seed + 7)

    def _trunc(self, text: str, budget: int) -> str:
        ids = self.tok.encode(text, add_special_tokens=False)
        return text if len(ids) <= budget else self.tok.decode(ids[:budget]).rstrip()

    def sample(self) -> tuple[str, str]:
        while True:
            msgs = self.ds[self.rng.randrange(len(self.ds))]["messages"]
            pairs = [(msgs[i]["content"].strip(), msgs[i + 1]["content"].strip())
                     for i in range(0, len(msgs) - 1, 2)
                     if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant"]
            if not pairs:
                continue
            u, a = pairs[self.rng.randrange(len(pairs))]
            if not u or not a:
                continue
            # budget: format overhead ~8 tokens; user gets a third, reply the rest
            u = self._trunc(u, (self.cap - 8) // 3)
            a = self._trunc(a, self.cap - 8 - len(self.tok.encode(u, add_special_tokens=False)))
            if a:
                return u, a


class VerbalRuleGen:
    """Yields batched segments. Each item:

        {input_ids [B,T], attention_mask [B,T], labels [B,T] (-100 masked),
         reset [B] bool, kind str, ans_pos [B] (query only, -1 otherwise),
         ans_ids [B]   (query only: the single value-token id)}

    All lanes share the segment KIND sequence (like the synthetic generator);
    keys/values differ per lane. Infinite iterator; `split` picks the value
    sub-pool (train / held).
    """

    def __init__(self, tokenizer, cfg: VerbalTaskConfig, split: str = "train") -> None:
        self.tok, self.cfg = tokenizer, cfg
        ids, words = [], []
        if cfg.value_source == "vocab":
            # every " word" single token that round-trips: lowercase alpha,
            # length-bounded, deterministic order (token id)
            for tid in range(len(tokenizer)):
                s = tokenizer.decode([tid])
                if (len(s) >= 2 and s[0] == " " and s[1:].isalpha()
                        and s[1:].islower()
                        and cfg.value_min_len <= len(s) - 1 <= cfg.value_max_len
                        and tokenizer.encode(s, add_special_tokens=False) == [tid]):
                    words.append(s[1:]); ids.append(tid)
        else:
            for w in _VALUE_CANDIDATES:
                t = tokenizer.encode(" " + w, add_special_tokens=False)
                if len(t) == 1:
                    words.append(w); ids.append(t[0])
        assert len(words) >= 32, f"only {len(words)} single-token values survive"
        if cfg.value_pool_cap and len(words) > cfg.value_pool_cap:
            words = words[:cfg.value_pool_cap]
            ids   = ids[:cfg.value_pool_cap]
        mod = cfg.heldout_val_mod
        keep = [i % mod == 0 for i in range(len(words))]
        sel = keep if split == "held" else [not k for k in keep]
        self.values    = [w for w, s in zip(words, sel) if s]
        self.value_ids = [i for i, s in zip(ids, sel) if s]
        self.chance    = 1.0 / len(self.values)
        self.pad_id    = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.turn_pool = (UltraChatTurns(tokenizer, cfg)
                          if cfg.distractor_source == "ultrachat" else None)

    # ── Segment builders (text → fixed-width tensors) ───────────────────────
    def _encode(self, user: str, assistant: str, cap: Optional[int] = None):
        """Return (ids, labels) — loss on assistant tokens (incl. EOS)."""
        prefix = self.tok.encode(f"User: {user}\nAssistant:", add_special_tokens=False)
        answer = self.tok.encode(" " + assistant, add_special_tokens=False)
        ids    = prefix + answer
        labels = [-100] * len(prefix) + answer[:]
        limit  = cap if cap is not None else self.cfg.max_seg_len
        assert len(ids) <= limit, f"segment too long: {len(ids)} > {limit}"
        return ids, labels

    def _present(self, key: str, val: str, rng: Optional[random.Random] = None):
        u, ack = (_PRESENT_TEMPLATES[rng.randrange(len(_PRESENT_TEMPLATES))]
                  if (rng is not None and self.cfg.phrasing_diversity)
                  else _PRESENT_TEMPLATES[0])
        return self._encode(u.format(key=key, val=val), ack)

    def _query(self, key: str, val: str, rng: Optional[random.Random] = None):
        u = (_QUERY_TEMPLATES[rng.randrange(len(_QUERY_TEMPLATES))]
             if (rng is not None and self.cfg.phrasing_diversity)
             else _QUERY_TEMPLATES[0])
        ids, labels = self._encode(u.format(key=key), val)
        # the value token is the LAST position (single-token value, no EOS after)
        return ids, labels, len(ids) - 1

    # ── Batched conversation stream ──────────────────────────────────────────
    def __iter__(self):
        cfg = self.cfg
        rng = random.Random(cfg.seed)
        B = cfg.batch_size
        while True:
            # per-conv K (shared across lanes: kinds stay aligned)
            K = (rng.randint(cfg.n_pairs, cfg.n_pairs_max)
                 if cfg.n_pairs_max > cfg.n_pairs else cfg.n_pairs)
            # per-lane fresh keys + values (variable-length pseudo-words)
            keys = [[pseudo_word(rng, rng.randint(cfg.key_syll_min, cfg.key_syll_max))
                     for _ in range(K)] for _ in range(B)]
            vals = [rng.sample(self.values, K) for _ in range(B)]
            # shared segment plan (kinds aligned across lanes)
            plan = [("present", k) for k in range(K)]
            for t in range(cfg.turns):
                if rng.random() < cfg.switch_p:
                    plan.append(("switch", rng.randrange(K)))
                plan.append(("distract", rng.randrange(len(_DISTRACTORS)))
                            if rng.random() < cfg.distractor_p
                            else ("query", t % K))
            for si, (kind, arg) in enumerate(plan):
                rows, labs, apos, aids = [], [], [], []
                tf = []          # teacher target: value TOKEN id on presentations
                for b in range(B):
                    if kind == "present":
                        ids, lab = self._present(keys[b][arg], vals[b][arg], rng)
                        p, a = -1, -1
                        tf.append(self.value_ids[self.values.index(vals[b][arg])])
                    elif kind == "switch":
                        new = rng.choice([v for v in self.values if v != vals[b][arg]])
                        vals[b][arg] = new
                        ids, lab = self._present(keys[b][arg], new, rng)
                        p, a = -1, -1
                        tf.append(self.value_ids[self.values.index(new)])
                    elif kind == "distract":
                        # per-lane real turn when injected (richer noise);
                        # shared canned pair otherwise
                        u, r = (self.turn_pool.sample() if self.turn_pool is not None
                                else _DISTRACTORS[arg])
                        ids, lab = self._encode(u, r, cap=self.cfg.distractor_max_len)
                        p, a = -1, -1
                        tf.append(-1)
                    else:  # query
                        ids, lab, p = self._query(keys[b][arg], vals[b][arg], rng)
                        a = ids[p]
                        tf.append(-1)
                    rows.append(ids); labs.append(lab); apos.append(p); aids.append(a)
                T = max(len(r) for r in rows)
                x  = torch.full((B, T), self.pad_id, dtype=torch.long)
                y  = torch.full((B, T), -100, dtype=torch.long)
                am = torch.zeros((B, T), dtype=torch.long)
                for b, (r, l) in enumerate(zip(rows, labs)):
                    x[b, :len(r)] = torch.tensor(r); y[b, :len(l)] = torch.tensor(l)
                    am[b, :len(r)] = 1
                yield {
                    "input_ids": x, "attention_mask": am, "labels": y,
                    "reset": torch.full((B,), si == 0, dtype=torch.bool),
                    "kind": kind,
                    "ans_pos": torch.tensor(apos), "ans_ids": torch.tensor(aids),
                    "tf_ids": torch.tensor(tf, dtype=torch.long),
                }
