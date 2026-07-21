"""Persona facts — synthetic recall conversations (phase 2 SFT, demo family).

Design (user, 2026-07-20): the most legible public demo of the bank is a chat
where a personal fact stated early ("my dog is named Biscuit") is recalled many
turns later, with the ablated-bank arm as the control. Recall is a TRAINED
behaviour (dsv4mini-switch-task: zero-shot perseveres), so this stream teaches
it explicitly, the same way math_school teaches bindings.

Two conversation kinds:
  * smalltalk — filler exchanges only (weather, plans, hobbies). Teaches the
                ChatML register and protects the no-question default; never
                graded (trivial 1.0 both arms, Δ=0).
  * recall    — 1..3 facts planted in early user turns (each fact = its own
                seg = its own bank write, sometimes acknowledged), small-talk
                filler pushes them out of context — WHICH, at one write per
                seg, means out of the FORWARD WINDOW after one seg and out of
                the FIFO after max_mem=8 writes — then 1..2 queries whose
                supervised answers restate the fact. Multi-fact conversations
                exercise selection at recall (superposition is the feature).
                Occasional UPDATE turns ("Actually, we renamed…") exercise
                supersession: the query then expects the NEW value.

Age = number of segs (= writes) between a fact's seg and its query's answer
decode. Filler mass is stratified in octave bins; a p_beyond fraction pushes
age past the FIFO horizon (8 writes) as an eviction-survival probe — report
that stratum separately, do not average it in.

Hybrid filler (user green light 2026-07-20): with `real_filler` set (e.g.
"HuggingFaceTB/smol-smoltalk"), a p_real fraction of filler exchanges is
sampled from REAL conversations (short turns only, disk-cached) — natural
register learned from supervised real assistant turns, while facts/queries
stay synthetic and mechanically gradable. Template filler remains the
fallback and the hermetic-test path.

Grading = word-boundary match of the fact value in the decoded answer
(case-insensitive). Values are picked to be rare-ish words (crimson, not blue)
so the ablated arm's babble cannot false-positive its way to a grade.

Segments match chat_defer_data/math_school exactly: {"input_ids" [1,T],
"loss_mask" [1,T], "attention_mask", "role", "write"} — loss only on assistant
text + closing <|im_end|>. Interface matches the `chat:` block of
code_defer_native (.next_conv() + .rng + module grade_conv).

Hermetic self-test (stub tokenizer, no downloads):
  python -m deepseek_v4_mini.persona_chat_data
Real-tokenizer smoke (decode one conv per kind + stats):
  python -m deepseek_v4_mini.persona_chat_data deepseek_v4_mini/configs/farm/v3_reach.yaml
"""
from __future__ import annotations

import random
import re
import sys

import torch

from .math_school_data import U_OPEN, A_OPEN, CLOSE

# ── fact slots ───────────────────────────────────────────────────────────────
# Each slot: statement templates, question templates, answer templates, value
# pool. Values are deliberately uncommon words/names: the grader is a
# word-boundary match and frequent words would let babble score.

PET_TYPES = ["dog", "cat", "parrot", "hamster", "rabbit"]
PET_NAMES = ["Biscuit", "Waffles", "Mochi", "Pretzel", "Noodle", "Pistachio",
             "Clementine", "Marzipan", "Turnip", "Gizmo", "Pumpernickel",
             "Crouton", "Paprika", "Fondue", "Gnocchi"]
PEOPLE = ["Ottilie", "Barnaby", "Perpetua", "Ignatius", "Wilhelmina",
          "Thaddeus", "Euphemia", "Leopoldine", "Casimir", "Apollonia",
          "Bartholomew", "Seraphina", "Archibald", "Philomena", "Montgomery"]
CITIES = ["Ljubljana", "Tromso", "Valparaiso", "Guanajuato", "Fremantle",
          "Coimbra", "Rovaniemi", "Antofagasta", "Trondheim", "Ouarzazate",
          "Matsumoto", "Bruges", "Cuenca", "Galway", "Dubrovnik"]
JOBS = ["beekeeper", "luthier", "cartographer", "glassblower", "falconer",
        "typesetter", "archivist", "milliner", "cooper", "saddler"]
COLORS = ["crimson", "turquoise", "vermilion", "chartreuse", "indigo",
          "ochre", "magenta", "cerulean"]
FOODS = ["lasagna", "gazpacho", "ratatouille", "moussaka", "tabbouleh",
         "goulash", "paella", "borscht", "falafel", "tiramisu"]
SIBLINGS = ["sister", "brother"]

# slot -> (statements, questions, answers, update-statements, pool)
# {v} = value, {p} = pet type / sibling word (fixed per conversation).
SLOTS = {
    "pet": (
        ["By the way, I have a {p} named {v}.",
         "I adopted a {p} last year, her name is {v}.",
         "My {p} {v} kept me up all night again."],
        ["What is my {p}'s name?",
         "Do you remember what my {p} is called?",
         "Remind me, what did I say my {p}'s name was?"],
        ["Your {p} is named {v}.",
         "You said your {p} is called {v}."],
        ["Actually, we ended up renaming our {p}. She goes by {v} now.",
         "Small correction: my {p} is called {v} these days."],
        PET_NAMES,
    ),
    "name": (
        ["My name is {v}, nice to meet you.",
         "I forgot to introduce myself earlier, I'm {v}.",
         "Oh, and you can call me {v}."],
        ["Do you remember my name?",
         "What's my name again?"],
        ["Your name is {v}.",
         "You told me your name is {v}."],
        ["Actually I go by my middle name, {v}. Please use that.",
         "Correction: call me {v} instead."],
        PEOPLE,
    ),
    "city": (
        ["I live in {v} these days.",
         "I moved to {v} a few months ago.",
         "Greetings from {v}, where I live."],
        ["Where do I live?",
         "Which city did I say I live in?"],
        ["You live in {v}.",
         "You said you live in {v}."],
        ["Actually we just moved again, to {v} this time.",
         "Update: I'm living in {v} now."],
        CITIES,
    ),
    "sibling": (
        ["My {p}'s name is {v}.",
         "I spent the weekend at my {p} {v}'s place."],
        ["What is my {p}'s name?",
         "Do you remember what my {p} is called?"],
        ["Your {p} is named {v}.",
         "You said your {p} is called {v}."],
        ["I misspoke earlier, my {p}'s name is actually {v}."],
        PEOPLE,
    ),
    "job": (
        ["I work as a {v}.",
         "My day job is being a {v}, believe it or not."],
        ["What do I do for a living?",
         "What did I say my job was?"],
        ["You work as a {v}.",
         "You said you are a {v}."],
        ["I changed careers recently, I'm a {v} now."],
        JOBS,
    ),
    "color": (
        ["My favorite color is {v}.",
         "I repainted my office {v}, my favorite color."],
        ["What's my favorite color?",
         "Which color did I say I like best?"],
        ["Your favorite color is {v}.",
         "You said your favorite color is {v}."],
        ["I've changed my mind about colors, {v} is my favorite now."],
        COLORS,
    ),
    "food": (
        ["My favorite dish is {v}.",
         "Nothing beats a good {v}, my absolute favorite."],
        ["What's my favorite dish?",
         "Which dish did I say was my favorite?"],
        ["Your favorite dish is {v}.",
         "You said your favorite dish is {v}."],
        ["My tastes changed, these days my favorite dish is {v}."],
        FOODS,
    ),
}

ACK_TMPL = [
    "Nice, {v} is a lovely name.",
    "Got it, I'll remember that.",
    "Noted!",
    "That's a great choice.",
]

# ── small talk (filler pairs; slots keep them varied) ────────────────────────
WEATHER = ["rainy", "windy", "foggy", "scorching", "freezing", "muggy"]
ACTIVITIES = ["run", "swim", "hike", "bike ride", "walk"]
HOBBIES = ["baking bread", "learning the ukulele", "repotting my plants",
           "fixing my old bike", "sketching", "doing a jigsaw puzzle"]
SHOWS = ["a documentary about volcanoes", "an old detective series",
         "a cooking competition", "a nature documentary"]

FILLERS = [
    ("It's been {w} all day here.",
     "I hope the weather clears up for you soon."),
    ("I went for a {a} this morning before work.",
     "That sounds like a great way to start the day."),
    ("I spent the evening {h} yesterday.",
     "That sounds relaxing, I hope it went well."),
    ("I've been watching {s} lately.",
     "That sounds interesting, tell me how it ends."),
    ("Work has been pretty busy this week.",
     "Hopefully things calm down for you soon."),
    ("I finally cleaned out the garage this weekend.",
     "That must feel satisfying to have done."),
    ("I'm thinking about planting tomatoes this year.",
     "Homegrown tomatoes are worth the effort."),
    ("My neighbor's music was way too loud last night.",
     "That's frustrating, I hope tonight is quieter."),
    ("I burned my toast twice this morning.",
     "Third time's the charm, as they say."),
    ("Traffic was terrible on the way home today.",
     "That's always draining, glad you made it back."),
    ("I found a great little bakery around the corner.",
     "A good bakery nearby is a real treasure."),
    ("I can't decide what to cook tonight.",
     "Something simple usually turns out best."),
]

# filler-pair count bins, octave-ish: age past the fact grows accordingly.
FILLER_BINS = [(0, 0), (1, 1), (2, 3), (4, 6)]
BEYOND_BIN = (7, 10)                     # pushes age past the 8-write FIFO


def extract_filler_pairs(rows, tok, *, max_tok: int = 96, cap: int = 20000):
    """Harvest short (user, assistant) exchange pairs from an SFT-format
    dataset (rows = iterable of {"messages": [{"role","content"}, ...]}).
    Char pre-filter first, exact token count second (both turns <= max_tok).
    User idea 2026-07-20: real conversations as filler = natural register,
    synthetic facts/queries stay mechanically gradable."""
    pairs = []
    for row in rows:
        msgs = row.get("messages") or []
        for i in range(len(msgs) - 1):
            u, a = msgs[i], msgs[i + 1]
            if u.get("role") != "user" or a.get("role") != "assistant":
                continue
            ut, at = u["content"].strip(), a["content"].strip()
            if not ut or not at or max(len(ut), len(at)) > 4 * max_tok:
                continue
            if max(len(tok(ut, add_special_tokens=False)["input_ids"]),
                   len(tok(at, add_special_tokens=False)["input_ids"])) > max_tok:
                continue
            pairs.append((ut, at))
            if len(pairs) >= cap:
                return pairs
    return pairs


def _find_sub(seq: list, sub: list):
    """Premier span [i, j) où seq[i:j] == sub, sinon None."""
    if not sub:
        return None
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return (i, i + len(sub))
    return None


def _canon(v: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(v.lower()) + r"\b")


def grade_recall(answers: list[str], truths: list[str]) -> float:
    """Fraction of answers containing their truth value (word-boundary,
    case-insensitive)."""
    ok = sum(bool(_canon(t).search(a.lower())) for a, t in zip(answers, truths))
    return ok / max(1, len(truths))


def grade_conv(conv: dict, texts: list[str]) -> float:
    """Eval entry point (same contract as math_school_data.grade_conv): the
    trainer decodes the assistant turns in order, the generator grades. Graded
    turns are the LAST len(truths) assistant turns."""
    truths = conv["info"]["truths"]
    if not truths:                        # smalltalk: nothing to grade
        return 1.0
    return grade_recall(texts[-len(truths):], truths)


# ── the stream ───────────────────────────────────────────────────────────────

class PersonaChatStream:
    def __init__(self, tok, *, p_smalltalk: float = 0.25,
                 n_facts: tuple = (1, 3), n_queries: tuple = (1, 2),
                 p_ack: float = 0.5, p_update: float = 0.15,
                 p_beyond: float = 0.15, value_weight: float = 1.0,
                 real_filler: str = None,
                 real_split: str = "train", real_cap: int = 20000,
                 real_max_tok: int = 96, p_real: float = 0.8,
                 real_cache_dir: str = None,
                 surprisal_ref: str = None, surprisal_device: str = "cpu",
                 surprisal_alpha: float = 2.0, surprisal_mode: str = "nll",
                 sif_a: float = 1e-4, seed: int = 0) -> None:
        self.tok = tok
        self.rng = random.Random(seed)
        self.p_smalltalk = float(p_smalltalk)
        self.n_facts = tuple(int(v) for v in n_facts)
        self.n_queries = tuple(int(v) for v in n_queries)
        self.p_ack = float(p_ack)
        self.p_update = float(p_update)
        self.p_beyond = float(p_beyond)
        self.value_weight = float(value_weight)
        self.p_real = float(p_real)
        self.real_pairs = None
        if real_filler:
            self.real_pairs = self._load_real_pairs(
                real_filler, real_split, int(real_cap), int(real_max_tok),
                real_cache_dir)
        self._enc = {}
        # teacher surprisal (label-free) : poids par token = nll^alpha sous un
        # LM de référence GELÉ — « l'information d'un seg = ce que le modèle de
        # référence n'a pas su prédire ». Généralise val_mask à tout corpus
        # (code/prose : pas de span étiquetable). Cible fixe (modèle gelé),
        # cache par texte de seg (les templates se répètent massivement).
        self.surp_ref_name = surprisal_ref
        self.surp_device = surprisal_device
        self.surp_alpha = float(surprisal_alpha)
        # mode 'sif' (décision user 2026-07-21, verdict analysis/freq_vs_surp*) :
        # poids = a/(a+p(token)) sur table unigram AUTO (300 convs du stream,
        # rng dédié) — zéro modèle ref, borné (typos plafonnés), bat la nll^2
        # sur les 2 axes (persona ET mix 14 sources). alpha ignoré en sif.
        assert surprisal_mode in ("nll", "sif"), surprisal_mode
        self.surp_mode = surprisal_mode
        self.sif_a = float(sif_a)
        self.surp_on = bool(surprisal_ref) or surprisal_mode == "sif"
        self._sif_p = None
        self._table_pass = False
        self._surp_model = None
        self._surp_cache = {}

    def _sif_table(self) -> dict:
        """Table unigram p(token) construite sur 300 convs du stream lui-même
        (rng dédié figé => identique entre instances train/eval)."""
        if self._sif_p is not None:
            return self._sif_p
        saved = self.rng
        self.rng = random.Random(4242)
        self._table_pass = True
        from collections import Counter
        cnt, tot = Counter(), 0
        for _ in range(300):
            for seg in self.next_conv()["segs"]:
                t = seg["input_ids"][0].tolist()
                cnt.update(t); tot += len(t)
        self._table_pass = False
        self.rng = saved
        self._sif_p = {t: c / tot for t, c in cnt.items()}
        self._sif_unseen = 0.5 / tot
        print(f"surprisal SIF: table unigram 300 convs ({tot} tokens, "
              f"{len(self._sif_p)} types, a={self.sif_a:g})", flush=True)
        return self._sif_p

    def _surp_weights(self, ids: torch.Tensor) -> torch.Tensor:
        """[T] float : poids de saillance par token — mode 'nll' : nll de la
        ref gelée ^ alpha (token 0 = poids moyen) ; mode 'sif' : a/(a+p)."""
        key = tuple(ids.tolist())
        w = self._surp_cache.get(key)
        if w is not None:
            return w
        if self.surp_mode == "sif":
            p = self._sif_table()
            a = self.sif_a
            w = torch.tensor([a / (a + p.get(t, self._sif_unseen))
                              for t in key])
            self._surp_cache[key] = w
            return w
        if self._surp_model is None:
            from transformers import AutoModelForCausalLM
            # sur GPU : fp16 (~270 Mo pour 135M) — permet de coloc la ref avec
            # le train 386M dans 8 Go de VRAM rig (le CPU des rigs est trop
            # lent pour le forward de ref, décision user 2026-07-21)
            m = AutoModelForCausalLM.from_pretrained(
                self.surp_ref_name,
                torch_dtype=(torch.float16 if self.surp_device != "cpu"
                             else torch.float32))
            self._surp_model = m.eval().requires_grad_(False).to(self.surp_device)
            self._surp_vocab = m.get_input_embeddings().num_embeddings
            print(f"surprisal ref: {self.surp_ref_name} gelé sur "
                  f"{self.surp_device} (vocab {self._surp_vocab}, "
                  f"alpha {self.surp_alpha})", flush=True)
        x = ids.clamp_max(self._surp_vocab - 1).unsqueeze(0).to(self.surp_device)
        with torch.no_grad():
            lg = self._surp_model(x).logits.float()
        import torch.nn.functional as F
        nll = F.cross_entropy(lg[0, :-1], x[0, 1:], reduction="none")
        w = torch.cat([nll.mean().reshape(1), nll]).pow(self.surp_alpha).cpu()
        self._surp_cache[key] = w
        return w

    def _load_real_pairs(self, name, split, cap, max_tok, cache_dir):
        """Filler pairs from a real SFT dataset, disk-cached (texts, small)."""
        import hashlib, os
        key = hashlib.sha256(f"{name}|{split}|{cap}|{max_tok}".encode()) \
            .hexdigest()[:16]
        path = (os.path.join(cache_dir, f"persona_filler_{key}.pt")
                if cache_dir else None)
        if path and os.path.exists(path):
            pairs = torch.load(path)
            print(f"persona filler: cache hit {path} — {len(pairs)} pairs")
            return pairs
        from datasets import load_dataset
        rows = load_dataset(name, split=split, streaming=True)
        pairs = extract_filler_pairs(rows, self.tok, max_tok=max_tok, cap=cap)
        print(f"persona filler: {name}[{split}] — {len(pairs)} pairs "
              f"(<= {max_tok} tok/turn)")
        if path:
            torch.save(pairs, path)
        return pairs

    grade_conv = staticmethod(grade_conv)

    # ── token plumbing (identical to math_school_data) ───────────────────────
    def _ids(self, s: str) -> torch.Tensor:
        if s not in self._enc:
            self._enc[s] = torch.tensor(
                self.tok(s, add_special_tokens=False)["input_ids"], dtype=torch.long)
        return self._enc[s]

    def _seg(self, pieces: list[tuple[str, bool]], role: str) -> dict:
        ids = torch.cat([self._ids(p) for p, _ in pieces])
        mask = torch.cat([torch.full((self._ids(p).numel(),), float(sup))
                          for p, sup in pieces])
        seg = {"input_ids": ids.unsqueeze(0), "loss_mask": mask.unsqueeze(0),
               "attention_mask": torch.ones(1, ids.numel(), dtype=torch.long),
               "role": role, "write": True}
        if self.surp_on and not self._table_pass:
            # l'échafaudage ChatML est injecté par la machine, pas du contenu :
            # exclu du pooling (sinon <|im_end|>/user, très surprenants pour une
            # ref base, dominent TOUTES les cibles -> zéro discriminabilité)
            body = torch.cat([torch.full((self._ids(p).numel(),),
                                         0.0 if p in (U_OPEN, A_OPEN, CLOSE)
                                         else 1.0)
                              for p, _ in pieces])
            seg["surp_w"] = (self._surp_weights(ids) * body).unsqueeze(0)
        return seg

    def _user(self, text: str) -> dict:
        return self._seg([(U_OPEN, False), (text + "\n", False), (CLOSE, False)],
                         "user")

    def _val_span(self, ids: list, v: str):
        """Span [i,j) des tokens de la valeur v dans ids (essaie l'espace de
        tête d'abord = tokenisation naturelle), sinon None."""
        for vs in (" " + v, v):
            vids = self.tok(vs, add_special_tokens=False)["input_ids"]
            if vids:
                span = _find_sub(ids, vids)
                if span is not None:
                    return span
        return None

    def _user_valued(self, text: str, v: str) -> dict:
        """Seg user qui ÉNONCE un fait : balise le span valeur avec un val_mask
        (séparé du loss_mask — le user n'est pas supervisé) pour que le teacher
        discriminant puisse pooler l'embedding de LA valeur (code propre par
        valeur) au lieu du gist moyen du chunk. Run 7-resume."""
        seg = self._user(text)
        ids = seg["input_ids"][0].tolist()
        vmask = torch.zeros(len(ids))
        span = self._val_span(ids, v)
        if span is not None:
            vmask[span[0]:span[1]] = 1.0
        seg["val_mask"] = vmask.unsqueeze(0)
        return seg

    def _assistant(self, text: str) -> dict:
        return self._seg([(A_OPEN, False), (text, True), ("\n", True),
                          (CLOSE, True)], "assistant")

    def _assistant_valued(self, text: str, v: str) -> dict:
        """Assistant answer whose VALUE token span is upweighted in the loss
        mask (run 7 : pression native persistante — la valeur ne peut venir que
        de la banque, donc la surpondérer concentre le gradient sur le chemin
        read→réponse, et ça SURVIT au retrait du teacher, contrairement au blend
        β qui n'est qu'un échafaudage). Tokenisation identique à _assistant
        (texte entier en un bloc) ; seul le masque change sur le span valeur."""
        w = self.value_weight
        if w == 1.0:
            return self._assistant(text)
        body = self._ids(text)
        m = torch.ones(body.numel())
        span = self._val_span(body.tolist(), v)
        if span is not None:
            m[span[0]:span[1]] = w
        pre = self._ids(A_OPEN); suf = self._ids("\n"); cl = self._ids(CLOSE)
        ids = torch.cat([pre, body, suf, cl])
        mask = torch.cat([torch.zeros(pre.numel()), m,
                          torch.ones(suf.numel() + cl.numel())])
        seg = {"input_ids": ids.unsqueeze(0), "loss_mask": mask.unsqueeze(0),
               "attention_mask": torch.ones(1, ids.numel(), dtype=torch.long),
               "role": "assistant", "write": True}
        if self.surp_on and not self._table_pass:
            keep = torch.cat([torch.zeros(pre.numel()), torch.ones(body.numel()),
                              torch.zeros(suf.numel() + cl.numel())])
            seg["surp_w"] = (self._surp_weights(ids) * keep).unsqueeze(0)
        return seg

    # ── pieces ───────────────────────────────────────────────────────────────
    def _filler_pair(self) -> list[dict]:
        if self.real_pairs and self.rng.random() < self.p_real:
            u, a = self.rng.choice(self.real_pairs)
        else:
            u, a = self.rng.choice(FILLERS)
            u = u.format(w=self.rng.choice(WEATHER),
                         a=self.rng.choice(ACTIVITIES),
                         h=self.rng.choice(HOBBIES), s=self.rng.choice(SHOWS))
        return [self._user(u), self._assistant(a)]

    def _sample_fact(self, used_slots: set, used_vals: set):
        slot = self.rng.choice([s for s in SLOTS if s not in used_slots])
        st, qs, ans, upd, pool = SLOTS[slot]
        v = self.rng.choice([x for x in pool if x not in used_vals])
        p = self.rng.choice(PET_TYPES if slot == "pet"
                            else SIBLINGS if slot == "sibling" else [""])
        return dict(slot=slot, v=v, p=p, st=st, qs=qs, ans=ans, upd=upd,
                    pool=pool)

    # ── conversations ────────────────────────────────────────────────────────
    def _conv_smalltalk(self) -> dict:
        segs = []
        for _ in range(self.rng.randint(2, 5)):
            segs += self._filler_pair()
        return {"kind": "smalltalk", "segs": segs,
                "info": {"truths": [], "ages": []}}

    def _conv_recall(self) -> dict:
        segs, facts = [], []
        used_slots, used_vals = set(), set()
        fact_seg = {}                    # slot -> index of its (latest) seg
        for _ in range(self.rng.randint(*self.n_facts)):
            f = self._sample_fact(used_slots, used_vals)
            used_slots.add(f["slot"]); used_vals.add(f["v"])
            fact_seg[f["slot"]] = len(segs)
            segs.append(self._user_valued(self.rng.choice(f["st"])
                                          .format(v=f["v"], p=f["p"]), f["v"]))
            if self.rng.random() < self.p_ack:
                segs.append(self._assistant(self.rng.choice(ACK_TMPL)
                                            .format(v=f["v"])))
            facts.append(f)

        # optional supersession: restate one fact with a NEW value
        updated = None
        if self.rng.random() < self.p_update:
            f = self.rng.choice(facts)
            nv = self.rng.choice([x for x in f["pool"] if x not in used_vals])
            used_vals.add(nv)
            f["old_v"], f["v"], updated = f["v"], nv, f["slot"]
            fact_seg[f["slot"]] = len(segs)
            segs.append(self._user_valued(self.rng.choice(f["upd"])
                                          .format(v=f["v"], p=f["p"]), f["v"]))

        beyond = self.rng.random() < self.p_beyond
        lo, hi = BEYOND_BIN if beyond else self.rng.choice(FILLER_BINS)
        for _ in range(self.rng.randint(lo, hi)):
            segs += self._filler_pair()

        queries, truths, ages, q_slots = [], [], [], []
        for f in self.rng.sample(facts, min(len(facts),
                                            self.rng.randint(*self.n_queries))):
            q = self.rng.choice(f["qs"]).format(p=f["p"])
            queries.append(q)
            truths.append(f["v"])
            q_slots.append(f["slot"])
            segs.append(self._user(q))
            # age = writes between the fact seg and this answer's decode
            ages.append(len(segs) - fact_seg[f["slot"]])
            # réponse de rappel : valeur hors contexte (le fait est dans un seg
            # passé, pont = banque seule) => span valeur surpondéré (run 7)
            segs.append(self._assistant_valued(
                self.rng.choice(f["ans"]).format(v=f["v"], p=f["p"]), f["v"]))
        return {"kind": "recall", "segs": segs,
                "info": {"truths": truths, "queries": queries, "ages": ages,
                         "q_slots": q_slots, "updated": updated,
                         "beyond": beyond,
                         "old_v": next((f["old_v"] for f in facts
                                        if f["slot"] == updated), None),
                         "facts": [(f["slot"], f["v"]) for f in facts]}}

    def next_conv(self) -> dict:
        if self.rng.random() < self.p_smalltalk:
            return self._conv_smalltalk()
        return self._conv_recall()


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
    assert grade_recall(["Your dog is named Biscuit."], ["Biscuit"]) == 1.0
    assert grade_recall(["your dog is named biscuit!"], ["Biscuit"]) == 1.0
    assert grade_recall(["I have no idea."], ["Biscuit"]) == 0.0
    assert grade_recall(["Biscuits are tasty."], ["Biscuit"]) == 0.0  # boundary

    tok = _StubTok()
    fake_rows = [{"messages": [
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "y" * 1000},
        {"role": "assistant", "content": "ok"},
        {"role": "assistant", "content": "orphan"},
    ]}]
    assert extract_filler_pairs(fake_rows, tok, max_tok=20) == \
        [("Hi there", "Hello!")]

    ps = PersonaChatStream(tok, seed=0)
    from collections import Counter
    kinds, ages = Counter(), Counter()
    n_updated = n_beyond = 0

    for _ in range(600):
        c = ps.next_conv()
        kinds[c["kind"]] += 1

        for s in c["segs"]:
            if s["role"] == "user":
                assert s["loss_mask"].sum() == 0
            else:
                n_open = len(A_OPEN)
                assert s["loss_mask"][0, :n_open].sum() == 0
                assert s["loss_mask"][0, n_open:].min() == 1

        # canonical texts grade 1.0 for every kind; wrong answers grade 0
        assert grade_conv(c, _assistant_texts(tok, c)) == 1.0
        info = c["info"]
        if c["kind"] == "recall":
            nt = len(info["truths"])
            assert nt >= 1 and len(info["ages"]) == nt
            assert grade_conv(c, ["no idea"] * len(_assistant_texts(tok, c))) \
                == 0.0
            for a in info["ages"]:
                ages[1 << max(0, a.bit_length() - 1)] += 1
                assert a >= 1
            if info["updated"] is not None:
                n_updated += 1
                # a query about the updated slot expects the NEW value only
                if info["updated"] in info["q_slots"]:
                    i = info["q_slots"].index(info["updated"])
                    new_v = dict(info["facts"])[info["updated"]]
                    assert info["truths"][i] == new_v != info["old_v"]
                    assert info["old_v"] not in info["truths"]
            n_beyond += info["beyond"]
            # values never collide across facts of one conversation
            vals = [v for _, v in info["facts"]]
            assert len(vals) == len(set(vals))

    assert kinds["smalltalk"] > 60 and kinds["recall"] > 300, kinds
    assert n_updated > 20 and n_beyond > 30
    assert max(ages) >= 16, f"no beyond-FIFO stratum sampled: {dict(ages)}"
    print(f"persona_chat self-test: OK ({dict(kinds)}, "
          f"age octaves {dict(sorted(ages.items()))}, "
          f"updated {n_updated}, beyond {n_beyond})")


# ── real-tokenizer smoke ─────────────────────────────────────────────────────

def _show(tok, conv, max_seg_chars=160):
    info = conv["info"]
    print(f"\n===== kind={conv['kind']} truths={info['truths']} "
          f"ages={info['ages']} segs={len(conv['segs'])} =====")
    for s in conv["segs"]:
        txt = tok.decode(s["input_ids"][0].tolist())
        sup = int(s["loss_mask"].sum().item())
        head = txt[:max_seg_chars].replace("\n", "\\n")
        print(f"  [{s['role']:9s} T={s['input_ids'].numel():4d} "
              f"loss_on={sup:3d}] {head}{'…' if len(txt) > max_seg_chars else ''}")


def main() -> None:
    if len(sys.argv) < 2:
        _self_test()
        return
    import yaml
    from transformers import AutoTokenizer
    raw = yaml.safe_load(open(sys.argv[1]))
    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    kw = {}
    if "--real" in sys.argv:
        kw = dict(real_filler="HuggingFaceTB/smol-smoltalk",
                  real_cache_dir="/mnt/tb/data_cache")
    ps = PersonaChatStream(tok, seed=0, **kw)
    for want in ("smalltalk", "recall"):
        for _ in range(100):
            c = ps.next_conv()
            if c["kind"] == want:
                _show(tok, c); break
    from collections import Counter
    kinds, toks, ages = Counter(), [], Counter()
    for _ in range(300):
        c = ps.next_conv()
        kinds[c["kind"]] += 1
        toks.append(sum(s["input_ids"].numel() for s in c["segs"]))
        for a in c["info"]["ages"]:
            ages[1 << max(0, a.bit_length() - 1)] += 1
    print(f"\nmix over 300 convs: {dict(kinds)}")
    print(f"recall age octaves (lower edge): {dict(sorted(ages.items()))}")
    print(f"tokens/conv: min {min(toks)} med {sorted(toks)[len(toks)//2]} "
          f"max {max(toks)}")


if __name__ == "__main__":
    main()
