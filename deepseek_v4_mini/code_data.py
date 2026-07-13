"""Per-file code-chunk stream for the bank-as-long-context experiment (dsv6).

Design (memory dsv6-bank-code-memory-defer): the bank is a cross-chunk memory,
tested on REAL within-file continuation. A conversation = up to K consecutive
chunks of a code file, so chunk i+1 genuinely continues chunk i (no cross-file
packing — that would break semantic continuity). One chunk = one forward = one
bank write; a blank <think> forward after chunk i must predict the start of
chunk i+1 from the bank alone.

Chunking (batch=1 regime): a file of len(t) tokens is cut into ceil(len(t)/L)
consecutive chunks — the FULL chunks of exactly L tokens PLUS the remainder as a
final short chunk (1..L tokens). No minimum file size by default (min_chunks=1);
web-text datasets should set min_chunks=2 so single-chunk docs (no deferred
target) don't dilute the deferred loss. Chunks are ragged, so batch=1 is the
reference regime: no padding, no attention mask, no FIFO corruption.

BATCHED training (batch>1, next_conv_batch): B same-depth conversations windowed
over FULL chunks only — every turn is a uniform [B, L] tensor, so no padding or
mask is ever needed and the bank batches natively. The ragged tail chunk is
excluded as an input in this mode (it stays available to the batch=1 eval paths);
depth diversity is preserved by drawing m per batch from an anchor file with the
same rule as next_conv.

MULTI-SOURCE MIX (pretraining-diversity phase): pass `sources=[{dataset:..,
weight:.., ...}, ..]` — each source is built (and disk-cached) separately, conv
sampling draws the SOURCE by weight then a file uniformly within it, so the mix
ratio is exact regardless of per-source file counts. `source_stream(i)` returns a
single-source view (same files, own rng) for per-domain eval.

Source datasets are STREAMED and materialized. Deterministic; train/held are
disjoint (every 10th kept file -> held).
"""
from __future__ import annotations

import hashlib
import os
import random
import sys
import time
import torch


def _load_source(tokenizer, *, split: str, seq_len: int, n_files: int,
                 dataset: str, data_dir: str = "", config_name: str = "",
                 content_key: str = "content", min_chunks: int = 1,
                 stream_skip: int = 0, max_chunks_per_file: int = 12,
                 stream_cap: int = 60000, cache_dir: str = "data_cache"
                 ) -> list[list[torch.Tensor]]:
    """Build (or load from disk cache) ONE source's ragged chunk lists."""
    L = int(seq_len)
    label = f"{split}:{dataset.split('/')[-1]}{'/' + config_name if config_name else ''}"

    # ── Tokenized-corpus disk cache: the streamed build is deterministic given
    # these knobs, so cache it (pod restarts / re-runs skip the 10-25 min pass).
    cache_path = None
    if cache_dir:
        key = "|".join(str(v) for v in (
            dataset, data_dir, split, L, max_chunks_per_file, n_files, stream_cap,
            content_key, config_name, min_chunks, stream_skip,
            getattr(tokenizer, "name_or_path", "?"), len(tokenizer)))
        h = hashlib.md5(key.encode()).hexdigest()[:16]
        cache_path = os.path.join(cache_dir, f"chunks_{split}_{h}.pt")
        if os.path.exists(cache_path):
            files = torch.load(cache_path)
            print(f"tokenize[{label}]: cache hit {cache_path} — {len(files)} files / "
                  f"{sum(len(f) for f in files)} chunks", flush=True)
            return files

    from datasets import load_dataset
    kw = {}
    if data_dir: kw["data_dir"] = data_dir
    if config_name: kw["name"] = config_name            # HF *config* (e.g. fineweb dumps)
    ds = load_dataset(dataset, split="train", streaming=True, **kw)
    if stream_skip:
        ds = ds.skip(int(stream_skip))                  # per-pod shard offset

    files: list[list[torch.Tensor]] = []                # each = list of 1-D chunk tensors
    kept = seen = 0
    # progress: live tqdm bar on a tty, periodic one-line prints in a log file
    # (nohup > train.log — \r bars would spam the log).
    pbar = None
    if sys.stderr.isatty():
        try:
            from tqdm import tqdm
            pbar = tqdm(total=stream_cap, desc=f"tokenize[{label}]", unit="file")
        except ImportError:
            pass
    t_start = last_log = time.time()
    for ex in ds:
        seen += 1
        if pbar is not None:
            pbar.update(1)
        elif time.time() - last_log >= 30.0:
            last_log = time.time()
            print(f"tokenize[{label}]: seen {seen}/{stream_cap} | kept "
                  f"{len(files)}/{n_files} files | {sum(len(f) for f in files)} chunks | "
                  f"{seen / max(last_log - t_start, 1e-9):.0f} files/s", flush=True)
        if seen > stream_cap:
            break
        content = ex.get(content_key) or ex.get("text") or ex.get("content")
        if not content:
            continue
        # min_chunks: web-text datasets (fineweb p50 ~400 tok) are dominated by
        # single-chunk docs that yield NO deferred target; filter them so the
        # deferred loss keeps its density. Cheap char pre-filter (~3 chars/token,
        # conservative) avoids tokenizing the mass of obviously-short docs.
        if min_chunks > 1 and len(content) < 3.0 * L * (min_chunks - 1):
            continue
        t = tokenizer.encode(content, add_special_tokens=False)
        if len(t) < 1 or len(t) <= L * (min_chunks - 1):
            continue
        # ceil division: full L-chunks + the remainder as a final short chunk,
        # capped at max_chunks_per_file.
        nc = min(-(-len(t) // L), max_chunks_per_file)
        is_held = (kept % 10 == 0)
        kept += 1
        if is_held != (split == "held"):
            continue
        tt = torch.tensor(t, dtype=torch.long)
        files.append([tt[j * L: (j + 1) * L] for j in range(nc)])  # last is ragged
        if len(files) >= n_files:
            break
    if pbar is not None:
        pbar.close()
    print(f"tokenize[{label}]: done in {time.time() - t_start:.0f}s — seen {seen}, "
          f"kept {len(files)} files / {sum(len(f) for f in files)} chunks", flush=True)
    assert len(files) >= 1, f"no files kept ({label}, seen={seen})"
    if cache_path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = cache_path + ".tmp"
        torch.save(files, tmp); os.replace(tmp, cache_path)  # atomic vs preemption
        print(f"tokenize[{label}]: cached → {cache_path}", flush=True)
    return files


def file_label_ids(tok, f, n_digits=6):
    """Stable synthetic address for a file (idea G2, 2026-07-12): a determinist
    arithmetic hash of its opening tokens rendered as '<<FILE:483920>>' token
    ids. Same file => same label across convs, processes and probes; synthetic
    (not the real path) so addressing is measured without semantic leakage."""
    ts = f[0][:16].tolist()
    h = sum((i + 1) * int(t) for i, t in enumerate(ts)) % 10 ** n_digits
    return torch.tensor(tok(f"<<FILE:{h:0{n_digits}d}>>")["input_ids"],
                        dtype=torch.long)


def docstring_pairs(text, tok, min_doc=8, min_body=32, max_pairs=4):
    """B3 (backlog 2026-07-13) : paires (docstring, corps) extraites d'un source
    Python — niveau REGEX (le corps court jusqu'au prochain def/class en début
    de ligne, les méthodes imbriquées débordent dedans ; docstrings triple-quote
    seulement). Suffisant pour la probe cross-modale docstring↔code ; le mode
    d'entraînement (defer corps depuis write doc-only) réutilisera ce helper si
    le zero-shot montre un signal. Retourne [(doc_ids, body_ids)] tokenisés."""
    import re
    out = []
    pat = re.compile(
        r"def\s+\w+\s*\([^)]*\)[^:\n]*:\s*\n\s+[rbuRBU]*(\"\"\"|''')(.*?)\1[ \t]*\n"
        r"(.*?)(?=\ndef\s|\nclass\s|\Z)", re.DOTALL)
    for m in pat.finditer(text):
        doc, body = m.group(2).strip(), m.group(3).strip("\n")
        if not doc or not body.strip():
            continue
        di = tok.encode(doc, add_special_tokens=False)
        bi = tok.encode(body, add_special_tokens=False)
        if len(di) >= min_doc and len(bi) >= min_body:
            out.append((torch.tensor(di, dtype=torch.long),
                        torch.tensor(bi, dtype=torch.long)))
        if len(out) >= max_pairs:
            break
    return out


class CodeChunkStream:
    def __init__(self, tokenizer, *, seq_len: int = 2048, chunks_per_conv: int = 3,
                 batch: int = 1, n_files: int = 800, split: str = "train",
                 seed: int = 0, dataset: str = "codeparrot/codeparrot-clean-valid",
                 data_dir: str = "", max_chunks_per_file: int = 12,
                 stream_cap: int = 60000, cache_dir: str = "data_cache",
                 content_key: str = "content", config_name: str = "",
                 min_chunks: int = 1, stream_skip: int = 0,
                 sources: list[dict] | None = None,
                 var_chunk: list | tuple | None = None) -> None:
        self.tok = tokenizer
        self.L = int(seq_len); self.K = int(chunks_per_conv); self.B = int(batch)
        self.rng = random.Random(seed + (0 if split == "train" else 101))
        # var_chunk=[lo, hi]: VARIABLE chunk lengths ~ U[lo, hi], re-cut at sampling
        # time from the cached fixed-L slices (contiguous, so cat() reconstructs the
        # token stream — the tokenized cache is untouched). Breaks the fixed-512
        # write positions (anti positional-shortcut; RL prerequisite). batch=1 only.
        self.var_chunk = tuple(int(v) for v in var_chunk) if var_chunk else None
        if self.var_chunk:
            lo, hi = self.var_chunk
            assert self.B == 1, "var_chunk: ragged variable chunks require batch=1"
            assert 1 <= lo <= hi <= self.L, f"var_chunk {self.var_chunk} vs seq_len {self.L}"

        common = dict(split=split, seq_len=self.L,
                      max_chunks_per_file=max_chunks_per_file, cache_dir=cache_dir)
        if sources:
            # weighted mix: n_files is the TOTAL budget, split by weight unless a
            # source pins its own n_files; weight also drives conv sampling below.
            tw = sum(float(s.get("weight", 1.0)) for s in sources)
            self.src_files: list[list] = []
            self.src_weights: list[float] = []
            self.src_names: list[str] = []
            for s in sources:
                w = float(s.get("weight", 1.0))
                nf = int(s.get("n_files", round(n_files * w / tw)))
                self.src_files.append(_load_source(
                    tokenizer, n_files=nf, dataset=s["dataset"],
                    data_dir=s.get("data_dir", ""), config_name=s.get("config_name", ""),
                    content_key=s.get("content_key", "content"),
                    min_chunks=int(s.get("min_chunks", 1)),
                    stream_skip=int(s.get("stream_skip", 0)),
                    stream_cap=int(s.get("stream_cap", stream_cap)), **common))
                self.src_weights.append(w)
                self.src_names.append(s.get("name") or s["dataset"].split("/")[-1])
        else:
            fl = _load_source(tokenizer, n_files=n_files, dataset=dataset,
                              data_dir=data_dir, config_name=config_name,
                              content_key=content_key, min_chunks=min_chunks,
                              stream_skip=stream_skip, stream_cap=stream_cap, **common)
            self.src_files = [fl]; self.src_weights = [1.0]
            self.src_names = [dataset.split("/")[-1]]
        self._wsum = sum(self.src_weights)
        self.files = [f for fl in self.src_files for f in fl]   # global view (eval)
        self.n_files = len(self.files)
        self.n_chunk = sum(len(f) for f in self.files)
        if len(self.src_files) > 1:
            mix = " + ".join(f"{n} {len(fl)}f/{sum(len(f) for f in fl)}c (w={w:g})"
                             for n, fl, w in zip(self.src_names, self.src_files, self.src_weights))
            print(f"mix[{split}]: {mix}", flush=True)

        # ── Batched-conv index (B>1): windows over FULL chunks only (uniform [B, L]
        # shapes, no padding/mask — the ragged tail chunk is excluded as an INPUT but
        # stays in self.files for the batch=1 eval paths). nfull = #leading full chunks.
        if self.B > 1:
            self.src_by_depth: list[dict[int, list]] = []
            for fl in self.src_files:
                by_d: dict[int, list] = {}
                for f in fl:
                    nfull = len(f) if f[-1].numel() == self.L else len(f) - 1
                    if len(f) >= 2 and nfull >= 1:
                        # m=1: single full-chunk write + EXTERNAL defer target (the
                        # successor chunk, possibly the ragged tail) — this is how
                        # >L-token-but-<2L files (1 full + ragged) stay trainable.
                        by_d.setdefault(1, []).append((f, nfull))
                    for m in range(2, min(self.K, nfull) + 1):
                        by_d.setdefault(m, []).append((f, nfull))
                self.src_by_depth.append(by_d)
            n_ok = sum(len(by_d.get(1, [])) for by_d in self.src_by_depth)
            assert n_ok >= self.B, \
                f"batch={self.B} but only {n_ok} files with a defer pair (split={split})"

    def source_stream(self, i: int) -> "CodeChunkStream":
        """Single-source VIEW (shared file lists, own rng) — per-domain eval."""
        v = object.__new__(CodeChunkStream)
        v.tok = self.tok; v.L = self.L; v.K = self.K; v.B = 1
        v.rng = random.Random(self.rng.randrange(1 << 30))
        v.B = 1                                         # views serve batch=1 eval paths
        v.src_files = [self.src_files[i]]; v.src_weights = [1.0]
        v.src_names = [self.src_names[i]]; v._wsum = 1.0
        v.var_chunk = self.var_chunk
        v.files = self.src_files[i]
        v.n_files = len(v.files); v.n_chunk = sum(len(f) for f in v.files)
        return v

    def _pick_file(self) -> list[torch.Tensor]:
        """Draw the SOURCE by weight, then a file uniformly within it."""
        fl = self.src_files[self._pick_source()]
        return fl[self.rng.randrange(len(fl))]

    def _reslice(self, f: list[torch.Tensor]) -> list[torch.Tensor]:
        """Variable-length re-cut of a cached file: chunk lengths ~ U[lo, hi].
        Cached chunks are contiguous slices, so cat() reconstructs the (capped)
        token stream; only the boundaries change, never the content."""
        lo, hi = self.var_chunk
        t = torch.cat(f)
        out, i = [], 0
        while i < t.numel():
            n = self.rng.randint(lo, hi)
            out.append(t[i:i + n]); i += n
        return out

    def next_conv(self) -> list[dict]:
        """A random-DEPTH window of consecutive chunks from ONE file (batch=1). The
        conversation length m is SAMPLED per call in [2, min(K, nc)] (or 1 when the
        file has a single chunk) so the bank is trained across a spread of horizons
        (1-hop … K-hop), not always the deepest. A random start offset is then drawn.
        Each seg: {input_ids [1, Lj], attention_mask [1, Lj]} — Lj varies (last ragged)."""
        f = self._pick_file()
        if self.var_chunk:
            f = self._reslice(f)
        nc = len(f)
        hi = min(self.K, nc)
        m = hi if hi < 2 else self.rng.randint(2, hi)  # randint inclusive => depth in [2, hi]
        st = self.rng.randrange(0, nc - m + 1)         # nc-m >= 0
        segs = []
        for j in range(st, st + m):
            ids = f[j].unsqueeze(0)                    # [1, Lj]
            segs.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
        return segs

    def next_conv_batch(self, defer_len: int = 16) -> list[dict]:
        """B conversations of the SAME depth m, batched turn-wise: list of m segs,
        each {input_ids [B, L], defer_tgt [B, defer_len]} — all INPUT chunks FULL
        (no padding, no mask, the bank and every trainer op batch natively).

        defer_tgt[i] = first defer_len tokens of each element's NEXT chunk. For
        i < m-1 that successor is a full in-window chunk; for the LAST turn it is
        the chunk just after the window when it exists — including the RAGGED tail
        (so >L-token files keep their final defer pair even though the ragged chunk
        is never an input). Missing/short targets are padded with -100 (CE ignore).

        Depth-diversity is preserved: m is drawn per BATCH from an ANCHOR file with
        the next_conv rule (m ~ U[2, min(K, nfull)]; m=1 when the anchor has a
        single full chunk + tail). Sources are drawn per element by weight (exact
        mix); a source with no file deep enough for this m is renormalized away."""
        assert self.B > 1, "next_conv_batch requires batch>1 (use next_conv)"
        dl = int(defer_len)
        # anchor: source by weight, then any file with a defer pair (by_depth[1])
        while True:
            si = self._pick_source()
            cands = self.src_by_depth[si].get(1)
            if cands:
                break
        f_a, nfull_a = cands[self.rng.randrange(len(cands))]
        m = self.rng.randint(2, min(self.K, nfull_a)) if nfull_a >= 2 else 1
        # fill the batch: anchor + B-1 files admitting depth m (weighted sources)
        picks = [(f_a, nfull_a)]
        ok = [i for i in range(len(self.src_files)) if self.src_by_depth[i].get(m)]
        wk = [self.src_weights[i] for i in ok]; wsum = sum(wk)
        while len(picks) < self.B:
            r = self.rng.random() * wsum
            si = ok[-1]
            for i, w in zip(ok, wk):
                r -= w
                if r <= 0:
                    si = i; break
            cl = self.src_by_depth[si][m]
            picks.append(cl[self.rng.randrange(len(cl))])
        starts = [self.rng.randrange(0, nf - m + 1) for _, nf in picks]
        segs = []
        for j in range(m):
            ids = torch.stack([f[st + j] for (f, _), st in zip(picks, starts)])  # [B, L]
            tgt = torch.full((self.B, dl), -100, dtype=torch.long)
            for b, ((f, _), st) in enumerate(zip(picks, starts)):
                if st + j + 1 < len(f):                            # successor exists
                    nx = f[st + j + 1][:dl]                        # ragged tail ok
                    tgt[b, :nx.numel()] = nx
            segs.append({"input_ids": ids, "attention_mask": torch.ones_like(ids),
                         "defer_tgt": tgt})
        return segs

    def next_conv_interleaved(self, n_streams: int, defer_len: int = 16,
                              label: bool = False, addr_prob: float = 0.0,
                              addr_cue_len: int = 16, addr_max: int = 2) -> list[dict]:
        """Idea G (2026-07-11): ONE bank lifetime holds n_streams files whose chunks
        are randomly interleaved (within-file order preserved) — school-style spaced
        practice instead of sequential no-reset chains. The total chunk budget matches
        next_conv (m ~ U[2, K]) split across the streams, so VRAM/compute are unchanged
        vs v2c; only the STRUCTURE differs. Each seg carries its own defer_tgt (the
        SAME file's successor chunk, -100-padded) because the next seg in the flat
        list usually belongs to another file. Kills the no-reset boundary confound:
        an off-topic LAST write no longer implies the current thread is dead — the
        read must select by content, not by a recency/boundary heuristic.
        n_streams: int = fixed F; (lo, hi) = F ~ U[lo, hi] SAMPLED per conv (like
        depth): some convs are 2 deep subjects, some are hi brief ones. F is then
        capped by m_total, so shallow convs stay naturally less fragmented.

        G2 extensions (2026-07-12, both OFF by default):
          label      prepend the file's stable synthetic address (file_label_ids)
                     to every chunk at WRITE time — the gist must encode it.
          addr_prob  after a seg, with this probability attach an ADDRESSED defer
                     toward a random OTHER live stream: cue = that stream's label
                     (50%) or the raw opening of its last written chunk (50%),
                     target = its next chunk's opening. The cue identifies the
                     thread but sits ~a chunk away from the target, so the gist
                     is the only bridge — trains content/label-addressed reads
                     that the blank defer (recency convention) never exercises."""
        assert self.B == 1, "interleave: ragged variable-depth streams require batch=1"
        dl = int(defer_len)
        m_total = self.rng.randint(2, self.K)
        if isinstance(n_streams, (list, tuple)):
            n_streams = self.rng.randint(int(n_streams[0]), int(n_streams[1]))
        F = min(int(n_streams), m_total)
        cuts = sorted(self.rng.sample(range(1, m_total), F - 1)) if F > 1 else []
        parts = [b - a for a, b in zip([0] + cuts, cuts + [m_total])]
        streams: list[list[dict]] = []
        files: list[list[torch.Tensor]] = []
        lbl_ids: list[torch.Tensor | None] = []
        for m in parts:
            f = self._pick_file()
            if self.var_chunk:
                f = self._reslice(f)
            m = min(m, len(f))
            st = self.rng.randrange(0, len(f) - m + 1)
            lb = file_label_ids(self.tok, f) if label else None
            q = []
            for j in range(st, st + m):
                ids = f[j].unsqueeze(0)                     # [1, Lj]
                if lb is not None:                          # write carries its address
                    ids = torch.cat([lb.unsqueeze(0), ids], dim=1)
                tgt = torch.full((1, dl), -100, dtype=torch.long)
                if j + 1 < len(f):                          # same-file successor
                    nx = f[j + 1][:dl]                      # (ragged tail ok)
                    tgt[0, :nx.numel()] = nx
                q.append({"input_ids": ids, "attention_mask": torch.ones_like(ids),
                          "defer_tgt": tgt, "_j": j})
            streams.append(q); files.append(f); lbl_ids.append(lb)
        order = [i for i, q in enumerate(streams) for _ in q]
        self.rng.shuffle(order)                             # uniform random merge
        segs, last_j, elig = [], {}, []                     # sid -> last written j
        for sid in order:
            seg = streams[sid].pop(0)
            last_j[sid] = seg.pop("_j")
            # eligible: some OTHER live stream still has a successor here
            cands = [s for s, j in last_j.items()
                     if s != sid and j + 1 < len(files[s])]
            if cands:
                elig.append((len(segs), list(cands), dict(last_j)))
            segs.append(seg)
        # addressed defers: prob-gated then CAPPED at addr_max per conv (each
        # extra forward pays a full fast-weight-read graph until the conv's
        # backward — uncapped p=0.5 OOMs the 8 GB rigs, post-mortem 2026-07-12);
        # sampled over ALL eligible positions so late/full banks stay represented
        if addr_prob > 0 and elig:
            picked = [e for e in elig if self.rng.random() < addr_prob]
            if len(picked) > addr_max:
                picked = self.rng.sample(picked, addr_max)
            for pos, cands, lj in picked:
                t = self.rng.choice(cands)
                f_t, j_t = files[t], lj[t]
                if lbl_ids[t] is not None and self.rng.random() < 0.5:
                    cue = lbl_ids[t]                        # explicit address
                else:
                    cue = f_t[j_t][:addr_cue_len]           # semantic address (raw)
                at = torch.full((1, dl), -100, dtype=torch.long)
                nx = f_t[j_t + 1][:dl]
                at[0, :nx.numel()] = nx
                segs[pos]["addr_cue"] = cue.unsqueeze(0)
                segs[pos]["addr_tgt"] = at
        return segs

    def _pick_source(self) -> int:
        if len(self.src_files) == 1:
            return 0
        r = self.rng.random() * self._wsum
        for i, w in enumerate(self.src_weights):
            r -= w
            if r <= 0:
                return i
        return len(self.src_files) - 1

    def conv_at_depth(self, n_chunks: int):
        """A window of EXACTLY n_chunks consecutive chunks from a random file with
        nc >= n_chunks (None if no such file). Same seg format as next_conv. Used by
        the depth-stratified eval to control conversation depth instead of sampling it."""
        if self.var_chunk:
            # conservative token filter: total >= n_chunks*hi guarantees the re-cut
            # yields at least n_chunks chunks whatever lengths the rng draws.
            hi = self.var_chunk[1]
            cands = [f for f in self.files
                     if sum(c.numel() for c in f) >= n_chunks * hi]
            if not cands:
                return None
            f = self._reslice(cands[self.rng.randrange(len(cands))])
        else:
            cands = [f for f in self.files if len(f) >= n_chunks]
            if not cands:
                return None
            f = cands[self.rng.randrange(len(cands))]
        st = self.rng.randrange(0, len(f) - n_chunks + 1)
        segs = []
        for j in range(st, st + n_chunks):
            ids = f[j].unsqueeze(0)
            segs.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
        return segs
