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


class CodeChunkStream:
    def __init__(self, tokenizer, *, seq_len: int = 2048, chunks_per_conv: int = 3,
                 batch: int = 1, n_files: int = 800, split: str = "train",
                 seed: int = 0, dataset: str = "codeparrot/codeparrot-clean-valid",
                 data_dir: str = "", max_chunks_per_file: int = 12,
                 stream_cap: int = 60000, cache_dir: str = "data_cache",
                 content_key: str = "content", config_name: str = "",
                 min_chunks: int = 1, stream_skip: int = 0,
                 sources: list[dict] | None = None) -> None:
        self.tok = tokenizer
        self.L = int(seq_len); self.K = int(chunks_per_conv); self.B = int(batch)
        self.rng = random.Random(seed + (0 if split == "train" else 101))

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
        v.files = self.src_files[i]
        v.n_files = len(v.files); v.n_chunk = sum(len(f) for f in v.files)
        return v

    def _pick_file(self) -> list[torch.Tensor]:
        """Draw the SOURCE by weight, then a file uniformly within it."""
        fl = self.src_files[self._pick_source()]
        return fl[self.rng.randrange(len(fl))]

    def next_conv(self) -> list[dict]:
        """A random-DEPTH window of consecutive chunks from ONE file (batch=1). The
        conversation length m is SAMPLED per call in [2, min(K, nc)] (or 1 when the
        file has a single chunk) so the bank is trained across a spread of horizons
        (1-hop … K-hop), not always the deepest. A random start offset is then drawn.
        Each seg: {input_ids [1, Lj], attention_mask [1, Lj]} — Lj varies (last ragged)."""
        f = self._pick_file()
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
