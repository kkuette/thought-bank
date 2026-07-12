"""dsv6 — inference-time bank probes on a code_defer_native checkpoint.

Five zero-training probes that together characterize WHAT the bank stores and
HOW the read consumes it, on real held-out data (per source when the config is
a weighted mix). All report the deferred CE of the SAME target — the opening
`defer_len` tokens of a file's 4th chunk — under controlled write sequences,
paired per file:

  swap        specificity: own bank vs another same-domain file's bank vs a
              cross-domain bank vs reset. Splits the GAP into file-SPECIFIC
              content vs generic register; xdom measures blind trust in the bank.
  dup         duplicate consecutive writes: (c1,c2,c2) and (c1,c1,c2) vs
              (c1,c2) and (c0,c1,c2) — is the write idempotent? does a
              duplicate evict useful history?
  distractor  one foreign chunk (same- or cross-domain) written mid-conversation
              or as the LAST write, vs a depth-matched control (c0,c1,c2,c2) —
              does the thread survive an interruption?
  order       (c2,c1,c0) and (c0,c2,c1) vs (c0,c1,c2) — recency weighting;
              plus the PURE-order condition (c1,c0,c2): old writes permuted,
              last write fixed — any delta is order encoding BEYOND recency.
  eviction    recall-by-lag: write a 12-chunk file (max_mem=8 => the first
              gists are physically evicted), then decode the opening of chunk
              j for lag 1..10 — retention profile across the eviction boundary.
              Note: recalling OLD openings is off-task for this model (it is
              only trained to predict the chunk after the LAST write), so read
              the lag PROFILE, not the absolute level.

Three later probes (2026-07-10) extend the battery toward the "more than a
window" claims — store-both-and-select, bank-as-workspace, gist-as-abstraction:

  cohab       TWO files A and B share the bank (interleaved a0,b0,a1,b1,a2,b2
              and blocked a0,a1,a2,b0,b1,b2) vs their mono banks; the SAME
              superposed bank is decoded against A's target AND B's target.
              Cohabitation cost per file + recency asymmetry (B always ends).
  reflect     k consecutive THOUGHT turns (blank forwards whose bank write is
              carried, exactly the deferred format) inserted between the writes
              and the final decode, k=0..3. CE(k) dropping below CE(0) = the
              bank serves as an iterative workspace on real data (dsv5f's cell,
              production conditions). Each thought evicts one oldest slot.
  invar       does the gist store the DEFINITION or the SURFACE? (a) reseg:
              the same c0..c2 tokens re-cut at shifted boundaries (offset 128,
              same content minus the first 128 tokens, farthest from target);
              (b) rename (code source only): identifiers consistently renamed
              qz0..qzN via regex, then re-tokenized — semantics ~preserved,
              token surface destroyed. Score each against the own-bank floor
              and the swap (different file) ceiling: invariance = how far the
              perturbed bank stays from "just another file's bank".
              Caveats: reseg chunks are off-512 (OOD length for fixed-L ckpts;
              clean on varlen ckpts); rename is regex-level (strings/attributes
              get renamed too — surface perturbation, not compiler-grade).

Usage (repo root):
    PYTHONPATH=. python deepseek_v4_mini/analysis/code_defer_bank_probes.py \
        deepseek_v4_mini/configs/code_defer_native_v2b_mix.yaml \
        checkpoints/code_defer_native_v2b_mix/final.pt [--probes swap,dup,...]

Stream seeds are fixed per probe (same file sampling as the published numbers).
"""
import argparse
import random
import statistics as st

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer

from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.model import ThoughtBankLM
from deepseek_v4_mini.code_data import CodeChunkStream

DL = 16  # defer_len used for all targets


def _load(cfg_path, ckpt_path, dev):
    raw = yaml.safe_load(open(cfg_path))
    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    for t in ("<think>", "<blank>"):
        if t not in tok.get_vocab():
            tok.add_special_tokens({"additional_special_tokens": [t]})
    ck = torch.load(ckpt_path, map_location="cpu")
    model = ThoughtBankLM(ThoughtBankConfig(**ck["cfg"])).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {ckpt_path} @step {ck.get('step', '?')}")
    return raw, tok, model


def _stream(raw, tok, seed):
    d = raw["data"]
    sd = dict(seq_len=int(d["seq_len"]), chunks_per_conv=int(d["chunks_per_conv"]),
              batch=1, n_files=int(d.get("n_files", 1500)),
              dataset=d.get("dataset", "codeparrot/codeparrot-clean-valid"),
              data_dir=d.get("data_dir", ""),
              stream_cap=int(d.get("stream_cap", 40000)),
              cache_dir=d.get("cache_dir", "data_cache"),
              content_key=d.get("content_key", "content"),
              config_name=d.get("config_name", ""),
              min_chunks=int(d.get("min_chunks", 1)),
              stream_skip=int(d.get("stream_skip", 0)),
              sources=d.get("sources"), seed=seed)
    return CodeChunkStream(tok, split="held", **sd)


def _mk_ops(model, tok, dev):
    think_id = tok.convert_tokens_to_ids("<think>")
    blank_id = tok.convert_tokens_to_ids("<blank>")

    def write_seq(chunks):
        bank = None
        for c in chunks:
            x = c.unsqueeze(0).to(dev)
            xt = torch.cat([x, torch.full((1, 1), think_id, dtype=torch.long, device=dev)], 1)
            with torch.no_grad():
                bank = model(xt, init_mem=bank)["mem_bank"]
        return bank

    def defer_ce(bank, gt):
        di = torch.full((1, DL), blank_id, dtype=torch.long, device=dev)
        with torch.no_grad():
            lg = model(di, init_mem=bank)["logits"].float()
        return float(F.cross_entropy(lg.reshape(-1, lg.size(-1)), gt.reshape(-1)))

    return write_seq, defer_ce


def _pools(stream, min_len=4):
    """Per-source shuffled file pools (files long enough for a c3 target)."""
    rng = random.Random(0)
    pools = []
    for si in range(len(stream.src_names)):
        vw = stream.source_stream(si)
        fs = [f for f in vw.files if len(f) >= min_len and f[3].numel() >= DL]
        rng.shuffle(fs)
        pools.append(fs)
    return pools


def _report(res, n, pairs):
    for k, v in res.items():
        print(f"  {k:>14}: CE {st.mean(v):6.3f}")
    for a, b, lab in pairs:
        dd = [x - y for x, y in zip(res[a], res[b])]
        md = st.mean(dd)
        se = st.stdev(dd) / n ** 0.5 if n > 1 else 0.0
        print(f"  d {lab}: {md:+.3f} +- {se:.3f}  |t|~{abs(md) / max(se, 1e-9):.1f}")


def probe_swap(raw, tok, model, dev, n_files):
    write_seq, defer_ce = _mk_ops(model, tok, dev)
    stream = _stream(raw, tok, seed=555)
    pools = _pools(stream)
    for si, nm in enumerate(stream.src_names):
        fs, other = pools[si], pools[1 - si] if len(pools) > 1 else pools[si]
        res = {k: [] for k in ("own", "swap", "xdom", "reset")}
        for i, f in enumerate(fs[:n_files]):
            g = fs[(i + 1) % min(len(fs), 200)]
            h = other[i % min(len(other), 200)]
            gt = f[3][:DL].unsqueeze(0).to(dev)
            res["own"].append(defer_ce(write_seq([f[0], f[1], f[2]]), gt))
            res["swap"].append(defer_ce(write_seq([g[0], g[1], g[2]]), gt))
            res["xdom"].append(defer_ce(write_seq([h[0], h[1], h[2]]), gt))
            res["reset"].append(defer_ce(None, gt))
        print(f"\n[{nm}] SWAP/specificity (n={len(res['own'])}, target = A's c3 opening)")
        _report(res, len(res["own"]), [
            ("swap", "own", "SPECIFIC (own vs same-domain swap)"),
            ("reset", "swap", "REGISTER (swap vs reset)"),
            ("reset", "xdom", "cross-domain (xdom vs reset)")])


def probe_dup(raw, tok, model, dev, n_files):
    write_seq, defer_ce = _mk_ops(model, tok, dev)
    stream = _stream(raw, tok, seed=321)
    pools = _pools(stream)
    for si, nm in enumerate(stream.src_names):
        res = {k: [] for k in ("normal3", "dup_last", "dup_mid", "ctrl2", "reset")}
        for f in pools[si][:n_files]:
            c0, c1, c2 = f[0], f[1], f[2]
            gt = f[3][:DL].unsqueeze(0).to(dev)
            res["normal3"].append(defer_ce(write_seq([c0, c1, c2]), gt))
            res["dup_last"].append(defer_ce(write_seq([c1, c2, c2]), gt))
            res["dup_mid"].append(defer_ce(write_seq([c1, c1, c2]), gt))
            res["ctrl2"].append(defer_ce(write_seq([c1, c2]), gt))
            res["reset"].append(defer_ce(None, gt))
        print(f"\n[{nm}] DUPLICATES (n={len(res['reset'])}, target = c3 opening)")
        _report(res, len(res["reset"]), [
            ("dup_last", "ctrl2", "dup-last vs ctrl-2 (does rewriting help?)"),
            ("dup_last", "normal3", "dup-last vs normal-3 (does it evict?)"),
            ("dup_mid", "ctrl2", "dup-mid vs ctrl-2")])


def probe_distractor(raw, tok, model, dev, n_files):
    write_seq, defer_ce = _mk_ops(model, tok, dev)
    stream = _stream(raw, tok, seed=777)
    pools = _pools(stream)
    for si, nm in enumerate(stream.src_names):
        fs, other = pools[si], pools[1 - si] if len(pools) > 1 else pools[si]
        res = {k: [] for k in ("ctrl3", "ctrl4_dup", "mid_same", "mid_xdom",
                               "last_same", "last_xdom", "reset")}
        for i, f in enumerate(fs[:n_files]):
            Ds = fs[(i + 1) % min(len(fs), 200)][0]
            Dx = other[i % min(len(other), 200)][0]
            a0, a1, a2 = f[0], f[1], f[2]
            gt = f[3][:DL].unsqueeze(0).to(dev)
            res["ctrl3"].append(defer_ce(write_seq([a0, a1, a2]), gt))
            res["ctrl4_dup"].append(defer_ce(write_seq([a0, a1, a2, a2]), gt))
            res["mid_same"].append(defer_ce(write_seq([a0, a1, Ds, a2]), gt))
            res["mid_xdom"].append(defer_ce(write_seq([a0, a1, Dx, a2]), gt))
            res["last_same"].append(defer_ce(write_seq([a0, a1, a2, Ds]), gt))
            res["last_xdom"].append(defer_ce(write_seq([a0, a1, a2, Dx]), gt))
            res["reset"].append(defer_ce(None, gt))
        print(f"\n[{nm}] DISTRACTOR (n={len(res['reset'])}, target = A's c3 opening)")
        _report(res, len(res["reset"]), [
            ("mid_same", "ctrl4_dup", "MID same-domain vs ctrl4"),
            ("mid_xdom", "ctrl4_dup", "MID cross-domain vs ctrl4"),
            ("last_same", "ctrl4_dup", "LAST same-domain vs ctrl4"),
            ("last_xdom", "ctrl4_dup", "LAST cross-domain vs ctrl4"),
            ("last_xdom", "reset", "worst case vs reset (does the thread survive?)")])


def probe_cued(raw, tok, model, dev, n_files):
    """Cued-query thread selection (idea-G follow-up, 2026-07-12). In the
    interleaved regime the blank defer query is AMBIGUOUS (which live thread?)
    and v2e resolves it by recency (last_xdom +2.42 while MID is null). Real
    usage always has a cue: the current window's tail identifies the thread.
    Query = the last CUE tokens of a2 fed IN CONTEXT (the model's IC mode —
    in-distribution for interleave-trained ckpts, whose IC chunks routinely
    follow another thread's write), scored on A's c3 opening. Banks vary what
    was written LAST:
      clean3       a0,a1,a2                 (baseline)
      junk_last    a0,a1,a2,Dx              (cross-domain junk written last)
      thread_last  a0,a1,a2,b0,b1           (a LIVE same-domain thread B last)
      reset        no bank
    Verdict: if the cued junk/thread costs collapse toward zero while the
    blank-query costs stay high, selection is CONTENT-driven and recency is
    only the blank-query fallback — no focus layer needed for the cued case."""
    write_seq, defer_ce = _mk_ops(model, tok, dev)
    CUE = 16

    def cued_ce(bank, cue, gt):
        C = cue.numel()                                     # cue length may vary (labels)
        di = torch.cat([cue.unsqueeze(0).to(dev), gt], dim=1)   # [1, C+DL]
        with torch.no_grad():
            lg = model(di, init_mem=bank)["logits"].float()
        # standard shift: logits at C-1 .. C+DL-2 predict gt[0..DL-1]
        pred = lg[:, C - 1:C + DL - 1]
        return float(F.cross_entropy(pred.reshape(-1, pred.size(-1)),
                                     gt.reshape(-1)))

    blank_id = tok.convert_tokens_to_ids("<blank>")

    def addr_ce(bank, cue, gt):
        """G2's TRAINED format: [cue, <blank>*DL], loss at blank positions
        (blank C+i predicts gt[i]) — the defer mode, where the bank is the
        only bridge; cued_ce's teacher-forced IC mode is NOT this mode."""
        C = cue.numel()
        di = torch.cat([cue.unsqueeze(0).to(dev),
                        torch.full((1, DL), blank_id, dtype=torch.long, device=dev)], 1)
        with torch.no_grad():
            lg = model(di, init_mem=bank)["logits"].float()[:, C:]
        return float(F.cross_entropy(lg.reshape(-1, lg.size(-1)), gt.reshape(-1)))

    # Two cue types, two questions:
    #   cont  = last CUE tokens of a2 (target continues it) — usage realism.
    #     Finding on v2e: local context makes the bank marginal (~0.016 nat),
    #     so junk costs nothing but nothing is proven about SELECTION.
    #   id    = CUE tokens from a1's interior (identifies the thread, target is
    #     NOT its continuation) — the bank must supply the content; clean_id vs
    #     reset_id shows whether the read is used, junk/thread_id whether the
    #     selection survives a foreign LAST write.
    # lbl = G2 addressing (2026-07-12): banks written with each chunk PREFIXED by
    # its file's stable synthetic label (code_data.file_label_ids), query = A's
    # label. Only meaningful on addr_label-trained ckpts; earlier ckpts serve as
    # the untrained control (labelled writes are mildly OOD for them).
    from deepseek_v4_mini.code_data import file_label_ids
    stream = _stream(raw, tok, seed=888)
    pools = _pools(stream)
    for si, nm in enumerate(stream.src_names):
        fs, other = pools[si], pools[1 - si] if len(pools) > 1 else pools[si]
        res = {k: [] for k in ("clean_blank", "junk_blank",
                               "clean_cont", "junk_cont", "thread_cont", "reset_cont",
                               "clean_id", "junk_id", "thread_id", "reset_id",
                               "clean_open", "junk_open", "thread_open", "reset_open",
                               "clean_lbl", "junk_lbl", "thread_lbl", "reset_lbl")}
        for i, f in enumerate(fs[:n_files]):
            B = fs[(i + 2) % min(len(fs), 200)]
            Dx = other[i % min(len(other), 200)][0]
            a0, a1, a2 = f[0], f[1], f[2]
            gt = f[3][:DL].unsqueeze(0).to(dev)
            cue_cont = a2[-CUE:]
            mid = max(0, a1.numel() // 2 - CUE // 2)
            cue_id = a1[mid:mid + CUE]
            bank_clean = write_seq([a0, a1, a2])
            bank_junk = write_seq([a0, a1, a2, Dx])
            bank_thread = write_seq([a0, a1, a2, B[0], B[1]])
            res["clean_blank"].append(defer_ce(bank_clean, gt))
            res["junk_blank"].append(defer_ce(bank_junk, gt))
            for tag, cue in (("cont", cue_cont), ("id", cue_id)):
                res[f"clean_{tag}"].append(cued_ce(bank_clean, cue, gt))
                res[f"junk_{tag}"].append(cued_ce(bank_junk, cue, gt))
                res[f"thread_{tag}"].append(cued_ce(bank_thread, cue, gt))
                res[f"reset_{tag}"].append(cued_ce(None, cue, gt))
            # open = G2's semantic-address task VERBATIM: cue = a2's opening,
            # target = c3's opening, defer mode — unlabelled banks
            cue_open = a2[:CUE]
            res["clean_open"].append(addr_ce(bank_clean, cue_open, gt))
            res["junk_open"].append(addr_ce(bank_junk, cue_open, gt))
            res["thread_open"].append(addr_ce(bank_thread, cue_open, gt))
            res["reset_open"].append(addr_ce(None, cue_open, gt))
            # labelled writes + label query, defer mode
            la, lb_, lx = (file_label_ids(tok, x) for x in (f, B, [Dx]))
            L = lambda lab, c: torch.cat([lab, c])
            bank_clean_l = write_seq([L(la, a0), L(la, a1), L(la, a2)])
            bank_junk_l = write_seq([L(la, a0), L(la, a1), L(la, a2), L(lx, Dx)])
            bank_thread_l = write_seq([L(la, a0), L(la, a1), L(la, a2),
                                       L(lb_, B[0]), L(lb_, B[1])])
            res["clean_lbl"].append(addr_ce(bank_clean_l, la, gt))
            res["junk_lbl"].append(addr_ce(bank_junk_l, la, gt))
            res["thread_lbl"].append(addr_ce(bank_thread_l, la, gt))
            res["reset_lbl"].append(addr_ce(None, la, gt))
        print(f"\n[{nm}] CUED thread selection (n={len(res['clean_blank'])}, "
              f"target = A's c3 opening, cont = a2 tail / id = a1 interior / "
              f"lbl = file label, {CUE} tokens)")
        _report(res, len(res["clean_blank"]), [
            ("junk_blank", "clean_blank", "JUNK-LAST cost, blank query (recency trap)"),
            ("junk_cont", "clean_cont", "JUNK-LAST cost, cont-cued (usage realism)"),
            ("clean_id", "reset_id", "BANK VALUE under id-cue (read used at all?)"),
            ("junk_id", "clean_id", "JUNK-LAST cost, id-cued (selection survives?)"),
            ("thread_id", "clean_id", "LIVE-THREAD-LAST cost, id-cued"),
            ("clean_open", "reset_open", "BANK VALUE, open-cue defer (G2 semantic task)"),
            ("junk_open", "clean_open", "JUNK-LAST cost, open-cue defer"),
            ("thread_open", "clean_open", "LIVE-THREAD-LAST cost, open-cue defer"),
            ("clean_lbl", "reset_lbl", "BANK VALUE, label-cue defer (G2 addressing)"),
            ("junk_lbl", "clean_lbl", "JUNK-LAST cost, label-cued"),
            ("thread_lbl", "clean_lbl", "LIVE-THREAD-LAST cost, label-cued")])


def probe_merge(raw, tok, model, dev, n_files):
    """v3 cascade brick test (user spec 2026-07-12): the tensor hierarchy merges
    stored matrices AT READ TIME via a function (v1 = average). Is the existing
    read robust to consuming an AVERAGE of independently-written banks? Write
    bank_A and bank_B (and 3 more for the depth-2 simulation) separately, then
    decode A's target from: its mono bank, avg of 2 banks (block-1 unit), avg
    of 4 banks (block-2 unit), vs reset. Superposition probes (2026-07-09)
    showed the read already consumes recency-weighted superpositions ~linearly
    — this measures the CAPACITY of that linearity, zero-shot, no new archi.
    Caveat: seed slots are averaged too (variance halves) — plain-average v1."""
    write_seq, defer_ce = _mk_ops(model, tok, dev)
    stream = _stream(raw, tok, seed=999)
    pools = _pools(stream)
    for si, nm in enumerate(stream.src_names):
        fs = pools[si]
        Ks = (2, 4, 8, 16, 32, 64)
        res = {k: [] for k in ("mono", *(f"avg{K}" for K in Ks), "reset")}
        pool_n = min(len(fs), 200)
        bank_cache = {}  # deep-block averages reuse others heavily: 1 write/file
        for i, f in enumerate(fs[:n_files]):
            gt = f[3][:DL].unsqueeze(0).to(dev)
            if i not in bank_cache:
                bank_cache[i] = write_seq([f[0], f[1], f[2]])
            bank_a = bank_cache[i]
            o_sum, n_o = bank_a.clone(), 1
            res["mono"].append(defer_ce(bank_a, gt))
            for K in Ks:
                while n_o < K:
                    j = (i + n_o) % pool_n
                    if j not in bank_cache:
                        g = fs[j]
                        bank_cache[j] = write_seq([g[0], g[1], g[2]])
                    o_sum = o_sum + bank_cache[j]
                    n_o += 1
                res[f"avg{K}"].append(defer_ce(o_sum / K, gt))
            res["reset"].append(defer_ce(None, gt))
        print(f"\n[{nm}] MERGE-BY-AVERAGE (n={len(res['mono'])}, target = A's c3 opening)")
        _report(res, len(res["mono"]), [
            ("avg2", "mono", "cost of 2-way average (block-1 unit)"),
            ("avg4", "mono", "cost of 4-way average (block-2 unit)"),
            ("avg8", "mono", "cost of 8-way average (block-3 unit)"),
            ("avg16", "mono", "cost of 16-way average (block-4 unit)"),
            ("avg32", "mono", "cost of 32-way average (deep-block unit)"),
            ("avg64", "mono", "cost of 64-way average (v3 block-3 accumulator)"),
            ("avg2", "reset", "avg2 vs reset (does A survive the merge?)"),
            ("avg16", "reset", "avg16 vs reset (old cascade floor)"),
            ("avg64", "reset", "avg64 vs reset (fractal cascade floor)")])


def probe_order(raw, tok, model, dev, n_files):
    write_seq, defer_ce = _mk_ops(model, tok, dev)
    stream = _stream(raw, tok, seed=777)
    pools = _pools(stream)
    for si, nm in enumerate(stream.src_names):
        res = {k: [] for k in ("fwd", "rev", "midswap", "oldswap", "reset")}
        for f in pools[si][:n_files]:
            a0, a1, a2 = f[0], f[1], f[2]
            gt = f[3][:DL].unsqueeze(0).to(dev)
            res["fwd"].append(defer_ce(write_seq([a0, a1, a2]), gt))
            res["rev"].append(defer_ce(write_seq([a2, a1, a0]), gt))
            res["midswap"].append(defer_ce(write_seq([a0, a2, a1]), gt))
            res["oldswap"].append(defer_ce(write_seq([a1, a0, a2]), gt))
            res["reset"].append(defer_ce(None, gt))
        print(f"\n[{nm}] ORDER (n={len(res['reset'])}, target = c3 opening)")
        _report(res, len(res["reset"]), [
            ("rev", "fwd", "reversed vs forward (recency-confounded)"),
            ("midswap", "fwd", "a2/a1 swapped vs forward (recency-confounded)"),
            ("oldswap", "fwd", "PURE order: old writes permuted, last fixed")])


def probe_eviction(raw, tok, model, dev, n_files):
    write_seq, defer_ce = _mk_ops(model, tok, dev)
    stream = _stream(raw, tok, seed=999)
    max_mem = model.cfg.max_mem if hasattr(model, "cfg") else 8
    rng = random.Random(0)
    for si, nm in enumerate(stream.src_names):
        vw = stream.source_stream(si)
        longs = [f for f in vw.files
                 if len(f) >= 12 and all(f[j].numel() >= DL for j in range(2, 12))]
        rng.shuffle(longs)
        longs = longs[:n_files]
        print(f"\n[{nm}] EVICTION / recall-by-lag: {len(longs)} 12-chunk files")
        if len(longs) < 8:
            print("  (too few long files - skip)")
            continue
        lags = {}
        for f in longs:
            bank = write_seq([f[j] for j in range(12)])
            for j in range(2, 12):
                gt = f[j][:DL].unsqueeze(0).to(dev)
                lag = 12 - j
                lags.setdefault(lag, []).append(defer_ce(None, gt) - defer_ce(bank, gt))
        print(f"  {'lag':>4} {'gist status':>12} {'recall GAP':>12} {'n':>4}")
        for lag in sorted(lags):
            g = lags[lag]
            status = "in bank" if lag <= max_mem else "EVICTED"
            se = st.stdev(g) / len(g) ** 0.5 if len(g) > 1 else 0.0
            print(f"  {lag:>4} {status:>12} {st.mean(g):>+9.3f}+-{se:.3f} {len(g):>4}")


def probe_cohab(raw, tok, model, dev, n_files):
    write_seq, defer_ce = _mk_ops(model, tok, dev)
    stream = _stream(raw, tok, seed=444)
    pools = _pools(stream)
    for si, nm in enumerate(stream.src_names):
        fs = pools[si]
        res = {k: [] for k in ("monoA_A", "inter_A", "block_A", "reset_A",
                               "monoB_B", "inter_B", "block_B", "reset_B")}
        for i, f in enumerate(fs[:n_files]):
            g = fs[(i + 1) % min(len(fs), 200)]           # B = neighbor file
            gtA = f[3][:DL].unsqueeze(0).to(dev)
            gtB = g[3][:DL].unsqueeze(0).to(dev)
            inter = write_seq([f[0], g[0], f[1], g[1], f[2], g[2]])
            block = write_seq([f[0], f[1], f[2], g[0], g[1], g[2]])
            res["monoA_A"].append(defer_ce(write_seq([f[0], f[1], f[2]]), gtA))
            res["inter_A"].append(defer_ce(inter, gtA))
            res["block_A"].append(defer_ce(block, gtA))
            res["reset_A"].append(defer_ce(None, gtA))
            res["monoB_B"].append(defer_ce(write_seq([g[0], g[1], g[2]]), gtB))
            res["inter_B"].append(defer_ce(inter, gtB))
            res["block_B"].append(defer_ce(block, gtB))
            res["reset_B"].append(defer_ce(None, gtB))
        n = len(res["monoA_A"])
        print(f"\n[{nm}] COHABITATION (n={n}, one bank decoded against BOTH targets; "
              f"B always written last)")
        _report(res, n, [
            ("inter_A", "monoA_A", "cohab cost for A, interleaved"),
            ("block_A", "monoA_A", "cohab cost for A, blocked (A oldest)"),
            ("inter_B", "monoB_B", "cohab cost for B, interleaved"),
            ("block_B", "monoB_B", "cohab cost for B, blocked (B recent)"),
            ("reset_A", "inter_A", "A GAP surviving cohabitation"),
            ("reset_B", "inter_B", "B GAP surviving cohabitation")])


def probe_reflect(raw, tok, model, dev, n_files):
    write_seq, defer_ce = _mk_ops(model, tok, dev)
    blank_id = tok.convert_tokens_to_ids("<blank>")
    stream = _stream(raw, tok, seed=606)
    pools = _pools(stream)

    def think_turn(bank):
        di = torch.full((1, DL), blank_id, dtype=torch.long, device=dev)
        with torch.no_grad():
            return model(di, init_mem=bank)["mem_bank"]

    KS = (0, 1, 2, 3)
    for si, nm in enumerate(stream.src_names):
        res = {f"k{k}": [] for k in KS}
        res["reset"] = []
        for f in pools[si][:n_files]:
            gt = f[3][:DL].unsqueeze(0).to(dev)
            bank = write_seq([f[0], f[1], f[2]])
            for k in KS:
                res[f"k{k}"].append(defer_ce(bank, gt))
                bank = think_turn(bank)                 # k -> k+1 thought turns
            res["reset"].append(defer_ce(None, gt))
        n = len(res["reset"])
        print(f"\n[{nm}] REFLECT-k (n={n}, k blank thought-turns carried before decode; "
              f"each thought evicts one oldest slot)")
        _report(res, n, [(f"k{k}", "k0", f"k={k} vs k=0") for k in KS[1:]])


_PY_KW = frozenset("""False None True and as assert async await break class continue
def del elif else except finally for from global if import in is lambda nonlocal
not or pass print raise return self set str try while with yield int len dict
list range object type import""".split())


def _rename_ids(text):
    """Consistently rename the most frequent identifiers to qz0..qzN (regex-level:
    strings/attributes included — surface perturbation, not compiler-grade)."""
    import re
    from collections import Counter
    words = Counter(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", text))
    cand = [w for w, _ in words.most_common(48) if w.lower() not in _PY_KW][:24]
    mapping = {w: f"qz{i}" for i, w in enumerate(cand)}
    return re.sub(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b",
                  lambda m: mapping.get(m.group(0), m.group(0)), text)


def probe_invar(raw, tok, model, dev, n_files):
    write_seq, defer_ce = _mk_ops(model, tok, dev)
    L = int(raw["data"]["seq_len"])
    stream = _stream(raw, tok, seed=888)
    pools = _pools(stream)
    for si, nm in enumerate(stream.src_names):
        fs = pools[si]
        is_code = "code" in nm.lower() or si == 0
        keys = ["own", "reseg", "swap", "reset"] + (["rename"] if is_code else [])
        res = {k: [] for k in keys}
        for i, f in enumerate(fs[:n_files]):
            g = fs[(i + 1) % min(len(fs), 200)]
            gt = f[3][:DL].unsqueeze(0).to(dev)
            toks = torch.cat([f[0], f[1], f[2]])
            res["own"].append(defer_ce(write_seq([f[0], f[1], f[2]]), gt))
            # (a) reseg: boundaries shifted by 128 (drops the first 128 tokens,
            # the ones FARTHEST from the target)
            sh = toks[128:]
            reseg = [sh[j * L:(j + 1) * L] for j in range(-(-sh.numel() // L))]
            res["reseg"].append(defer_ce(write_seq(reseg), gt))
            # (b) rename: same definition, destroyed surface (code only)
            if is_code:
                rt = torch.tensor(
                    tok.encode(_rename_ids(tok.decode(toks)), add_special_tokens=False),
                    dtype=torch.long)
                ren = [rt[j * L:(j + 1) * L] for j in range(-(-rt.numel() // L))]
                res["rename"].append(defer_ce(write_seq(ren), gt))
            res["swap"].append(defer_ce(write_seq([g[0], g[1], g[2]]), gt))
            res["reset"].append(defer_ce(None, gt))
        n = len(res["own"])
        print(f"\n[{nm}] INVARIANCE (n={n}, target = ORIGINAL c3 opening; "
              f"invariance = perturbed bank stays near own, far from swap)")
        pairs = [("reseg", "own", "reseg cost (segmentation invariance)"),
                 ("swap", "own", "swap distance (specificity ceiling)")]
        if is_code:
            pairs.insert(1, ("rename", "own", "rename cost (surface invariance)"))
        _report(res, n, pairs)


PROBES = {"swap": probe_swap, "dup": probe_dup, "distractor": probe_distractor,
          "order": probe_order, "eviction": probe_eviction,
          "cohab": probe_cohab, "reflect": probe_reflect, "invar": probe_invar,
          "cued": probe_cued, "merge": probe_merge}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("ckpt")
    ap.add_argument("--probes", default="swap,dup,distractor,order,eviction")
    ap.add_argument("--n-files", type=int, default=48)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw, tok, model = _load(a.cfg, a.ckpt, dev)
    for p in a.probes.split(","):
        PROBES[p.strip()](raw, tok, model, dev, a.n_files)


if __name__ == "__main__":
    main()
