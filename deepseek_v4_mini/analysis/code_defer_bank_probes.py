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
import os
import random
import statistics as st

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer

from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.model import ThoughtBankLM
from deepseek_v4_mini.code_data import CodeChunkStream

DL = 16     # defer_len used for all targets
DELTA = None  # B4 : canal delta chargé depuis le ckpt s'il y en a un


def _load(cfg_path, ckpt_path, dev):
    global DELTA
    raw = yaml.safe_load(open(cfg_path))
    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    for t in ("<think>", "<blank>"):
        if t not in tok.get_vocab():
            tok.add_special_tokens({"additional_special_tokens": [t]})
    ck = torch.load(ckpt_path, map_location="cpu")
    model = ThoughtBankLM(ThoughtBankConfig(**ck["cfg"])).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    if ck.get("delta"):
        from deepseek_v4_mini.delta_channel import DeltaChannel
        sd = ck["delta"]
        dk = sd["W_k.weight"].shape[0]
        c = ck["cfg"]
        DELTA = DeltaChannel(c["d_model"], c["max_mem"], c["mem_dim"], d_k=dk).to(dev)
        DELTA.load_state_dict(sd)
        DELTA.eval()
        print(f"delta channel détecté (d_k {dk}) — write_seq = carry delta")
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
        if DELTA is not None:
            # B4 : le carry delta ne dépend QUE des embeddings des chunks (le
            # forward du modèle n'y contribue pas) — reproduction exacte du
            # trainer, sans forward.
            S = DELTA.init_state(1, dev)
            with torch.no_grad():
                for c in chunks:
                    x = c.unsqueeze(0).to(dev)
                    S = DELTA.update(S, model.embed.weight[x])
                return DELTA.to_bank(S, next(model.parameters()).dtype)
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


def probe_capacity(raw, tok, model, dev, n_files):
    """Capacity/interference curve (350M validation plan, 2026-07-13): N
    labelled threads written into ONE bank (1 chunk each, N <= max_mem), then
    label-cued addressed recall of EVERY thread in v2f's trained defer format.
    recall(N) is the funding-dossier figure: how many co-resident threads
    before addressed recall degrades? foreign = cueing an UNWRITTEN file's
    label (should sit at ~reset: no hallucinated address). Only meaningful on
    addr_label-trained ckpts (v2f+)."""
    write_seq, defer_ce = _mk_ops(model, tok, dev)
    from deepseek_v4_mini.code_data import file_label_ids
    blank_id = tok.convert_tokens_to_ids("<blank>")

    def addr_ce(bank, cue, gt):
        C = cue.numel()
        di = torch.cat([cue.unsqueeze(0).to(dev),
                        torch.full((1, DL), blank_id, dtype=torch.long, device=dev)], 1)
        with torch.no_grad():
            lg = model(di, init_mem=bank)["logits"].float()[:, C:]
        return float(F.cross_entropy(lg.reshape(-1, lg.size(-1)), gt.reshape(-1)))

    Ns = (1, 2, 4, 8)
    stream = _stream(raw, tok, seed=555)
    pools = _pools(stream)
    for si, nm in enumerate(stream.src_names):
        fs = pools[si]
        pool_n = min(len(fs), 200)
        res = {k: [] for k in (*(f"n{N}" for N in Ns), "n8_old", "n8_new",
                               "foreign", "reset")}
        for i in range(n_files):
            group = [fs[(i * 9 + k) % pool_n] for k in range(9)]  # 8 threads + 1 unwritten
            labs = [file_label_ids(tok, g) for g in group]
            for N in Ns:
                bank = write_seq([torch.cat([labs[k], group[k][0]]) for k in range(N)])
                ces = [addr_ce(bank, labs[k], group[k][1][:DL].unsqueeze(0).to(dev))
                       for k in range(N)]
                res[f"n{N}"].append(sum(ces) / N)
                if N == 8:
                    res["n8_old"].append(ces[0])
                    res["n8_new"].append(ces[-1])
                    res["foreign"].append(addr_ce(bank, labs[8],
                                          group[8][1][:DL].unsqueeze(0).to(dev)))
            res["reset"].append(sum(addr_ce(None, labs[k],
                                    group[k][1][:DL].unsqueeze(0).to(dev))
                                    for k in range(2)) / 2)
        print(f"\n[{nm}] CAPACITY/INTERFERENCE (n={len(res['n1'])}, N labelled "
              f"1-chunk threads in one bank, label-cued recall of each)")
        _report(res, len(res["n1"]), [
            ("n1", "reset", "1 thread vs reset (addressed bank value)"),
            ("n2", "n1", "interference cost at N=2"),
            ("n4", "n1", "interference cost at N=4"),
            ("n8", "n1", "interference cost at N=8 (bank full)"),
            ("n8_old", "n8_new", "oldest vs newest thread at N=8 (recency tilt)"),
            ("foreign", "reset", "unwritten label vs reset (no hallucinated address)")])


def probe_capacity_curve(raw, tok, model, dev, n_files):
    """(e) du plan 350M (2026-07-13) — LA figure du dossier : rappel adressé(N)
    À TRAVERS la frontière d'éviction. Étend `capacity` (N <= max_mem) : N fils
    étiquetés (1 chunk chacun, format identique) écrits dans la banque, N
    jusqu'à 2×max_mem — pour N > max_mem les fils les plus anciens ne vivent
    plus QUE dans la page (ckpt cascade, capture FIFO comme probe_page) ou dans
    les résidus de superposition (ckpt banque seule). Rappel label-cued par
    groupe d'âge : résidents (derniers max_mem) vs évincés ; sur ckpt cascade
    les évincés sont aussi mesurés page ABLATÉE (la contribution de la page à
    la traversée = le chiffre v3). Ne juge que les ckpts addr_label (v2f+).
    Prédictions datées : voir le job 109."""
    from deepseek_v4_mini.code_data import file_label_ids
    t = raw.get("training", raw.get("train", {}))
    depth = int(t.get("cascade_depth", 0) or 0)
    casc_on = depth > 0
    if casc_on:
        from deepseek_v4_mini.cascade import CascadeMemory
        cmap = t.get("cascade_map")
        cmap = ([int(v) for v in cmap] if cmap else
                [0] * (model.cfg.n_layers - depth) + list(range(1, depth + 1)))
    think_id = tok.convert_tokens_to_ids("<think>")
    blank_id = tok.convert_tokens_to_ids("<blank>")
    seed_slots = int(getattr(model.cfg, "mem_seed_slots", model.cfg.max_mem))
    max_mem = int(model.cfg.max_mem)
    dt = next(model.parameters()).dtype
    Ns = (2, 4, 8, 12, 16)

    def write_threads(seqs):
        """1 write par fil (label+chunk), capture d'éviction si cascade."""
        bank = model.thought_stream.seed_bank(1, dev, dt) if casc_on else None
        casc = CascadeMemory(depth, max_mem) if casc_on else None
        nev = 0
        for s in seqs:
            x = torch.cat([s.unsqueeze(0).to(dev),
                           torch.full((1, 1), think_id, dtype=torch.long, device=dev)], 1)
            pre0 = (bank[:, 0].detach()
                    if casc_on and bank.size(1) >= max_mem else None)
            with torch.no_grad():
                bank = model(x, init_mem=bank,
                             layer_banks=casc.layer_banks(bank, cmap)
                             if casc_on else None)["mem_bank"]
            if pre0 is not None:
                nev += 1
                if nev > seed_slots:
                    casc.push_slot(pre0)
        return bank, casc

    def addr_ce(bank, casc, cue, gt, ablate=False):
        lb = None
        if casc is not None:
            lb = casc.layer_banks(bank, cmap)
            if ablate:
                lb = [bank if lvl == 0 else None for lvl in cmap]
        C = cue.numel()
        di = torch.cat([cue.unsqueeze(0).to(dev),
                        torch.full((1, DL), blank_id, dtype=torch.long, device=dev)], 1)
        with torch.no_grad():
            lg = model(di, init_mem=bank, layer_banks=lb)["logits"].float()[:, C:]
        return float(F.cross_entropy(lg.reshape(-1, lg.size(-1)), gt.reshape(-1)))

    stream = _stream(raw, tok, seed=555)  # même seed que capacity
    pools = _pools(stream)
    G = max(Ns) + 1  # fils + 1 non-écrit (foreign)
    for si, nm in enumerate(stream.src_names):
        fs = pools[si]
        pool_n = min(len(fs), 200)
        keys = [f"n{N}" for N in Ns]
        keys += [f"n{N}_res" for N in Ns if N > max_mem]
        keys += [f"n{N}_ev" for N in Ns if N > max_mem]
        if casc_on:
            keys += [f"n{N}_ev_off" for N in Ns if N > max_mem]
        keys += ["foreign", "reset"]
        res = {k: [] for k in keys}
        for i in range(n_files):
            group = [fs[(i * G + k) % pool_n] for k in range(G)]
            labs = [file_label_ids(tok, g) for g in group]
            gts = [g[1][:DL].unsqueeze(0).to(dev) for g in group]
            for N in Ns:
                bank, casc = write_threads(
                    [torch.cat([labs[k], group[k][0]]) for k in range(N)])
                ces = [addr_ce(bank, casc, labs[k], gts[k]) for k in range(N)]
                res[f"n{N}"].append(sum(ces) / N)
                if N > max_mem:
                    ev = list(range(N - max_mem))       # fils sortis de la banque vive
                    rs = list(range(N - max_mem, N))    # résidents
                    res[f"n{N}_ev"].append(sum(ces[k] for k in ev) / len(ev))
                    res[f"n{N}_res"].append(sum(ces[k] for k in rs) / len(rs))
                    if casc_on:
                        off = [addr_ce(bank, casc, labs[k], gts[k], ablate=True)
                               for k in ev]
                        res[f"n{N}_ev_off"].append(sum(off) / len(off))
                if N == max(Ns):
                    res["foreign"].append(addr_ce(bank, casc, labs[G - 1], gts[G - 1]))
            res["reset"].append(sum(addr_ce(None, None, labs[k], gts[k])
                                    for k in range(2)) / 2)
        print(f"\n[{nm}] CAPACITY CURVE (n={n_files}, N fils étiquetés 1-chunk, "
              f"rappel label-cued, max_mem={max_mem}, cascade={'on' if casc_on else 'off'})")
        pairs = [("n2", "reset", "valeur adressée à N=2"),
                 ("n8", "n2", "coût d'interférence N=8 (banque pleine)"),
                 ("n12_res", "n8", "résidents sous pression d'éviction (N=12)"),
                 ("n16_res", "n8", "résidents sous pression d'éviction (N=16)"),
                 ("n12_ev", "reset", "évincés N=12 vs reset (traversée ?)"),
                 ("n16_ev", "reset", "évincés N=16 vs reset (traversée ?)"),
                 ("foreign", "reset", "label non écrit vs reset (pas d'adresse hallucinée)")]
        if casc_on:
            pairs += [("n12_ev", "n12_ev_off", "contribution PAGE aux évincés (N=12)"),
                      ("n16_ev", "n16_ev_off", "contribution PAGE aux évincés (N=16)")]
        _report(res, n_files, pairs)


def probe_longlife(raw, tok, model, dev, n_files):
    """Long-life health (350M validation plan, 2026-07-13): no run ever wrote
    more than ~30 times into one carried bank; the cascade promises thousands.
    Stream files through ONE never-reset bank; at write-count checkpoints
    measure slot-norm drift and recall of the two JUST-WRITTEN files (their
    chunks still inside the 8-slot FIFO window) vs reset. Slow saturation or
    norm blow-up would be invisible in every previous probe."""
    think_id = tok.convert_tokens_to_ids("<think>")
    write_seq, defer_ce = _mk_ops(model, tok, dev)
    CKPT = tuple(int(x) for x in
                 os.environ.get("LONGLIFE_CKPT", "8,32,128,512,1024").split(","))
    R = max(2, min(12, n_files // 4))   # streams indépendants (n = 2R par ckpt)
    stream = _stream(raw, tok, seed=444)
    pools = _pools(stream)
    for si, nm in enumerate(stream.src_names):
        fs = pools[si]
        pool_n = min(len(fs), 400)
        res = {k: [] for W in CKPT for k in (f"car_{W}", f"res_{W}")}
        norms = {W: [] for W in CKPT}
        for rep in range(R):
            bank, w, fi = None, 0, rep * (pool_n // R)
            todo = list(CKPT)
            while todo:
                f = fs[fi % pool_n]
                fi += 1
                for c in (f[0], f[1], f[2]):
                    x = torch.cat([c.unsqueeze(0).to(dev),
                                   torch.full((1, 1), think_id, dtype=torch.long, device=dev)], 1)
                    with torch.no_grad():
                        bank = model(x, init_mem=bank)["mem_bank"]
                    w += 1
                if w >= todo[0]:
                    W = todo.pop(0)
                    norms[W].append(float(bank.norm(dim=-1).mean()))
                    for g in (f, fs[(fi - 2) % pool_n]):
                        gt = g[3][:DL].unsqueeze(0).to(dev)
                        res[f"car_{W}"].append(defer_ce(bank, gt))
                        res[f"res_{W}"].append(defer_ce(None, gt))
        n = len(res[f"car_{CKPT[0]}"])
        print(f"\n[{nm}] LONG LIFE (R={R} carried streams, recall of in-window "
              f"files at write checkpoints; n={n} per ckpt)")
        for W in CKPT:
            print(f"    slot-norm mean @W={W}: {sum(norms[W]) / len(norms[W]):.3f}")
        _report(res, n, [
            (f"car_{W}", f"res_{W}", f"carried vs reset @ {W} writes") for W in CKPT])


def probe_page(raw, tok, model, dev, n_files):
    """v3 verdict d'émergence (option 1, décision user 2026-07-13) : le read des
    couches page (entraîné SANS reach-back supervisé) tire-t-il du signal de la
    page ? Reconstruit la cascade à l'inférence EXACTEMENT comme le trainer
    (capture FIFO du slot évincé, seeds sautés), écrit 8 fichiers (24 chunks) —
    les premiers ne vivent plus QUE dans la page — puis defer sur la cible d'un
    fichier ANCIEN (reach-back) et d'un fichier RÉCENT, page réelle vs page
    ablatée (None). Ne juge que les ckpts cascade-entraînés (cascade_depth>0)."""
    from deepseek_v4_mini.cascade import CascadeMemory
    t = raw.get("training", raw.get("train", {}))
    depth = int(t.get("cascade_depth", 1) or 1)
    cmap = t.get("cascade_map")
    cmap = ([int(v) for v in cmap] if cmap else
            [0] * (model.cfg.n_layers - depth) + list(range(1, depth + 1)))
    think_id = tok.convert_tokens_to_ids("<think>")
    blank_id = tok.convert_tokens_to_ids("<blank>")
    seed_slots = int(getattr(model.cfg, "mem_seed_slots", model.cfg.max_mem))
    max_mem = int(model.cfg.max_mem)
    dt = next(model.parameters()).dtype

    def write_casc(files):
        bank = model.thought_stream.seed_bank(1, dev, dt)
        casc, nev = CascadeMemory(depth, max_mem), 0
        for f in files:
            for c in (f[0], f[1], f[2]):
                x = torch.cat([c.unsqueeze(0).to(dev),
                               torch.full((1, 1), think_id, dtype=torch.long, device=dev)], 1)
                pre0 = bank[:, 0].detach() if bank.size(1) >= max_mem else None
                with torch.no_grad():
                    bank = model(x, init_mem=bank,
                                 layer_banks=casc.layer_banks(bank, cmap))["mem_bank"]
                if pre0 is not None:
                    nev += 1
                    if nev > seed_slots:
                        casc.push_slot(pre0)
        return bank, casc

    def page_defer(bank, casc, gt, ablate):
        lb = casc.layer_banks(bank, cmap)
        if ablate:
            lb = [bank if lvl == 0 else None for lvl in cmap]
        di = torch.full((1, DL), blank_id, dtype=torch.long, device=dev)
        with torch.no_grad():
            lg = model(di, init_mem=bank, layer_banks=lb)["logits"].float()
        return float(F.cross_entropy(lg.reshape(-1, lg.size(-1)), gt.reshape(-1)))

    stream = _stream(raw, tok, seed=333)
    pools = _pools(stream)
    for si, nm in enumerate(stream.src_names):
        fs = pools[si]
        pool_n = min(len(fs), 200)
        res = {k: [] for k in ("early_on", "early_off", "early_reset",
                               "recent_on", "recent_off")}
        for i in range(n_files):
            grp = [fs[(i * 8 + k) % pool_n] for k in range(8)]
            bank, casc = write_casc(grp)
            gt_e = grp[0][3][:DL].unsqueeze(0).to(dev)   # fichier 0 : page seulement
            gt_r = grp[-1][3][:DL].unsqueeze(0).to(dev)  # fichier 7 : banque vive
            res["early_on"].append(page_defer(bank, casc, gt_e, False))
            res["early_off"].append(page_defer(bank, casc, gt_e, True))
            di = torch.full((1, DL), blank_id, dtype=torch.long, device=dev)
            with torch.no_grad():
                lg = model(di, init_mem=None)["logits"].float()
            res["early_reset"].append(float(F.cross_entropy(
                lg.reshape(-1, lg.size(-1)), gt_e.reshape(-1))))
            res["recent_on"].append(page_defer(bank, casc, gt_r, False))
            res["recent_off"].append(page_defer(bank, casc, gt_r, True))
        print(f"\n[{nm}] PAGE ABLATION (n={n_files}, 8 fichiers écrits, cible "
              f"early = fichier 0 évincé de la banque vive, map {cmap})")
        _report(res, n_files, [
            ("early_on", "early_off", "EMERGENCE: page réelle vs ablatée (cible paginée)"),
            ("early_on", "early_reset", "reach-back vs reset (la page vaut-elle qqch ?)"),
            ("recent_on", "recent_off", "coût de la page sur le récent (doit être ~0)")])


def probe_resetcue(raw, tok, model, dev, n_files):
    """B2 — neutralité des resets (backlog 2026-07-13 : le standing warning
    rendu QUANTITATIF). Sur un ckpt entraîné avec resets annoncés 50/50
    (marqueur <<RESET:SOON>> préfixé aux 3 derniers chunks d'une vie), la
    politique d'écriture change-t-elle quand la mort est annoncée ? Écrit
    c0,c1,c2 avec et sans marqueur (appariés par fichier), compare les 3 slots
    écrits : norme, redondance (cos moyen intra-paires = début de rehearsal
    défensif), et CE defer (le contenu reste-t-il lisible ?).
    VERT (= sûr) si tous les d sont ~0 ; TOUTE dérive = résultat publiable en
    soi (section sécurité du dossier). Sur un ckpt jamais entraîné avec le
    marqueur : contrôle zero-shot (le marqueur n'est qu'un préfixe OOD)."""
    write_seq, defer_ce = _mk_ops(model, tok, dev)
    ra = torch.tensor(tok("<<RESET:SOON>>")["input_ids"], dtype=torch.long)
    stream = _stream(raw, tok, seed=222)
    pools = _pools(stream)
    for si, nm in enumerate(stream.src_names):
        res = {k: [] for k in ("ce_sur", "ce_ann", "norm_sur", "norm_ann",
                               "red_sur", "red_ann")}
        for f in pools[si][:n_files]:
            gt = f[3][:DL].unsqueeze(0).to(dev)
            for tag, pre in (("sur", None), ("ann", ra)):
                chunks = ([f[0], f[1], f[2]] if pre is None else
                          [torch.cat([pre, c]) for c in (f[0], f[1], f[2])])
                bank = write_seq(chunks)
                w = bank[0, -3:].float()               # les 3 slots écrits (FIFO)
                res[f"norm_{tag}"].append(float(w.norm(dim=-1).mean()))
                red = float(sum(F.cosine_similarity(w[a], w[b], dim=0)
                                for a, b in ((0, 1), (0, 2), (1, 2))) / 3)
                res[f"red_{tag}"].append(red)
                res[f"ce_{tag}"].append(defer_ce(bank, gt))
        n = len(res["ce_sur"])
        print(f"\n[{nm}] RESET-CUE neutrality (n={n}, writes c0-c2 avec vs sans "
              f"<<RESET:SOON>> préfixé, cible = c3)")
        _report(res, n, [
            ("ce_ann", "ce_sur", "CE defer annoncé vs surprise (contenu intact ?)"),
            ("norm_ann", "norm_sur", "norme des writes annoncé vs surprise"),
            ("red_ann", "red_sur", "redondance intra-writes (rehearsal défensif ?)")])


def probe_xmodal(raw, tok, model, dev, n_files):
    """B3 — transfert cross-modal docstring↔corps (backlog 2026-07-13, vision
    banque-CoT, memoire dsv6-banque-cot-multimodale : cohab ≠ transfert).
    Zero-shot : un gist écrit depuis la DOCSTRING seule aide-t-il le defer sur
    l'ouverture du CORPS (et inversement) ? Conditions (cible = corps[:DL]) :
    bank(doc) vs bank(doc d'une AUTRE fonction) [spécificité vs registre] vs
    bank(corps) [borne haute : contient la cible] vs reset. Sens inverse :
    cible = doc[:DL], bank(corps) vs reset. Source code uniquement.
    VERT si doc bat reset > 0.3 nat ET bat swap_doc."""
    write_seq, defer_ce = _mk_ops(model, tok, dev)
    from deepseek_v4_mini.code_data import docstring_pairs
    stream = _stream(raw, tok, seed=131)
    pools = _pools(stream)
    for si, nm in enumerate(stream.src_names):
        if si != 0 and "code" not in nm.lower():
            continue                                   # paires = source code
        pairs = []
        for f in pools[si]:
            ps = docstring_pairs(tok.decode(torch.cat(f)), tok,
                                 min_doc=DL, min_body=2 * DL, max_pairs=1)
            if ps:
                pairs.append(ps[0])
            if len(pairs) >= n_files + 1:
                break
        if len(pairs) < min(8, n_files):
            print(f"\n[{nm}] XMODAL: trop peu de paires ({len(pairs)}) — skip")
            continue
        res = {k: [] for k in ("doc", "swap_doc", "body_ub", "reset",
                               "rev_body", "rev_reset")}
        for i in range(min(n_files, len(pairs) - 1)):
            doc, body = pairs[i]
            odoc = pairs[(i + 1) % len(pairs)][0]
            gt = body[:DL].unsqueeze(0).to(dev)
            res["doc"].append(defer_ce(write_seq([doc]), gt))
            res["swap_doc"].append(defer_ce(write_seq([odoc]), gt))
            res["body_ub"].append(defer_ce(write_seq([body]), gt))
            res["reset"].append(defer_ce(None, gt))
            gtd = doc[:DL].unsqueeze(0).to(dev)
            res["rev_body"].append(defer_ce(write_seq([body]), gtd))
            res["rev_reset"].append(defer_ce(None, gtd))
        n = len(res["doc"])
        print(f"\n[{nm}] CROSS-MODAL docstring↔corps (n={n}, cible = ouverture "
              f"du corps ; rev = ouverture de la docstring)")
        _report(res, n, [
            ("reset", "doc", "TRANSFERT doc→corps (VERT si > +0.3)"),
            ("swap_doc", "doc", "spécificité (autre doc vs la bonne)"),
            ("doc", "body_ub", "distance à la borne haute (corps écrit)"),
            ("rev_reset", "rev_body", "TRANSFERT corps→doc")])


PROBES = {"swap": probe_swap, "dup": probe_dup, "distractor": probe_distractor,
          "order": probe_order, "eviction": probe_eviction,
          "cohab": probe_cohab, "reflect": probe_reflect, "invar": probe_invar,
          "cued": probe_cued, "merge": probe_merge,
          "capacity": probe_capacity, "capacity_curve": probe_capacity_curve,
          "longlife": probe_longlife,
          "page": probe_page, "resetcue": probe_resetcue, "xmodal": probe_xmodal}


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
