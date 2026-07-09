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


PROBES = {"swap": probe_swap, "dup": probe_dup, "distractor": probe_distractor,
          "order": probe_order, "eviction": probe_eviction}


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
