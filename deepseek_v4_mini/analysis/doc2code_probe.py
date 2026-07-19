"""Cross-register transfer probe: does a NATURAL-LANGUAGE gist (docstring) help
the deferred continuation of the CODE it describes?

Usage:
    PYTHONPATH=. python deepseek_v4_mini/analysis/doc2code_probe.py \
        deepseek_v4_mini/configs/rl_defer_grpo_97m.yaml \
        checkpoints/code_defer_native_v2c_varlen/final.pt [n_pairs]

Conditions per function (one function per source file, mined from
codeparrot-clean-valid, docstring >= 150 chars, body opening >= 40 chars):
  reset    : defer the body opening from the seed bank alone
  doc      : write [docstring text only] -> defer body opening
  sig+doc  : write [def line + docstring] -> defer (in-distribution upper ref)
  doc_shuf : write an UNRELATED function's docstring (specificity control)

Transfer = reset - doc ; specificity = doc_shuf - doc.
2026-07-10 result on the 97M v2c final (n=96, zero-shot): transfer
+0.683 +/- 0.067, specificity +0.169 +/- 0.066 (p ~ 0.01, 61/96 wins),
sig+doc +1.087. One direction (doc -> code) only.
"""
import ast
import statistics
import sys

import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from deepseek_v4_mini.config import ThoughtBankConfig
from deepseek_v4_mini.model import ThoughtBankLM
import deepseek_v4_mini.rl_defer_grpo as rl


def main(cfg_path: str, ckpt_path: str, n_pairs: int = 96) -> None:
    raw = yaml.safe_load(open(cfg_path))
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    tok.add_special_tokens({"additional_special_tokens": ["<think>", "<blank>"]})
    think_id = tok.convert_tokens_to_ids("<think>")
    blank_id = tok.convert_tokens_to_ids("<blank>")
    mcfg = dict(raw["model"])
    mcfg["vocab_size"] = len(tok)
    model = ThoughtBankLM(ThoughtBankConfig(**mcfg)).to(device).eval()
    ck = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ck["model"])

    # -- mine (docstring, def-line, body-opening) triples ---------------------
    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train",
                      streaming=True)
    pairs = []
    for i, ex in enumerate(ds):
        if i > 20000 or len(pairs) >= n_pairs:
            break
        src = ex["content"]
        if len(src) > 60000:
            continue
        try:
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node)
            if not doc or len(doc) < 150 or len(node.body) < 2:
                continue
            body = node.body[1:]                 # after the docstring stmt
            seg = ast.get_source_segment(src, body[0])
            if seg is None or len(seg) < 40:
                continue
            defline = src.splitlines()[node.lineno - 1].strip()
            pairs.append({"doc": doc, "def": defline, "body": seg})
            break                                # one per file (diversity)
    print(f"mined {len(pairs)} (docstring, body) pairs")

    def write(bank, text):
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=500)["input_ids"].to(device)
        xt = torch.cat([ids, torch.full((1, 1), think_id, dtype=torch.long,
                                        device=device)], 1)
        return model(xt, init_mem=bank)["mem_bank"]

    def dce(bank, text):
        t = tok(text, return_tensors="pt", truncation=True,
                max_length=16)["input_ids"].to(device)
        return float(rl.defer_ce(model, bank, t[:, :16], blank_id))

    res = {"reset": [], "doc": [], "sigdoc": [], "shuf": []}
    with torch.no_grad():
        for i, p in enumerate(pairs):
            seed = rl.conv_seed_bank(model)
            other = pairs[(i + 7) % len(pairs)]["doc"]
            res["reset"].append(dce(seed, p["body"]))
            res["doc"].append(dce(write(seed, p["doc"]), p["body"]))
            res["sigdoc"].append(dce(
                write(seed, p["def"] + '\n    """' + p["doc"] + '"""'),
                p["body"]))
            res["shuf"].append(dce(write(seed, other), p["body"]))

    m = {k: statistics.mean(v) for k, v in res.items()}
    n = len(res["reset"])

    def sd(a, b):
        d = [x - y for x, y in zip(res[a], res[b])]
        return statistics.mean(d), statistics.pstdev(d) / (len(d) ** 0.5)

    print(f"\nbody-opening CE (16 tok, bank only) n={n}")
    print(f"  reset           : {m['reset']:.3f}")
    print(f"  foreign doc     : {m['shuf']:.3f}")
    print(f"  own doc         : {m['doc']:.3f}")
    print(f"  signature + doc : {m['sigdoc']:.3f}")
    for lbl, a, b in [("TRANSFER (reset - doc)", "reset", "doc"),
                      ("SPECIFICITY (shuf - doc)", "shuf", "doc"),
                      ("sig+doc ref (reset - sigdoc)", "reset", "sigdoc")]:
        mu, se = sd(a, b)
        print(f"  {lbl}: {mu:+.3f} +/- {se:.3f} (SEM)")
    wins = sum(1 for x, y in zip(res["shuf"], res["doc"]) if y < x)
    print(f"  own-doc beats foreign-doc: {wins}/{n} pairs")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 96)
