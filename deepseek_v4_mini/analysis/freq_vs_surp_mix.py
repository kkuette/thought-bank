"""Pondérations de pooling sur le MIX PHASE 1 RÉEL (14 sources, cache NFS) :
uniform (average) vs freq-inverse (1/p)^a vs SIF a/(a+p) vs surprisal nll^2.
Pas de val_mask sur corpus réel => juge = DISCRIMINABILITÉ : cos moyen entre
cibles de chunks différents (bas = bon), intra-source (le dur : même registre)
et global. + top tokens pondérés sur un chunk code et un chunk prose (sanité).

Usage :
  python -m deepseek_v4_mini.analysis.freq_vs_surp_mix <config.yaml> [n_par_source]
(config = n'importe laquelle avec le bloc data du mix, ex. sft_persona_350m.yaml)
"""
import os, sys, random
from collections import Counter

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deepseek_v4_mini.code_data import CodeChunkStream

CFG = sys.argv[1] if len(sys.argv) > 1 else "deepseek_v4_mini/configs/sft_persona_350m.yaml"
N_PER_SRC = int(sys.argv[2]) if len(sys.argv) > 2 else 24
T_MAX = 256          # tokens par chunk pour la passe ref (vitesse)
REF = "HuggingFaceTB/SmolLM2-135M-Instruct"

raw = yaml.safe_load(open(CFG))
d = raw["data"]
tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
add = [x for x in ("<think>", "<blank>") if x not in tok.get_vocab()]
if add:
    tok.add_special_tokens({"additional_special_tokens": add})

sd = dict(seq_len=int(d["seq_len"]), chunks_per_conv=int(d["chunks_per_conv"]),
          batch=1, n_files=int(d.get("n_files", 800)),
          stream_cap=int(d.get("stream_cap", 60000)),
          cache_dir=d.get("cache_dir", "data_cache"),
          min_chunks=int(d.get("min_chunks", 1)),
          sources=d.get("sources"), seed=0)
stream = CodeChunkStream(tok, split="train", **sd)
names = stream.src_names
print(f"mix chargé : {len(names)} sources, {stream.n_chunk} chunks")

rng = random.Random(1789)
per_src: list[list[torch.Tensor]] = []
for files in stream.src_files:
    picks = []
    flat = [c for f in files for c in (f if isinstance(f, list) else [f])]
    for c in rng.sample(flat, min(N_PER_SRC, len(flat))):
        ids = c if torch.is_tensor(c) else torch.tensor(c)
        picks.append(ids.flatten()[:T_MAX])
    per_src.append(picks)

# table unigram sur tout l'échantillon (proxy corpus ; en prod : à la tokenisation)
cnt, tot = Counter(), 0
for picks in per_src:
    for ids in picks:
        cnt.update(ids.tolist()); tot += len(ids)
p = {t: c / tot for t, c in cnt.items()}
P_UNSEEN = 0.5 / tot

# nll sous la ref gelée (fp16 GPU si dispo)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ref = AutoModelForCausalLM.from_pretrained(
    REF, dtype=(torch.float16 if dev == "cuda" else torch.float32)).to(dev).eval()
ref_vocab = ref.get_input_embeddings().num_embeddings

@torch.no_grad()
def nll_of(ids: torch.Tensor) -> torch.Tensor:
    x = ids.clamp(max=ref_vocab - 1).unsqueeze(0).to(dev)
    logits = ref(x).logits.float()
    lp = torch.log_softmax(logits[0, :-1], dim=-1)
    nll = -lp.gather(1, x[0, 1:].unsqueeze(1)).squeeze(1)
    return torch.cat([nll[:1], nll]).cpu()      # 1er token : proxy = nll du 2e

D, M = 768, 512
g = torch.Generator().manual_seed(1789)
proj = torch.randn(D, M, generator=g) / D ** 0.5
emb = torch.nn.Embedding(len(tok), D)
torch.manual_seed(0); torch.nn.init.normal_(emb.weight, std=0.02)

def wvec(ids, scheme):
    t = ids.tolist()
    if scheme == "uniform":
        return torch.ones(len(t))
    if scheme.startswith("inv"):
        a = float(scheme.split("^")[1])
        return torch.tensor([(1.0 / p.get(x, P_UNSEEN)) ** a for x in t])
    if scheme.startswith("sif"):
        a = float(scheme.split("=")[1])
        return torch.tensor([a / (a + p.get(x, P_UNSEEN)) for x in t])
    if scheme == "nll2":
        return nll_of(ids).pow(2.0)
    raise ValueError(scheme)

SCHEMES = ["uniform", "inv^1.0", "inv^2.0", "sif=1e-4", "sif=1e-3", "nll2"]

nll_cache = {}
def target(ids, scheme):
    e = emb(ids).float()
    w = wvec(ids, scheme).unsqueeze(-1)
    return ((e * w).sum(0) / w.sum().clamp_min(1e-6)) @ proj

print(f"\n{'scheme':10s}  {'intra-source':>12s}  {'global':>8s}   (cos moyen inter-chunks, bas=bon)")
for scheme in SCHEMES:
    all_t, intra = [], []
    for picks in per_src:
        ts = torch.stack([target(ids, scheme) for ids in picks])
        ts = ts / ts.norm(dim=1, keepdim=True)
        cc = (ts @ ts.T)[~torch.eye(len(ts), dtype=torch.bool)]
        intra.append(float(cc.mean()))
        all_t.append(ts)
    T = torch.cat(all_t)
    cg = (T @ T.T)[~torch.eye(len(T), dtype=torch.bool)]
    print(f"{scheme:10s}  {sum(intra)/len(intra):12.3f}  {float(cg.mean()):8.3f}")

# sanité qualitative : top-6 tokens pondérés, 1 chunk code + 1 chunk prose
for si, label in ((0, names[0]), (7 if len(names) > 7 else 1, names[min(7, len(names)-1)])):
    ids = per_src[si][0]
    print(f"\n[{label}] extrait : {tok.decode(ids[:48].tolist())!r}")
    for scheme in ("inv^1.0", "sif=1e-4", "nll2"):
        w = wvec(ids, scheme)
        top = w.topk(min(6, len(w))).indices
        print(f"  {scheme:9s} top: " +
              " | ".join(tok.decode([ids[i]]).strip() or "␣" for i in top))
