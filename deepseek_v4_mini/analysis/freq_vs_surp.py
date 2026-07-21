"""Pondération FRÉQUENCE-INVERSE (sans ref, table unigram) vs SURPRISAL (nll ref)
sur les mêmes segs porteurs : discriminabilité (cos inter-faits, bas=bon) et
fidélité à la cible valeur (cos vs val_mask, haut=bon). Même harnais que
surp_vs_valmask.py. La table de fréquences est estimée sur 400 convs du
stream lui-même (proxy corpus)."""
import sys, torch
from collections import Counter
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))))
from transformers import AutoTokenizer
from deepseek_v4_mini.persona_chat_data import PersonaChatStream

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
D, M = 768, 512
g = torch.Generator().manual_seed(1789)
proj = torch.randn(D, M, generator=g) / D ** 0.5
emb = torch.nn.Embedding(49154, D)
torch.manual_seed(0); torch.nn.init.normal_(emb.weight, std=0.02)

# table unigram sur 400 convs (seed indépendante)
cnt, tot = Counter(), 0
s0 = PersonaChatStream(tok, seed=97)
for _ in range(400):
    for seg in s0.next_conv()["segs"]:
        ids = seg["input_ids"][0].tolist()
        cnt.update(ids); tot += len(ids)
p = {t: c / tot for t, c in cnt.items()}
P_UNSEEN = 0.5 / tot

SCAF = set()
for txt in ("<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>\n"):
    SCAF.update(tok(txt, add_special_tokens=False)["input_ids"])

def wfreq(ids, alpha, sif_a=None):
    w = []
    for t in ids.tolist():
        if t in SCAF:
            w.append(0.0); continue
        pt = p.get(t, P_UNSEEN)
        w.append((sif_a / (sif_a + pt)) if sif_a else (1.0 / pt) ** alpha)
    return torch.tensor(w)

s = PersonaChatStream(tok, seed=11, surprisal_ref="HuggingFaceTB/SmolLM2-135M-Instruct",
                      surprisal_device="cuda", surprisal_alpha=2.0)
segs = []
while len(segs) < 40:
    c = s.next_conv()
    if c["kind"] != "recall":
        continue
    for seg in c["segs"]:
        vm = seg.get("val_mask")
        if vm is not None and float(vm.sum()) > 0:
            segs.append(seg)

def report(name, weight_fn):
    cos_same, tgts = [], []
    for seg in segs:
        ids = seg["input_ids"][0]
        e = emb(ids).float()
        vm = seg["val_mask"][0]
        t_val = ((e * vm.unsqueeze(-1)).sum(0) / vm.sum()) @ proj
        w = weight_fn(seg).unsqueeze(-1)
        t_w = ((e * w).sum(0) / w.sum().clamp_min(1e-6)) @ proj
        cos_same.append(float(torch.cosine_similarity(t_val, t_w, dim=0)))
        tgts.append(t_w)
    T = torch.stack(tgts); T = T / T.norm(dim=1, keepdim=True)
    off = (T @ T.T)[~torch.eye(len(T), dtype=torch.bool)]
    print(f"{name:28s} cos vs val_mask = {sum(cos_same)/len(cos_same):.3f}"
          f" | cos inter-faits = {off.mean():.3f}")

report("surprisal nll^2 (ref)", lambda s_: s_["surp_w"][0])
for a in (0.5, 1.0, 2.0):
    report(f"freq-inverse (1/p)^{a}", lambda s_, a=a: wfreq(s_["input_ids"][0], a))
report("SIF a=1e-4", lambda s_: wfreq(s_["input_ids"][0], 0, sif_a=1e-4))
report("SIF a=1e-3", lambda s_: wfreq(s_["input_ids"][0], 0, sif_a=1e-3))
