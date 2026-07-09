"""CPU smoke test for verbal_tasks.py + SmolBankLM end-to-end.

Checks:
  1. value pool     : enough single-token values, train/held disjoint
  2. sample conv    : decoded text is well-formed (printed for eyeballing)
  3. segment chain  : a full conversation runs through SmolBankLM with the
                      bank carried; loss finite on every supervised segment
  4. metric hook    : query answer-token accuracy extracts cleanly; at init
                      (untrained graft, random or real host) it sits near
                      chance both with carried and reset bank
  5. TBPTT backward : loss summed over a 2-segment window backprops through
                      the carried bank without error

Usage: PYTHONPATH=. python deepseek_v4_mini/analysis/verbal_tasks_smoke.py [--hf]
"""
import sys

import torch

sys.path.insert(0, ".")
from transformers import AutoTokenizer

from deepseek_v4_mini.smollm_graft import GraftConfig, SmolBankLM
from deepseek_v4_mini.verbal_tasks import VerbalRuleGen, VerbalTaskConfig

torch.manual_seed(0)
USE_HF = "--hf" in sys.argv

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
if USE_HF:
    from transformers import AutoModelForCausalLM
    host = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
else:
    from transformers import LlamaConfig, LlamaForCausalLM
    host = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tok), hidden_size=576, intermediate_size=1536,
        num_hidden_layers=4, num_attention_heads=9, num_key_value_heads=3,
        max_position_embeddings=512, tie_word_embeddings=True,
    ))
host.eval()
model = SmolBankLM(host, GraftConfig(d_model=host.config.hidden_size))
model.eval()

# 1. pools  (--uc: real UltraChat turns as distractors)
cfg = VerbalTaskConfig(batch_size=4, n_pairs=2, turns=6, seed=1,
                       distractor_source=("ultrachat" if "--uc" in sys.argv else "canned"),
                       distractor_p=(0.5 if "--uc" in sys.argv else 0.35))
gtr = VerbalRuleGen(tok, cfg, split="train")
ghe = VerbalRuleGen(tok, cfg, split="held")
inter = set(gtr.values) & set(ghe.values)
print(f"1. values: train {len(gtr.values)} / held {len(ghe.values)} "
      f"(chance {gtr.chance:.3f}), overlap {len(inter)}")
assert not inter and len(ghe.values) >= 8

# 2. decode one conversation (lane 0)
it = iter(gtr)
segs, kinds = [], []
while True:
    s = next(it)
    if s["reset"][0] and segs:
        first_next = s
        break
    segs.append(s); kinds.append(s["kind"])
print(f"2. conversation: {len(segs)} segments  kinds={kinds}")
for s in segs:
    ids = s["input_ids"][0][s["attention_mask"][0].bool()]
    print("   |", tok.decode(ids))

# 3+4. run the chain, collect query acc (carried) then again with reset banks
@torch.no_grad()
def run_chain(carry: bool):
    bank, hits, tot = None, 0, 0
    losses = []
    for s in segs:
        out = model(s["input_ids"], attention_mask=s["attention_mask"],
                    init_mem=(bank if carry else None), labels=s["labels"])
        losses.append(float(out["loss"]))
        bank = out["mem_bank"]
        if s["kind"] == "query":
            for b in range(s["input_ids"].size(0)):
                p = int(s["ans_pos"][b])
                # logits at p-1 predict position p (the value token)
                pred = int(out["logits"][b, p - 1].argmax())
                hits += int(pred == int(s["ans_ids"][b])); tot += 1
    return hits / max(tot, 1), tot, losses

acc_c, nq, losses = run_chain(carry=True)
acc_r, _, _ = run_chain(carry=False)
assert all(torch.isfinite(torch.tensor(losses))), "non-finite loss"
print(f"3. chain: {len(segs)} forwards, losses finite (CE first/last "
      f"{losses[0]:.2f}/{losses[-1]:.2f})")
print(f"4. query acc @init: carried {acc_c:.3f} / reset {acc_r:.3f} "
      f"({nq} queries, chance {gtr.chance:.3f}) — gap is the TRAINED signal, ~0 now")

# 5. TBPTT backward across 2 segments through the carried bank
model.train()
model.read.fw_o.weight.data.normal_(0.0, 0.02)
o1 = model(segs[0]["input_ids"], attention_mask=segs[0]["attention_mask"],
           labels=segs[0]["labels"])
o2 = model(segs[1]["input_ids"], attention_mask=segs[1]["attention_mask"],
           init_mem=o1["mem_bank"], labels=segs[1]["labels"])
(o1["loss"] + o2["loss"]).backward()
g = model.write.thought_head.weight.grad
assert g is not None and float(g.abs().sum()) > 0
print(f"5. TBPTT window-2 backward through the bank: |g(write)|₁ = {float(g.abs().sum()):.2e}  OK")
print("SMOKE OK — verbal task plumbing validated")
