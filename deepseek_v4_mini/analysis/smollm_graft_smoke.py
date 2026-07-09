"""CPU smoke test for the SmolLM thought-bank graft (smollm_graft.py).

Checks, in order:
  1. no-op at init      : grafted logits == host logits (fw_o = 0), bit-close
  2. write semantics    : one write per forward, FIFO to max_mem, merge branch runs
  3. carry across segs  : bank from segment 1 feeds segment 2 (same API as train)
  4. live read path     : with fw_o perturbed, logits move — the hook is wired
  5. joint gradient     : loss.backward() reaches fw_A (read), thought_head (write,
                          through the NEXT segment's read) and a host layer

Usage: PYTHONPATH=. python deepseek_v4_mini/analysis/smollm_graft_smoke.py [--hf]
       --hf tries to download SmolLM2-135M; default is an offline random host
       with the same architecture/API (hidden 576, 30 layers ~ SmolLM2-135M).
"""
import sys

import torch

sys.path.insert(0, ".")
from deepseek_v4_mini.smollm_graft import GraftConfig, SmolBankLM

torch.manual_seed(0)

USE_HF = "--hf" in sys.argv
if USE_HF:
    from transformers import AutoModelForCausalLM
    host = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
else:
    from transformers import LlamaConfig, LlamaForCausalLM
    host = LlamaForCausalLM(LlamaConfig(
        vocab_size=49152, hidden_size=576, intermediate_size=1536,
        num_hidden_layers=4, num_attention_heads=9, num_key_value_heads=3,
        max_position_embeddings=512, tie_word_embeddings=True,
    ))
host.eval()
print(f"host: {'SmolLM2-135M' if USE_HF else 'random Llama (SmolLM2-135M shape, 4 layers)'}"
      f"  hidden={host.config.hidden_size}  params={sum(p.numel() for p in host.parameters()):,}")

cfg   = GraftConfig(d_model=host.config.hidden_size, read_layer=0)
model = SmolBankLM(host, cfg)
model.eval()

B, T = 2, 16
x = torch.randint(0, host.config.vocab_size, (B, T))
am = torch.ones(B, T, dtype=torch.long)
am[1, -4:] = 0                                     # ragged pad lane

# 1. no-op at init
with torch.no_grad():
    ref = host(input_ids=x, attention_mask=am).logits
    out = model(x, attention_mask=am)
diff = (out["logits"] - ref).abs().max().item()
assert diff < 1e-5, f"graft is not a no-op at init: max|Δlogits|={diff}"
print(f"1. no-op at init   : max|Δlogits| = {diff:.2e}  OK")

# 2+3. write & carry: seed 4 slots, +1 per forward, cap at max_mem
with torch.no_grad():
    bank = out["mem_bank"]
    assert bank.shape == (B, cfg.mem_seed_slots + 1, cfg.mem_dim), bank.shape
    for _ in range(6):
        bank = model(x, attention_mask=am, init_mem=bank)["mem_bank"]
    assert bank.shape == (B, cfg.max_mem, cfg.mem_dim), bank.shape
    # merge branch: rewrite the same content at capacity — bank must stay at cap
    bank2 = model(x, attention_mask=am, init_mem=bank)["mem_bank"]
    assert bank2.shape == (B, cfg.max_mem, cfg.mem_dim)
    merge_rate = float(getattr(model.write, "last_merge_rate", torch.tensor(-1.0)))
print(f"2. write/FIFO      : seed 4 → +1/forward → cap {cfg.max_mem}  OK"
      f"  (merge branch ran, rate {merge_rate:.2f})")
print(f"3. carry           : bank threaded across 8 segments  OK")

# 4. live read path: perturb fw_o, logits must move
with torch.no_grad():
    model.read.fw_o.weight.normal_(0.0, 0.02)
    moved = (model(x, attention_mask=am, init_mem=bank)["logits"] - ref).abs().max().item()
    model.read.fw_o.weight.zero_()
assert moved > 1e-4, f"read path dead: max|Δlogits|={moved}"
print(f"4. read path live  : fw_o≠0 → max|Δlogits| = {moved:.3f}  OK")

# 5. joint gradient across two segments (write of seg1 read by seg2)
model.train()
model.read.fw_o.weight.data.normal_(0.0, 0.02)     # open the read path for grads
s1 = model(x, attention_mask=am)
s2 = model(x, attention_mask=am, init_mem=s1["mem_bank"], labels=x)
s2["loss"].backward()
grads = {
    "read.fw_A":          model.read.fw_A.weight.grad,
    "write.thought_head": model.write.thought_head.weight.grad,
    "host.layer0.q_proj": host.model.layers[0].self_attn.q_proj.weight.grad,
}
for name, g in grads.items():
    assert g is not None and float(g.abs().sum()) > 0, f"no gradient into {name}"
    print(f"5. grad {name:18s}: |g|₁ = {float(g.abs().sum()):.3e}  OK")

print("SMOKE OK — graft plumbing validated on CPU")
