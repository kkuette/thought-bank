"""dsv6 native — QUALITATIVE sample: decode what the bank produces at the deferred
turn on held code. For each held chunk N: write the bank on [chunk, <think>], then
greedy-decode defer_len tokens of chunk N+1 from BLANK input, (a) with the carried
bank and (b) with no bank (reset ablation), against the ground-truth continuation
and the turn-0 full-context ceiling.

    python -m deepseek_v4_mini.code_defer_sample \
        deepseek_v4_mini/configs/code_defer_native_v1.yaml \
        checkpoints/code_defer_native_v1/final.pt  [n_examples]
"""
import sys, yaml, torch
from transformers import AutoTokenizer
from .config import ThoughtBankConfig
from .model import ThoughtBankLM
from .code_data import CodeChunkStream


def _fill(x, tok, w):
    return torch.full((x.size(0), w), tok, dtype=x.dtype, device=x.device)


def _greedy_from_bank(model, x_ref, blank_id, bank, defer_len, ban=()):
    """Greedy-decode defer_len tokens from the deferred (all-<blank>) turn. Position i
    reads the bank; we fill position i with the model's own argmax so later positions
    see a coherent prefix (bank stays fixed = init_mem). `ban` = special/action token
    ids excluded from the argmax (<think>/<blank> are control tokens, not content —
    and RL on the <think> row inflates its logit enough to win greedy everywhere)."""
    B = x_ref.size(0)
    di = _fill(x_ref, blank_id, defer_len)
    out = torch.zeros(B, defer_len, dtype=torch.long, device=x_ref.device)
    for i in range(defer_len):
        o = model(di, init_mem=bank)
        lg = o["logits"].float()[:, i]
        for b in ban:
            lg[:, b] = float("-inf")
        nt = lg.argmax(-1)
        out[:, i] = nt
        if i + 1 < defer_len:
            di[:, i + 1] = nt      # feed prediction forward (bank unchanged)
    return out


@torch.no_grad()
def main(cfg_path, ckpt_path, n_ex=6, split="held"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = yaml.safe_load(open(cfg_path)); d = raw["data"]
    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    for t in ("<think>", "<blank>"):
        if t not in tok.get_vocab():
            tok.add_special_tokens({"additional_special_tokens": [t]})
    think_id = tok.convert_tokens_to_ids("<think>")
    blank_id = tok.convert_tokens_to_ids("<blank>")

    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ThoughtBankConfig(**ck["cfg"])
    model = ThoughtBankLM(cfg).to(dev); model.load_state_dict(ck["model"]); model.eval()
    print(f"loaded {ckpt_path} @step {ck.get('step','?')} | {model.num_params():,} params\n")

    L, K = int(d["seq_len"]), int(d["chunks_per_conv"])
    defer_len = int(d.get("defer_len", 16))
    stream = CodeChunkStream(tok, split=split, seq_len=L, chunks_per_conv=K, batch=1,
                             n_files=int(d.get("n_files", 600)),
                             dataset=d.get("dataset", "codeparrot/codeparrot-clean-valid"),
                             data_dir=d.get("data_dir", ""),
                             stream_cap=int(d.get("stream_cap", 25000)),
                             seed=123)

    def dec(ids):
        return tok.decode(ids, skip_special_tokens=False).replace("\n", "\\n")

    shown = 0
    while shown < n_ex:
        segs = stream.next_conv(); bank = None
        for i, s in enumerate(segs):
            x = s["input_ids"].to(dev)
            bank_in = bank
            o = model(torch.cat([x, _fill(x, think_id, 1)], 1), init_mem=bank)
            bank = o["mem_bank"]
            if i >= len(segs) - 1 or shown >= n_ex:
                continue
            gt = segs[i + 1]["input_ids"][:, :defer_len].to(dev)         # ground truth
            dl = gt.size(1)                                              # ragged remainder may be < defer_len
            ban = (think_id, blank_id)
            car = _greedy_from_bank(model, x, blank_id, bank, dl, ban)    # bank
            res = _greedy_from_bank(model, x, blank_id, None, dl, ban)    # no bank
            acc = (car[0] == gt[0]).float().mean().item()
            acc_r = (res[0] == gt[0]).float().mean().item()
            print("=" * 78)
            print(f"[ex {shown}]  context tail: …{dec(x[0, -60:].tolist())}")
            print(f"  GROUND TRUTH next : {dec(gt[0].tolist())}")
            print(f"  BANK (carried)    : {dec(car[0].tolist())}   [tok acc {acc:.2f}]")
            print(f"  RESET (no bank)   : {dec(res[0].tolist())}   [tok acc {acc_r:.2f}]")
            shown += 1
            if shown >= n_ex:
                break


if __name__ == "__main__":
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    sp = sys.argv[4] if len(sys.argv) > 4 else "held"
    main(sys.argv[1], sys.argv[2], n, sp)
