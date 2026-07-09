"""dsv6 native — QUALITATIVE decode on a LONG conversation. Take one held file with
many chunks, write them into the bank one at a time, and at increasing depths (d
writes accumulated) greedy-decode the opening of the NEXT chunk from the bank ALONE,
against the ground truth and the no-bank reset. Shows whether the bank keeps
producing sensible continuations deep into the conversation.

    python -m deepseek_v4_mini.code_defer_sample_deep \
        deepseek_v4_mini/configs/code_defer_native_v1.yaml \
        checkpoints/code_defer_native_ragged/final.pt
"""
import sys, yaml, torch
from transformers import AutoTokenizer
from .config import ThoughtBankConfig
from .model import ThoughtBankLM
from .code_data import CodeChunkStream


def _fill(x, tok, w):
    return torch.full((x.size(0), w), tok, dtype=x.dtype, device=x.device)


def _greedy(model, ref, blank_id, bank, dl):
    """Greedy-decode dl tokens from the deferred (all-<blank>) turn, bank fixed."""
    di = _fill(ref, blank_id, dl)
    out = torch.zeros(ref.size(0), dl, dtype=torch.long, device=ref.device)
    for i in range(dl):
        o = model(di, init_mem=bank)
        nt = o["logits"].float()[:, i].argmax(-1)
        out[:, i] = nt
        if i + 1 < dl:
            di[:, i + 1] = nt
    return out


@torch.no_grad()
def main(cfg_path, ckpt_path, depths=(1, 2, 4, 6, 8, 10)):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = yaml.safe_load(open(cfg_path)); d = raw["data"]
    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    for tkn in ("<think>", "<blank>"):
        if tkn not in tok.get_vocab():
            tok.add_special_tokens({"additional_special_tokens": [tkn]})
    think_id = tok.convert_tokens_to_ids("<think>")
    blank_id = tok.convert_tokens_to_ids("<blank>")

    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ThoughtBankConfig(**ck["cfg"])
    model = ThoughtBankLM(cfg).to(dev); model.load_state_dict(ck["model"]); model.eval()
    print(f"loaded {ckpt_path} @step {ck.get('step','?')} | max_mem {cfg.max_mem} seed {cfg.mem_seed_slots}\n")

    L, K = int(d["seq_len"]), int(d["chunks_per_conv"])
    defer_len = int(d.get("defer_len", 16))
    sd = dict(seq_len=L, chunks_per_conv=K, batch=1,
              n_files=int(d.get("n_files", 1500)),
              dataset=d.get("dataset", "codeparrot/codeparrot-clean-valid"),
              data_dir=d.get("data_dir", ""),
              stream_cap=int(d.get("stream_cap", 40000)),
              cache_dir=d.get("cache_dir", "data_cache"),
              content_key=d.get("content_key", "content"),
              config_name=d.get("config_name", ""),
              min_chunks=int(d.get("min_chunks", 1)),
              stream_skip=int(d.get("stream_skip", 0)),
              sources=d.get("sources"), seed=2024)
    stream = CodeChunkStream(tok, split="held", **sd)
    # weighted mix => one long conversation PER SOURCE
    views = ([(nm, stream.source_stream(i)) for i, nm in enumerate(stream.src_names)]
             if len(stream.src_files) > 1 else [("", stream)])

    def dec(ids):
        return tok.decode(ids, skip_special_tokens=False).replace("\n", "\\n")

    for nm, vw in views:
        # deepest held file this source has (web text runs shorter than code)
        f = max(vw.files, key=len)
        need = min(max(depths) + 1, len(f))
        tag = f" [{nm}]" if nm else ""
        print("#" * 80)
        print(f"long conversation{tag}: 1 held file, {len(f)} chunks of {L} tokens\n")

        bank = None
        for j in range(need):
            x = f[j].unsqueeze(0).to(dev)
            o = model(torch.cat([x, _fill(x, think_id, 1)], 1), init_mem=bank)
            bank = o["mem_bank"]
            d_now = j + 1                               # chunks written so far
            if d_now in depths and j + 1 < len(f):
                gt = f[j + 1][:defer_len].unsqueeze(0).to(dev)
                dl = gt.size(1)
                car = _greedy(model, gt, blank_id, bank, dl)
                res = _greedy(model, gt, blank_id, None, dl)
                acc = (car[0] == gt[0]).float().mean().item()
                print("=" * 80)
                print(f"[after {d_now} writes]  bank slots = {bank.size(1)} (seed {cfg.mem_seed_slots} + {d_now} writes)")
                print(f"  GROUND TRUTH next : {dec(gt[0].tolist())}")
                print(f"  BANK (carried)    : {dec(car[0].tolist())}   [tok acc {acc:.2f}]")
                print(f"  RESET (no bank)   : {dec(res[0].tolist())}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
