"""dsv6 native — depth-stratified GAP curve on a trained checkpoint.

Loads final.pt and runs evaluate_by_depth on the held split with a CONTROLLED,
large n per depth => the reliable "does the bank hold as the conversation deepens?"
curve the 8-conv live eval was too noisy/sparse to give.

    python -m deepseek_v4_mini.code_defer_depthcurve \
        deepseek_v4_mini/configs/code_defer_native_v1.yaml \
        checkpoints/code_defer_native_ragged/final.pt
"""
import sys, yaml, torch
from transformers import AutoTokenizer
from .config import ThoughtBankConfig
from .model import ThoughtBankLM
from .code_data import CodeChunkStream
from .code_defer_native import evaluate_by_depth


def main(cfg_path, ckpt_path, n_per=48):
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
    print(f"loaded {ckpt_path} @step {ck.get('step','?')} | {model.num_params():,} params "
          f"| max_mem {cfg.max_mem} mem_seed {cfg.mem_seed_slots}\n")

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
              sources=d.get("sources"), seed=777)
    stream = CodeChunkStream(tok, split="held", **sd)
    # weighted mix => one curve PER SOURCE (a blended curve would hide a domain failure)
    views = ([(nm, stream.source_stream(i)) for i, nm in enumerate(stream.src_names)]
             if len(stream.src_files) > 1 else [("", stream)])

    depths = [1, 2, 3, 4, 5, 6, 8, 10, 12]        # #chunks written before predicting the next
    for nm, vw in views:
        bd = evaluate_by_depth(model, vw, dev, think_id, blank_id, defer_len,
                               depths, n_per, amp=False)
        tag = f" [{nm}]" if nm else ""
        print(f"depth-stratified GAP (n_per={n_per}), held{tag}:")
        print(f"{'writes':>7} {'GAP':>9} {'car':>8} {'res':>8} {'n':>5}")
        for dd in depths:
            r = bd[dd]
            print(f"{dd:>7} {r['gap']:>+9.3f} {r['car']:>8.3f} {r['res']:>8.3f} {r['n']:>5}")
        print()


if __name__ == "__main__":
    npq = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    main(sys.argv[1], sys.argv[2], npq)
