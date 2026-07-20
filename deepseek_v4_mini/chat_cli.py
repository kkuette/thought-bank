"""dsv6 native — interactive chat with a checkpoint, bank carried across turns.

Reproduces the persona-eval protocol exactly (evaluate_math in code_defer_native):
each user turn is forwarded once to WRITE the bank; the reply is greedy-decoded
from the bare "<|im_start|>assistant\n" prefix with the bank as ONLY memory (the
conversation history is never in the window); the reply is then forwarded too so
the model's own words enter the bank. So what the model remembers across turns
is what the bank carries — that's the demo.

If the config has cascade_depth > 0, the FULL v3 bank runs like in training:
evicted live slots overflow into the cascade (seeds don't descend), deep layers
read the cascade levels via cascade_map, and --bank restores the cascade state
saved in the artifact. The --both/--ablated arm decodes with NO memory at all
(bank None + no cascade), same as the trainer's ablation.

    python -m deepseek_v4_mini.chat_cli \
        deepseek_v4_mini/configs/sft_persona_350m.yaml \
        /mnt/tb/checkpoints/v350_sft_persona/step_200.pt \
        [--bank /mnt/tb/checkpoints/v350_sft_persona/bank_step_200.pt] \
        [--both] [--max-new 48] [--temp 0.7 --top-p 0.9 --seed 0]

Options:
  --bank PATH   seed the session with a saved bank artifact (bank_step_N.pt):
                live bank + cascade state, dims checked against the model
  --both        decode each reply twice: live bank vs ablated (None) — the demo
                pair; only the live reply is written back into the bank
  --ablated     run the whole session with bank reads disabled (control arm)
  --max-new N   decode budget per reply (default 48)
  --temp T      sampling temperature; 0 = greedy (default 0)
  --top-p P     nucleus cutoff, only with --temp > 0 (default 0.9)
  --seed N      sampling seed (default: unseeded)

REPL commands: /reset (fresh bank+cascade), /save PATH (dump bank), /quit.
"""
import argparse

import torch
import yaml
from transformers import AutoTokenizer

from .cascade import CascadeMemory
from .config import ThoughtBankConfig
from .model import ThoughtBankLM

U_OPEN = "<|im_start|>user\n"
A_OPEN = "<|im_start|>assistant\n"
CLOSE = "<|im_end|>\n"


def _ids(tok, s, device):
    return torch.tensor(tok(s, add_special_tokens=False)["input_ids"],
                        dtype=torch.long, device=device).unsqueeze(0)


class Session:
    """Bank + cascade state, advanced exactly like the trainer's conv loop."""

    def __init__(self, model, cfg, tcfg, device, amp):
        self.model, self.cfg, self.device, self.amp = model, cfg, device, amp
        self.depth = int(tcfg.get("cascade_depth", 0) or 0)
        self.map = None
        if self.depth > 0:
            _cmap = tcfg.get("cascade_map")
            self.map = ([int(v) for v in _cmap] if _cmap else
                        [0] * (cfg.n_layers - self.depth)
                        + list(range(1, self.depth + 1)))
            assert len(self.map) == cfg.n_layers
        self.seed_slots = int(getattr(cfg, "mem_seed_slots", 0) or 0)
        self.reset()

    def reset(self):
        self.bank = None
        self.casc = CascadeMemory(self.depth, self.cfg.max_mem) \
            if self.depth else None
        self.n_evict = 0

    def load(self, path):
        bk = torch.load(path, map_location="cpu", weights_only=False)
        b = bk["bank"]
        if b is not None:
            if b.size(-1) != self.cfg.mem_dim or b.size(1) > self.cfg.max_mem:
                raise SystemExit(
                    f"banque incompatible: fichier {tuple(b.shape)} vs modèle "
                    f"max_mem {self.cfg.max_mem} x mem_dim {self.cfg.mem_dim}")
            self.bank = b.to(self.device)
        if self.depth and bk.get("casc") is not None:
            sd = bk["casc"]
            if sd["M"] != self.cfg.max_mem:
                raise SystemExit(f"cascade incompatible: M {sd['M']} vs "
                                 f"max_mem {self.cfg.max_mem}")
            self.casc = CascadeMemory.from_state(sd, self.device)
        self.n_evict = int(bk.get("n_evict", 0) or 0)
        print(f"bank seeded from {path} (step {bk.get('step', '?')}, writes "
              f"{bk.get('w_total', '?')}, cascade "
              f"{self.casc.stats() if self.casc else 'off'})", flush=True)

    def layer_banks(self, bank):
        if self.casc is None or bank is None:
            return None
        return self.casc.layer_banks(bank, self.map)

    @torch.no_grad()
    def write(self, x):
        """Forward a segment to advance bank + cascade (logits discarded)."""
        if self.casc is not None and self.bank is None:
            self.bank = self.model.thought_stream.seed_bank(
                x.size(0), self.device, next(self.model.parameters()).dtype)
        pre0 = (self.bank[:, 0].detach()
                if self.casc is not None and self.bank is not None
                and self.bank.size(1) >= self.cfg.max_mem else None)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            o = self.model(x, init_mem=self.bank,
                           layer_banks=self.layer_banks(self.bank))
        self.bank = o["mem_bank"]
        if pre0 is not None:
            self.n_evict += 1
            if self.n_evict > self.seed_slots:
                self.casc.push_slot(pre0)


def _pick(logits, temp, top_p):
    """Next token from the last-position logits: greedy if temp == 0, else
    temperature + nucleus (top-p) sampling."""
    if temp <= 0:
        return logits.argmax(-1, keepdim=True)
    probs = torch.softmax(logits.float() / temp, dim=-1)
    if top_p < 1.0:
        sp, si = probs.sort(dim=-1, descending=True)
        keep = sp.cumsum(-1) - sp < top_p           # keep at least the top token
        sp = sp * keep
        idx = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1)
        return si.gather(-1, idx)
    return torch.multinomial(probs, 1)


@torch.no_grad()
def _decode(model, prefix, bank, lb, max_new, stop_id, amp, temp=0.0, top_p=0.9):
    out = prefix
    for _ in range(max_new):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            o = model(out, init_mem=bank, layer_banks=lb)
        nt = _pick(o["logits"][:, -1], temp, top_p)
        out = torch.cat([out, nt], dim=1)
        if int(nt) == stop_id:
            break
    return out[:, prefix.size(1):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("ckpt")
    ap.add_argument("--bank", default=None, help="bank_step_N.pt to seed from")
    ap.add_argument("--both", action="store_true",
                    help="decode live AND ablated each turn")
    ap.add_argument("--ablated", action="store_true",
                    help="disable bank reads for the whole session")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--temp", type=float, default=0.0,
                    help="sampling temperature, 0 = greedy")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    raw = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = raw.get("training", {})
    amp = bool(tcfg.get("amp", False))

    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    add = [x for x in ("<think>", "<blank>") if x not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    stop_id = tok.convert_tokens_to_ids("<|im_end|>")

    mcfg = dict(raw["model"])
    mcfg["vocab_size"] = len(tok)
    cfg = ThoughtBankConfig(**mcfg)
    model = ThoughtBankLM(cfg).to(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    sess = Session(model, cfg, tcfg, device, amp)
    print(f"loaded {args.ckpt} (step {ck.get('step', '?')}) | "
          f"{model.num_params():,} params | max_mem {cfg.max_mem} | cascade "
          f"{'d' + str(sess.depth) if sess.depth else 'off'} | "
          f"{'ABLATED (no bank)' if args.ablated else 'bank live'}", flush=True)
    if args.bank:
        sess.load(args.bank)

    a_open = _ids(tok, A_OPEN, device)
    print("REPL: /reset /save PATH /quit — Ctrl-D quits\n", flush=True)
    while True:
        try:
            user = input("you> ").strip()
        except EOFError:
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            sess.reset()
            print("bank + cascade reset")
            continue
        if user.startswith("/save"):
            path = user.split(None, 1)[1] if " " in user else "bank_session.pt"
            torch.save({"step": -1,
                        "bank": None if sess.bank is None
                        else sess.bank.detach().cpu(),
                        "casc": None if sess.casc is None
                        else sess.casc.state_dict(),
                        "n_evict": sess.n_evict, "w_total": -1}, path)
            print(f"bank -> {path}")
            continue

        sess.write(_ids(tok, U_OPEN + user + CLOSE, device))
        if args.ablated:
            rd_bank, rd_lb = None, None
        else:
            rd_bank, rd_lb = sess.bank, sess.layer_banks(sess.bank)
        g = _decode(model, a_open, rd_bank, rd_lb, args.max_new, stop_id, amp,
                    args.temp, args.top_p)
        reply = tok.decode(g[0].tolist()).replace("<|im_end|>", "").strip()
        print(f"bot> {reply}")
        if args.both and not args.ablated:
            g0 = _decode(model, a_open, None, None, args.max_new, stop_id, amp,
                         args.temp, args.top_p)
            print(f"abl> {tok.decode(g0[0].tolist()).replace('<|im_end|>', '').strip()}")
        sess.write(_ids(tok, A_OPEN + reply + "\n" + CLOSE, device))


if __name__ == "__main__":
    main()
