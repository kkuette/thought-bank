"""SFT loop: SmolLM2 + thought-bank graft on verbalized bank tasks.

Recipe (memory dsv4mini-real-data-sft-smollm, lessons dsv4y/dsv4z/dsv5d/dsv5f):
  1. TWO optimizer groups — host at a low LR (protect the pretrained circuit),
     graft (read+write) at full LR with linear warmup;
  2. REPLAY — a fraction of conversations are pure UltraChat turns (no bank
     task); the bank conversations' distractor replies are themselves real
     supervised turns, so replay is woven into the task data too;
  3. RAMP — the proportion of bank conversations ramps up (the new objective
     arrives gradually: the dsv4z/dsv5f lesson applied at data level);
  4. TBPTT — one segment = one forward = one write; the bank is carried and
     detached every `bptt_window` segments; loss averaged per conversation.

Eval = the content-gap protocol: query-answer accuracy with the bank CARRIED
vs RESET at every segment, on train-values and held-values splits. The
carried-vs-reset gap is the whole claim; held values test decode
generalization to words never used as labels.

Usage:
  PYTHONUNBUFFERED=1 python -m deepseek_v4_mini.sft_train deepseek_v4_mini/configs/sft_smollm_v1.yaml
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from typing import Optional

import torch
import yaml
from torch.optim import AdamW

from .smollm_graft import GraftConfig, SmolBankLM
from .verbal_tasks import VerbalRuleGen, VerbalTaskConfig, UltraChatTurns


# ── Conversation plumbing ────────────────────────────────────────────────────

class ConvStream:
    """Group the per-segment stream of VerbalRuleGen into conversations."""

    def __init__(self, gen: VerbalRuleGen) -> None:
        self.it = iter(gen)
        self.buf = None

    def next_conv(self) -> list[dict]:
        segs = [self.buf] if self.buf is not None else [next(self.it)]
        self.buf = None
        while True:
            s = next(self.it)
            if bool(s["reset"][0]):
                self.buf = s
                return segs
            segs.append(s)


def replay_conversation(gen: VerbalRuleGen, pool: UltraChatTurns,
                        n_turns: int, batch: int) -> list[dict]:
    """A pure-replay conversation: real dialogue turns only, no bank task."""
    segs = []
    for si in range(n_turns):
        rows, labs = [], []
        for _ in range(batch):
            u, a = pool.sample()
            ids, lab = gen._encode(u, a, cap=gen.cfg.distractor_max_len)
            rows.append(ids); labs.append(lab)
        T = max(len(r) for r in rows)
        x  = torch.full((batch, T), gen.pad_id, dtype=torch.long)
        y  = torch.full((batch, T), -100, dtype=torch.long)
        am = torch.zeros((batch, T), dtype=torch.long)
        for b, (r, l) in enumerate(zip(rows, labs)):
            x[b, :len(r)] = torch.tensor(r); y[b, :len(l)] = torch.tensor(l)
            am[b, :len(r)] = 1
        segs.append({"input_ids": x, "attention_mask": am, "labels": y,
                     "kind": "replay", "ans_pos": torch.full((batch,), -1),
                     "ans_ids": torch.full((batch,), -1),
                     "tf_ids": torch.full((batch,), -1)})
    return segs


# ── Eval: content-gap on query answers ──────────────────────────────────────

@torch.no_grad()
def eval_gap(model, streams: dict[str, ConvStream], device, n_convs: int,
             amp_dtype) -> dict:
    model.eval()
    out: dict[str, float] = {}
    for split, stream in streams.items():
        for mode in ("carried", "reset"):
            hits = tot = 0
            for _ in range(n_convs):
                segs = stream.next_conv()
                bank = None
                for s in segs:
                    x  = s["input_ids"].to(device)
                    am = s["attention_mask"].to(device)
                    with torch.autocast(device.type, dtype=amp_dtype,
                                        enabled=device.type == "cuda"):
                        o = model(x, attention_mask=am,
                                  init_mem=(bank if mode == "carried" else None))
                    bank = o["mem_bank"]
                    if s["kind"] == "query":
                        for b in range(x.size(0)):
                            p = int(s["ans_pos"][b])
                            pred = int(o["logits"][b, p - 1].argmax())
                            hits += int(pred == int(s["ans_ids"][b])); tot += 1
            out[f"acc_{split}_{mode}"] = hits / max(tot, 1)
    model.train()
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main(cfg_path: str) -> None:
    raw = yaml.safe_load(open(cfg_path))
    t   = raw["training"]
    d   = raw["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16
    torch.manual_seed(int(t.get("seed", 0)))

    # host + graft
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    if raw["host"] == "random":                     # CPU smoke path
        from transformers import LlamaConfig, LlamaForCausalLM
        host = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(tok), hidden_size=576, intermediate_size=1536,
            num_hidden_layers=4, num_attention_heads=9, num_key_value_heads=3,
            max_position_embeddings=512, tie_word_embeddings=True))
    else:
        from transformers import AutoModelForCausalLM
        host = AutoModelForCausalLM.from_pretrained(
            raw["host"], torch_dtype=torch.float32)
    gcfg  = GraftConfig(d_model=host.config.hidden_size,
                        **{k: v for k, v in raw.get("graft", {}).items()})
    model = SmolBankLM(host, gcfg).to(device)
    n_graft = sum(p.numel() for m in (model.read, model.write) for p in m.parameters())
    print(f"host {raw['host']}  params {sum(p.numel() for p in host.parameters()):,}"
          f"  | graft {n_graft:,}  read_layer {gcfg.read_layer}", flush=True)

    # data: bank conversations + replay pool + eval streams
    def _vcfg(split_seed: int, **over) -> VerbalTaskConfig:
        import dataclasses
        fields = {f.name for f in dataclasses.fields(VerbalTaskConfig)}
        base = {k: v for k, v in d.items() if k in fields}
        base["seed"] = split_seed
        base.update(over)
        return VerbalTaskConfig(**base)

    gen_train = VerbalRuleGen(tok, _vcfg(int(t.get("seed", 0))), split="train")
    train_stream = ConvStream(gen_train)
    replay_pool = (gen_train.turn_pool if gen_train.turn_pool is not None
                   else UltraChatTurns(tok, _vcfg(int(t.get("seed", 0)),
                                                  distractor_source="ultrachat")))
    eval_bs = int(d.get("eval_batch_size", d["batch_size"]))
    eval_streams = {
        "train": ConvStream(VerbalRuleGen(tok, _vcfg(1234, batch_size=eval_bs), split="train")),
        "held":  ConvStream(VerbalRuleGen(tok, _vcfg(5678, batch_size=eval_bs), split="held")),
    }

    # two optimizers (host protected on AdamW; graft on AdamW or Muon).
    # opt_bank "muon": the graft's 2-D matrices (read hypernets fw_A/fw_B/fw_o
    # + write heads) get Muon with rms_match — the synthetic recipe's
    # optimizer (dsv4m/dsv4w); non-matrix params (biases, norms) ride Muon's
    # bundled AdamW at lr_bank.
    lr_host, lr_bank = float(t["lr_host"]), float(t["lr_bank"])
    wd_bank = float(t.get("weight_decay", 0.01))
    graft_params = [p for m in (model.read, model.write) for p in m.parameters()]
    host_params  = [p for p in host.parameters() if p.requires_grad]
    opt_host = AdamW([{"params": host_params, "lr": lr_host,
                       "weight_decay": 0.0}])
    opt_kind = str(t.get("opt_bank", "adamw"))
    if opt_kind == "muon":
        from .train import Muon
        muon_lr = float(t.get("muon_lr", 3.0e-3))
        mats = [p for p in graft_params if p.ndim == 2]
        rest = [p for p in graft_params if p.ndim != 2]
        opt_graft = Muon(mats, lr=muon_lr, momentum=0.95, nesterov=True,
                         ns_steps=10, wd=wd_bank, rms_match=True,
                         adam_params=rest, adam_lr=lr_bank, adam_wd=wd_bank)
        graft_peak = muon_lr
        print(f"graft optimizer: Muon lr {muon_lr} rms_match "
              f"({sum(p.numel() for p in mats):,} matrix params) + bundled "
              f"AdamW lr {lr_bank} ({sum(p.numel() for p in rest):,})",
              flush=True)
    else:
        opt_graft = AdamW([{"params": graft_params, "lr": lr_bank,
                            "weight_decay": wd_bank}])
        graft_peak = lr_bank

    def _set_lrs(scale: float, decay: float) -> float:
        """Warmup×decay on the graft, decay on the host; return graft lr."""
        opt_host.param_groups[0]["lr"] = lr_host * decay
        g_lr = graft_peak * scale * decay
        for gp in opt_graft.param_groups:
            gp["lr"] = g_lr
        ad = getattr(opt_graft, "_adam", None)
        if ad is not None:
            for gp in ad.param_groups:
                gp["lr"] = lr_bank * scale * decay
        return g_lr

    warmup = int(t.get("warmup_steps", 100))
    # WSD decay: after the teacher anneal ends, ramp both LR groups down to
    # wsd_floor×peak by the final step (dsv4m lesson — the terminal WSD decay
    # is what consolidates GENERALIZATION, not just train).
    wsd_decay = bool(t.get("wsd_decay", False))
    wsd_floor = float(t.get("wsd_floor", 0.0))

    steps      = int(t["steps"])
    W          = int(t.get("bptt_window", 8))
    clip       = float(t.get("grad_clip", 1.0))
    log_every  = int(t.get("log_every", 20))
    eval_every = int(t.get("eval_every", 200))
    save_every = int(t.get("save_every", 500))
    save_dir   = t["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(t["metrics_file"]), exist_ok=True)
    metrics = open(t["metrics_file"], "a", buffering=1)
    writer = None
    if bool(t.get("tensorboard", True)):
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(os.path.dirname(t["metrics_file"]), "tb")
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"tensorboard → {tb_dir}", flush=True)

    # bank-conversation proportion ramp (data-level curriculum)
    p0, p1   = float(t.get("bank_p0", 0.25)), float(t.get("bank_p1", 0.75))
    ramp_end = int(t.get("bank_ramp_steps", steps // 3))
    rng = random.Random(int(t.get("seed", 0)) + 99)

    # ── Teacher bootstrap (the dsv5f recipe, one level up) ──────────────────
    # The joint write/read gradient is stuck at the ignore-bank fixed point
    # from scratch at EVERY scale tried (credit assignment, not features —
    # decode-neq-apply). Kick: on presentation segments, blend the written
    # slot toward a FIXED code of the value word (host embedding × frozen
    # random projection, RMS 1 — the Fourier-codes analogue) + cosine distill;
    # β anneals 1→0 so the write ends up owning its own code. Teacher is
    # never active at eval (eval_gap calls the bare model).
    tf_cfg = raw.get("teacher", {}) or {}
    tf_on  = bool(tf_cfg.get("enabled", False))
    tf_dw  = float(tf_cfg.get("distill_weight", 2.0))
    tf_a0, tf_a1 = (int(v) for v in tf_cfg.get("anneal", [800, 1600]))
    # ans_ce_below trigger (train.py ce_below transposed, PULL-EARLIER only —
    # the fixed [a0,a1] stays as fallback): β holds at 1 until the EMA of the
    # CE at ANSWER positions (not the diluted conversation loss) proves the
    # read exploits the teacher-blended bank, then the anneal runs
    # [now, now+anneal_len]. Threshold = ln(n_values) − margin (chance-anchored).
    tf_trigger = str(tf_cfg.get("anneal_trigger", ""))
    tf_alen    = int(tf_cfg.get("anneal_len", tf_a1 - tf_a0))
    tf_margin  = float(tf_cfg.get("margin", 0.5))
    tf_ans_ema: Optional[float] = None
    tf_fired = False
    tf_proj = tf_codes = None
    if tf_on:
        g = torch.Generator().manual_seed(1789)
        tf_proj = torch.randn(host.config.hidden_size, gcfg.mem_dim,
                              generator=g).to(device) / host.config.hidden_size ** 0.5
        with torch.no_grad():
            emb = host.get_input_embeddings().weight.detach().to(device)
            c = emb @ tf_proj
            tf_codes = (c / c.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-6))
        print(f"teacher ON: fixed value codes (emb×proj, RMS 1), "
              f"distill_w {tf_dw}, anneal [{tf_a0},{tf_a1}]", flush=True)

    def _beta(s: int) -> float:
        if not tf_on:
            return 0.0
        a0, a1 = (tf_a0, tf_a1)
        if tf_fired and tf_fired_at + tf_alen < a1:      # pulled earlier
            a0, a1 = tf_fired_at, tf_fired_at + tf_alen
        if s >= a1:
            return 0.0
        return 1.0 if s <= a0 else 1.0 - (s - a0) / max(1, a1 - a0)

    tf_fired_at = 0

    model.train()
    ema = {"bank": None, "replay": None}
    tf_distill_last: Optional[float] = None
    t0 = time.time()
    for step in range(1, steps + 1):
        # LR schedule: linear warmup on the graft group; optional WSD decay
        # of BOTH groups after the teacher anneal ends (post-β=0 consolidation
        # at decaying LR closes the train/held gap — dsv4m).
        scale = min(1.0, step / max(1, warmup))
        decay = 1.0
        if wsd_decay:
            a1 = tf_a1                                     # mirror _beta's a1
            if tf_on and tf_fired and tf_fired_at + tf_alen < tf_a1:
                a1 = tf_fired_at + tf_alen
            if step > a1 and steps > a1:
                frac = 1.0 - (step - a1) / max(1, steps - a1)   # 1→0 linear
                decay = wsd_floor + (1.0 - wsd_floor) * frac
        lr_now = _set_lrs(scale, decay)

        bank_p = p0 + (p1 - p0) * min(1.0, step / max(1, ramp_end))
        is_bank = rng.random() < bank_p
        segs = (train_stream.next_conv() if is_bank
                else replay_conversation(gen_train, replay_pool,
                                         n_turns=int(d.get("replay_turns", 6)),
                                         batch=int(d["batch_size"])))

        opt_host.zero_grad(set_to_none=True)
        opt_graft.zero_grad(set_to_none=True)
        bank, win_loss, conv_loss, n_sup = None, 0.0, 0.0, 0
        for i, s in enumerate(segs):
            x  = s["input_ids"].to(device)
            am = s["attention_mask"].to(device)
            y  = s["labels"].to(device)
            with torch.autocast(device.type, dtype=amp_dtype,
                                enabled=device.type == "cuda"):
                o = model(x, attention_mask=am, init_mem=bank, labels=y)
            bank = o["mem_bank"]
            seg_loss = o["loss"]
            tf_ids = s.get("tf_ids")
            beta = _beta(step)
            if tf_on and beta > 0.0 and tf_ids is not None and bool((tf_ids >= 0).all()):
                w0  = bank[:, -1]
                t_s = tf_codes[tf_ids.to(device)].to(w0.dtype)
                distill = (1.0 - torch.nn.functional.cosine_similarity(
                    w0.float(), t_s.detach().float(), dim=1)).mean()
                seg_loss = seg_loss + tf_dw * beta * distill
                code = (beta * t_s + (1.0 - beta) * w0).unsqueeze(1)
                bank = torch.cat([bank[:, :-1], code], dim=1)
                tf_distill_last = float(distill.detach())
            if tf_on and s["kind"] == "query":
                # answer-position CE telemetry (trigger signal, no grad)
                with torch.no_grad():
                    pos = s["ans_pos"].to(device)
                    lg  = o["logits"][torch.arange(x.size(0), device=device), pos - 1]
                    ace = torch.nn.functional.cross_entropy(
                        lg.float(), s["ans_ids"].to(device))
                tf_ans_ema = (float(ace) if tf_ans_ema is None
                              else 0.95 * tf_ans_ema + 0.05 * float(ace))
            win_loss = win_loss + seg_loss / len(segs)
            conv_loss += float(o["loss"].detach()); n_sup += 1
            if (i + 1) % W == 0 or i == len(segs) - 1:
                win_loss.backward()
                win_loss = 0.0
                bank = bank.detach()
        torch.nn.utils.clip_grad_norm_(host_params + graft_params, clip)
        opt_host.step()
        opt_graft.step()

        if (tf_on and tf_trigger == "ans_ce_below" and not tf_fired
                and step < tf_a0 and tf_ans_ema is not None
                and tf_ans_ema < math.log(len(gen_train.values)) - tf_margin):
            tf_fired, tf_fired_at = True, step
            print(f"[teacher] ans-CE EMA {tf_ans_ema:.3f} < "
                  f"ln({len(gen_train.values)})−{tf_margin} → anneal pulled to "
                  f"[{step},{step + tf_alen}]", flush=True)

        kind = "bank" if is_bank else "replay"
        cl = conv_loss / max(n_sup, 1)
        ema[kind] = cl if ema[kind] is None else 0.95 * ema[kind] + 0.05 * cl
        if step % log_every == 0:
            msg = {"step": step, "loss": round(cl, 4), "kind": kind,
                   "ema_bank": None if ema["bank"] is None else round(ema["bank"], 4),
                   "ema_replay": None if ema["replay"] is None else round(ema["replay"], 4),
                   "bank_p": round(bank_p, 3), "lr_bank": lr_now,
                   "tok_s": None}
            print(f"step {step:5d}  loss {cl:.3f} ({kind})  "
                  f"ema bank {msg['ema_bank']} replay {msg['ema_replay']}  "
                  f"bank_p {bank_p:.2f}  {(time.time()-t0)/step:.2f}s/step", flush=True)
            metrics.write(json.dumps(msg) + "\n")
            if writer is not None:
                writer.add_scalar(f"loss/{kind}", cl, step)
                for k in ("bank", "replay"):
                    if ema[k] is not None:
                        writer.add_scalar(f"loss/ema_{k}", ema[k], step)
                writer.add_scalar("sched/bank_p", bank_p, step)
                writer.add_scalar("sched/lr_bank", lr_now, step)
                if tf_on:
                    writer.add_scalar("teacher/beta", _beta(step), step)
                    if tf_distill_last is not None:
                        writer.add_scalar("teacher/distill", tf_distill_last, step)
                    if tf_ans_ema is not None:
                        writer.add_scalar("teacher/ans_ce_ema", tf_ans_ema, step)
                # write-head telemetry (last conversation's final segment)
                for name, val in (("alpha", model.write.last_write_alpha),
                                  ("redund", model.write.last_write_redundancy),
                                  ("merge_rate", getattr(model.write, "last_merge_rate", None))):
                    if val is not None:
                        writer.add_scalar(f"write/{name}",
                                          float(torch.as_tensor(val).detach()), step)

        if step % eval_every == 0 or step == steps:
            ev = eval_gap(model, eval_streams, device,
                          n_convs=int(t.get("eval_convs", 4)), amp_dtype=amp_dtype)
            gap_tr = ev["acc_train_carried"] - ev["acc_train_reset"]
            gap_he = ev["acc_held_carried"] - ev["acc_held_reset"]
            print(f"[eval @{step}] train {ev['acc_train_carried']:.3f}/"
                  f"{ev['acc_train_reset']:.3f} (gap {gap_tr:+.3f})  "
                  f"held {ev['acc_held_carried']:.3f}/{ev['acc_held_reset']:.3f}"
                  f" (gap {gap_he:+.3f})  chance {gen_train.chance:.3f}", flush=True)
            metrics.write(json.dumps({"step": step, **{k: round(v, 4) for k, v in ev.items()},
                                      "gap_train": round(gap_tr, 4),
                                      "gap_held": round(gap_he, 4)}) + "\n")
            if writer is not None:
                for k, v in ev.items():
                    writer.add_scalar(f"eval/{k}", v, step)
                writer.add_scalar("eval/gap_train", gap_tr, step)
                writer.add_scalar("eval/gap_held", gap_he, step)
                # memory pictures: one carried conversation, three figures
                from .bank_viz import log_bank_figures
                model.eval()
                log_bank_figures(writer, model, eval_streams["train"].next_conv(),
                                 device, step, amp_dtype)
                model.train()

        if step % save_every == 0 or step == steps:
            ck = {"step": step,
                  "read": model.read.state_dict(),
                  "write": model.write.state_dict(),
                  "graft_cfg": gcfg.__dict__}
            if step == steps and bool(t.get("save_host_final", True)):
                ck["host"] = host.state_dict()
            torch.save(ck, os.path.join(save_dir, f"step_{step}.pt" if step != steps
                                        else "final.pt"))
    if writer is not None:
        writer.close()
    print("done.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
