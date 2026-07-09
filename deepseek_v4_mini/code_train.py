"""dsv6 — bank as cross-chunk long-context memory on real code + deferred <think>.

Design: memory dsv6-bank-code-memory-defer. Per code chunk (seq_len tokens):
  (1) in-context forward: standard next-token LM loss + ONE bank write;
  (2) deferred <think> forward: input [<think>, t0..t_{M-2}] (t = start of the
      NEXT chunk), predict [t0..t_{M-1}] — position 0 predicts t0 from the bank
      ALONE (nothing of the next chunk in-window = the causal memory claim).
Dual loss L = L_incontext + defer_weight * L_defer, summed over the conversation
of K chunks, bank carried (TBPTT = whole conversation).

Host is NOT frozen (freeze => ignore-bank fixed point): two optimizers, host on
AdamW at a small real LR, graft on Muon. The in-context code-LM loss is its own
anti-forgetting anchor (native objective). Success = deferred gap > 0 (carried
beats init_mem=None) AND in-context loss stays a competent LM (may rise then
converge; monotone divergence = damage).

    PYTHONUNBUFFERED=1 python -m deepseek_v4_mini.code_train deepseek_v4_mini/configs/code_defer_v1.yaml
"""
import os, sys, math, time, yaml, dataclasses
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

from .smollm_graft import GraftConfig, SmolBankLM
from .code_data import CodeChunkStream


def _append_think(x, am, think_id):
    """Append one <think> token so the model learns to emit it after context."""
    col = torch.full((x.size(0), 1), think_id, dtype=x.dtype, device=x.device)
    ones = torch.ones((x.size(0), 1), dtype=am.dtype, device=am.device)
    return torch.cat([x, col], 1), torch.cat([am, ones], 1)


@torch.no_grad()
def evaluate(model, stream, device, amp_dtype, think_id, defer_len, n_conv):
    model.eval()
    ic_loss = ic_n = 0.0
    d_car = d_res = d_car0 = d_res0 = dn = 0.0
    think_rel = 0.0
    for _ in range(n_conv):
        segs = stream.next_conv(); bank = None
        for i, s in enumerate(segs):
            x = s["input_ids"].to(device); am = s["attention_mask"].to(device)
            xt, amt = _append_think(x, am, think_id)          # emit <think> after context
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                o = model(xt, attention_mask=amt, init_mem=bank, labels=xt)
            bank = o["mem_bank"]; ic_loss += float(o["loss"]); ic_n += 1
            if i < len(segs) - 1:
                nxt = segs[i + 1]["input_ids"][:, :defer_len].to(device)          # [B, M] targets
                di = torch.full((x.size(0), defer_len), think_id, device=device)  # all-<think>, NO context
                for mode in ("car", "res"):
                    mem = bank if mode == "car" else None
                    with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                        od = model(di, attention_mask=torch.ones_like(di), init_mem=mem)
                    lg = od["logits"].float()                  # [B, M, V]: position i predicts nxt[:, i]
                    lall = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), nxt.reshape(-1))
                    l0 = torch.nn.functional.cross_entropy(lg[:, 0], nxt[:, 0])   # pos0 = pure bank
                    if mode == "car":
                        d_car += float(lall); d_car0 += float(l0)
                        think_rel += (model._last_read_rel or 0.0)
                    else: d_res += float(lall); d_res0 += float(l0)
                dn += 1
    model.train()
    return {"ic_ppl": math.exp(ic_loss / max(ic_n, 1)),
            "defer_car": d_car / max(dn, 1), "defer_res": d_res / max(dn, 1),
            "defer_gap": (d_res - d_car) / max(dn, 1),
            "defer_car0": d_car0 / max(dn, 1), "defer_res0": d_res0 / max(dn, 1),
            "defer_gap0": (d_res0 - d_car0) / max(dn, 1),
            "think_read_rel": think_rel / max(dn, 1)}


def main(cfg_path: str) -> None:
    raw = yaml.safe_load(open(cfg_path)); t = raw["training"]; d = raw["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16

    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    if "<think>" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<think>"]})
    think_id = tok.convert_tokens_to_ids("<think>")
    host = AutoModelForCausalLM.from_pretrained(raw["host"], torch_dtype=torch.float32)
    host.resize_token_embeddings(len(tok))
    gcfg = GraftConfig(d_model=host.config.hidden_size, **raw.get("graft", {}))
    model = SmolBankLM(host, gcfg).to(device)
    n_graft = sum(p.numel() for m in (model.read, model.write) for p in m.parameters())
    print(f"host {raw['host']} {sum(p.numel() for p in host.parameters()):,} | graft {n_graft:,} "
          f"| <think>={think_id} | read_layer {gcfg.read_layer}", flush=True)

    # teacher-forcing bootstrap (port of the SFT/synthetic teacher): force the
    # bank WRITE toward a fixed random projection of the FIRST continuation token
    # so the read has a legible target (breaks the ignore-bank fixed point);
    # blend the bank slot β·t+(1-β)·w0 + distill, β anneals 1→0. Teacher OFF at eval.
    tf_cfg = raw.get("teacher", {}) or {}
    tf_on = bool(tf_cfg.get("enabled", False))
    tf_dw = float(tf_cfg.get("distill_weight", 2.0))
    tf_a0, tf_a1 = (int(v) for v in tf_cfg.get("anneal", [1000, 2000]))
    tf_proj = None
    if tf_on:
        emb_dim = host.get_input_embeddings().weight.size(1)
        gpu_g = torch.Generator(device="cpu").manual_seed(1789)
        tf_proj = (torch.randn(emb_dim, gcfg.mem_dim, generator=gpu_g) / emb_dim ** 0.5).to(device)
        print(f"teacher ON: distill_w {tf_dw}, anneal [{tf_a0},{tf_a1}] "
              f"(target = proj of mean-pooled chunk gist)", flush=True)

    def _beta(s: int) -> float:
        if not tf_on or s >= tf_a1:
            return 0.0
        return 1.0 if s <= tf_a0 else 1.0 - (s - tf_a0) / max(1, tf_a1 - tf_a0)

    L, K = int(d["seq_len"]), int(d["chunks_per_conv"])
    defer_len = int(d.get("defer_len", 16))
    sd = dict(seq_len=L, chunks_per_conv=K, batch=int(d["batch_size"]),
              n_files=int(d.get("n_files", 800)),
              dataset=d.get("dataset", "codeparrot/codeparrot-clean-valid"),
              data_dir=d.get("data_dir", ""), stream_cap=int(d.get("stream_cap", 60000)),
              seed=int(t.get("seed", 0)))
    train_stream = CodeChunkStream(tok, split="train", **sd)
    eval_stream  = CodeChunkStream(tok, split="held", **sd)
    print(f"corpus: train {train_stream.n_chunk} chunks / held {eval_stream.n_chunk} | "
          f"seq_len {L}  K {K}  defer_len {defer_len}", flush=True)

    # optimizers: host AdamW (small real LR, NOT frozen), graft on Muon
    lr_host = float(t["lr_host"]); lr_bank = float(t["lr_bank"]); wd = float(t.get("weight_decay", 0.01))
    host_params = [p for p in host.parameters() if p.requires_grad]
    graft_params = [p for m in (model.read, model.write) for p in m.parameters()]
    opt_host = AdamW([{"params": host_params, "lr": lr_host, "weight_decay": 0.0}])
    graft_opt = str(t.get("graft_optimizer", "muon")).lower()
    muon_lr = float(t.get("muon_lr", 7.5e-4))
    graft_lr = float(t.get("graft_lr", lr_bank))
    if graft_opt == "adamw":
        # AdamW on the whole graft: per-param adaptive step self-limits a runaway
        # read (Muon's RMS-matched update let think_inject blow to 2x then crash).
        opt_graft = AdamW([{"params": graft_params, "lr": graft_lr, "weight_decay": wd}])
        print(f"graft optimizer: AdamW lr {graft_lr} ({sum(p.numel() for p in graft_params):,}) "
              f"| host AdamW lr {lr_host}", flush=True)
    else:
        from .train import Muon
        mats = [p for p in graft_params if p.ndim == 2]; rest = [p for p in graft_params if p.ndim != 2]
        opt_graft = Muon(mats, lr=muon_lr, momentum=0.95, nesterov=True, ns_steps=10, wd=wd,
                         rms_match=True, adam_params=rest, adam_lr=lr_bank, adam_wd=wd)
        print(f"graft optimizer: Muon lr {muon_lr} rms_match ({sum(p.numel() for p in mats):,}) "
              f"+ AdamW lr {lr_bank} ({sum(p.numel() for p in rest):,}) | host AdamW lr {lr_host}", flush=True)

    steps = int(t["steps"]); warmup = int(t.get("warmup_steps", 100))
    lam = float(t.get("defer_weight", 1.0))
    wsd = bool(t.get("wsd_decay", True)); wsd_floor = float(t.get("wsd_floor", 0.0))
    decay_start = int(t.get("wsd_decay_start", int(steps * 0.6)))
    log_every, eval_every = int(t.get("log_every", 20)), int(t.get("eval_every", 200))
    save_every, save_dir = int(t.get("save_every", 500)), t["save_dir"]
    metrics_file = t.get("metrics_file"); os.makedirs(save_dir, exist_ok=True)
    if metrics_file: os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    import json
    writer = None
    if metrics_file:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(os.path.dirname(metrics_file), "tb")
        writer = SummaryWriter(tb_dir); print(f"tensorboard → {tb_dir}", flush=True)

    def set_lr(step):
        scale = min(1.0, step / max(1, warmup))
        decay = 1.0
        if wsd and step > decay_start:
            frac = 1.0 - (step - decay_start) / max(1, steps - decay_start)
            decay = wsd_floor + (1.0 - wsd_floor) * frac
        opt_host.param_groups[0]["lr"] = lr_host * decay
        base = graft_lr if graft_opt == "adamw" else muon_lr
        for gp in opt_graft.param_groups: gp["lr"] = base * scale * decay
        ad = getattr(opt_graft, "_adam", None)
        if ad:
            for gp in ad.param_groups: gp["lr"] = lr_bank * scale * decay
        return base * scale * decay

    model.train(); ema_ic = ema_d = None; t0 = time.time()
    for step in range(1, steps + 1):
        lr_now = set_lr(step)
        segs = train_stream.next_conv()
        opt_host.zero_grad(set_to_none=True); opt_graft.zero_grad(set_to_none=True)
        bank = None; total = 0.0; ic_v = d_v = 0.0; distill_v = 0.0; distill_n = 0
        for i, s in enumerate(segs):
            x = s["input_ids"].to(device); am = s["attention_mask"].to(device)
            xt, amt = _append_think(x, am, think_id)          # emit <think> after context
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                o = model(xt, attention_mask=amt, init_mem=bank, labels=xt)
            total = total + o["loss"]; ic_v += float(o["loss"].detach()); bank = o["mem_bank"]
            if i < len(segs) - 1:
                nxt = segs[i + 1]["input_ids"][:, :defer_len].to(device)          # [B, M] targets
                beta = _beta(step)
                if tf_on and beta > 0.0:
                    # teacher: force the write toward a clean code of THE CHUNK IT JUST
                    # SAW (mean-pooled gist of chunk N) — a target the write CAN produce
                    # (in-context), unlike the future. Blend into the last slot, distill,
                    # carry the blend. The read learns to decode this stored gist.
                    with torch.no_grad():
                        emb_chunk = host.get_input_embeddings().weight[x].float()   # [B, L, emb_dim]
                        tcode = emb_chunk.mean(dim=1) @ tf_proj.float()             # whole-chunk gist
                        tcode = tcode / tcode.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                    w0 = bank[:, -1]
                    distill = (1.0 - torch.nn.functional.cosine_similarity(
                        w0.float(), tcode, dim=1)).mean()
                    total = total + tf_dw * beta * distill
                    distill_v += float(distill.detach()); distill_n += 1
                    blended = (beta * tcode.to(w0.dtype) + (1.0 - beta) * w0).unsqueeze(1)
                    bank = torch.cat([bank[:, :-1], blended], dim=1)
                di = torch.full((x.size(0), defer_len), think_id, device=device)  # all-<think>, NO context
                with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    od = model(di, attention_mask=torch.ones_like(di), init_mem=bank)
                lg = od["logits"].float()                      # [B, M, V]: position i predicts nxt[:, i]
                dloss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), nxt.reshape(-1))
                total = total + lam * dloss; d_v += float(dloss.detach())
                # deferred forward's own write is discarded (do NOT carry od bank)
        total.backward()
        torch.nn.utils.clip_grad_norm_(host_params + graft_params, float(t.get("grad_clip", 1.0)))
        opt_host.step(); opt_graft.step()
        ic_v /= K; d_v /= max(K - 1, 1)
        ema_ic = ic_v if ema_ic is None else 0.95 * ema_ic + 0.05 * ic_v
        ema_d  = d_v  if ema_d  is None else 0.95 * ema_d  + 0.05 * d_v
        if step % log_every == 0:
            print(f"step {step:5d}  ic {ema_ic:.3f} (ppl {math.exp(ema_ic):.1f})  defer {ema_d:.3f}  "
                  f"β {_beta(step):.2f}  lr {lr_now:.2e}  {(time.time()-t0)/step:.2f}s/step", flush=True)
            if writer is not None:
                writer.add_scalar("train/ic_loss", ema_ic, step)
                writer.add_scalar("train/ic_ppl", math.exp(ema_ic), step)
                writer.add_scalar("train/defer_loss", ema_d, step)
                writer.add_scalar("sched/lr", lr_now, step)
                writer.add_scalar("sched/beta", _beta(step), step)
                if distill_n:
                    writer.add_scalar("train/distill", distill_v / distill_n, step)
        if step % eval_every == 0 or step == steps:
            m = evaluate(model, eval_stream, device, amp_dtype, think_id, defer_len, int(t.get("eval_convs", 8)))
            print(f"[eval @{step}] ic_ppl {m['ic_ppl']:.1f} | defer car {m['defer_car']:.3f} "
                  f"res {m['defer_res']:.3f} GAP {m['defer_gap']:+.3f} | pos0 GAP0 {m['defer_gap0']:+.3f} "
                  f"| think_inject {m['think_read_rel']:.3f}", flush=True)
            if metrics_file:
                with open(metrics_file, "a") as f:
                    f.write(json.dumps({"step": step, **m}) + "\n")
            if writer is not None:
                for k, v in m.items():
                    writer.add_scalar(f"eval/{k}", v, step)
        if step % save_every == 0 or step == steps:
            torch.save({"step": step, "read": model.read.state_dict(),
                        "write": model.write.state_dict(), "graft_cfg": gcfg.__dict__},
                       os.path.join(save_dir, f"step_{step}.pt" if step != steps else "final.pt"))
    print("done.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
