"""dsv6 native — replay a nan-guard dump offline and localize the NaN.

Takes the `nan_conv_stepN_gG.pt` dump written by the trainer's nan-guard
(the offending conv's segs + entry bank + cascade state) plus the weights of
the moment (`nan_weights.pt`, saved on first guard trip), replays the conv
forward+backward, and reports where the first non-finite value appears —
with grad_checkpoint ON, OFF, or both (the user's hypothesis: the
checkpoint recompute diverges from the forward on the hard MoE routing,
producing inconsistent grads).

    python -m deepseek_v4_mini.nan_repro \
        deepseek_v4_mini/configs/sft_persona_350m.yaml \
        /mnt/tb/checkpoints/v350_sft_persona/nan_weights.pt \
        /mnt/tb/checkpoints/v350_sft_persona/nan_conv_step33_g2.pt \
        [--gc both] [--anomaly] [--trials 3]

Chat segs only (the persona SFT is p_chat 1.0). Weights file: either the
trainer's nan_weights.pt ({step, model}) or any regular ckpt ({model, ...}).
--trials N replays N times per mode (routing nondeterminism = the suspect,
one replay can miss). --anomaly wraps backward in autograd anomaly mode to
name the op that produced the first NaN grad (slow).
"""
import argparse

import torch
import torch.nn.functional as F
import yaml

from .cascade import CascadeMemory
from .config import ThoughtBankConfig
from .model import ThoughtBankLM


def _first_nonfinite_hooks(model, record):
    """Forward hooks: record the first module whose OUTPUT goes non-finite."""
    hs = []
    for name, mod in model.named_modules():
        if not name:
            continue

        def hook(m, inp, out, _n=name):
            if record["fwd"] is not None:
                return
            ts = out if isinstance(out, (tuple, list)) else (out,)
            for t in ts:
                if torch.is_tensor(t) and t.is_floating_point() \
                        and not bool(torch.isfinite(t).all()):
                    record["fwd"] = _n
                    return
        hs.append(mod.register_forward_hook(hook))
    return hs


def replay(model, cfg, dump, device, amp, grad_ckpt, cascade_map, balw,
           anomaly=False, sac=False):
    """One forward+backward replay of the dumped conv. Returns a report dict.
    sac=True = checkpoint avec save-topk (le fix gc_save_topk du trainer)."""
    from torch.utils.checkpoint import checkpoint as _ckpt
    ctx = None
    if sac:
        from torch.utils.checkpoint import (create_selective_checkpoint_contexts,
                                            CheckpointPolicy)
        _save = {torch.ops.aten.topk.default}

        def _pol(c, op, *a, **k):
            return (CheckpointPolicy.MUST_SAVE if op in _save
                    else CheckpointPolicy.PREFER_RECOMPUTE)

        ctx = lambda: create_selective_checkpoint_contexts(_pol)

    def fwd(*a, **k):
        if grad_ckpt and torch.is_grad_enabled():
            if ctx is not None:
                return _ckpt(model, *a, use_reentrant=False, context_fn=ctx, **k)
            return _ckpt(model, *a, use_reentrant=False, **k)
        return model(*a, **k)

    model.zero_grad(set_to_none=True)
    record = {"fwd": None}
    hooks = _first_nonfinite_hooks(model, record)

    bank = None if dump["bank_in"] is None else dump["bank_in"].to(device)
    casc = None if dump["casc"] is None \
        else CascadeMemory.from_state(dump["casc"], device)
    seed_slots = int(getattr(cfg, "mem_seed_slots", 0) or 0)
    n_evict = int(dump.get("n_evict", 0) or 0)

    total = 0.0
    seg_ce = []
    for s in dump["segs"]:
        assert "loss_mask" in s, "repro: chat segs only"
        x = s["input_ids"].to(device)
        lmask = s["loss_mask"].to(device)
        if casc is not None and bank is None:
            bank = model.thought_stream.seed_bank(
                x.size(0), device, next(model.parameters()).dtype)
        pre0 = (bank[:, 0].detach()
                if casc is not None and bank is not None
                and bank.size(1) >= cfg.max_mem else None)
        lb = casc.layer_banks(bank, cascade_map) if casc is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            o = fwd(x, init_mem=bank, layer_banks=lb)
        lg = o["logits"].float()
        loss = balw * o["balance_loss"].float()
        m = lmask[:, 1:].reshape(-1)
        if float(m.sum()) > 0:
            ce_tok = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)),
                                     x[:, 1:].reshape(-1), reduction="none")
            loss = loss + (ce_tok * m).sum() / m.sum()
        bank = o["mem_bank"]
        if pre0 is not None:
            n_evict += 1
            if n_evict > seed_slots:
                casc.push_slot(pre0)
        seg_ce.append(float(loss.detach()))
        total = total + loss

    loss_ok = bool(torch.isfinite(total).all())
    bank_ok = bank is None or bool(torch.isfinite(bank).all())
    grad_ok, gn, bad = None, None, []
    if loss_ok:
        with torch.autograd.set_detect_anomaly(anomaly):
            total.backward()
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        grad_ok = not bad
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e18))
    for h in hooks:
        h.remove()
    return {"loss_ok": loss_ok, "bank_ok": bank_ok, "grad_ok": grad_ok,
            "grad_norm": gn, "first_nonfinite_fwd": record["fwd"],
            "seg_loss": seg_ce,
            "bad_grads": bad[:8] if loss_ok and not grad_ok else []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("weights", help="nan_weights.pt or any step ckpt")
    ap.add_argument("dump", help="nan_conv_stepN_gG.pt")
    ap.add_argument("--gc", choices=["on", "off", "sac", "both", "all"],
                    default="both", help="sac = GC + save-topk (le fix) ; "
                    "all = off, on, sac")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--anomaly", action="store_true")
    args = ap.parse_args()

    raw = yaml.safe_load(open(args.config))
    t = raw.get("training", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(t.get("amp", False))
    balw = float(raw["model"].get("balance_loss_weight", 0.0))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    add = [x for x in ("<think>", "<blank>") if x not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    mcfg = dict(raw["model"])
    mcfg["vocab_size"] = len(tok)
    cfg = ThoughtBankConfig(**mcfg)

    depth = int(t.get("cascade_depth", 0) or 0)
    cascade_map = None
    if depth:
        _cmap = t.get("cascade_map")
        cascade_map = ([int(v) for v in _cmap] if _cmap else
                       [0] * (cfg.n_layers - depth)
                       + list(range(1, depth + 1)))

    model = ThoughtBankLM(cfg).to(device)
    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.train()
    raw_dump = torch.load(args.dump, map_location="cpu", weights_only=False)
    # deux formats : nan_conv_stepN_gG.pt (une conv, loss nan) ou
    # nan_gradstep_N.pt (toutes les convs du step, grads nan)
    convs = raw_dump["convs"] if "convs" in raw_dump \
        else [{k: raw_dump[k] for k in ("segs", "bank_in", "casc", "n_evict")}]
    print(f"dump step {raw_dump['step']}: {len(convs)} conv(s) | weights step "
          f"{ck.get('step', '?')}", flush=True)

    # modes = liste de (grad_ckpt, sac)
    modes = {"on": [(True, False)], "off": [(False, False)],
             "sac": [(True, True)], "both": [(False, False), (True, False)],
             "all": [(False, False), (True, False), (True, True)]}[args.gc]
    for ci, dump in enumerate(convs):
        dump.setdefault("step", raw_dump["step"])
        for gc, sac in modes:
            for i in range(args.trials):
                r = replay(model, cfg, dump, device, amp, gc, cascade_map, balw,
                           anomaly=args.anomaly, sac=sac)
                print(f"conv {ci} gc={'SAC' if sac else ('ON ' if gc else 'OFF')} trial {i}: "
                      f"loss {'ok' if r['loss_ok'] else 'NON FINIE'} | "
                      f"bank {'ok' if r['bank_ok'] else 'NON FINIE'} | "
                      f"grads {'ok' if r['grad_ok'] else ('NON FINIS' if r['grad_ok'] is not None else '-')} "
                      f"(norm {r['grad_norm'] if r['grad_norm'] is None else round(r['grad_norm'], 3)}) | "
                      f"1er fwd non fini: {r['first_nonfinite_fwd']}", flush=True)
                if r["bad_grads"]:
                    print(f"   grads NaN: {r['bad_grads']}", flush=True)
                if r["seg_loss"] and not r["loss_ok"]:
                    print(f"   loss par seg: "
                          f"{[round(v, 3) for v in r['seg_loss']]}", flush=True)


if __name__ == "__main__":
    main()
