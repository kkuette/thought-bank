"""dsv6 — FROM-SCRATCH native bank as cross-chunk long-context memory on real code.

Pivot from the graft (code_train.py): grafting the bank onto a pretrained-frozen
host hit the hard ignore-bank tension — a host trained WITHOUT a bank has no slot
for the read (make it consume => generalization drifts; bound it => it carries
nothing; between => blow-up). See memory dsv6-bank-code-memory-defer §VERDICT.

Here the bank is NATIVE (ThoughtBankLM, read/write inside every DualModalBlock),
co-adapted with the model from init — no graft to force. Same deferred structure:
per code chunk (seq_len tokens):
  (1) in-context forward on [chunk, <think>]: next-token LM loss (+ MoE balance) —
      the model learns to emit <think> after context; the per-token write fills
      the bank as it reads the chunk;
  (2) deferred forward on defer_len <blank> tokens (NO context in-window): position
      i predicts the i-th token of the NEXT chunk from the BANK ALONE.
Dual loss L = L_incontext + defer_weight * L_defer over K chunks, bank carried
(TBPTT = whole conversation). Teacher (optional): distill the last bank slot toward
a random projection of the mean-pooled chunk gist, β anneals 1->0.

Success = deferred GAP > 0 (carried beats init_mem=None), STABLE across the anneal
and WSD decay (not the graft's spike-then-crash), while in-context ppl stays sane.

    PYTHONUNBUFFERED=1 python -m deepseek_v4_mini.code_defer_native \
        deepseek_v4_mini/configs/code_defer_native_v1.yaml
"""
import os, sys, math, time, yaml, json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .config import ThoughtBankConfig
from .model import ThoughtBankLM
from .train import Muon, _split_muon_params
from .code_data import CodeChunkStream
from .cascade import CascadeMemory


def _fill(x_ref, tok_id, width):
    """[B, width] tensor filled with tok_id, on x_ref's device/dtype."""
    return torch.full((x_ref.size(0), width), tok_id, dtype=x_ref.dtype, device=x_ref.device)


def _append(x, tok_id):
    return torch.cat([x, _fill(x, tok_id, 1)], dim=1)


def _ic_loss(model, xt, bank, balw, amp, layer_banks=None):
    """In-context next-token CE on xt (=[chunk, <think>]) + MoE balance. Returns
    (loss, new_bank, ce_detached)."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        o = model(xt, init_mem=bank, layer_banks=layer_banks)
    lg = o["logits"].float()
    ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), xt[:, 1:].reshape(-1))
    loss = ce + balw * o["balance_loss"].float()
    return loss, o["mem_bank"], float(ce.detach())


@torch.no_grad()
def evaluate(model, stream, device, think_id, blank_id, defer_len, n_conv, balw, amp,
             delta=None):
    model.eval()
    mdt = next(model.parameters()).dtype
    ic_loss = ic_n = 0.0
    d_car = d_res = d_car0 = d_res0 = dn = 0.0
    cont = cont0 = 0.0
    # GAP by conversation depth: hop-1 (first pair, i==0) vs deep (i>=4)
    c1 = r1 = n1 = 0.0
    cd = rd = nd_deep = 0.0
    for _ in range(n_conv):
        segs = stream.next_conv(); bank = None
        dstate = delta.init_state(1, device) if delta is not None else None
        for i, s in enumerate(segs):
            x = s["input_ids"].to(device)
            bank_in = bank                                # carried bank BEFORE this chunk's write
            xt = _append(x, think_id)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                o = model(xt, init_mem=bank)
            bank = o["mem_bank"]
            if delta is not None:                         # B4: carry = delta state
                dstate = delta.update(dstate, model.embed.weight[x])
                bank = delta.to_bank(dstate, mdt)
            lg = o["logits"].float()
            ic_loss += float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)),
                                             xt[:, 1:].reshape(-1))); ic_n += 1
            if i < len(segs) - 1:
                nxt = segs[i + 1]["input_ids"][:, :defer_len].to(device)   # [B, M]
                dl = nxt.size(1)                       # ragged: remainder chunk may be < defer_len
                # turn-0 CEILING: predict the SAME next-chunk tokens with chunk N in-window
                # (teacher-forced continuation). cont vs defer_car = cost of routing the
                # info through the bank instead of attention (user's turn0-vs-turn1 diff).
                ctx = torch.cat([x, nxt[:, :dl - 1]], dim=1)               # [B, L+M-1]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    oc = model(ctx, init_mem=bank_in)
                pc = oc["logits"].float()[:, x.size(1) - 1: x.size(1) - 1 + dl]  # [B,M,V]
                cont += float(F.cross_entropy(pc.reshape(-1, pc.size(-1)), nxt.reshape(-1)))
                cont0 += float(F.cross_entropy(pc[:, 0], nxt[:, 0]))
                # turn-1 DEFERRED: same targets from the bank ALONE (carried vs reset ablation)
                di = _fill(x, blank_id, dl)
                lall_m = {}
                for mode in ("car", "res"):
                    mem = bank if mode == "car" else None
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        od = model(di, init_mem=mem)
                    lg = od["logits"].float()                              # [B, M, V]
                    lall = F.cross_entropy(lg.reshape(-1, lg.size(-1)), nxt.reshape(-1))
                    l0 = F.cross_entropy(lg[:, 0], nxt[:, 0])              # pos0 = pure bank
                    lall_m[mode] = float(lall)
                    if mode == "car": d_car += float(lall); d_car0 += float(l0)
                    else:             d_res += float(lall); d_res0 += float(l0)
                dn += 1
                lc, lr = lall_m["car"], lall_m["res"]         # by conversation depth
                if i == 0:  c1 += lc; r1 += lr; n1 += 1
                if i >= 4:  cd += lc; rd += lr; nd_deep += 1
    model.train()
    dnc = max(dn, 1)
    return {"ic_ppl": math.exp(ic_loss / max(ic_n, 1)),
            "defer_car": d_car / dnc, "defer_res": d_res / dnc,
            "defer_gap": (d_res - d_car) / dnc,
            "defer_car0": d_car0 / dnc, "defer_res0": d_res0 / dnc,
            "defer_gap0": (d_res0 - d_car0) / dnc,
            "cont": cont / dnc, "cont0": cont0 / dnc,                       # turn-0 ceiling
            "headroom": (d_car - cont) / dnc,                              # bank-only vs full-context
            "headroom0": (d_car0 - cont0) / dnc,
            "gap_hop1": (r1 - c1) / max(n1, 1),                            # GAP at depth-1 (first pair)
            "gap_deep": (rd - cd) / max(nd_deep, 1),                       # GAP at depth>=4
            "n_deep": nd_deep}


@torch.no_grad()
def evaluate_by_depth(model, stream, device, think_id, blank_id, defer_len,
                      depths, n_per, amp, delta=None):
    """GAP as a function of conversation DEPTH d = #chunks written into the bank
    before the deferred prediction. For each d: write d chunks (carry the bank),
    then predict the (d+1)-th chunk's opening from the bank ALONE (carried) vs reset
    (init_mem=None). Depth is CONTROLLED via conv_at_depth (not sampled), so buckets
    are populated with equal n — the reliable 'does memory hold as the conversation
    deepens?' curve. Returns {d: {'gap': .., 'car': .., 'res': .., 'n': ..}}."""
    model.eval()
    out = {}
    for d in depths:
        gv = cv = rv = 0.0; n = 0
        for _ in range(n_per):
            segs = stream.conv_at_depth(d + 1)          # d writes + 1 target
            if segs is None:
                break
            bank = None
            dstate = delta.init_state(1, device) if delta is not None else None
            for j in range(d):
                x = segs[j]["input_ids"].to(device)
                xt = _append(x, think_id)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    o = model(xt, init_mem=bank)
                bank = o["mem_bank"]
                if delta is not None:
                    dstate = delta.update(dstate, model.embed.weight[x])
                    bank = delta.to_bank(dstate, next(model.parameters()).dtype)
            nxt = segs[d]["input_ids"][:, :defer_len].to(device)
            dl = nxt.size(1)
            di = _fill(nxt, blank_id, dl)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                oc = model(di, init_mem=bank)
                orr = model(di, init_mem=None)
            V = oc["logits"].size(-1)
            car = float(F.cross_entropy(oc["logits"].float().reshape(-1, V), nxt.reshape(-1)))
            res = float(F.cross_entropy(orr["logits"].float().reshape(-1, V), nxt.reshape(-1)))
            gv += res - car; cv += car; rv += res; n += 1
        nn = max(n, 1)
        out[d] = {"gap": gv / nn, "car": cv / nn, "res": rv / nn, "n": n}
    model.train()
    return out


def main(cfg_path: str, resume: bool = False) -> None:
    raw = yaml.safe_load(open(cfg_path)); t = raw["training"]; d = raw["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(t.get("amp", False))            # native MoE/sinkhorn: fp32 by default

    tok = AutoTokenizer.from_pretrained(raw["tokenizer"])
    add = [x for x in ("<think>", "<blank>") if x not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    think_id = tok.convert_tokens_to_ids("<think>")
    blank_id = tok.convert_tokens_to_ids("<blank>")

    mcfg = dict(raw["model"]); mcfg["vocab_size"] = len(tok)
    cfg = ThoughtBankConfig(**mcfg)
    model = ThoughtBankLM(cfg).to(device)
    print(f"native ThoughtBankLM {model.num_params():,} params | d_model {cfg.d_model} "
          f"n_layers {cfg.n_layers} n_experts {cfg.n_experts} mem_dim {cfg.mem_dim} "
          f"max_mem {cfg.max_mem} | <think>={think_id} <blank>={blank_id} vocab {cfg.vocab_size}",
          flush=True)

    # init_from: CONTINUED pretraining — model weights from a finished run's
    # checkpoint, fresh optimizer/schedule/data (unlike --resume, which restores
    # the full training state of THIS run). Used by the var-chunk phase (v2c).
    init_from = t.get("init_from")
    if init_from:
        ck0 = torch.load(init_from, map_location="cpu")
        model.load_state_dict(ck0["model"])
        print(f"init_from: model weights <- {init_from} (step {ck0.get('step', '?')})",
              flush=True)

    # B4 (backlog 2026-07-13) : canal DeltaNet inter-tours À LA PLACE du carry
    # de banque — modèle strictement inchangé, seul le canal inter-chunks change
    # (o["mem_bank"] ignoré, l'état delta est porté et présenté en pseudo-banque).
    # Config : delta_channel: {d_k: 64}. Voir delta_channel.py.
    dc_cfg = t.get("delta_channel")
    delta = None
    if dc_cfg:
        from .delta_channel import DeltaChannel
        _dk = int(dc_cfg.get("d_k", 64)) if isinstance(dc_cfg, dict) else 64
        delta = DeltaChannel(cfg.d_model, cfg.max_mem, cfg.mem_dim, d_k=_dk).to(device)
        print(f"delta channel ON: d_k {_dk} d_v {delta.d_v} "
              f"({sum(p.numel() for p in delta.parameters()):,} params) — "
              f"carry inter-chunks = état delta, o['mem_bank'] ignoré", flush=True)

    # teacher: distill the last bank slot toward a fixed random projection of the
    # mean-pooled chunk gist (a target the write CAN produce), β anneals 1->0.
    tf_cfg = raw.get("teacher", {}) or {}
    tf_on = bool(tf_cfg.get("enabled", False))
    tf_dw = float(tf_cfg.get("distill_weight", 2.0))
    tf_a0, tf_a1 = (int(v) for v in tf_cfg.get("anneal", [200, 1000]))
    tf_proj = None
    if tf_on:
        g = torch.Generator(device="cpu").manual_seed(1789)
        tf_proj = (torch.randn(cfg.d_model, cfg.mem_dim, generator=g) / cfg.d_model ** 0.5).to(device)
        print(f"teacher ON: distill_w {tf_dw}, anneal [{tf_a0},{tf_a1}] "
              f"(target = proj of mean-pooled chunk gist)", flush=True)

    def _beta(s):
        if not tf_on or s >= tf_a1: return 0.0
        return 1.0 if s <= tf_a0 else 1.0 - (s - tf_a0) / max(1, tf_a1 - tf_a0)

    L, K = int(d["seq_len"]), int(d["chunks_per_conv"])
    defer_len = int(d.get("defer_len", 16))
    sd = dict(seq_len=L, chunks_per_conv=K, batch=int(d["batch_size"]),
              n_files=int(d.get("n_files", 800)),
              dataset=d.get("dataset", "codeparrot/codeparrot-clean-valid"),
              data_dir=d.get("data_dir", ""), stream_cap=int(d.get("stream_cap", 60000)),
              cache_dir=d.get("cache_dir", "data_cache"),
              content_key=d.get("content_key", "content"),
              config_name=d.get("config_name", ""),
              min_chunks=int(d.get("min_chunks", 1)),
              stream_skip=int(d.get("stream_skip", 0)),
              sources=d.get("sources"),
              var_chunk=d.get("var_chunk"),
              seed=int(t.get("seed", 0)))
    train_stream = CodeChunkStream(tok, split="train", **sd)
    eval_stream  = CodeChunkStream(tok, split="held", **{**sd, "batch": 1})  # eval = batch=1 paths
    print(f"corpus: train {train_stream.n_chunk} chunks / held {eval_stream.n_chunk} | "
          f"seq_len {L}  K {K}  defer_len {defer_len}", flush=True)
    # per-domain eval views: on a weighted mix, GAP/depth are reported PER SOURCE
    # (a blended number would hide "the bank works on code but not on web text").
    eval_views = ([(nm, eval_stream.source_stream(i))
                   for i, nm in enumerate(eval_stream.src_names)]
                  if len(eval_stream.src_files) > 1 else [("", eval_stream)])

    # single native optimizer: Muon (2-D weights) + bundled AdamW (embed/norm/1-D)
    lr = float(t.get("lr", 3e-4)); muon_lr = float(t.get("muon_lr", 3e-3))
    wd = float(t.get("weight_decay", 0.01)); balw = float(cfg.balance_loss_weight)
    muon_p, adam_p = _split_muon_params(model)
    if delta is not None:
        # B4 : les ~50k params du canal delta vont dans le bundle AdamW (module
        # neuf, pas de piège √cols Muon à gérer) ; état optimiseur sauvé avec.
        adam_p = adam_p + list(delta.parameters())
    # Per-module lr_scale: legacy Muon scaling (update * √cols) ties the per-matrix
    # update RMS to SHAPE (≈ √(cols/rows)), so changing mem_dim silently rescales the
    # effective lr of every mem_dim-shaped matrix: 64→512 made the read hypernet
    # (fw_A/fw_B, [.., mem_dim]) ~2.8x FASTER and the write head (thought_head/
    # write_gate, [mem_dim, ..]) ~2.8x SLOWER at fixed muon_lr — the v2 GAP collapse.
    # Restore the mem_dim=64-validated per-module effective RMS via group lr scales.
    ref_dim = float(t.get("muon_ref_mem_dim", 64))
    s_read  = (ref_dim / cfg.mem_dim) ** 0.5          # cols = mem_dim grew → scale down
    s_write = (cfg.mem_dim / ref_dim) ** 0.5          # rows = mem_dim grew → scale up
    names = {id(p): n for n, p in model.named_parameters()}
    g_read  = [p for p in muon_p if ("fw_A" in names[id(p)] or "fw_B" in names[id(p)])]
    g_write = [p for p in muon_p if ("thought_head" in names[id(p)] or "write_gate" in names[id(p)])]
    ids = {id(p) for p in g_read} | {id(p) for p in g_write}
    g_rest  = [p for p in muon_p if id(p) not in ids]
    groups = [{"params": g_rest},
              {"params": g_read,  "lr_scale": s_read},
              {"params": g_write, "lr_scale": s_write}]
    opt = Muon(groups, lr=muon_lr, momentum=0.95, nesterov=True, ns_steps=10, wd=wd,
               adam_params=adam_p, adam_lr=lr, adam_wd=wd)
    print(f"optimizer: Muon lr {muon_lr} ({sum(p.numel() for p in muon_p):,}) "
          f"+ AdamW lr {lr} ({sum(p.numel() for p in adam_p):,}) | "
          f"lr_scale read {s_read:.3f} ({sum(p.numel() for p in g_read):,}) "
          f"write {s_write:.3f} ({sum(p.numel() for p in g_write):,}) "
          f"[ref mem_dim {ref_dim:.0f}]", flush=True)
    _G, _B = int(t.get("grad_accum", 1)), train_stream.B
    print(f"grad_accum {_G} x batch {_B} (effective batch = {_G * _B} convs"
          f"{', batched full-chunk windows' if _B > 1 else ''}) "
          f"| K {K} (conv depth up to max_mem)", flush=True)

    steps = int(t["steps"]); warmup = int(t.get("warmup_steps", 100))
    grad_accum = int(t.get("grad_accum", 1))          # convs per optimizer step (effective batch)
    # no_reset_files N > 1: chain N consecutive files into one bank lifetime — the bank
    # is carried (detached) across file boundaries instead of reset, so every file after
    # the first STARTS with the previous file's gists in its slots (dirty-bank regime).
    # Boundary defer stays masked for free: batch=1 derives defer targets within-file only.
    no_reset_files = int(t.get("no_reset_files", 1))
    if no_reset_files > 1:
        assert int(d["batch_size"]) == 1, "no_reset_files requires batch_size 1 (ragged mode)"
        assert grad_accum % no_reset_files == 0, "grad_accum must be a multiple of no_reset_files"
    # interleave_files F > 1 (idea G): each conv = F files' chunks randomly interleaved
    # in ONE bank lifetime (same total chunk budget as next_conv). Trains content-based
    # selection without the no-reset boundary confound (v2d probes 2026-07-11: no-reset
    # learned "off-topic last write => new file" and collapses on mid-file distractors).
    # scalar = fixed F; [lo, hi] = F sampled U[lo, hi] per conv (subject-count diversity)
    _ilv = t.get("interleave_files", 1)
    interleave_files = (tuple(int(v) for v in _ilv)
                        if isinstance(_ilv, (list, tuple)) else int(_ilv))
    ilv_on = (max(interleave_files) if isinstance(interleave_files, tuple)
              else interleave_files) > 1
    if ilv_on:
        assert int(d["batch_size"]) == 1, "interleave_files requires batch_size 1 (ragged mode)"
        # D+G (2026-07-12): no_reset_files>1 + interleave = the carry is INTERLEAVED
        # — bank lives across groups of mixed-thread convs. The carry/init logic is
        # sampler-independent, so the combination is free; boundary defers stay
        # within-file (per-seg defer_tgt). Probes must check whether the v2d
        # boundary heuristic partially returns (group boundaries correlate with
        # "all previous threads dead" until pages make old threads resumable).
    # G2 (2026-07-12): addressed defers — cue (file label 50% / raw chunk opening
    # 50%) + blanks toward a NON-last live stream; trains the content/label-
    # addressed read that the blank defer's recency convention never exercises.
    addr_prob = float(t.get("addr_prob", 0.0))
    addr_label = bool(t.get("addr_label", False))
    addr_max = int(t.get("addr_max", 2))    # cap/conv: each addr forward = a full
    #                                         read graph held until backward (8 GB!)
    if addr_prob > 0 or addr_label:
        assert ilv_on, "addr_prob/addr_label require interleave_files (multi-thread bank)"
    # B2 (backlog 2026-07-13) : resets ANNONCÉS — un marqueur tokenisé
    # <<RESET:SOON>> (pattern file_label_ids : texte, pas de token spécial,
    # vocab inchangé) est préfixé aux `reset_announce_chunks` derniers chunks
    # d'une VIE de banque avec prob `reset_announce` (0.5 = 50/50 annoncé/
    # surprise). On MESURE seulement (probe resetcue : la politique d'écriture
    # bouge-t-elle quand la mort est annoncée ?) — standing warning : aucune
    # perte/reward attachée à l'annonce.
    ra_prob = float(t.get("reset_announce", 0.0))
    ra_chunks = int(t.get("reset_announce_chunks", 3))
    ra_ids = (torch.tensor(tok("<<RESET:SOON>>")["input_ids"], dtype=torch.long)
              if ra_prob > 0 else None)
    # Cascade v3 (spec user 2026-07-12, débordement en 2 temps × fractale max_mem) :
    # cascade_depth = nombre de niveaux au-dessus de la banque vive (0 = off,
    # 1 = v3-lite page, 4 = complet). cascade_map[i] = niveau lu par la couche i
    # (0 = banque vive) ; défaut : les `depth` dernières couches lisent 1..depth.
    cascade_depth = int(t.get("cascade_depth", 0) or 0)
    cascade_map = None
    if cascade_depth > 0:
        _cmap = t.get("cascade_map")
        cascade_map = ([int(v) for v in _cmap] if _cmap else
                       [0] * (cfg.n_layers - cascade_depth)
                       + list(range(1, cascade_depth + 1)))
        assert len(cascade_map) == cfg.n_layers and max(cascade_map) <= cascade_depth
        assert not bool(getattr(cfg, "mem_write_gate_merge", False)), \
            "cascade: gate_merge réordonne les slots, la capture d'éviction suppose FIFO pur"
        assert train_stream.B == 1, "cascade v1 = mode ragged batch=1 (alignement conv)"
        _seed_slots = int(getattr(cfg, "mem_seed_slots", cfg.max_mem))
    if delta is not None:
        assert cascade_depth == 0, "delta_channel remplace le canal — pas de cascade"
        assert not tf_on, "delta_channel: teacher incompatible (manipule les slots)"
    lam = float(t.get("defer_weight", 1.0))
    wsd = bool(t.get("wsd_decay", True)); wsd_floor = float(t.get("wsd_floor", 0.0))
    decay_start = int(t.get("wsd_decay_start", int(steps * 0.66)))
    # decay shape over the decay window (p = fraction of the window elapsed):
    #   linear : 1-p (legacy)
    #   step   : DeepSeek-V2/V3 — x0.316 immediately at decay_start, x0.1 for the last
    #            quarter (same 3:1 phase ratio as their 60%/90%-of-total boundaries).
    #            Leaves the read-destroying full-LR zone in ONE step.
    #   1-sqrt : Hägele et al. 2024 WSD-cooldown winner — fast early drop, long low tail
    #   cosine : Chinchilla/LLaMA classic
    decay_shape = str(t.get("wsd_decay_shape", "linear"))
    log_every, eval_every = int(t.get("log_every", 20)), int(t.get("eval_every", 200))
    eval_depths = list(t.get("eval_depths", []) or [])   # [] => depth-stratified eval OFF
    eval_depth_convs = int(t.get("eval_depth_convs", 8))
    save_every, save_dir = int(t.get("save_every", 500)), t["save_dir"]
    metrics_file = t.get("metrics_file"); os.makedirs(save_dir, exist_ok=True)
    if metrics_file: os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    writer = None
    if metrics_file:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(os.path.dirname(metrics_file), "tb")
        writer = SummaryWriter(tb_dir); print(f"tensorboard → {tb_dir}", flush=True)

    def set_lr(step):
        scale = min(1.0, step / max(1, warmup))
        decay = 1.0
        if wsd and step > decay_start:
            p = (step - decay_start) / max(1, steps - decay_start)   # window progress 0->1
            if decay_shape == "step":
                decay = 0.316 if p <= 0.75 else 0.1
            elif decay_shape == "1-sqrt":
                decay = wsd_floor + (1.0 - wsd_floor) * (1.0 - p ** 0.5)
            elif decay_shape == "cosine":
                decay = wsd_floor + (1.0 - wsd_floor) * 0.5 * (1.0 + math.cos(math.pi * p))
            else:                                                    # linear (legacy)
                decay = wsd_floor + (1.0 - wsd_floor) * (1.0 - p)
        for gp in opt.param_groups:
            gp["lr"] = muon_lr * scale * decay * gp.get("lr_scale", 1.0)
        ad = getattr(opt, "_adam", None)
        if ad:
            for gp in ad.param_groups: gp["lr"] = lr * scale * decay
        return muon_lr * scale * decay

    def _save_ck(step, path):
        """Full training state: preemption-safe resume on rented/spot GPUs."""
        tmp = path + ".tmp"
        torch.save({"step": step, "model": model.state_dict(), "cfg": cfg.__dict__,
                    "opt": opt.state_dict(),
                    "adam": opt._adam.state_dict() if opt._adam else None,
                    "delta": delta.state_dict() if delta is not None else None,
                    "ema_ic": ema_ic, "ema_d": ema_d,
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": (torch.cuda.get_rng_state_all()
                                 if torch.cuda.is_available() else None),
                    "rng_train_stream": train_stream.rng.getstate(),
                    "rng_eval_stream": eval_stream.rng.getstate()}, tmp)
        os.replace(tmp, path)                        # atomic: no torn file on preemption

    start_step = 0; ema_ic = ema_d = ema_a = None
    if resume:
        import glob, re
        cks = {}
        for p in glob.glob(os.path.join(save_dir, "step_*.pt")):
            mt = re.match(r"step_(\d+)\.pt$", os.path.basename(p))
            if mt: cks[int(mt.group(1))] = p
        fin = os.path.join(save_dir, "final.pt")
        if os.path.exists(fin):
            print(f"resume: {fin} exists — training already complete, nothing to do.", flush=True)
            return
        if cks:
            start_step = max(cks)
            ck = torch.load(cks[start_step], map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            if ck.get("opt") is not None: opt.load_state_dict(ck["opt"])
            if ck.get("adam") is not None and opt._adam is not None:
                opt._adam.load_state_dict(ck["adam"])
            if ck.get("delta") is not None and delta is not None:
                delta.load_state_dict(ck["delta"])
            ema_ic, ema_d = ck.get("ema_ic"), ck.get("ema_d")
            if ck.get("rng_torch") is not None: torch.set_rng_state(ck["rng_torch"])
            if ck.get("rng_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(ck["rng_cuda"])
            if ck.get("rng_train_stream") is not None:
                train_stream.rng.setstate(ck["rng_train_stream"])
            if ck.get("rng_eval_stream") is not None:
                eval_stream.rng.setstate(ck["rng_eval_stream"])
            print(f"resume: restored {cks[start_step]} @step {start_step} "
                  f"(opt {'yes' if ck.get('opt') else 'NO — old-format ck'})", flush=True)
        else:
            print("resume: no checkpoint found, starting fresh.", flush=True)

    model.train(); t0 = time.time()
    for step in range(start_step + 1, steps + 1):
        lr_now = set_lr(step)
        opt.zero_grad(set_to_none=True)
        ic_v = d_v = a_v = 0.0; ic_cnt = d_cnt = a_cnt = 0; distill_v = 0.0; distill_n = 0
        # gradient accumulation: G independent conversations (batch=1 each, bank reset
        # between them) summed into one optimizer step => effective batch = G files,
        # variance reduced without padding/GPU-batching the ragged chunks.
        bank_carry = None
        casc_carry, nev_carry = None, 0
        dstate_carry = None
        for _g in range(grad_accum):
            segs = (train_stream.next_conv_batch(defer_len) if train_stream.B > 1
                    else train_stream.next_conv_interleaved(
                        interleave_files, defer_len,
                        label=addr_label, addr_prob=addr_prob, addr_max=addr_max)
                    if ilv_on else train_stream.next_conv())
            # B2 : la vie se termine à la fin de la DERNIÈRE conv du groupe
            # no_reset ((_g+1) % nrf == 0 ; nrf=1 => chaque conv est une vie).
            if (ra_ids is not None and (_g + 1) % no_reset_files == 0
                    and train_stream.rng.random() < ra_prob):
                for s_ in segs[-ra_chunks:]:
                    s_["input_ids"] = torch.cat(
                        [ra_ids.unsqueeze(0), s_["input_ids"]], dim=1)
            if no_reset_files > 1 and _g % no_reset_files != 0:
                bank = bank_carry                     # dirty start: previous file's gists
                casc, n_evict = casc_carry, nev_carry
                dstate = dstate_carry
            else:
                bank = None
                casc = CascadeMemory(cascade_depth, cfg.max_mem) if cascade_depth else None
                n_evict = 0
                dstate = delta.init_state(_B, device) if delta is not None else None
            total = 0.0
            for i, s in enumerate(segs):
                x = s["input_ids"].to(device)
                xt = _append(x, think_id)
                if casc is not None and bank is None:
                    # seed explicite : les niveaux profonds lisent None (vide),
                    # jamais la banque vive par accident au premier chunk
                    bank = model.thought_stream.seed_bank(
                        x.size(0), device, next(model.parameters()).dtype)
                # capture d'éviction AVANT le write : FIFO pur => le slot 0 de la
                # banque pleine est celui qui déborde vers la page (grain slot,
                # spec « débordement en 2 temps » — les seeds ne descendent pas)
                pre0 = (bank[:, 0].detach()
                        if casc is not None and bank.size(1) >= cfg.max_mem else None)
                lb = casc.layer_banks(bank, cascade_map) if casc is not None else None
                loss, bank, ce = _ic_loss(model, xt, bank, balw, amp, lb)
                if delta is not None:
                    # B4 : le write du modèle reste actif DANS le chunk (même
                    # forward), mais le carry inter-chunks = l'état delta
                    dstate = delta.update(dstate, model.embed.weight[x])
                    bank = delta.to_bank(dstate, next(model.parameters()).dtype)
                if pre0 is not None:
                    n_evict += 1
                    if n_evict > _seed_slots:
                        casc.push_slot(pre0)
                total = total + loss; ic_v += ce; ic_cnt += 1
                # deferred target: batched segs carry their own defer_tgt (incl. the
                # LAST turn's external successor, -100-padded); batch=1 derives it
                # from the next in-conv chunk as before.
                nxt = s.get("defer_tgt")
                if nxt is None and i < len(segs) - 1:
                    nxt = segs[i + 1]["input_ids"][:, :defer_len]
                if nxt is not None and bool((nxt != -100).any()):
                    nxt = nxt.to(device)
                    dl = nxt.size(1)                   # ragged: remainder chunk may be < defer_len
                    beta = _beta(step)
                    if tf_on and beta > 0.0:
                        with torch.no_grad():
                            gist = model.embed.weight[x].float().mean(dim=1) @ tf_proj.float()
                            gist = gist / gist.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                        w0 = bank[:, -1]
                        distill = (1.0 - F.cosine_similarity(w0.float(), gist, dim=1)).mean()
                        total = total + tf_dw * beta * distill
                        distill_v += float(distill.detach()); distill_n += 1
                        blended = (beta * gist.to(w0.dtype) + (1.0 - beta) * w0).unsqueeze(1)
                        bank = torch.cat([bank[:, :-1], blended], dim=1)
                    di = _fill(x, blank_id, dl)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        od = model(di, init_mem=bank,
                                   layer_banks=casc.layer_banks(bank, cascade_map)
                                   if casc is not None else None)
                    lg = od["logits"].float()
                    dloss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), nxt.reshape(-1),
                                            ignore_index=-100)
                    total = total + lam * dloss; d_v += float(dloss.detach()); d_cnt += 1
                    # deferred forward's own write is discarded (do NOT carry od bank)
                # G2 addressed defer: [cue, blanks] toward a NON-last stream; loss
                # only on the blank positions (cue is context, not supervision)
                ac, at = s.get("addr_cue"), s.get("addr_tgt")
                if ac is not None and bool((at != -100).any()):
                    ac, at = ac.to(device), at.to(device)
                    di = torch.cat([ac, _fill(ac, blank_id, at.size(1))], dim=1)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        oa = model(di, init_mem=bank,
                                   layer_banks=casc.layer_banks(bank, cascade_map)
                                   if casc is not None else None)
                    lga = oa["logits"].float()[:, ac.size(1):]
                    aloss = F.cross_entropy(lga.reshape(-1, lga.size(-1)),
                                            at.reshape(-1), ignore_index=-100)
                    total = total + lam * aloss
                    a_v += float(aloss.detach()); a_cnt += 1
                    # addressed forward's write is discarded too
            (total / grad_accum).backward()          # mean over the G accumulated convs
            if no_reset_files > 1:
                # graph freed per file; the carried bank is data, not gradient path
                bank_carry = bank.detach() if bank is not None else None
                casc_carry, nev_carry = casc, n_evict
                if delta is not None:
                    dstate_carry = dstate.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(t.get("grad_clip", 1.0)))
        opt.step()
        ic_v /= max(ic_cnt, 1); d_v /= max(d_cnt, 1)
        ema_ic = ic_v if ema_ic is None else 0.95 * ema_ic + 0.05 * ic_v
        ema_d  = d_v  if ema_d  is None else 0.95 * ema_d  + 0.05 * d_v
        if a_cnt:
            a_v /= a_cnt
            ema_a = a_v if ema_a is None else 0.95 * ema_a + 0.05 * a_v
        if step % log_every == 0:
            addr_s = f"addr {ema_a:.3f}  " if ema_a is not None else ""
            print(f"step {step:5d}  ic {ema_ic:.3f} (ppl {math.exp(ema_ic):.1f})  defer {ema_d:.3f}  "
                  f"{addr_s}β {_beta(step):.2f}  lr {lr_now:.2e}  "
                  f"{(time.time()-t0)/max(step - start_step, 1):.2f}s/step", flush=True)
            if writer is not None:
                writer.add_scalar("train/ic_loss", ema_ic, step)
                writer.add_scalar("train/ic_ppl", math.exp(ema_ic), step)
                writer.add_scalar("train/defer_loss", ema_d, step)
                if ema_a is not None:
                    writer.add_scalar("train/addr_loss", ema_a, step)
                writer.add_scalar("sched/lr", lr_now, step)
                writer.add_scalar("sched/beta", _beta(step), step)
                if distill_n:
                    writer.add_scalar("train/distill", distill_v / distill_n, step)
        if step % eval_every == 0 or step == steps:
            # eval_depth_sources (mix large, ex. divmix 13 sources) : la courbe
            # par profondeur coûte 4x le GAP top-level (eval_depths x
            # eval_depth_convs convs PAR source) et ne sert de comparaison que
            # sur les ancres — la restreindre à cette liste ramène l'éval de
            # 13x40 à 13x8 + 2x32 convs. None (défaut) = toutes les sources.
            depth_srcs = t.get("eval_depth_sources")
            if cascade_depth and casc is not None:
                print(f"[cascade @{step}] {casc.stats()} (dernière conv du step)")
            for src_name, es in eval_views:
                tag = f" [{src_name}]" if src_name else ""
                pfx = f"{src_name}/" if src_name else ""
                m = evaluate(model, es, device, think_id, blank_id, defer_len,
                             int(t.get("eval_convs", 8)), balw, amp, delta=delta)
                print(f"[eval @{step}]{tag} ic_ppl {m['ic_ppl']:.1f} | defer car {m['defer_car']:.3f} "
                      f"res {m['defer_res']:.3f} GAP {m['defer_gap']:+.3f} GAP0 {m['defer_gap0']:+.3f} "
                      f"| ceil(t0) {m['cont']:.3f} headroom {m['headroom']:+.3f} "
                      f"| GAP hop1 {m['gap_hop1']:+.3f} deep(>=4) {m['gap_deep']:+.3f} (n={int(m['n_deep'])})",
                      flush=True)
                if metrics_file:
                    with open(metrics_file, "a") as f:
                        f.write(json.dumps({"step": step, "source": src_name, **m}) + "\n")
                if writer is not None:
                    for k, v in m.items():
                        writer.add_scalar(f"eval/{pfx}{k}", v, step)
                if eval_depths and (depth_srcs is None or not src_name
                                    or src_name in depth_srcs):
                    bd = evaluate_by_depth(model, es, device, think_id, blank_id,
                                           defer_len, eval_depths, eval_depth_convs, amp,
                                           delta=delta)
                    curve = "  ".join(f"d{d}:{bd[d]['gap']:+.3f}(n{bd[d]['n']})" for d in eval_depths)
                    print(f"[eval @{step}]{tag} GAP by depth (writes→predict next): {curve}", flush=True)
                    if metrics_file:
                        with open(metrics_file, "a") as f:
                            f.write(json.dumps({"step": step, "source": src_name, "gap_by_depth":
                                {str(d): bd[d] for d in eval_depths}}) + "\n")
                    if writer is not None:
                        for d in eval_depths:
                            writer.add_scalar(f"eval_depth/{pfx}gap_d{d}", bd[d]["gap"], step)
        if step % save_every == 0 or step == steps:
            _save_ck(step, os.path.join(save_dir,
                     "final.pt" if step == steps else f"step_{step}.pt"))
    print("done.", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--resume"]
    main(args[0], resume="--resume" in sys.argv[1:])
