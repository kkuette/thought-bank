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


def _chat_loss(model, x, lmask, bank, balw, amp, layer_banks=None):
    """Chat-templated segment: next-token CE restricted to supervised positions
    (loss_mask marks the assistant answer + closing <|im_end|>; template/user
    tokens are masked). No <think> append — the template carries its own stop.
    Returns (loss, new_bank, ce_detached_or_None). User segs (mask all-zero)
    still forward (their WRITE is the point) but contribute no CE."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        o = model(x, init_mem=bank, layer_banks=layer_banks)
    lg = o["logits"].float()
    m = lmask[:, 1:].reshape(-1)                        # targets = positions 1..T-1
    loss = balw * o["balance_loss"].float()
    ce_f = None
    if float(m.sum()) > 0:
        ce_tok = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)),
                                 x[:, 1:].reshape(-1), reduction="none")
        ce = (ce_tok * m).sum() / m.sum()
        loss = loss + ce
        ce_f = float(ce.detach())
    return loss, o["mem_bank"], ce_f


@torch.no_grad()
def _greedy(model, prefix, bank, max_new, stop_id, amp):
    """Greedy-decode max_new tokens after prefix from the CURRENT bank (reads
    only; the decode forward's write is discarded). Returns generated ids."""
    out = prefix
    for _ in range(max_new):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            o = model(out, init_mem=bank)
        nt = o["logits"][:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, nt], dim=1)
        if int(nt) == stop_id:
            break
    return out[:, prefix.size(1):]


def _age_bucket(age):
    for name, hi in (("<=4", 4), ("5-8", 8), ("9-16", 16)):
        if age <= hi:
            return name
    return ">16"


AGE_BUCKETS = ("<=4", "5-8", "9-16", ">16")


@torch.no_grad()
def evaluate_math(model, stream, tok, device, amp, n_conv, max_new=24):
    """Chat eval (math_school | persona): canonical segments advance the bank
    (teacher-forced writes). Only the GRADED assistant turns (the last
    len(truths) — the answers to memory queries) are greedy-decoded TWICE —
    live bank vs ABLATED (None) — and graded by the generator's verifiers.
    grade_live - grade_ablated per kind = the working-memory efficacy figure
    (memory dsv6-grpo-m2-integre). Probe fine ajoutée (verdict run 5i : le
    grade exact-match est aveugle tant que le canal n'existe pas) : nll
    teacher-forcée du tour gradé live vs ablatée — Δnll>0 = la banque aide,
    sensible bien avant l'exact-match. Ventilée par âge (writes fait→réponse)
    quand le stream fournit info.ages. Bank-only (no cascade), like evaluate().
    Kinds sans truths (smalltalk) = contrôles : nll seule, pas de décodage.
    Returns {kind: {...}} + clé "_by_age" (à pop avant itération par kind)."""
    from .math_school_data import A_OPEN, grade_conv
    from .persona_chat_data import grade_recall
    grade = getattr(stream, "grade_conv", grade_conv)   # persona ships its own
    model.eval()
    a_open = torch.tensor(tok(A_OPEN, add_special_tokens=False)["input_ids"],
                          dtype=torch.long, device=device).unsqueeze(0)
    stop_id = tok.convert_tokens_to_ids("<|im_end|>")
    # kind -> [nll_sum, nll_n, gl, ga, n, ans_nll_live, ans_nll_abl, n_ans]
    agg = {}
    by_age = {}                               # bucket -> [n, dg_sum, dnll_sum]
    for _ in range(n_conv):
        conv = stream.next_conv()
        info = conv.get("info", {})
        truths = info.get("truths", []) or []
        ages = info.get("ages", []) or []
        a_idx = [i for i, s in enumerate(conv["segs"])
                 if s["role"] == "assistant"]
        graded = set(a_idx[-len(truths):]) if truths else set()
        bank = None
        live_txt, abl_txt = [], []
        nll_s, nll_n = 0.0, 0
        qi = 0
        a = agg.setdefault(conv["kind"], [0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0])
        for i, s in enumerate(conv["segs"]):
            x = s["input_ids"].to(device)
            lmask = s["loss_mask"].to(device)
            if i in graded:
                live_txt.append(tok.decode(_greedy(
                    model, a_open, bank, max_new, stop_id, amp)[0].tolist()))
                abl_txt.append(tok.decode(_greedy(
                    model, a_open, None, max_new, stop_id, amp)[0].tolist()))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                o = model(x, init_mem=bank)
            m = lmask[:, 1:].reshape(-1)
            if float(m.sum()) > 0:
                lg = o["logits"].float()
                ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)),
                                     x[:, 1:].reshape(-1), reduction="none")
                nll = float((ce * m).sum() / m.sum())
                nll_s += nll; nll_n += 1
                if i in graded:
                    with torch.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=amp):
                        oa = model(x, init_mem=None)
                    lga = oa["logits"].float()
                    cea = F.cross_entropy(
                        lga[:, :-1].reshape(-1, lga.size(-1)),
                        x[:, 1:].reshape(-1), reduction="none")
                    nll_a = float((cea * m).sum() / m.sum())
                    a[5] += nll; a[6] += nll_a; a[7] += 1
                    if qi < len(ages):
                        b = by_age.setdefault(_age_bucket(ages[qi]),
                                              [0, 0.0, 0.0])
                        b[0] += 1
                        b[2] += nll_a - nll
                        if qi < len(truths):
                            b[1] += (grade_recall([live_txt[-1]], [truths[qi]])
                                     - grade_recall([abl_txt[-1]], [truths[qi]]))
                    qi += 1
            bank = o["mem_bank"]
        a[0] += nll_s; a[1] += nll_n
        a[2] += grade(conv, live_txt)
        a[3] += grade(conv, abl_txt)
        a[4] += 1
    model.train()
    out = {k: {"nll": v[0] / max(v[1], 1), "grade": v[2] / v[4],
               "grade_abl": v[3] / v[4], "n": v[4],
               "ans_nll": v[5] / max(v[7], 1),
               "ans_nll_abl": v[6] / max(v[7], 1), "n_ans": v[7]}
           for k, v in agg.items()}
    out["_by_age"] = {k: {"n": v[0], "dgrade": v[1] / v[0],
                          "dnll": v[2] / v[0]} for k, v in by_age.items()}
    return out


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

    # DDP (opt-in via torchrun): data parallelism WITHOUT the DDP wrapper — the
    # conv loop runs many forwards (in-context + defer + addr + reach) per single
    # backward, which trips DDP's reducer ("marked ready twice"). Instead: each
    # rank runs its own convs (bank/cascade/carry are rank-local state, exactly
    # the mono-GPU semantics), gradients are all-reduced manually before
    # opt.step(). Muon/AdamW are deterministic, so identical grads => identical
    # weights on every rank, no sync drift. Effective batch = B * G * world_size.
    ddp_world = int(os.environ.get("WORLD_SIZE", "1"))
    ddp_rank = int(os.environ.get("RANK", "0"))
    if ddp_world > 1:
        import datetime
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)         # BEFORE init: NCCL binds the current
        #                                           device — after, every rank also grows
        #                                           a ~0.5GB stray context on cuda:0
        torch.distributed.init_process_group(
            "nccl", timeout=datetime.timedelta(hours=2))  # rank0 evals while others wait
        device = torch.device(f"cuda:{local_rank}")
        if ddp_rank != 0:
            sys.stdout = open(os.devnull, "w")   # one log stream; errors keep stderr
        print(f"ddp: world {ddp_world} (this = rank {ddp_rank}, cuda:{local_rank})",
              flush=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(t.get("amp", False))            # native MoE/sinkhorn: fp32 by default

    # OPT-IN speed levers (all default off => every existing config bit-identical).
    # tf32: the biggest phase-1 win — MoE/sinkhorn run in fp32, TF32 accelerates
    # exactly those matmuls on Ampere/Ada with a tiny precision cost. Measured in
    # the bringup sweep before committing.
    if bool(t.get("tf32", False)):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("tf32: enabled (fp32 matmul -> tf32)", flush=True)

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

    # `base` is the eager module — the source of truth for state_dict / named params
    # / optimizer grouping (torch.compile wraps in OptimizedModule and prefixes keys
    # with `_orig_mod.`, which would break checkpoint format and the name-based Muon
    # groups below). `model` is what we CALL forward on; when compiled it shares the
    # same Parameter tensors as `base`, so the optimizer built from `base` still
    # drives it. Opt-in; graph breaks (MoE/sinkhorn/einsum) measured in the sweep.
    base = model
    if bool(t.get("compile", False)):
        # dynamic=False : nos shapes sont statiques par construction ([B,512] et
        # [B,16] ; m ne change que le NOMBRE d'appels) — un graphe statique par
        # shape. Sans ça, la transition automatique statique->dynamique fait
        # choisir au recompute du grad_checkpoint un graphe différent du forward
        # (CheckpointError "different number of tensors", pytorch #166926).
        # cache_size_limit relevé : le mem_bank flippe requires_grad (write on/off)
        # => dynamo recompile a chaque flip ; la limite par defaut (8) est crevee
        # vers step 60 (observe pod 45185048 2026-07-17) et le fallback eager
        # desynchronise les graphes entre rangs DDP => deadlock NCCL (100% util,
        # ~95W). On monte la limite pour que les 8 rangs recompilent en lockstep
        # (depth_sync garantit m identique) sans jamais tomber en fallback.
        from torch._dynamo import config as _dynamo_config
        _dynamo_config.cache_size_limit = 256
        _dynamo_config.accumulated_cache_size_limit = 1024
        # compile_cache_dir (opt-in) : cache inductor+triton PERSISTANT — les
        # kernels compilés survivent aux restarts (préemption pod, resume) au
        # lieu de repayer la compilation à froid à chaque boot. Sur pod :
        # pointer sous /workspace (volume persistant), tar-able vers HF avec le
        # data cache pour réchauffer un pod NEUF (clé de cache = arch GPU +
        # version torch : A100+même image => hits). Env lues à la PREMIÈRE
        # compilation (premier forward), donc les poser ici suffit ; setdefault
        # => un export shell garde la main.
        _cc = t.get("compile_cache_dir")
        if _cc:
            os.makedirs(os.path.join(_cc, "triton"), exist_ok=True)
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _cc)
            os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_cc, "triton"))
            print(f"compile: cache persistant → {_cc}", flush=True)
        model = torch.compile(model, dynamic=False)
        print("compile: torch.compile(model, dynamic=False) enabled "
              "(dynamo cache_size_limit=256)", flush=True)

    # grad_checkpoint (opt-in): rematerialize each model forward during backward.
    # The conv loop keeps EVERY chunk's graph alive until the single end-of-conv
    # backward (whole-conv TBPTT), so activation peak = O(K * B * depth-draw) —
    # the step-to-step conv-depth variance is exactly what OOMs B>=8 on 80GB.
    # Checkpointing stores only the inputs per forward => peak ~ one chunk's
    # activations; gradients are EXACT (same graph, recomputed), ~+30% compute.
    grad_ckpt = bool(t.get("grad_checkpoint", False))
    _ckpt_ctx = None
    if grad_ckpt:
        from torch.utils.checkpoint import checkpoint as _ckpt
        # gc_save_topk (défaut ON) : les grads du checkpoint ne sont « exacts »
        # que si le recompute est bit-identique au forward — faux en pratique :
        # cuBLAS peut choisir un autre algo au recompute (workspace différent
        # pendant le backward), les scores bougent d'un ULP, et les TOPK durs
        # (routage MoE, sélection de blocs CSA) FLIPPENT => gradients calculés
        # sur un autre graphe que la loss. Suspect n°1 des NaN (incident phase 1
        # step 2520, run 5e step 33 — tous deux GC ON). Fix : selective
        # activation checkpointing — les sorties de topk sont SAUVÉES au
        # forward et rejouées au recompute (coût mémoire = des indices).
        if bool(t.get("gc_save_topk", True)):
            from torch.utils.checkpoint import (create_selective_checkpoint_contexts,
                                                CheckpointPolicy)
            _save_ops = {torch.ops.aten.topk.default}

            def _sac_policy(ctx, op, *args, **kwargs):
                return (CheckpointPolicy.MUST_SAVE if op in _save_ops
                        else CheckpointPolicy.PREFER_RECOMPUTE)

            _ckpt_ctx = lambda: create_selective_checkpoint_contexts(_sac_policy)
            print("grad_checkpoint: ON + save-topk (routage figé au recompute)",
                  flush=True)
        else:
            print("grad_checkpoint: ON (rematerialized forwards)", flush=True)

    def _fwd(*a, **k):
        if grad_ckpt and torch.is_grad_enabled():
            if _ckpt_ctx is not None:
                return _ckpt(model, *a, use_reentrant=False,
                             context_fn=_ckpt_ctx, **k)
            return _ckpt(model, *a, use_reentrant=False, **k)
        return model(*a, **k)

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

    # DDP: model init uses the (unseeded) default RNG, so ranks start with
    # different weights — broadcast rank0's so the manual all-reduce keeps
    # everyone bit-identical from step 1. (tf_proj is Generator-seeded: already
    # identical everywhere.)
    if ddp_world > 1:
        with torch.no_grad():
            for p in base.parameters():
                torch.distributed.broadcast(p.data, src=0)
            if delta is not None:
                for p in delta.parameters():
                    torch.distributed.broadcast(p.data, src=0)

    # teacher: distill the last bank slot toward a fixed random projection of the
    # mean-pooled chunk gist (a target the write CAN produce), β anneals 1->0.
    tf_cfg = raw.get("teacher", {}) or {}
    tf_on = bool(tf_cfg.get("enabled", False))
    tf_dw = float(tf_cfg.get("distill_weight", 2.0))
    tf_a0, tf_a1 = (int(v) for v in tf_cfg.get("anneal", [200, 1000]))
    # target: 'chunk' (défaut) = proj du gist moyen du chunk (hash de contenu) ;
    # 'value' = proj de l'embedding des tokens VALEUR (val_mask) = code propre
    # DISCRIMINANT par valeur (recette 47M nn.Embedding(rule_id) : 0.03→0.99).
    # En mode 'value' le teacher ne tire QUE les segs porteurs de valeur ; les
    # autres writes (filler, question, ack) restent natifs.
    # 'surprisal' = généralisation label-free de 'value' : pooling pondéré par
    # la nll^alpha d'un LM de référence gelé (surp_w posé par le générateur,
    # clé gen.surprisal_ref) — les tokens imprévisibles (l'information) dominent
    # la cible, les templates pèsent ~0. Marche sur tout corpus, tous les segs.
    tf_target = str(tf_cfg.get("target", "chunk"))
    assert tf_target in ("chunk", "value", "surprisal"), tf_target
    tf_proj = None
    if tf_on:
        g = torch.Generator(device="cpu").manual_seed(1789)
        tf_proj = (torch.randn(cfg.d_model, cfg.mem_dim, generator=g) / cfg.d_model ** 0.5).to(device)
        _tdesc = {"value": "proj embed valeur (discriminant)",
                  "surprisal": "proj pooling pondéré nll ref (label-free)",
                  "chunk": "proj gist chunk"}[tf_target]
        print(f"teacher ON: distill_w {tf_dw}, anneal [{tf_a0},{tf_a1}], "
              f"target={tf_target} ({_tdesc})", flush=True)

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
              surprisal_mode=d.get("surprisal_mode", "none"),
              sif_a=float(d.get("sif_a", 1e-4)),
              pack_convs=bool(d.get("pack_convs", False)),
              pack_same_source=bool(d.get("pack_same_source", False)),
              seed=int(t.get("seed", 0)))
    # DDP: per-rank seed offset => each rank samples different convs (random
    # sampling with per-rank RNG — no distributed sampler needed). Rank0 builds
    # the tokenized cache alone first (concurrent misses race on the .tmp
    # rename), the barrier releases the others onto a guaranteed cache hit.
    train_seed = sd["seed"] + 9973 * ddp_rank
    # depth_sync (opt-in): rank-invariant anchor/m rng in next_conv_batch, so all
    # ranks run the same conv depth per step (a DDP step lasts as long as the
    # deepest rank — independent draws cost ~2.6x the mean step time).
    depth_sync = bool(d.get("depth_sync", False))
    if ddp_world > 1 and ddp_rank != 0:
        torch.distributed.barrier()
    train_stream = CodeChunkStream(tok, split="train", **{**sd, "seed": train_seed},
                                   depth_sync_seed=sd["seed"] if depth_sync else None)
    eval_stream  = CodeChunkStream(tok, split="held",
                                   **{**sd, "batch": 1, "surprisal_mode": "none"})  # eval = batch=1 paths
    if ddp_world > 1 and ddp_rank == 0:
        torch.distributed.barrier()               # cache built — release the other ranks
    print(f"corpus: train {train_stream.n_chunk} chunks / held {eval_stream.n_chunk} | "
          f"seq_len {L}  K {K}  defer_len {defer_len}", flush=True)
    # per-domain eval views: on a weighted mix, GAP/depth are reported PER SOURCE
    # (a blended number would hide "the bank works on code but not on web text").
    eval_views = ([(nm, eval_stream.source_stream(i))
                   for i, nm in enumerate(eval_stream.src_names)]
                  if len(eval_stream.src_files) > 1 else [("", eval_stream)])
    # eval_sources (opt-in) : restreint l'éval per-source à des ANCRES — sur un
    # mix 14 sources l'éval complète re-teste tout le corpus à chaque palier
    # (14x8 convs + depth), ce qui domine le wall-clock d'un SFT court. None
    # (défaut) = toutes les sources (comportement historique).
    ev_srcs = t.get("eval_sources")
    if ev_srcs:
        eval_views = [(nm, es) for nm, es in eval_views if nm in ev_srcs]

    # ── chat mode (opt-in `chat:` block — phase 2 SFT, marche 2) ─────────────
    # Chat-templated convs (math school) mixed into the conv stream at p_chat:
    # a chat conv occupies one grad-accum slot and RIDES the same life carry
    # (no_reset/cascade) as code convs — cross-domain interleaving for free.
    # Segments carry loss_mask (CE on assistant answers only); defer/addr/
    # reach never fire on them (no defer_tgt). Absent block => bit-identical.
    chat_cfg = raw.get("chat") or {}
    chat_stream = chat_eval = None
    p_chat = float(chat_cfg.get("p_chat", 0.5))
    chat_w = float(chat_cfg.get("weight", 1.0))
    chat_eval_convs = int(chat_cfg.get("eval_convs", 24))
    chat_max_new = int(chat_cfg.get("max_new", 24))
    if chat_cfg:
        sname = chat_cfg.get("stream", "math_school")
        if sname == "math_school":
            from .math_school_data import MathSchoolStream as _ChatStream
        elif sname == "persona":
            from .persona_chat_data import PersonaChatStream as _ChatStream
        else:
            raise ValueError(f"chat.stream: unknown stream {sname!r} "
                             "(math_school | persona)")
        gen_kw = dict(chat_cfg.get("gen", {}) or {})
        chat_stream = _ChatStream(tok, seed=train_seed + 1, **gen_kw)
        chat_eval = _ChatStream(tok, seed=1234, **gen_kw)
        print(f"chat mode ON: {sname} p_chat {p_chat} weight {chat_w} "
              f"eval_convs {chat_eval_convs} (masked-CE SFT convs in the "
              f"life carry)", flush=True)

    # single native optimizer: Muon (2-D weights) + bundled AdamW (embed/norm/1-D)
    lr = float(t.get("lr", 3e-4)); muon_lr = float(t.get("muon_lr", 3e-3))
    wd = float(t.get("weight_decay", 0.01)); balw = float(cfg.balance_loss_weight)
    muon_p, adam_p = _split_muon_params(base)
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
    names = {id(p): n for n, p in base.named_parameters()}
    g_read  = [p for p in muon_p if ("fw_A" in names[id(p)] or "fw_B" in names[id(p)])]
    g_write = [p for p in muon_p if ("thought_head" in names[id(p)] or "write_gate" in names[id(p)])]
    ids = {id(p) for p in g_read} | {id(p) for p in g_write}
    g_rest  = [p for p in muon_p if id(p) not in ids]
    groups = [{"params": g_rest},
              {"params": g_read,  "lr_scale": s_read},
              {"params": g_write, "lr_scale": s_write}]
    opt = Muon(groups, lr=muon_lr, momentum=0.95, nesterov=True, ns_steps=10, wd=wd,
               adam_params=adam_p, adam_lr=lr, adam_wd=wd,
               adam_fused=bool(t.get("adam_fused", False)))
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
    # no_reset_files == 0 : UNE vie infinie — la banque n'est JAMAIS reset, ni entre
    # convs ni entre steps d'optimiseur ; elle évolue en continu sur tout le run
    # (décision user 2026-07-20, run 5c). Non sauvegardée dans les ckpts : un resume
    # repart banque vide.
    nrf_never = (no_reset_files == 0)
    if nrf_never:
        assert int(d["batch_size"]) == 1, "no_reset_files=0 requires batch_size 1 (ragged mode)"
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
    # OPTION 2 (verdict page 2026-07-13, FINDINGS d70b595) : reach-back
    # SUPERVISÉ. L'émergence est réfutée (2 seeds : ablater la page ne change
    # rien) mais la cible existe (early évincé −0.37..−0.72 vs reset via
    # résidus de superposition) et la page est gratuite quand non lue —
    # recette v2f : créer le mécanisme par SFT. Un POOL de cibles est porté à
    # travers la vie carried : chaque seg écrit y dépose (cue = ouverture du
    # chunk, label compris si addr_label ; tgt = son defer_tgt même-fichier) ;
    # dans les convs SUIVANTES, avec prob reach_prob, un defer adressé vise
    # une entrée dont les slots ont quitté la banque vive (âge >= max_mem
    # writes) — la seule route vers la cible est la page (ou les résidus).
    # STRATIFICATION par âge (réserve user 2026-07-13 : les cibles du bloc le
    # plus mergé pourraient ne pas apprendre) : perte loggée par strate
    #   s1 [M, 2M)  ~ page p0 fraîche      s2 [2M, 4M) ~ page mergée (p1)
    #   s3 [4M, ∞)  ~ au-delà de la page à depth 1 = DÉTRUIT (contrôle
    #                 négatif : ne DOIT pas s'améliorer si le read lit la page)
    # Même garde VRAM que G2 : chaque forward reach = un graphe de read complet
    # jusqu'au backward => cap reach_max par conv.
    reach_prob = float(t.get("reach_prob", 0.0))
    reach_max = int(t.get("reach_max", 2))
    reach_cue_len = int(t.get("reach_cue_len", 16))
    if reach_prob > 0:
        assert cascade_depth > 0, "reach_prob: il faut une page (cascade_depth >= 1)"
        assert int(t.get("no_reset_files", 1)) > 1, \
            "reach_prob: le pool vit dans le carry (no_reset_files > 1)"
    # ── Optimisations budget-compute (2026-07-23, tous OPT-IN, défauts =
    # comportement historique bit-identique). Contexte pod 10B : 8 boucles data
    # single-core à 99% (host-bound — B24→B32 était gratuit), m moyen ~3.9 vs
    # K=8 => coûts fixes par step (Muon + all-reduce + host) amortis sur moitié
    # moins de tokens que le pire cas dimensionnant la VRAM.
    #   prefetch        : thread producteur qui tire les convs batchées EN AVANCE
    #                     (queue depth 2, pin_memory) — le host tourne pendant le
    #                     compute GPU au lieu de le sérialiser. Chemin batché PUR.
    #                     NB resume : la rng du stream sauvée court <=2 convs en
    #                     avance du step exécuté (les dumps nan portent les segs
    #                     eux-mêmes, la repro d'incident n'en dépend pas).
    #   chunk_budget N  : accumulation DYNAMIQUE — on enchaîne des convs (banque
    #                     reset entre elles, distribution d'entraînement INTACTE)
    #                     jusqu'à N chunks par step avant l'opt.step => step time
    #                     uniforme au pire-cas VRAM, ~m̄/N fois moins d'opt-steps
    #                     par token (grads normalisés par le nb de convs, =
    #                     sémantique grad_accum). depth_sync requis : la suite
    #                     des m est rank-invariante => même nb de convs partout,
    #                     DDP reste en lockstep. tokens/step change (~N/m̄ x) :
    #                     re-checker lr/schedules au moment de config.
    #   allreduce_bf16  : all-reduce des buckets de grads en bf16 (÷2 le volume
    #                     NCCL). Perte de précision ~1 ULP bf16 sur la somme de
    #                     W grads fp32 — à valider par A/B court avant un long run.
    prefetch = bool(t.get("prefetch", False))
    chunk_budget = int(t.get("chunk_budget", 0) or 0)
    ar_bf16 = bool(t.get("allreduce_bf16", False))
    if prefetch:
        assert train_stream.B > 1, "prefetch: chemin batché (batch_size > 1) uniquement"
        assert (chat_stream is None and ra_prob == 0.0 and not ilv_on
                and no_reset_files == 1), \
            "prefetch: chemin batché PUR (pas de chat/reset_announce/interleave/no_reset)"
    if chunk_budget:
        assert train_stream.B > 1 and depth_sync, \
            "chunk_budget: chemin batché + depth_sync requis (lockstep du nb de convs)"
        assert grad_accum == 1, "chunk_budget remplace grad_accum (laisser grad_accum: 1)"
        assert chat_stream is None and ra_prob == 0.0, \
            "chunk_budget: chemin batché pur (pas de chat/reset_announce)"
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
    # step == steps déclenche l'éval même avec eval_every énorme : sur un mix
    # large ça bloque ~40 min de GPU en fin de run (et le rang 0 seul en DDP).
    skip_final_eval = bool(t.get("skip_final_eval", False))
    save_every, save_dir = int(t.get("save_every", 500)), t["save_dir"]
    metrics_file = t.get("metrics_file"); os.makedirs(save_dir, exist_ok=True)
    if metrics_file: os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    if ddp_rank != 0:
        metrics_file = None                       # IO (metrics/tb/eval/save) = rank0 only
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
        torch.save({"step": step, "model": base.state_dict(), "cfg": cfg.__dict__,
                    "opt": opt.state_dict(),
                    "adam": opt._adam.state_dict() if opt._adam else None,
                    "delta": delta.state_dict() if delta is not None else None,
                    "ema_ic": ema_ic, "ema_d": ema_d,
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": (torch.cuda.get_rng_state_all()
                                 if torch.cuda.is_available() else None),
                    "rng_train_stream": train_stream.rng.getstate(),
                    "rng_eval_stream": eval_stream.rng.getstate(),
                    "rng_chat_stream": (chat_stream.rng.getstate()
                                        if chat_stream is not None else None)}, tmp)
        os.replace(tmp, path)                        # atomic: no torn file on preemption

    def _save_bank(step, path):
        """La banque COMPLÈTE (vive + cascade) : artefact autonome — rechargeable
        au resume, échangeable entre runs (bank_init), inspectable hors trainer."""
        if bank_carry is None:
            return
        tmp = path + ".tmp"
        torch.save({"step": step, "bank": bank_carry.detach().cpu(),
                    "casc": (casc_carry.state_dict()
                             if casc_carry is not None else None),
                    "n_evict": nev_carry, "w_total": wt_carry}, tmp)
        os.replace(tmp, path)

    def _load_bank(path, tag):
        _bk = torch.load(path, map_location="cpu", weights_only=False)
        casc_ld = (CascadeMemory.from_state(_bk["casc"], device=device)
                   if _bk.get("casc") is not None else None)
        print(f"{tag}: banque chargée depuis {path} (step d'origine "
              f"{_bk.get('step')}, cascade {'oui' if casc_ld else 'non'})",
              flush=True)
        return (_bk["bank"].to(device), casc_ld,
                int(_bk.get("n_evict", 0)), int(_bk.get("w_total", 0)))

    _bank_loaded, _casc_loaded, _nev_loaded, _wt_loaded = None, None, 0, 0
    _bi = t.get("bank_init")                 # chemin explicite : seed la vie avec
    if _bi:                                  # une banque venue d'un autre run
        _bank_loaded, _casc_loaded, _nev_loaded, _wt_loaded = \
            _load_bank(_bi, "bank_init")

    start_step = 0; ema_ic = ema_d = ema_a = ema_chat = None
    ema_reach = [None, None, None]              # EMA de perte par strate d'âge
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
            base.load_state_dict(ck["model"])
            if ck.get("opt") is not None: opt.load_state_dict(ck["opt"])
            if ck.get("adam") is not None and opt._adam is not None:
                opt._adam.load_state_dict(ck["adam"])
            if ck.get("delta") is not None and delta is not None:
                delta.load_state_dict(ck["delta"])
            ema_ic, ema_d = ck.get("ema_ic"), ck.get("ema_d")
            if ck.get("rng_torch") is not None: torch.set_rng_state(ck["rng_torch"])
            if ck.get("rng_cuda") is not None and torch.cuda.is_available():
                # migration de world size (ex. 8->6 GPUs) : le ck porte un état
                # RNG par device du node d'origine — ne restaurer que les nôtres
                torch.cuda.set_rng_state_all(ck["rng_cuda"][:torch.cuda.device_count()])
            # DDP: the checkpoint holds rank0's stream RNG — restoring it on every
            # rank would make them all sample the SAME convs. Rank0 resumes its
            # exact stream; other ranks re-seed on (rank, start_step) instead.
            if ck.get("rng_train_stream") is not None and ddp_rank == 0:
                train_stream.rng.setstate(ck["rng_train_stream"])
            elif ddp_rank != 0:
                train_stream.rng.seed(train_seed + start_step)
            # depth_sync: rng_m must stay in LOCKSTEP across ranks after resume —
            # deterministic re-seed on (base seed, start_step), same everywhere.
            if train_stream.rng_m is not train_stream.rng:
                train_stream.rng_m.seed(sd["seed"] + start_step)
            if ck.get("rng_eval_stream") is not None:
                eval_stream.rng.setstate(ck["rng_eval_stream"])
            if ck.get("rng_chat_stream") is not None and chat_stream is not None:
                if ddp_rank == 0:
                    chat_stream.rng.setstate(ck["rng_chat_stream"])
                else:
                    chat_stream.rng.seed(train_seed + 1 + start_step)
            _bp = os.path.join(save_dir, f"bank_step_{start_step}.pt")
            if os.path.exists(_bp):
                _bank_loaded, _casc_loaded, _nev_loaded, _wt_loaded = \
                    _load_bank(_bp, "resume")
            print(f"resume: restored {cks[start_step]} @step {start_step} "
                  f"(opt {'yes' if ck.get('opt') else 'NO — old-format ck'})", flush=True)
        else:
            print("resume: no checkpoint found, starting fresh.", flush=True)

    _pf_q = None
    if prefetch:
        # Producteur UNIQUE de train_stream à partir d'ici (l'ordre des tirages
        # rng est préservé : queue FIFO, un seul thread). Démarré APRÈS le
        # resume pour produire depuis l'état rng restauré. Daemon : meurt avec
        # le process (fin de run / préemption).
        import threading, queue as _pyqueue
        _pf_q = _pyqueue.Queue(maxsize=2)

        def _pf_worker():
            while True:
                segs_ = train_stream.next_conv_batch(defer_len)
                for s_ in segs_:
                    for k_, v_ in s_.items():
                        if torch.is_tensor(v_):
                            s_[k_] = v_.pin_memory()
                _pf_q.put(segs_)

        threading.Thread(target=_pf_worker, daemon=True).start()
        print(f"prefetch: ON (queue 2, pin_memory, H2D non_blocking)", flush=True)
    if chunk_budget:
        print(f"chunk_budget: {chunk_budget} chunks/step (accumulation dynamique, "
              f"grads /= nb convs)", flush=True)

    model.train(); t0 = time.time()
    _win_data = 0.0; _win_chunks = 0    # fenêtre log_every : temps d'attente data + chunks vus
    # carries hoisted out of the step loop: with nrf_never they persist across
    # optimizer steps (une vie = le run entier) ; sinon ils sont reset par step.
    bank_carry = _bank_loaded
    casc_carry, nev_carry = _casc_loaded, _nev_loaded
    dstate_carry = None
    rpool_carry, wt_carry = [], _wt_loaded
    for step in range(start_step + 1, steps + 1):
        _t_step0 = time.time()
        lr_now = set_lr(step)
        opt.zero_grad(set_to_none=True)
        ic_v = d_v = a_v = 0.0; ic_cnt = d_cnt = a_cnt = 0; distill_v = 0.0; distill_n = 0
        # dist ventilée porteur/filler : porteur = seg avec val_mask (fait
        # énoncé/màj). Si fait descend et fill reste ~1.0 = le write imite le
        # contenu ; les deux plats = le write n'imite rien (lever distill_w/α).
        dist_c = dist_f = 0.0; dist_cn = dist_fn = 0
        chat_v = 0.0; chat_cnt = 0
        _step_convs = []                     # trace repro nan-guard (voir plus bas)
        reach_v = [0.0, 0.0, 0.0]; reach_cnt = [0, 0, 0]
        # gradient accumulation: G independent conversations (batch=1 each, bank reset
        # between them) summed into one optimizer step => effective batch = G files,
        # variance reduced without padding/GPU-batching the ragged chunks.
        if not nrf_never:
            bank_carry = None
            casc_carry, nev_carry = None, 0
            dstate_carry = None
            rpool_carry, wt_carry = [], 0
        n_conv = 0; step_chunks = 0; data_t = 0.0
        while (step_chunks < chunk_budget) if chunk_budget else (n_conv < grad_accum):
            _g = n_conv
            _t_d = time.time()
            is_chat = (chat_stream is not None
                       and train_stream.rng.random() < p_chat)
            if _pf_q is not None:
                # data_t mesure l'ATTENTE réelle (0 si le producteur est en
                # avance) ; H2D non_blocking depuis la mémoire pinnée, ordonné
                # sur le stream par défaut donc sûr vis-à-vis des forwards.
                segs = [{k: (v.to(device, non_blocking=True)
                             if torch.is_tensor(v) else v) for k, v in s.items()}
                        for s in _pf_q.get()]
            else:
                segs = (chat_stream.next_conv()["segs"] if is_chat
                        else train_stream.next_conv_batch(defer_len) if train_stream.B > 1
                        else train_stream.next_conv_interleaved(
                            interleave_files, defer_len,
                            label=addr_label, addr_prob=addr_prob, addr_max=addr_max)
                        if ilv_on else train_stream.next_conv())
            data_t += time.time() - _t_d
            step_chunks += len(segs)
            # B2 : la vie se termine à la fin de la DERNIÈRE conv du groupe
            # no_reset ((_g+1) % nrf == 0 ; nrf=1 => chaque conv est une vie).
            if (ra_ids is not None and not is_chat
                    and (_g + 1) % max(no_reset_files, 1) == 0
                    and train_stream.rng.random() < ra_prob):
                for s_ in segs[-ra_chunks:]:
                    s_["input_ids"] = torch.cat(
                        [ra_ids.unsqueeze(0), s_["input_ids"]], dim=1)
            if (nrf_never and bank_carry is not None) or (
                    no_reset_files > 1 and _g % no_reset_files != 0):
                bank = bank_carry                     # dirty start: previous file's gists
                casc, n_evict = casc_carry, nev_carry
                if cascade_depth and casc is None:    # banque chargée sans cascade
                    casc = CascadeMemory(cascade_depth, cfg.max_mem)
                dstate = dstate_carry
                reach_pool, w_total = rpool_carry, wt_carry
            else:
                bank = None
                casc = CascadeMemory(cascade_depth, cfg.max_mem) if cascade_depth else None
                n_evict = 0
                dstate = delta.init_state(_B, device) if delta is not None else None
                reach_pool, w_total = [], 0           # le pool meurt avec la vie
            total = 0.0
            reach_n = 0                               # cap VRAM par conv
            # trace repro nan-guard : entrées de la conv AVANT forward (segs =
            # tenseurs CPU du générateur, banque d'entrée = ref détachée) — si
            # le guard grad-norm trip en fin de step, on dumpe tout le step
            _step_convs.append({
                "segs": segs,
                "bank_in": None if bank is None else bank.detach(),
                "casc": None if casc is None else casc.state_dict(),
                "n_evict": n_evict})
            for i, s in enumerate(segs):
                x = s["input_ids"].to(device)
                chat_seg = "loss_mask" in s          # chat segs: no <think>,
                xt = x if chat_seg else _append(x, think_id)  # masked CE
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
                if chat_seg:
                    loss, bank, ce = _chat_loss(_fwd, xt, s["loss_mask"].to(device),
                                                bank, balw, amp, lb)
                    loss = chat_w * loss
                    if ce is not None:
                        chat_v += ce; chat_cnt += 1
                    ce = None
                else:
                    loss, bank, ce = _ic_loss(_fwd, xt, bank, balw, amp, lb)
                if delta is not None:
                    # B4 : le write du modèle reste actif DANS le chunk (même
                    # forward), mais le carry inter-chunks = l'état delta
                    dstate = delta.update(dstate, model.embed.weight[x])
                    bank = delta.to_bank(dstate, next(model.parameters()).dtype)
                if pre0 is not None:
                    n_evict += 1
                    if n_evict > _seed_slots:
                        casc.push_slot(pre0)
                total = total + loss
                if ce is not None:
                    ic_v += ce; ic_cnt += 1
                # teacher par SEG (run 6) : le blend s'applique à CHAQUE seg —
                # chat inclus — plus seulement aux segs à cible defer. Le slot
                # fraîchement écrit est tiré vers un gist prédictible du seg :
                # le CE de la réponse peut alors trouver le routage read→réponse
                # (recette anti point-fixe 47M). target=value (run 7-resume) :
                # cible = proj de l'embedding des tokens VALEUR (val_mask), code
                # discriminant par valeur, et NE tire que les segs porteurs.
                beta = _beta(step)
                vmask = s.get("val_mask")
                surpw = s.get("surp_w") if tf_target == "surprisal" else None
                fire = tf_on and beta > 0.0 and (
                    (tf_target != "value" or vmask is not None) and
                    (tf_target != "surprisal" or surpw is not None))
                if fire:
                    with torch.no_grad():
                        emb = model.embed.weight[x].float()          # [B,T,D]
                        if tf_target == "value" and vmask is not None:
                            vm = vmask.to(device).unsqueeze(-1).float()  # [B,T,1]
                            pooled = (emb * vm).sum(dim=1) / vm.sum(dim=1).clamp_min(1.0)
                        elif surpw is not None:
                            sw = surpw.to(device).unsqueeze(-1).float()  # [B,T,1]
                            pooled = (emb * sw).sum(dim=1) / sw.sum(dim=1).clamp_min(1e-6)
                        else:
                            pooled = emb.mean(dim=1)
                        gist = pooled @ tf_proj.float()
                        gist = gist / gist.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
                    w0 = bank[:, -1]
                    distill = (1.0 - F.cosine_similarity(w0.float(), gist, dim=1)).mean()
                    total = total + tf_dw * beta * distill
                    distill_v += float(distill.detach()); distill_n += 1
                    if vmask is not None:
                        dist_c += float(distill.detach()); dist_cn += 1
                    else:
                        dist_f += float(distill.detach()); dist_fn += 1
                    blended = (beta * gist.to(w0.dtype) + (1.0 - beta) * w0).unsqueeze(1)
                    bank = torch.cat([bank[:, :-1], blended], dim=1)
                # deferred target: batched segs carry their own defer_tgt (incl. the
                # LAST turn's external successor, -100-padded); batch=1 derives it
                # from the next in-conv chunk as before.
                nxt = s.get("defer_tgt")
                if nxt is None and not chat_seg and i < len(segs) - 1:
                    nxt = segs[i + 1]["input_ids"][:, :defer_len]
                if nxt is not None and bool((nxt != -100).any()):
                    nxt = nxt.to(device)
                    dl = nxt.size(1)                   # ragged: remainder chunk may be < defer_len
                    di = _fill(x, blank_id, dl)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        od = _fwd(di, init_mem=bank,
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
                        oa = _fwd(di, init_mem=bank,
                                  layer_banks=casc.layer_banks(bank, cascade_map)
                                  if casc is not None else None)
                    lga = oa["logits"].float()[:, ac.size(1):]
                    aloss = F.cross_entropy(lga.reshape(-1, lga.size(-1)),
                                            at.reshape(-1), ignore_index=-100)
                    total = total + lam * aloss
                    a_v += float(aloss.detach()); a_cnt += 1
                    # addressed forward's write is discarded too
                # OPTION 2 : defer reach-back vers une entrée du pool dont les
                # slots ont quitté la banque vive (âge >= max_mem writes) —
                # même format que l'addr defer ([cue, blanks], perte sur les
                # blanks, write jeté), mais la cible vient d'une conv PASSÉE
                # de la vie : la page (ou ses résidus) est le seul pont.
                if reach_prob > 0 and reach_n < reach_max:
                    M_ = cfg.max_mem
                    elig = [e for e in reach_pool if w_total - e["w"] >= M_]
                    if elig and train_stream.rng.random() < reach_prob:
                        e = train_stream.rng.choice(elig)
                        age = w_total - e["w"]
                        sb = 0 if age < 2 * M_ else (1 if age < 4 * M_ else 2)
                        rc = e["cue"].to(device)
                        rt = e["tgt"].to(device)
                        di = torch.cat([rc, _fill(rc, blank_id, rt.size(1))], dim=1)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                            orr = _fwd(di, init_mem=bank,
                                       layer_banks=casc.layer_banks(bank, cascade_map)
                                       if casc is not None else None)
                        lgr = orr["logits"].float()[:, rc.size(1):]
                        rloss = F.cross_entropy(lgr.reshape(-1, lgr.size(-1)),
                                                rt.reshape(-1), ignore_index=-100)
                        total = total + lam * rloss
                        reach_v[sb] += float(rloss.detach()); reach_cnt[sb] += 1
                        reach_n += 1
                # dépôt au pool APRÈS usage (une entrée n'est jamais éligible
                # dans sa propre conv : âge < M garanti par m_total <= K = M)
                if reach_prob > 0:
                    w_total += 1                      # ce seg vient d'écrire 1 gist
                    tg = s.get("defer_tgt")
                    if tg is not None and bool((tg != -100).any()):
                        reach_pool.append({"cue": s["input_ids"][:, :reach_cue_len],
                                           "tgt": tg, "w": w_total})
                        if len(reach_pool) > 64:      # borne mémoire (tenseurs CPU 16+16 tok)
                            reach_pool.pop(0)
            # nan-guard (incident run 5e step 33 : chat nan -> opt.step sur grads
            # NaN = poids morts ; le pod phase 1 avait son guard, ce chemin non).
            # Conv non finie => pas de backward, et la banque de cette conv ne
            # rentre JAMAIS dans le carry (en never-reset un carry NaN est éternel).
            tot_ok = bool(torch.isfinite(total).all()) if torch.is_tensor(total) \
                else math.isfinite(total)
            if not tot_ok:
                # dump repro : la conv fautive + son état d'entrée, pour rejouer
                # offline GC on/off (hypothèse user : recompute du checkpoint qui
                # diverge du forward sur le routage MoE => grads incohérents)
                _dp = os.path.join(save_dir, f"nan_conv_step{step}_g{_g}.pt")
                try:
                    torch.save({"step": step, "g": _g,
                                "segs": [{k: (v.cpu() if torch.is_tensor(v) else v)
                                          for k, v in s.items()} for s in segs],
                                "bank_in": None if bank_carry is None
                                else bank_carry.detach().cpu(),
                                "casc": None if casc is None else casc.state_dict(),
                                "n_evict": n_evict}, _dp)
                except Exception as e:
                    _dp = f"dump raté: {e}"
                # les poids du MOMENT du nan (une seule fois par run) : sans eux
                # le repro depuis le dernier ckpt peut ne pas trigger
                _wp = os.path.join(save_dir, "nan_weights.pt")
                if not os.path.exists(_wp):
                    torch.save({"step": step, "model": base.state_dict()}, _wp)
                print(f"[nan-guard] step {step} conv {_g}: loss non finie, "
                      f"conv sautée (carry préservé) — repro {_dp}", flush=True)
                continue
            # mean over the accumulated convs ; en mode chunk_budget le nb de
            # convs n'est connu qu'en fin de step => division des grads après.
            (total / (1.0 if chunk_budget else grad_accum)).backward()
            if no_reset_files > 1 or nrf_never:
                # graph freed per file; the carried bank is data, not gradient path
                # guard local (supersède le nan_to_num du pod) : banque non
                # finie => carry RESET vie neuve, sinon carry propre
                nb = bank.detach() if bank is not None else None
                if nb is not None and not bool(torch.isfinite(nb).all()):
                    print(f"[nan-guard] step {step} conv {_g}: banque non finie, "
                          f"carry RESET (vie neuve)", flush=True)
                    bank_carry, casc_carry, nev_carry = None, None, 0
                    rpool_carry, wt_carry = [], 0
                else:
                    bank_carry = nb
                    casc_carry, nev_carry = casc, n_evict
                    if reach_prob > 0:
                        rpool_carry, wt_carry = reach_pool, w_total
                if delta is not None:
                    dstate_carry = dstate.detach()
            n_conv += 1
        if chunk_budget and n_conv > 1:
            # sémantique grad_accum restaurée : grads = moyenne sur les convs
            _gs = [p.grad for p in base.parameters() if p.grad is not None]
            if delta is not None:
                _gs += [p.grad for p in delta.parameters() if p.grad is not None]
            torch._foreach_div_(_gs, float(n_conv))
        _win_data += data_t; _win_chunks += step_chunks
        _prof = os.environ.get("TB_DDP_PROF")
        if _prof:
            torch.cuda.synchronize(); _t_bwd = time.time()
        if ddp_world > 1:
            # manual grad sync: average across ranks BEFORE clip so every rank
            # computes the same norm and the same update (ranks stay identical).
            # Buckets of ~64MB: one flat all-reduce per bucket instead of one
            # NCCL call per tensor (the MoE makes that hundreds of tiny calls).
            from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
            grads = [p.grad for p in base.parameters() if p.grad is not None]
            if delta is not None:
                grads += [p.grad for p in delta.parameters() if p.grad is not None]
            bucket, nbytes = [], 0
            for g_ in grads + [None]:
                if g_ is None or nbytes + g_.numel() * g_.element_size() > 64 << 20:
                    if bucket:
                        flat = _flatten_dense_tensors(bucket)
                        if ar_bf16:
                            f16 = flat.to(torch.bfloat16)
                            torch.distributed.all_reduce(f16)
                            flat = f16.to(flat.dtype)
                        else:
                            torch.distributed.all_reduce(flat)
                        flat.div_(ddp_world)
                        for b_, s_ in zip(bucket, _unflatten_dense_tensors(flat, bucket)):
                            b_.copy_(s_)
                    bucket, nbytes = [], 0
                if g_ is not None:
                    bucket.append(g_); nbytes += g_.numel() * g_.element_size()
        if _prof:
            torch.cuda.synchronize(); _t_ar = time.time()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            float(t.get("grad_clip", 1.0)))
        # DDP-safe (pod 10B) : après l'all-reduce les grads sont identiques sur
        # tous les ranks => ce check prend la même branche partout (pas de désync).
        if not bool(torch.isfinite(gn)):
            # LE cas intéressant pour l'hypothèse GC : loss finie mais grads NaN
            # (recompute qui diverge du forward). Dump du step complet + poids
            # (encore sains puisque le step est sauté).
            if ddp_rank == 0:
                # dump rank0 seulement : les convs diffèrent par rank, 8 writes
                # concurrents du même fichier = corruption
                _dp = os.path.join(save_dir, f"nan_gradstep_{step}.pt")
                try:
                    torch.save({"step": step,
                                "convs": [{"segs": [{k: (v.cpu() if torch.is_tensor(v)
                                                         else v) for k, v in s.items()}
                                                    for s in c["segs"]],
                                           "bank_in": None if c["bank_in"] is None
                                           else c["bank_in"].cpu(),
                                           "casc": c["casc"],
                                           "n_evict": c["n_evict"]}
                                          for c in _step_convs]}, _dp)
                except Exception as e:
                    _dp = f"dump raté: {e}"
                _wp = os.path.join(save_dir, "nan_weights.pt")
                if not os.path.exists(_wp):
                    torch.save({"step": step, "model": base.state_dict()}, _wp)
                # diagnostic pod 10B : quels modules portent les grads non finis
                bad = [n for n, p in model.named_parameters()
                       if p.grad is not None and not torch.isfinite(p.grad).all()]
                print(f"[nan-guard] step {step}: grad norm non finie ({len(bad)} "
                      f"tenseurs: {bad[:6]}), opt.step SAUTÉ — repro {_dp}", flush=True)
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
        if _prof:
            torch.cuda.synchronize()
            print(f"[prof step {step}] fwd+bwd {_t_bwd - _t_step0:.2f}s  "
                  f"allreduce {_t_ar - _t_bwd:.2f}s  clip+opt {time.time() - _t_ar:.2f}s  "
                  f"data-wait {data_t:.2f}s  chunks {step_chunks} ({n_conv} convs)",
                  flush=True)
        ic_v /= max(ic_cnt, 1); d_v /= max(d_cnt, 1)
        if ic_v == ic_v:   # un batch NaN isolé ne doit pas polluer l'EMA à vie
            ema_ic = ic_v if ema_ic is None else 0.95 * ema_ic + 0.05 * ic_v
        if d_v == d_v:
            ema_d  = d_v  if ema_d  is None else 0.95 * ema_d  + 0.05 * d_v
        if a_cnt:
            a_v /= a_cnt
            ema_a = a_v if ema_a is None else 0.95 * ema_a + 0.05 * a_v
        if chat_cnt:
            chat_v /= chat_cnt
            ema_chat = chat_v if ema_chat is None else 0.95 * ema_chat + 0.05 * chat_v
        for _s in range(3):
            if reach_cnt[_s]:
                rv = reach_v[_s] / reach_cnt[_s]
                ema_reach[_s] = (rv if ema_reach[_s] is None
                                 else 0.9 * ema_reach[_s] + 0.1 * rv)
        if step % log_every == 0:
            addr_s = f"addr {ema_a:.3f}  " if ema_a is not None else ""
            if ema_chat is not None:
                addr_s = f"chat {ema_chat:.3f}  " + addr_s
            if reach_prob > 0 and any(v is not None for v in ema_reach):
                # s1 ~ page p0, s2 ~ page mergée, s3 ~ détruit (contrôle : ne
                # doit PAS baisser si c'est bien la page qui est lue)
                addr_s += "reach " + "/".join(
                    "—" if v is None else f"{v:.2f}" for v in ema_reach) + "  "
            mem_s = (f"mem {torch.cuda.memory_allocated()/2**30:.1f}/"
                     f"{torch.cuda.max_memory_allocated()/2**30:.1f}G  "
                     if torch.cuda.is_available() else "")
            # probes utiles seulement : ic/defer masqués quand le step n'en a
            # pas vu (SFT pur p_chat=1.0 : ils étaient affichés à 0.000 fixe),
            # distill affiché dès que le teacher contribue (il n'était QUE dans
            # tensorboard pendant que β pilotait la moitié de la loss)
            ic_s = (f"ic {ema_ic:.3f} (ppl {math.exp(ema_ic):.1f})  "
                    if ic_cnt else "")
            d_s = f"defer {ema_d:.3f}  " if d_cnt else ""
            dist_s = (f"dist {distill_v / distill_n:.3f}  " if distill_n else "")
            if dist_cn or dist_fn:
                dist_s += ("[fait " + (f"{dist_c / dist_cn:.3f}" if dist_cn else "—")
                           + "/fill " + (f"{dist_f / dist_fn:.3f}" if dist_fn else "—")
                           + "]  ")
            _n_log = min(log_every, step - start_step)
            print(f"step {step:5d}  {ic_s}{d_s}"
                  f"{addr_s}{dist_s}β {_beta(step):.2f}  lr {lr_now:.2e}  {mem_s}"
                  f"{(time.time()-t0)/max(step - start_step, 1):.2f}s/step  "
                  f"chunks {_win_chunks / max(_n_log, 1):.1f}/step  "
                  f"data {_win_data / max(_n_log, 1):.2f}s", flush=True)
            if writer is not None:
                writer.add_scalar("train/ic_loss", ema_ic, step)
                writer.add_scalar("train/ic_ppl", math.exp(ema_ic), step)
                writer.add_scalar("train/defer_loss", ema_d, step)
                if ema_a is not None:
                    writer.add_scalar("train/addr_loss", ema_a, step)
                if ema_chat is not None:
                    writer.add_scalar("train/chat_loss", ema_chat, step)
                for _s, _v in enumerate(ema_reach):
                    if _v is not None:
                        writer.add_scalar(f"train/reach_s{_s + 1}", _v, step)
                writer.add_scalar("sched/lr", lr_now, step)
                writer.add_scalar("sched/beta", _beta(step), step)
                # perf : chunks/step + attente data (fenêtre log_every) — pour
                # régresser s/step = intercept + pente*m et chiffrer le host-bound
                writer.add_scalar("perf/chunks_per_step", _win_chunks / max(_n_log, 1), step)
                writer.add_scalar("perf/data_wait_s", _win_data / max(_n_log, 1), step)
                writer.add_scalar("perf/s_per_step",
                                  (time.time() - t0) / max(step - start_step, 1), step)
                if distill_n:
                    writer.add_scalar("train/distill", distill_v / distill_n, step)
                if dist_cn:
                    writer.add_scalar("train/distill_fait", dist_c / dist_cn, step)
                if dist_fn:
                    writer.add_scalar("train/distill_fill", dist_f / dist_fn, step)
            _win_data = 0.0; _win_chunks = 0
        if (step % eval_every == 0
                or (step == steps and not skip_final_eval)) and ddp_rank == 0:
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
                # eval sur `base` (non-compile) : les convs d'eval ont des largeurs
                # B=1 toutes differentes => un graphe dynamo par shape, premier
                # eval bloque >13 min a step 500 (pod 45191495). Eager = 1-2 min.
                m = evaluate(base, es, device, think_id, blank_id, defer_len,
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
                    bd = evaluate_by_depth(base, es, device, think_id, blank_id,
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
            if chat_eval is not None:
                chat_eval.rng.seed(1234)          # same conv set every eval
                mm = evaluate_math(model, chat_eval, tok, device, amp,
                                   chat_eval_convs, chat_max_new)
                by_age = mm.pop("_by_age", {})
                for kind in sorted(mm):
                    v = mm[kind]
                    if v["n_ans"]:
                        print(f"[math @{step}] {kind:10s} nll {v['nll']:.3f} "
                              f"grade {v['grade']:.2f} abl {v['grade_abl']:.2f} "
                              f"Δg {v['grade'] - v['grade_abl']:+.2f} | ans nll "
                              f"{v['ans_nll']:.3f} abl {v['ans_nll_abl']:.3f} "
                              f"Δnll {v['ans_nll_abl'] - v['ans_nll']:+.3f} "
                              f"(n={v['n']})", flush=True)
                    else:                     # contrôle sans truths (smalltalk)
                        print(f"[math @{step}] {kind:10s} nll {v['nll']:.3f} "
                              f"(n={v['n']}, contrôle)", flush=True)
                if by_age:
                    curve = "  ".join(
                        f"{b}: Δg {by_age[b]['dgrade']:+.2f} "
                        f"Δnll {by_age[b]['dnll']:+.3f} (n{by_age[b]['n']})"
                        for b in AGE_BUCKETS if b in by_age)
                    print(f"[math @{step}] recall par âge (writes fait→réponse)"
                          f" : {curve}", flush=True)
                if metrics_file:
                    with open(metrics_file, "a") as f:
                        f.write(json.dumps({"step": step, "math": mm,
                                            "math_by_age": by_age}) + "\n")
                if writer is not None:
                    for kind, v in mm.items():
                        writer.add_scalar(f"eval_math/{kind}/nll", v["nll"], step)
                        writer.add_scalar(f"eval_math/{kind}/grade", v["grade"], step)
                        writer.add_scalar(f"eval_math/{kind}/grade_abl",
                                          v["grade_abl"], step)
                        if v["n_ans"]:
                            writer.add_scalar(
                                f"eval_math/{kind}/ans_dnll",
                                v["ans_nll_abl"] - v["ans_nll"], step)
        if (step % save_every == 0 or step == steps) and ddp_rank == 0:
            _save_ck(step, os.path.join(save_dir,
                     "final.pt" if step == steps else f"step_{step}.pt"))
            if nrf_never:
                _save_bank(step, os.path.join(save_dir,
                           "bank_final.pt" if step == steps
                           else f"bank_step_{step}.pt"))
    if ddp_world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    print("done.", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--resume"]
    main(args[0], resume="--resume" in sys.argv[1:])
