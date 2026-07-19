# Vast.ai — bring-up puis run 350M

Décision 2026-07-13 : bring-up (mesure débit + pic VRAM) sur Vast.ai, le vrai
run 350M ensuite sur le GPU que le bring-up désigne (4090 batch 1 vs A100 80GB
B=4 — coût/conv quasi égal, l'A100 achète du wall-clock ÷3.7 et de la marge).

## 0. Prérequis (une fois)

```bash
pip install vastai
vastai set api-key <clé depuis https://cloud.vast.ai/account/>
```

## 1. Trouver une 4090 vérifiée

```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 verified=true reliability>0.98 \
  disk_space>=60 inet_down>=200' -o 'dph+' | head -15
```

- `verified=true` + `reliability>0.98` : on paie ~0.05 $/h de plus qu'un hôte
  marketplace nu, c'est l'assurance anti-« hôte disparu mi-run ».
- `inet_down>=200` : le bring-up télécharge ~2-4 GB de HF streaming.

## 2. Lancer le bring-up (~1-2 h, < 1 $)

```bash
vastai create instance <OFFER_ID> \
  --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
  --disk 60 \
  --onstart-cmd 'bash -lc "apt-get update -qq && apt-get install -y -qq git && \
    curl -fsSL https://raw.githubusercontent.com/kkuette/thought-bank/claude/status-check-2fa903/vast/bringup_350m.sh \
    -o /workspace/bringup_350m.sh && bash /workspace/bringup_350m.sh"'
```

Suivi : `vastai ssh <INSTANCE_ID>` puis `tail -f /workspace/bringup.log`.
Le verdict est le bloc `=== BRINGUP RESULT ===` en fin de log (steps/h + pic
VRAM). **Détruire l'instance après** (`vastai destroy instance <ID>`) — la
facturation court tant qu'elle existe, même stoppée le stockage est facturé.

## 3. Dépouillement → devis

- `sec_per_step` net (timestamps step 5→40 du log, pas le brut qui inclut le
  warm-up des caches HF) → `2000 steps × sec/step / 3600` = heures de 4090.
- Pic VRAM : si > ~21 GB sur 24, pas de marge probes/cascade → A100.
- Publier les 2 nombres dans EXPERIMENTS.md (ligne bring-up) et décider
  4090 (~0.29-0.39 $/h) vs A100 80GB (~0.78-1.50 $/h, B=4).

## Pièges connus

- **Pas de stockage persistant** sur instance détruite : le vrai run devra
  syncer ses checkpoints hors du pod (HF Hub ou B2/S3, toutes les N steps,
  `--resume` au restart). Script à écrire AVANT le vrai run, pas pour le
  bring-up (aucun checkpoint voulu).
- **Jamais de secrets dans onstart** (visible dans les métadonnées d'instance).
  Le repo est public, pas de token nécessaire pour le bring-up.
- Les datasets HF gated : aucun dans les ancres (codeparrot/fineweb = publics).
- Préemption : le bring-up est court et relançable tel quel (setup idempotent).
