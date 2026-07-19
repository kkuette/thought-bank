# Ferme de rigs — mode d'emploi

Architecture : `server0` (Unraid) exporte le share NFS `llm_research` ; chaque rig
(Debian 12 natif) monte le share sur `/mnt/tb` et fait tourner un worker par GPU.
Pas de scheduler : la file est un répertoire, un job est un fichier bash.

## 1. Côté Unraid (fait une fois)

- `Settings → NFS → Enable`.
- Share `llm_research` : `Export: Yes`, `Security: Private`, règle :
  `<SOUS_RESEAU>/24(sec=sys,rw,no_root_squash,insecure)`
  — `insecure` requis pour les clients WSL2 (NAT = ports source >1024) ; sans
  effet pour les rigs natifs.
- Unraid sert du NFSv3 (v4 refusé) → monter avec `vers=3,nolock`.
- Le share sur pool cache SSD de préférence (l'array shfs est lent en écriture).

## 2. Côté rig (une fois par rig, depuis Debian 12 fraîche)

```bash
git clone https://github.com/kkuette/thought-bank /opt/thought-bank
sudo /opt/thought-bank/scripts/farm/setup_rig.sh <IP_NAS> llm_research
sudo reboot
# après reboot :
nvidia-smi                # les 6 cartes doivent apparaître
for i in 0 1 2 3 4 5; do sudo systemctl enable --now tb-worker@$i; done
```

## 3. Lancer des jobs

Un job = un fichier `*.job` (script bash) déposé dans `/mnt/tb/queue/`.
Il s'exécute sur UN GPU (déjà isolé via `CUDA_VISIBLE_DEVICES`), cwd = repo,
venv actif, variables `TB_MNT`, `WORKER` disponibles. Priorité = ordre
lexicographique du nom (`00_urgent.job` passe avant `50_sweep.job`).

Exemple — une seed du 97M v2c :

```bash
cat > /mnt/tb/queue/10_v2c_97m_seed44.job <<'EOF'
cd deepseek_v4_mini
python code_defer_native.py \
  --config configs/v2c_varlen.yaml \
  --seed 44 \
  --out $TB_MNT/checkpoints/v2c_97m_s44 \
  --log $TB_MNT/runs/v2c_97m_s44
EOF
```

Cycle de vie : `queue/*.job` → `queue/running/<worker>__nom.job` →
`queue/done/` ou `queue/failed/`. Log worker : `runs/<nom>.workerlog`.

Surveiller : `ls /mnt/tb/queue/running/` et `tail -f /mnt/tb/runs/*.workerlog`.
Rejouer un échec : `mv queue/failed/x.job queue/`.

## 4. Pré-tokenisation (VM `data`, CPU, sans GPU)

VM Debian dédiée : NFS monté, venv `/opt/tb-venv` (torch CPU), repo
`/opt/thought-bank`, `HF_HOME=/mnt/tb/data/hf_cache`, SSH par clé uniquement.
Usage :

```bash
ssh <user>@<IP_VM_DATA>
export TB_REPO=/opt/thought-bank
/opt/tb-venv/bin/python /mnt/tb/scripts/prebuild_data.py \
  --cache-dir /mnt/tb/data_cache \
  /opt/thought-bank/deepseek_v4_mini/configs/<config>.yaml
```

(`TB_REPO` : le script est lancé depuis la copie NAS, hors repo — la variable
pointe l'import `deepseek_v4_mini`.) L'alternative Docker-sur-Unraid
(`prebuild_data.sh`) reste disponible mais la VM est le chemin par défaut.

### Ancienne méthode (server0, Docker, CPU)

`prebuild_data.py` rejoue la construction de données du trainer (mêmes clés de
cache md5, y compris les tokens spéciaux `<think>`/`<blank>`) et remplit le
cache partagé `/mnt/tb/data_cache/`. Validé : 4/4 cache hits sur la config v2c.

Depuis le terminal Unraid :

```bash
/mnt/user/llm_research/scripts/prebuild_data.sh \
  deepseek_v4_mini/configs/code_defer_native_v2c_varlen.yaml
```

**Convention** : toute config destinée à la ferme déclare
`data.cache_dir: /mnt/tb/data_cache` (le défaut du trainer est un `data_cache/`
local qui raterait le cache partagé). Le cache des corpus actuels est déjà
semé (16 entrées, ~1 Go). Le HF cache partagé (`data/hf_cache`) évite aussi
les re-téléchargements.

## Notes matérielles

- Risers PCIe x1 : sans effet (jobs mono-GPU, données lues en séquentiel sur NFS).
- Power limit 150 W/carte via `gpu-powerlimit.service` (boot).
- 8 GB VRAM : OK 97M (~3,7 GB/conv batchée), 135M à valider en batch 1 ragged,
  350M impossible (A100 spot requis).
- Premier job de validation sur UNE carte avant toute campagne : run 97M 500 steps
  (VRAM réelle, débit vs 3090 attendu ~55-60 %, stabilité riser sous charge).
