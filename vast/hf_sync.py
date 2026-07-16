#!/usr/bin/env python3
"""dsv6 — sidecar de sync checkpoints -> HF Hub (repo privé), pour les runs
Vast.ai (décision user 2026-07-13 : HF Hub plutôt que B2/S3).

Tourne À CÔTÉ du trainer (process séparé, ne touche pas au train) :
  - poll save_dir toutes les POLL_S secondes ;
  - un fichier *.pt dont le mtime a changé ET est resté stable STABLE_S
    secondes (pas d'upload d'un torch.save en cours) est uploadé ;
  - last.pt est écrasé dans le repo (l'historique des jalons save_every
    est conservé : ckpt_*.pt gardent leur nom) ;
  - après upload, vérification de taille via l'API avant de marquer « sauvé »
    (un step n'est sûr que si son checkpoint est DEHORS du pod).

Restauration (avant de lancer le trainer avec --resume) :
    python vast/hf_sync.py --restore --repo <user>/<repo> --dir <save_dir>

Sync (à lancer en fond au début du run) :
    python vast/hf_sync.py --repo <user>/<repo> --dir <save_dir> [--dry-run]

HF_TOKEN attendu dans l'environnement (jamais dans l'onstart : le poser via
`vastai create instance ... --env '-e HF_TOKEN=...'` ou un fichier lu au boot).
"""
import argparse, os, sys, time

POLL_S = 60
STABLE_S = 30


def log(msg):
    print(f"[hf_sync {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def api_and_repo(repo):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    return api


def remote_size(api, repo, name):
    try:
        info = api.get_paths_info(repo, [name])
        return info[0].size if info else None
    except Exception:
        return None


def restore(repo, save_dir):
    from huggingface_hub import hf_hub_download
    os.makedirs(save_dir, exist_ok=True)
    try:
        p = hf_hub_download(repo, "last.pt", local_dir=save_dir)
        log(f"restauré {p} ({os.path.getsize(p)/1e9:.2f} GB) — lancer le trainer avec --resume")
    except Exception as e:
        log(f"pas de last.pt distant ({type(e).__name__}) — démarrage from scratch")


def sync_loop(repo, save_dir, dry_run=False):
    api = None if dry_run else api_and_repo(repo)
    seen = {}                                  # path -> (mtime, size) du dernier upload
    log(f"watch {save_dir} -> {repo} (poll {POLL_S}s, stabilité {STABLE_S}s"
        + (", DRY-RUN)" if dry_run else ")"))
    while True:
        try:
            names = [n for n in os.listdir(save_dir) if n.endswith(".pt")]
        except FileNotFoundError:
            names = []
        for n in sorted(names):
            p = os.path.join(save_dir, n)
            try:
                st = os.stat(p)
            except FileNotFoundError:
                continue
            key = (st.st_mtime, st.st_size)
            if seen.get(p) == key:
                continue
            if time.time() - st.st_mtime < STABLE_S:
                continue                       # écriture peut-être en cours
            st2 = os.stat(p)
            if (st2.st_mtime, st2.st_size) != key:
                continue                       # a bougé pendant la vérif
            log(f"upload {n} ({st.st_size/1e9:.2f} GB)…")
            if dry_run:
                seen[p] = key
                log(f"  DRY-RUN ok {n}")
                continue
            try:
                api.upload_file(path_or_fileobj=p, path_in_repo=n,
                                repo_id=repo, commit_message=f"sync {n}")
                rs = remote_size(api, repo, n)
                if rs == st.st_size:
                    seen[p] = key
                    log(f"  SAUVÉ {n} (taille distante vérifiée)")
                else:
                    log(f"  ÉCHEC vérif taille {n} (distant={rs}) — retry au prochain poll")
            except Exception as e:
                log(f"  upload raté {n}: {type(e).__name__}: {e} — retry au prochain poll")
        time.sleep(POLL_S)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="ex: kkuette/dsv6-350m")
    ap.add_argument("--dir", required=True, help="save_dir du trainer")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.dry_run and not (os.environ.get("HF_TOKEN") or os.path.exists(
            os.path.expanduser("~/.cache/huggingface/token"))):
        sys.exit("HF_TOKEN absent de l'environnement")
    if a.restore:
        restore(a.repo, a.dir)
    else:
        sync_loop(a.repo, a.dir, dry_run=a.dry_run)
