#!/usr/bin/env bash
# Setup d'un rig de la ferme depuis une Debian 12 fraîche (netinstall, ssh actif).
# Usage: sudo ./setup_rig.sh <IP_server0> [share]
# Idempotent : relançable sans casser l'existant.
set -euo pipefail

NFS_HOST="${1:?Usage: setup_rig.sh <IP_NAS> [share]}"
NFS_SHARE="${2:-llm_research}"
MNT=/mnt/tb
POWER_LIMIT=150   # W par carte (3070/3070Ti : −40% conso pour −5-10% perf)

echo "== Paquets de base =="
apt-get update
apt-get install -y --no-install-recommends \
  nfs-common git python3-venv python3-pip curl ca-certificates \
  linux-headers-amd64 build-essential

echo "== Driver NVIDIA (non-free) =="
if ! grep -q non-free /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null; then
  sed -i 's/main\( contrib\)\?$/main contrib non-free non-free-firmware/' /etc/apt/sources.list
  apt-get update
fi
apt-get install -y nvidia-driver firmware-misc-nonfree

echo "== Montage NFS =="
mkdir -p "$MNT"
# NFSv3 : server0/Unraid a refuse le v4 lors de la mise en place (2026-07-10)
# automount : évite la course au boot quand le réseau (pont wifi) est lent à monter
FSTAB_LINE="${NFS_HOST}:/mnt/user/${NFS_SHARE} ${MNT} nfs vers=3,nolock,hard,noatime,_netdev,x-systemd.automount,x-systemd.mount-timeout=90 0 0"
grep -qF "$FSTAB_LINE" /etc/fstab || echo "$FSTAB_LINE" >> /etc/fstab
mount -a
mountpoint -q "$MNT" && echo "NFS OK: $MNT" || { echo "ECHEC montage NFS"; exit 1; }

echo "== Arborescence partagée (no-op si déjà là) =="
mkdir -p "$MNT"/{data,checkpoints,runs,queue,queue/running,queue/done,queue/failed}

echo "== Repo =="
if [ ! -d /opt/thought-bank ]; then
  git clone https://github.com/kkuette/thought-bank /opt/thought-bank
fi

echo "== Env Python =="
if [ ! -d /opt/tb-venv ]; then
  python3 -m venv /opt/tb-venv
  # TMPDIR : /tmp est un tmpfs limité par la RAM (trixie) — le wheel torch (780 Mo) le remplit
  export TMPDIR=/var/tmp
  /opt/tb-venv/bin/pip install --upgrade pip
  /opt/tb-venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
  /opt/tb-venv/bin/pip install --no-cache-dir -r /opt/thought-bank/requirements.txt
fi

echo "== Power limit au boot =="
cat > /etc/systemd/system/gpu-powerlimit.service <<EOF
[Unit]
Description=Power-limit GPUs a ${POWER_LIMIT}W
After=multi-user.target
[Service]
Type=oneshot
# post-mortem 2026-07-12 : au boot le driver peut ne pas etre pret — sans
# retry le oneshot echoue et les GPU restent a 310W (suspect n°1 de la coupure)
RemainAfterExit=yes
TimeoutStartSec=600
ExecStartPre=/bin/sh -c 'until /usr/bin/nvidia-smi >/dev/null 2>&1; do sleep 5; done'
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/usr/bin/nvidia-smi -pl ${POWER_LIMIT}
[Install]
WantedBy=multi-user.target
EOF

echo "== Workers (un par GPU, démarrés au boot) =="
install -m 755 "$(dirname "$0")/gpu_worker.sh" /usr/local/bin/gpu_worker.sh 2>/dev/null || \
  install -m 755 /opt/thought-bank/scripts/farm/gpu_worker.sh /usr/local/bin/gpu_worker.sh
cat > /etc/systemd/system/tb-worker@.service <<'EOF'
[Unit]
Description=thought-bank GPU worker %i
After=gpu-powerlimit.service remote-fs.target
Requires=remote-fs.target
# post-mortem 2026-07-12 : NFS pas pret au boot => crash-loop rapide => la
# rate-limit systemd abandonne le worker pour de bon (gpu1 mort au redemarrage)
StartLimitIntervalSec=0
[Service]
Environment=GPU_ID=%i
Environment=TB_MNT=/mnt/tb
Environment=TB_REPO=/opt/thought-bank
Environment=TB_VENV=/opt/tb-venv
ExecStart=/usr/local/bin/gpu_worker.sh
Restart=always
RestartSec=30
User=root
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable gpu-powerlimit.service

echo
echo "Reboot pour charger le driver, puis :"
echo "  nvidia-smi                                    # verifier les 6 cartes"
echo "  for i in 0 1 2 3 4 5; do systemctl enable --now tb-worker@\$i; done"
