#!/bin/bash
# install.sh RANK [--nets "A.B.C A.B.D"] [--devs "DEV0 DEV1"] [--mtu N] [--dry-run]
# Run as root on each host, from this folder. Installs the fabric setup persistently and starts it:
#   /etc/default/inkling2-fabric             FABRIC_HOST = RANK + 1, the netdevs, the two /24s, the MTU
#   /usr/local/sbin/inkling2-fabric.sh       the script
#   /etc/systemd/system/inkling2-fabric.service   enabled, runs at boot
#   /etc/udev/rules.d/99-inkling2-fabric.rules    re-runs the unit when the driver recreates a port (by MAC)
# --dry-run prints what would be written and changes nothing. uninstall.sh removes all of it.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
rank=${1:?usage: install.sh RANK [--nets "A.B.C A.B.D"] [--devs "DEV0 DEV1"] [--mtu N] [--dry-run]}
shift
nets="198.18.0 198.18.1"
devs="enp1s0f0np0 enP2p1s0f0np0"
mtu=9000
dry=0
while [ $# -gt 0 ]; do
  case $1 in
    --nets) nets=$2; shift 2;;
    --devs) devs=$2; shift 2;;
    --mtu) mtu=$2; shift 2;;
    --dry-run) dry=1; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
case ${rank} in 0|1|2|3) ;; *) echo "RANK is 0..3" >&2; exit 2;; esac
read -r -a d <<< "${devs}"
[ ${#d[@]} = 2 ] || { echo "--devs needs two netdevs" >&2; exit 2; }
macs=()
for x in "${d[@]}"; do
  [ -r "/sys/class/net/${x}/address" ] || { echo "no netdev ${x} on this host (ip -br link)" >&2; exit 1; }
  macs+=("$(cat "/sys/class/net/${x}/address")")
done
conf="FABRIC_HOST=$((rank + 1))
FABRIC_DEVS=\"${devs}\"
FABRIC_NETS=\"${nets}\"
FABRIC_MTU=${mtu}"
rules=$(sed -e "s/@MAC0@/${macs[0]}/" -e "s/@MAC1@/${macs[1]}/" "${here}/99-inkling2-fabric.rules.in")
if [ ${dry} = 1 ]; then
  printf '== /etc/default/inkling2-fabric\n%s\n== /etc/udev/rules.d/99-inkling2-fabric.rules\n%s\n' "${conf}" "${rules}"
  echo "== then: install the script and unit, systemctl enable --now inkling2-fabric, udevadm control --reload"
  exit 0
fi
[ "$(id -u)" = 0 ] || { echo "run as root (or --dry-run)" >&2; exit 1; }
printf '%s\n' "${conf}" > /etc/default/inkling2-fabric
install -m 0755 "${here}/inkling2-fabric.sh" /usr/local/sbin/inkling2-fabric.sh
install -m 0644 "${here}/inkling2-fabric.service" /etc/systemd/system/inkling2-fabric.service
printf '%s\n' "${rules}" > /etc/udev/rules.d/99-inkling2-fabric.rules
systemctl daemon-reload
systemctl enable inkling2-fabric.service
systemctl restart inkling2-fabric.service
udevadm control --reload
systemctl --no-pager status inkling2-fabric.service | head -n 12 || true
for x in "${d[@]}"; do ip -br address show dev "${x}"; done
