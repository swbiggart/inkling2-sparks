#!/bin/bash
# uninstall.sh [--flush]: remove what install.sh installed (run as root). --flush also removes the fabric addresses
# from the two netdevs now; otherwise they stay until the next reboot.
set -uo pipefail
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }
devs="enp1s0f0np0 enP2p1s0f0np0"
# shellcheck source=/dev/null
[ -r /etc/default/inkling2-fabric ] && . /etc/default/inkling2-fabric && devs=${FABRIC_DEVS:-${devs}}
systemctl disable --now inkling2-fabric.service 2>/dev/null
rm -f /etc/systemd/system/inkling2-fabric.service /usr/local/sbin/inkling2-fabric.sh \
  /etc/udev/rules.d/99-inkling2-fabric.rules /etc/default/inkling2-fabric
systemctl daemon-reload
udevadm control --reload
if [ "${1:-}" = --flush ]; then
  for x in ${devs}; do ip address flush dev "${x}" && echo "flushed ${x}"; done
fi
echo "inkling2-fabric removed"
