#!/bin/bash
# EXPERIMENTAL, UNVALIDATED for this release. ring-fabric.sh RANK [PREFIX]: address one host's four ports for the
# switchless four-host ring (runtime only: nothing persists past a reboot). Run as root on each host.
#
# Cabling: rank r's port 0 goes to rank r+1's port 1 (mod 4), two cables a pair, one per PCIe half. Link L joins
# rank L (port 0, host octet 10) and rank L+1 (port 1, host octet 11). Half 0 of link L is PREFIX.L.0/24, half 1 is
# PREFIX.(L+4).0/24, so the ring uses eight /24s:
#   enp1s0f0np0    half 0, port 0 -> PREFIX.r.10        enp1s0f1np1    half 0, port 1 -> PREFIX.(r-1).11
#   enP2p1s0f0np0  half 1, port 0 -> PREFIX.(r+4).10    enP2p1s0f1np1  half 1, port 1 -> PREFIX.(r-1+4).11
# PREFIX is the first two octets (default 198.19; any unused private /16 works).
set -euo pipefail
r=${1:?usage: ring-fabric.sh RANK [PREFIX]}
p=${2:-198.19}
case ${r} in 0|1|2|3) ;; *) echo "RANK is 0..3" >&2; exit 2;; esac
prev=$(( (r + 3) % 4 ))
add() { # netdev address/24
  local d=$1 a=$2
  ip link set dev "${d}" mtu 9000
  echo 1 > "/proc/sys/net/ipv4/conf/${d}/arp_ignore"
  echo 2 > "/proc/sys/net/ipv4/conf/${d}/arp_announce"
  ip address replace "${a}" dev "${d}"
  ip link set dev "${d}" up
  echo "${d} ${a}"
}
add enp1s0f0np0 "${p}.${r}.10/24"
add enp1s0f1np1 "${p}.${prev}.11/24"
add enP2p1s0f0np0 "${p}.$((r + 4)).10/24"
add enP2p1s0f1np1 "${p}.$((prev + 4)).11/24"
