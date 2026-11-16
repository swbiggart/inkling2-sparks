#!/bin/bash
# inkling2-fabric.sh: bring this host's switch-facing 200G port up for the Inkling-2 TP4 fabric, on both PCIe halves:
# MTU, one address on each half's netdev (a /24 a half), and the ARP settings for two ports on one L2 segment.
# Settings come from /etc/default/inkling2-fabric (see inkling2-fabric.conf.example), or from the environment:
#   FABRIC_HOST  this host's number on the fabric, rank + 1 (1..4): the last octet of both addresses
#   FABRIC_DEVS  the two netdevs, half 0 then half 1 (default: enp1s0f0np0 enP2p1s0f0np0)
#   FABRIC_NETS  the two /24s as their first three octets, half 0 then half 1 (no default)
#   FABRIC_MTU   default 9000
# Runs at boot (inkling2-fabric.service) and again whenever the mlx5 driver recreates a port (the udev rule), since a
# cable change re-probes the NIC and the ports come back down, at MTU 1500, without addresses.
set -u
conf=${FABRIC_CONF:-/etc/default/inkling2-fabric}
# shellcheck source=/dev/null
[ -r "${conf}" ] && . "${conf}"
host=${FABRIC_HOST:?FABRIC_HOST: this host is rank + 1 (1..4)}
read -r -a devs <<< "${FABRIC_DEVS:-enp1s0f0np0 enP2p1s0f0np0}"
read -r -a nets <<< "${FABRIC_NETS:?FABRIC_NETS: two /24 prefixes, e.g. \"198.18.0 198.18.1\"}"
mtu=${FABRIC_MTU:-9000}
if [ ${#devs[@]} != 2 ] || [ ${#nets[@]} != 2 ]; then
  echo "inkling2-fabric: FABRIC_DEVS and FABRIC_NETS need two entries each" >&2; exit 2
fi
# the netdevs can appear a few seconds after boot or a re-probe
i=0
while [ ${i} -lt 60 ] && ! { [ -e /sys/class/net/${devs[0]} ] && [ -e /sys/class/net/${devs[1]} ]; }; do
  sleep 1; i=$((i + 1))
done
rc=0
for h in 0 1; do
  d=${devs[$h]}
  if [ ! -e /sys/class/net/${d} ]; then echo "inkling2-fabric: no ${d}" >&2; rc=1; continue; fi
  ip link set dev "${d}" mtu "${mtu}"
  # two ports on one L2 segment: answer ARP only for the port's own address, and announce from it
  sysctl -q -w "net.ipv4.conf.${d}.arp_ignore=1" "net.ipv4.conf.${d}.arp_announce=2"
  ip address replace "${nets[$h]}.${host}/24" dev "${d}"
  ip link set dev "${d}" up
  echo "inkling2-fabric: ${d} ${nets[$h]}.${host}/24 mtu ${mtu}"
done
exit ${rc}
