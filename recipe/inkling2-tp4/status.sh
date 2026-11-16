#!/usr/bin/env bash
# status.sh CONFIG [--bw [SECS]]: each rank's process state, MemAvailable and last log lines, then each host's
# fabric health: both fabric ports' speed, MTU, carrier and address, the HCA state, and the NIC re-probe check (a
# "PCIe slot power" message in the kernel log since boot means the NIC was re-probed after a cable change and
# transmits at about 13 Gb/s per half until the host reboots).
# --bw: also a short ib_write_bw (64 KiB, 4 QPs, SECS seconds, default 3) from each rank to the next one, on each
# fabric half in turn, one flow at a time. Expect about 100 Gb/s a flow on a healthy 200G switch port; a re-probed
# NIC shows about 13. Best run while the server is idle or stopped. For the switch layout (two fabric halves).
# Exit status: 1 if any fabric check warned or any flow was below BW_MIN Gb/s (default 80).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
conf=${1:?usage: status.sh CONFIG [--bw [SECS]]}
bw=0; secs=3
if [ "${2:-}" = --bw ]; then bw=1; secs=${3:-3}; fi
# shellcheck source=/dev/null
source "$here/release-defaults.sh"
# shellcheck source=/dev/null
source "$conf"
release_settings || exit 2
remote=$ROOT/recipe
rc=0
for r in 0 1 2 3; do
  # rank.sh and this config as status.conf: the server's own copy (config.sh, from start.sh) is left alone, since
  # stop.sh needs the config the server was started with
  h=${HOSTS[$r]}
  ssh "$h" "mkdir -p '$remote'" && scp -q "$here/rank.sh" "$here/release-defaults.sh" "$h:$remote/" && scp -q "$conf" "$h:$remote/status.conf" \
    || { echo "copy to $h failed" >&2; exit 1; }
done
for r in 0 1 2 3; do
  ssh "${HOSTS[$r]}" "LINES_SHOWN=${LINES_SHOWN:-3} bash '$remote/rank.sh' '$remote/status.conf' status $r"
done
echo "fabric:"
for r in 0 1 2 3; do
  ssh "${HOSTS[$r]}" "bash '$remote/rank.sh' '$remote/status.conf' fabric $r" || rc=1
done
if [ $bw = 1 ]; then
  echo "ib_write_bw, rank r -> r+1, ${secs} s a flow:"
  port=${BW_PORT:-29700}
  for h in 0 1; do
    for r in 0 1 2 3; do
      n=$(( (r + 1) % 4 ))
      addr=$(ssh "${HOSTS[$n]}" "bash '$remote/rank.sh' '$remote/status.conf' addr $n $h")
      if [ -z "$addr" ]; then echo "  half $h rank $r -> $n: no address on rank $n"; rc=1; continue; fi
      ssh -n "${HOSTS[$n]}" "bash '$remote/rank.sh' '$remote/status.conf' bw-server $n $h $port $secs" &
      spid=$!
      sleep 1.5
      g=$(ssh -n "${HOSTS[$r]}" "bash '$remote/rank.sh' '$remote/status.conf' bw-client $r $h $port $secs $addr")
      wait $spid 2>/dev/null
      flag=OK
      if [ "$g" = FAIL ] || awk -v g="$g" -v m="${BW_MIN:-80}" 'BEGIN { exit !(g + 0 < m) }'; then flag=LOW; rc=1; fi
      echo "  half $h (${FABRIC_HCAS[$h]}) rank $r -> $n: $g Gb/s $flag"
      port=$((port + 1))
    done
  done
fi
exit $rc
