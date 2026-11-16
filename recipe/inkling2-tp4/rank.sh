#!/usr/bin/env bash
# rank.sh CONFIG COMMAND RANK [ARGS]: one host's rank of the TP4 server. start.sh, stop.sh, status.sh and check.sh
# copy this file, release-defaults.sh and CONFIG to every host and run it there over ssh.
#   start                 launch the rank in the background (setsid, nohup); pid in $RUN/rank-R.pid, log in
#                         $RUN/rank-R.log
#   stop                  SIGTERM the rank's pid (from its pid file), wait up to $STOP_WAIT s (default 60), then
#                         SIGKILL it
#   status                the pid's state, MemAvailable, the log's last lines
#   fabric                the fabric ports: speed, MTU, carrier, HCA state, and the NIC re-probe check (exit 1 on
#                         any warning)
#   run LOG SECS CMD...   CMD in the foreground with the server's environment, under `timeout SECS`, output to LOG
#                         ({R} in CMD becomes the rank); the exit status is CMD's
#   bw-server H PORT SECS          ib_write_bw server on fabric half H (index into FABRIC_HCAS)
#   bw-client H PORT SECS ADDR     ib_write_bw client to ADDR on half H; prints Gb/s
#   addr H                         this host's IPv4 address on fabric half H
#   env                            the server's environment, one VAR=value a line (setup.sh --container writes it
#                                  as the container's env file; ENV_LIBS replaces LIBS in LD_LIBRARY_PATH,
#                                  ENV_BIN_DIR the folder of tp_mailbox.fatbin)
set -uo pipefail
conf=$1; cmd=$2; r=$3
shift 3
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
source "$here/release-defaults.sh"
# shellcheck source=/dev/null
source "$conf"
release_settings || exit 2
pidf=$RUN/rank-$r.pid
log=$RUN/rank-$r.log
mkdir -p "$RUN"

alive() { [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; }
mem() { awk '/MemAvailable/{printf "%.1f GiB", $2 / 1048576}' /proc/meminfo; }
# the server's environment, in envs
server_env() {
  local graph=${GRAPH//\{R\}/$r}
  release_env
  # TF_TP_MASTER: the fabric TP wire's rendezvous (TF_INK2_ROCE_MULTI) is the NCCL bootstrap address. The release's
  # settings (release-defaults.sh) come before EXTRA_ENV, so an entry there overrides one.
  envs=(LD_LIBRARY_PATH="$LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" CUDA_VISIBLE_DEVICES=0
        NCCL_COMM_ID="$MASTER:$NCCL_PORT" NCCL_SOCKET_IFNAME="$SOCKET_IFNAME" TF_TP_MASTER="$MASTER"
        "${NCCL_ENV[@]}" "${release_envs[@]}" ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"})
  if [ -n "$graph" ]; then envs+=(NCCL_GRAPH_FILE="$graph"); fi
}
# the kernel log since boot (journal first, then dmesg); fails when neither is readable by this user
klog() {
  local out
  out=$(journalctl -k -b -q --no-pager -o short-monotonic 2>/dev/null) && [ -n "$out" ] && { printf '%s\n' "$out"; return 0; }
  dmesg 2>/dev/null
}

case $cmd in
start)
  if alive; then echo "rank $r already running (pid $(cat "$pidf"))"; exit 1; fi
  server_env
  [ -f "$log" ] && mv -f "$log" "$log.prev"
  echo "$(date -u +%FT%TZ) rank $r start; MemAvailable $(mem)" > "$log"
  # shellcheck disable=SC2086
  setsid nohup env "${envs[@]}" "$BIN" serve "$MODEL" --backend cuda --tp 4 --rank "$r" --context "$CONTEXT" \
    --parallel "$PARALLEL" $(release_args) --host "$HTTP_HOST" --port "$HTTP_PORT" $EXTRA_ARGS >> "$log" 2>&1 < /dev/null &
  echo $! > "$pidf"
  echo "rank $r on $(hostname): pid $(cat "$pidf"), log $log";;
stop)
  if ! alive; then echo "rank $r not running"; rm -f "$pidf"; exit 0; fi
  p=$(cat "$pidf")
  kill -TERM "$p"
  for _ in $(seq 1 "${STOP_WAIT:-60}"); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
  if kill -0 "$p" 2>/dev/null; then kill -KILL "$p"; echo "rank $r: pid $p killed after ${STOP_WAIT:-60} s"; else echo "rank $r: pid $p ended"; fi
  rm -f "$pidf";;
status)
  if alive; then s="running (pid $(cat "$pidf"))"; else s="not running"; fi
  echo "rank $r on $(hostname): $s; MemAvailable $(mem)"
  tail -n "${LINES_SHOWN:-3}" "$log" 2>/dev/null | cut -c1-240;;
fabric)
  warn=0
  devs=("${FABRIC_DEVS[@]}"); hcas=("${FABRIC_HCAS[@]}")
  for i in "${!devs[@]}"; do
    d=${devs[$i]}; n=/sys/class/net/$d
    if [ ! -e "$n" ]; then echo "  rank $r $d: MISSING (no such netdev)"; warn=1; continue; fi
    carrier=$(cat "$n/carrier" 2>/dev/null || echo 0)
    speed=$(cat "$n/speed" 2>/dev/null || echo unknown)
    mtu=$(cat "$n/mtu")
    addr=$(ip -4 -o address show dev "$d" 2>/dev/null | awk '{print $4}' | head -1)
    hca=${hcas[$i]:-}
    state=$(cat "/sys/class/infiniband/$hca/ports/1/state" 2>/dev/null | awk '{print $2}')
    notes=()
    [ "$carrier" = 1 ] || notes+=("no carrier")
    [ "$speed" = "$FABRIC_SPEED" ] || notes+=("speed $speed Mb/s, want $FABRIC_SPEED")
    [ "$mtu" = "$FABRIC_MTU" ] || notes+=("MTU $mtu, want $FABRIC_MTU")
    [ -n "$addr" ] || notes+=("no IPv4 address")
    [ "$state" = ACTIVE ] || notes+=("HCA ${hca:-?} ${state:-not found}")
    line="  rank $r $d (${hca:-?}): ${speed} Mb/s, MTU $mtu, carrier $carrier, ${addr:-no address}, HCA ${state:-?}"
    if [ ${#notes[@]} = 0 ]; then echo "$line: OK"; else echo "$line: WARN ($(IFS=';'; echo "${notes[*]}"))"; warn=1; fi
  done
  if kl=$(klog); then
    # the driver logs the PCIe slot power line for every port at its boot probe, seconds after boot; the same line
    # later than RE_PROBE_AFTER seconds of uptime (default 120) means a port was probed again at run time
    hits=$(printf '%s\n' "$kl" | grep -E 'PCIe slot power capability was not advertised|insufficient power on the PCIe slot' |
      awk -v t="${RE_PROBE_AFTER:-120}" '{ s = $0; sub(/^[^[]*\[ */, "", s); sub(/\].*/, "", s); if (s + 0 > t) n++ } END { print n + 0 }')
    if [ "$hits" -gt 0 ]; then
      echo "  rank $r WARN: the NIC was re-probed after boot ($hits kernel log lines: PCIe slot power), most likely"
      echo "    after a cable change. Transmit is capped at about 13 Gb/s per PCIe half until this host reboots."
      warn=1
    else
      echo "  rank $r NIC re-probe check: OK (no PCIe slot power message after the boot probe)"
    fi
  else
    echo "  rank $r NIC re-probe check: SKIPPED (cannot read the kernel log; add the user to the adm or"
    echo "    systemd-journal group, or check by hand: journalctl -k -b | grep 'PCIe slot power')"
  fi
  exit $warn;;
run)
  out=$1; secs=$2; shift 2
  server_env
  args=()
  for a in "$@"; do args+=("${a//\{R\}/$r}"); done
  mkdir -p "$(dirname "$out")"
  timeout "$secs" env "${envs[@]}" "${args[@]}" > "$out" 2>&1
  rc=$?
  echo "rank $r on $(hostname): exit $rc, log $out"
  exit $rc;;
env)
  [ -n "${ENV_LIBS:-}" ] && LIBS=$ENV_LIBS
  [ -n "${ENV_BIN_DIR:-}" ] && TP_FATBIN=$ENV_BIN_DIR/tp_mailbox.fatbin
  LD_LIBRARY_PATH=""
  server_env
  printf '%s\n' "${envs[@]}";;
addr)
  ip -4 -o address show dev "${FABRIC_DEVS[$1]}" | awk '{print $4}' | cut -d/ -f1 | head -1;;
bw-server)
  timeout $(( $3 + 20 )) ib_write_bw -d "${FABRIC_HCAS[$1]}" -x "${FABRIC_GID:-3}" -p "$2" -D "$3" -s 65536 -q 4 -F \
    --report_gbits > /dev/null 2>&1;;
bw-client)
  out=$(timeout $(( $3 + 20 )) ib_write_bw -d "${FABRIC_HCAS[$1]}" -x "${FABRIC_GID:-3}" -p "$2" -D "$3" -s 65536 -q 4 \
    -F --report_gbits "$4" 2>&1)
  bw=$(printf '%s\n' "$out" | awk '/^ *[0-9]+ +[0-9]+ +[0-9.]+ +[0-9.]+/{print $4}' | tail -1)
  echo "${bw:-FAIL}";;
*) echo "usage: rank.sh CONFIG start|stop|status|fabric|run|addr|env|bw-server|bw-client RANK [ARGS]" >&2; exit 2;;
esac
