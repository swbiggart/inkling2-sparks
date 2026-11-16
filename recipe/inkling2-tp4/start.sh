#!/usr/bin/env bash
# start.sh CONFIG: start Inkling-2-Small at TP4 on the four hosts in CONFIG (see config.example.sh): rank 0 first
# (it serves HTTP and holds the NCCL bootstrap), then ranks 1, 2, 3 as followers; then wait until rank 0 answers
# /v1/models and every follower has loaded (READY_WAIT seconds, default 900), or a rank ends; then, with XROWS=1,
# one warm-up prompt (WARMUP=0 skips it). With HTTP_HOST not loopback and no API key configured it warns first (it
# still starts).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
conf=${1:?usage: start.sh CONFIG}
# shellcheck source=/dev/null
source "$here/release-defaults.sh"
# shellcheck source=/dev/null
source "$conf"
release_settings || exit 2
http_open_warning # HTTP_HOST not loopback and no API key: a warning, not a refusal
remote=$ROOT/recipe
for r in 0 1 2 3; do
  h=${HOSTS[$r]}
  ssh "$h" "mkdir -p '$remote' '$RUN'" && scp -q "$here/rank.sh" "$here/release-defaults.sh" "$h:$remote/" && scp -q "$conf" "$h:$remote/config.sh" || {
    echo "copy to $h failed" >&2; exit 1; }
done
for r in 0 1 2 3; do
  ssh "${HOSTS[$r]}" "bash '$remote/rank.sh' '$remote/config.sh' start $r" || { echo "rank $r did not start; stop.sh $conf" >&2; exit 1; }
done
probe=$HTTP_HOST
[ "$probe" = 0.0.0.0 ] && probe=127.0.0.1
start=$SECONDS
while :; do
  # rank 0 answers HTTP once its own load is done; a request then waits for the followers, so wait for them too
  # (401: the server was started with an API key, EXTRA_ARGS --api-key / --api-key-file, and is up)
  code=$(ssh "${HOSTS[0]}" "curl -s -o /dev/null -w '%{http_code}' -m 5 http://$probe:$HTTP_PORT/v1/models")
  if [ "$code" = 200 ] || [ "$code" = 401 ]; then
    following=0
    for r in 1 2 3; do ssh "${HOSTS[$r]}" "grep -q 'following rank 0' '$RUN/rank-$r.log'" && following=$((following + 1)); done
    if [ $following = 3 ]; then echo "ready after $((SECONDS - start)) s: http://$probe:$HTTP_PORT on ${HOSTS[0]}"; break; fi
  fi
  for r in 0 1 2 3; do
    if ! ssh "${HOSTS[$r]}" "test -f '$RUN/rank-$r.pid' && kill -0 \$(cat '$RUN/rank-$r.pid') 2>/dev/null"; then
      echo "rank $r ended before the server was ready:" >&2
      ssh "${HOSTS[$r]}" "tail -20 '$RUN/rank-$r.log'" >&2
      exit 1
    fi
  done
  if (( SECONDS - start > ${READY_WAIT:-900} )); then echo "not ready after ${READY_WAIT:-900} s" >&2; exit 1; fi
  sleep 10
done
bash "$here/warmup.sh" "$conf"
for r in 0 1 2 3; do ssh "${HOSTS[$r]}" "LINES_SHOWN=0 bash '$remote/rank.sh' '$remote/config.sh' status $r"; done
