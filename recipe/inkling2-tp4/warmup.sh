#!/usr/bin/env bash
# warmup.sh CONFIG: with XROWS=1 (release-defaults.sh) and WARMUP=1 (the default), one prompt of about 2,300 tokens
# (a 2,048-row chunk and a shorter one, 1 reply token) to the running server. The by-rows layer sums' first use costs
# a first long prompt about 0.4-0.8 s; after this a user's first prompt runs near the steady rate. About 2 s, once.
# start.sh and setup.sh --container start run it once the server is ready. WARMUP=0 skips it.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
conf=${1:?usage: warmup.sh CONFIG}
# shellcheck source=/dev/null
source "$here/release-defaults.sh"
# shellcheck source=/dev/null
source "$conf"
release_settings || exit 2
[ "${XROWS:-0}" = 1 ] && [ "${WARMUP:-1}" = 1 ] || exit 0
probe=$HTTP_HOST
[ "$probe" = 0.0.0.0 ] && probe=127.0.0.1
start=$SECONDS
body='{"model":"model","max_tokens":1,"temperature":0,"prompt":"'$(printf 'The quick brown fox jumps over the lazy dog. %.0s' $(seq 1 230))'"}'
if ssh "${HOSTS[0]}" "curl -sf -m 300 -H 'Content-Type: application/json' -d '$body' http://$probe:$HTTP_PORT/v1/completions > /dev/null"; then
  echo "warm-up: one prompt of about 2,300 tokens in $((SECONDS - start)) s"
else
  echo "warm-up request failed (the server is up; its first long prompt may take about a second longer)" >&2
fi
