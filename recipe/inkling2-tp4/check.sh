#!/usr/bin/env bash
# check.sh CONFIG IDS [PROMPT [STEPS]] [-- MORE ARGS]: the decode check at W=4 on the four hosts (the server must be
# stopped: it needs the GPUs' memory). Every rank runs DECODE_BIN with the server's environment and settings
# (NCCL, TF_INK2_MODE, the digest manifest): it runs the prompt's first PROMPT tokens (default 3040) through the
# whole-prompt forward, then decodes STEPS more (default 80) one token at a time, and checks every decode step's
# logits against the same row of the whole-prompt forward, bit for bit, across all 48 layers.
# IDS is a safetensors file of prompt token ids (release asset data/long_4000.safetensors): a file on this machine
# is copied to ROOT/data/ on every host; otherwise IDS is taken as a path on the hosts (the same on all four).
# Logs in RUN/check-decode-R.log; rank 0's verdict is printed. Exit 0 on PASS.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
conf=${1:?usage: check.sh CONFIG IDS [PROMPT [STEPS]] [-- MORE ARGS]}
ids=${2:?usage: check.sh CONFIG IDS [PROMPT [STEPS]] [-- MORE ARGS]}
shift 2
prompt=3040; steps=80
if [ $# -gt 0 ] && [ "$1" != -- ]; then prompt=$1; shift; fi
if [ $# -gt 0 ] && [ "$1" != -- ]; then steps=$1; shift; fi
[ "${1:-}" = -- ] && shift
more=("$@")
# shellcheck source=/dev/null
source "$here/release-defaults.sh"
# shellcheck source=/dev/null
source "$conf"
release_settings || exit 2
remote=$ROOT/recipe
local_ids=""
# (a file already at ROOT/data/ is taken as the hosts' copy: when this machine is one of the hosts, copying it onto
# itself would truncate it)
if [ -f "$ids" ] && [ "$ids" != "$ROOT/data/$(basename "$ids")" ]; then local_ids=$ids; ids=$ROOT/data/$(basename "$ids"); fi
for r in 0 1 2 3; do
  h=${HOSTS[$r]}
  if ssh "$h" "test -f '$RUN/rank-$r.pid' && kill -0 \$(cat '$RUN/rank-$r.pid') 2>/dev/null"; then
    echo "rank $r is serving on $h: stop.sh $conf first" >&2; exit 1
  fi
  ssh "$h" "mkdir -p '$remote' '$RUN'" && scp -q "$here/rank.sh" "$here/release-defaults.sh" "$h:$remote/" && scp -q "$conf" "$h:$remote/config.sh" || {
    echo "copy to $h failed" >&2; exit 1; }
  if [ -n "$local_ids" ]; then
    ssh "$h" "mkdir -p '$ROOT/data'" && scp -q "$local_ids" "$h:$ids" || { echo "copy of $local_ids to $h failed" >&2; exit 1; }
  fi
done
q() { printf '%q ' "$@"; }
cmd=$(q "$DECODE_BIN" "$MODEL" --ids "$ids" --prompt "$prompt" --steps "$steps" --world 4 --rank '{R}' ${more[@]+"${more[@]}"})
start=$SECONDS
pids=()
for r in 0 1 2 3; do
  ssh -n "${HOSTS[$r]}" "bash '$remote/rank.sh' '$remote/config.sh' run $r '$RUN/check-decode-$r.log' ${CHECK_SECS:-1800} $cmd" &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "decode check: all ranks done in $((SECONDS - start)) s"
ssh "${HOSTS[0]}" "grep -hE 'PASS|FAIL' '$RUN/check-decode-0.log' | tail -3"
if [ $rc != 0 ]; then
  echo "a rank failed; last lines of each log:" >&2
  for r in 0 1 2 3; do ssh "${HOSTS[$r]}" "tail -5 '$RUN/check-decode-$r.log'" >&2; done
  exit 1
fi
ssh "${HOSTS[0]}" "grep -q PASS '$RUN/check-decode-0.log'"
