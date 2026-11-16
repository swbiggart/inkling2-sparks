#!/usr/bin/env bash
# bench.sh CONFIG COMMAND [ARGS...]: the speed sweep, the prompt set, the needle tests and the speed ledger
# (tools/inkling2_bench.py), and a GPU-free install check. Process ideas from 0xSero's inkling2-exl3 (MIT).
#   check [--sums]       GPU-free, the server may run or not: the config's values; on every host the binaries, the
#                        kernel image (tp_mailbox.fatbin), the libraries and that the binaries resolve against them
#                        (ldd), the plan files, the rank folder (config, shards, digests, the DSpark folds when
#                        DSPARK=1) and Python 3.9+ on rank 0's host; --sums also checks every file of the release's
#                        SHA256SUMS (bin/ and lib/ are checked always). Exit 1 if anything failed; a host whose ssh
#                        fails, or whose check ends before its last line, counts as failed.
#   sweep [ARGS]         against the running server: decode at 1/2/4/8 streams within MAX_STREAMS (DSpark and
#                        plain), the README's one-stream workloads, cold prompts of 2k and 8k tokens; medians of 3,
#                        JSON out (~6 minutes with the defaults)
#   prompts [ARGS]       the 16-row prompt set at temperature 0 and 0.7 (seeded), 2,048 tokens a row (~25 minutes)
#   needle [ARGS]        8 concurrent sessions with a passphrase each (needs MAX_STREAMS=8 and a CONTEXT that fits
#                        them, e.g. 32768, to run all 8 at once); --sessions 1 --tokens 120000 for one deep needle
#   ledger SWEEP.json [ARGS]   (no server, runs here) bytes a step, the bandwidth floor and % of it per row
# sweep, prompts and needle copy the tool to rank 0's host (BENCH_DIR, default RUN/bench), run it there against
# http://127.0.0.1:HTTP_PORT (HTTP_HOST when it is not loopback) and copy the JSON back to BENCH_OUT (default
# ./bench-results). ARGS go to tools/inkling2_bench.py (its --help lists them).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
usage="usage: bench.sh CONFIG check|sweep|prompts|needle|ledger [ARGS...]"
conf=${1:?${usage}}
cmd=${2:?${usage}}
shift 2
# shellcheck source=/dev/null
source "$here/release-defaults.sh"
# shellcheck source=/dev/null
source "$conf"
release_settings || exit 2
tools=$here/../../tools
[ -f "$tools/inkling2_bench.py" ] || tools=$here/../tools
tool=$tools/inkling2_bench.py
[ -f "$tool" ] || { echo "no tools/inkling2_bench.py next to the recipe" >&2; exit 2; }

if [ "$cmd" = ledger ]; then exec python3 "$tool" ledger "$@"; fi

if [ "$cmd" = check ]; then
  sums=0; [ "${1:-}" = --sums ] && sums=1
  fails=0
  ok() { printf '  %-5s %s\n' OK "$1"; }
  bad() { printf '  %-5s %s\n' FAIL "$1"; fails=$((fails + 1)); }
  echo "config $conf:"
  [ "${#HOSTS[@]}" = 4 ] && ok "HOSTS: 4 hosts (${HOSTS[*]})" || bad "HOSTS: ${#HOSTS[@]} hosts, TP4 needs 4"
  case ${HTTP_PORT:-} in ''|*[!0-9]*) bad "HTTP_PORT=${HTTP_PORT:-}: a port number";; *) ok "HTTP ${HTTP_HOST:-127.0.0.1}:${HTTP_PORT}";; esac
  case ${DSPARK} in 0|1) ok "DSPARK=${DSPARK}";; *) bad "DSPARK=${DSPARK}: 0 or 1";; esac
  case ${PARALLEL} in auto|[1-8]) ok "CONTEXT=${CONTEXT}, PARALLEL=${PARALLEL}, MAX_STREAMS=${MAX_STREAMS}";;
    *) bad "PARALLEL=${PARALLEL}: auto or 1-8";; esac
  for v in ROOT BIN DECODE_BIN MODEL LIBS RUN; do [ -n "${!v:-}" ] && ok "$v=${!v}" || bad "$v is not set"; done
  [ "${ROCE_MULTI}" = 1 ] && [ "${#FABRIC_HCAS[@]}" -lt 1 ] && bad "ROCE_MULTI=1 needs FABRIC_HCAS"
  fatbin=${TP_FATBIN:-$(dirname "$BIN")/tp_mailbox.fatbin}
  q() { printf '%q ' "$@"; }
  for r in 0 1 2 3; do
    h=${HOSTS[$r]:-}
    [ -n "$h" ] || continue
    echo "rank $r ($h):"
    # one ssh a host; the remote half prints OK/FAIL lines, then an end line, and changes nothing
    out=$(ssh -o ConnectTimeout=10 "$h" "bash -s -- $(q "$r" "$ROOT" "$BIN" "$DECODE_BIN" "$fatbin" "$ROCE_MULTI" "$LIBS" "${PLAN:-$ROOT/plan}" "$MODEL" "$DSPARK" "$sums")" 2>&1 <<'REMOTE'
r=$1 root=$2 bin=$3 dec=$4 fat=$5 multi=$6 libs=$7 plan=$8 model=$9 ds=${10} sums=${11}
ok() { printf '  %-5s %s\n' OK "$1"; }
bad() { printf '  %-5s %s\n' FAIL "$1"; }
for b in "$bin" "$dec"; do [ -x "$b" ] && ok "$(basename "$b") ($(stat -c %s "$b") bytes)" || bad "$b: missing or not executable"; done
if [ "$multi" = 1 ]; then [ -s "$fat" ] && ok "kernel image $(basename "$fat")" || bad "$fat: missing (ROCE_MULTI=1 needs it)"; fi
for l in libnccl.so.2 libcublas.so.13 libcublasLt.so.13; do [ -e "$libs/$l" ] && ok "lib $l" || bad "$libs/$l: missing"; done
for b in "$bin" "$dec"; do
  [ -x "$b" ] || continue
  nf=$(LD_LIBRARY_PATH=$libs ldd "$b" 2>&1 | grep -c 'not found')
  [ "$nf" = 0 ] && ok "$(basename "$b") resolves its libraries" || bad "$(basename "$b"): $nf libraries not found: $(LD_LIBRARY_PATH=$libs ldd "$b" 2>&1 | grep 'not found' | awk '{print $1}' | tr '\n' ' ')"
done
[ -s "$plan/inkling2-digests-w4.tsv" ] && ok "plan digests" || bad "$plan/inkling2-digests-w4.tsv: missing"
if [ -d "$model" ]; then
  n=$(ls "$model" | grep -c '^model-.*\.safetensors$'); want=$(ls "$model" | grep -m1 -oE 'of-[0-9]+' | cut -c4- | sed 's/^0*//')
  [ -s "$model/config.json" ] && [ "$n" -gt 0 ] && [ "$n" = "${want:-0}" ] && ok "rank folder: $n shards" || bad "$model: config.json or shards missing ($n of ${want:-?})"
  ls "$model"/inkling2-digests-w4*.tsv > /dev/null 2>&1 && ok "rank folder digests" || bad "$model: no inkling2-digests-w4*.tsv"
  if [ "$ds" = 1 ]; then [ -s "$model/inkling2-dspark-fold.safetensors" ] && ok "DSpark folds" || bad "$model/inkling2-dspark-fold.safetensors: missing (DSPARK=1; setup.sh folds)"; fi
else bad "$model: no rank folder"; fi
s=$root/SHA256SUMS; [ -f "$s" ] || s=$root/.setup/assets.SHA256SUMS
if [ -f "$s" ]; then
  if [ "$sums" = 1 ]; then pat='.'; what=every; else pat=' [*]?(\./)?(bin|lib)/'; what='bin/ and lib/'; fi
  lines=$(grep -E "$pat" "$s" | grep -vE '\.md$')
  if [ -z "$lines" ]; then t="no matching lines"; rc=1; else t=$(printf '%s\n' "$lines" | (cd "$root" && sha256sum -c --quiet 2>&1)); rc=$?; fi
  [ $rc = 0 ] && ok "$what files match $(basename "$s")" || bad "files differ from $(basename "$s"): $(echo "$t" | head -3 | tr '\n' ' ')"
else bad "no SHA256SUMS in $root"; fi
if [ "$r" = 0 ]; then
  v=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)
  python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null && ok "python3 $v (bench clients)" || bad "python3 ${v:-missing}: the bench clients need 3.9+"
fi
echo "  END check rank $r"
REMOTE
    )
    rc=$?
    printf '%s\n' "$out" | grep -v '^  END check rank '
    fails=$((fails + $(printf '%s\n' "$out" | grep -c '^  FAIL')))
    if [ "$rc" != 0 ]; then bad "rank $r ($h): ssh exit $rc, the check did not run or did not finish"
    elif ! printf '%s\n' "$out" | grep -qx "  END check rank $r"; then bad "rank $r ($h): the check ended before its last line"; fi
  done
  [ "$fails" = 0 ] && echo "check: all OK" || echo "check: $fails failed"
  [ "$fails" = 0 ]; exit $?
fi

case $cmd in sweep|prompts|needle) ;; *) echo "$usage" >&2; exit 2;; esac
h=${HOSTS[0]}
dir=${BENCH_DIR:-$RUN/bench}
host=${HTTP_HOST:-127.0.0.1}
case $host in 0.0.0.0|::|localhost|127.*) host=127.0.0.1;; esac
case $host in *:*) host="[$host]";; esac
stamp=$(date -u +%Y%m%dT%H%M%SZ)
name=$cmd-$stamp.json
extra=()
if [ "$cmd" = sweep ]; then
  cap=$MAX_STREAMS; case $PARALLEL in [1-8]) cap=$PARALLEL;; esac
  extra=(--max-streams "$cap")
fi
ssh "$h" "mkdir -p '$dir'" && scp -q "$tool" "$tools/inkling2_prompts.json" "$tools/bench_prompts.json" "$h:$dir/" || {
  echo "copy to $h:$dir failed" >&2; exit 1; }
q() { printf '%q ' "$@"; }
ssh "$h" "cd '$dir' && python3 -I inkling2_bench.py $cmd http://$host:$HTTP_PORT $(q ${extra[@]+"${extra[@]}"} "$@") --out $name"
rc=$?
out=${BENCH_OUT:-./bench-results}
mkdir -p "$out" && scp -q "$h:$dir/$name" "$out/" && echo "results: $out/$name"
exit $rc
