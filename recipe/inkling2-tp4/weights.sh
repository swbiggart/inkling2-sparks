#!/usr/bin/env bash
# weights.sh CONFIG start PLAN_DIR [copy-rank.sh options]   start each rank's weight copy on its host
# weights.sh CONFIG status                                   each host's copy: running or ended, last log lines
#
# PLAN_DIR (on this machine) holds ranges-w4-r0.tsv .. ranges-w4-r3.tsv and inkling2-digests-w4.tsv (the release's
# plan files, or your own from tf-inkling2-plan; see weights/README.md). `start` copies them with
# tools/zig/inkling2_rank_copy.py and weights/ to ROOT/recipe/weights on every host and runs weights/copy-rank.sh
# there in the background (setsid nohup) into MODEL: seed (optional), fetch from the pinned Hugging Face revision,
# manifest, verify, check. Logs and pids in RUN/weights-R.{log,pid}. Options, passed to copy-rank.sh:
#   --seed-from DIR   first copy the runs a checked rank folder DIR on that host already holds (same path on all)
#   --server A:P      from a full checkpoint served by weights/checkpoint.py serve (run `checkpoint.py check` on it
#                     first) instead of Hugging Face; setup.sh's CHECKPOINT does both (and gives each rank its address)
#   --drafter         also the DSpark drafter's tensors (added by itself while DSPARK=1 in release-defaults.sh)
#   --folds BIN       then the DSpark folds file, built on the host by BIN (added by itself while DSPARK=1:
#                     ROOT/bin/tf-inkling2-fold; skipped with a note while the release is not installed there)
#   --jobs J          parallel range requests (default 8)
#   --check-gib G     size of the sample check --http reads again (default 3; 0: headers only; all: every run)
# RANK_COPY overrides the path of inkling2_rank_copy.py on this machine.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
conf=${1:?usage: weights.sh CONFIG start PLAN_DIR [options] | weights.sh CONFIG status}
cmd=${2:?usage: weights.sh CONFIG start PLAN_DIR [options] | weights.sh CONFIG status}
# shellcheck source=/dev/null
source "$here/release-defaults.sh"
# shellcheck source=/dev/null
source "$conf"
release_settings || exit 2
remote=$ROOT/recipe/weights
case $cmd in
start)
  plan=${3:?usage: weights.sh CONFIG start PLAN_DIR [options]}
  shift 3
  tool=${RANK_COPY:-}
  if [ -z "$tool" ]; then
    for c in "$here/../../tools/zig/inkling2_rank_copy.py" "$here/../tools/inkling2_rank_copy.py"; do
      [ -f "$c" ] && { tool=$c; break; }
    done
  fi
  [ -f "$tool" ] || { echo "inkling2_rank_copy.py not found: set RANK_COPY" >&2; exit 2; }
  for f in ranges-w4-r0.tsv ranges-w4-r1.tsv ranges-w4-r2.tsv ranges-w4-r3.tsv inkling2-digests-w4.tsv; do
    [ -f "$plan/$f" ] || { echo "missing $plan/$f" >&2; exit 2; }
  done
  # DSPARK=1 (release-defaults.sh) needs the drafter's tensors in every rank folder
  case " $* " in *" --drafter "*) ;; *) [ "${DSPARK:-0}" = 1 ] && set -- "$@" --drafter;; esac
  case " $* " in *" --folds "*) ;; *) [ "${DSPARK:-0}" = 1 ] && set -- "$@" --folds "$ROOT/bin/tf-inkling2-fold";; esac
  opts=""; [ $# -gt 0 ] && opts=$(printf '%q ' "$@")
  for r in 0 1 2 3; do
    h=${HOSTS[$r]}
    if ssh "$h" "test -f '$RUN/weights-$r.pid' && kill -0 \$(cat '$RUN/weights-$r.pid') 2>/dev/null"; then
      echo "rank $r: a copy is already running on $h" >&2; continue
    fi
    ssh "$h" "mkdir -p '$remote' '$RUN'" &&
      scp -q "$tool" "$here/weights/copy-rank.sh" "$here/weights/fold-rank.sh" "$here/weights/seed_rank_folder.py" "$here/weights/netrun.py" "$plan/ranges-w4-r$r.tsv" \
        "$plan/inkling2-digests-w4.tsv" "$h:$remote/" || { echo "copy to $h failed" >&2; exit 1; }
    # cd on its own: a backgrounded `cd && cmd` list would keep ssh's output open (and ssh waiting) until the copy ends
    ssh "$h" "cd '$remote' || exit 1; setsid nohup bash copy-rank.sh $r '$MODEL' $opts > '$RUN/weights-$r.log' 2>&1 < /dev/null & echo \$! > '$RUN/weights-$r.pid'; echo \"rank $r on \$(hostname): copy pid \$(cat '$RUN/weights-$r.pid'), log $RUN/weights-$r.log\""
  done;;
status)
  for r in 0 1 2 3; do
    ssh "${HOSTS[$r]}" "if test -f '$RUN/weights-$r.pid' && kill -0 \$(cat '$RUN/weights-$r.pid') 2>/dev/null; then s=running; else s=ended; fi; echo \"rank $r on \$(hostname): copy \$s\"; tail -n ${LINES_SHOWN:-4} '$RUN/weights-$r.log' 2>/dev/null | cut -c1-200"
  done;;
*) echo "usage: weights.sh CONFIG start PLAN_DIR [options] | weights.sh CONFIG status" >&2; exit 2;;
esac
