#!/usr/bin/env bash
# stop.sh CONFIG: stop the TP4 server. Rank 0 first (SIGTERM; the followers end by themselves when it does), then
# any follower still running, each by the pid in its pid file.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
conf=${1:?usage: stop.sh CONFIG}
# shellcheck source=/dev/null
source "$here/release-defaults.sh"
# shellcheck source=/dev/null
source "$conf"
release_settings || exit 2
remote=$ROOT/recipe
for r in 0 1 2 3; do
  ssh "${HOSTS[$r]}" "STOP_WAIT=${STOP_WAIT:-60} bash '$remote/rank.sh' '$remote/config.sh' stop $r"
done
