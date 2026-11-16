# setup/lib.sh: setup.sh's release unpacking, server-config writer and weights copy jobs, sourced by setup.sh (and by
# setup/test-lib.sh with fixtures). They use setup.sh's globals: die, sha, cache; the config writer its settings (see
# serve_config); the copy jobs rq, q, HOSTS, ROOT, RUN, MODEL_DIR.

# valid_root NAME: a release tarball's one top folder: tensorfold-inkling2-tp4-*, a plain name.
valid_root() {
  case $1 in tensorfold-inkling2-tp4-?*) ;; *) return 1;; esac
  [[ $1 =~ ^[A-Za-z0-9._+-]+$ ]]
}

# unpack_release TGZ SHA256: sets rel_dir to the release folder of TGZ (its bytes already checked against SHA256),
# unpacked once into the cache under its checksum: ${cache}/rel-SHA256/ROOT. The listing is checked before anything
# is written: exactly one top folder ROOT (valid_root), every member a regular file or a folder under it with a plain
# relative path (no absolute path, no `.` or `..` component, no unusual characters), ROOT/SHA256SUMS among them. It is
# unpacked into a fresh temporary folder made by mktemp in the cache, checked again there (one entry ROOT, only files
# and folders, every file of its SHA256SUMS), marked complete and renamed into place in one step. The only folder ever
# removed is that temporary one: no path is derived from a member name before the checks, and nothing else in the
# cache is touched.
unpack_release() {
  local tgz=$1 got=$2 final names types n root bad t
  [[ ${got} =~ ^[0-9a-f]{64}$ ]] || die "release: bad sha256 '${got}'"
  final=${cache}/rel-${got}
  if [ -e "${final}" ]; then
    root=$(cat "${final}/.complete" 2> /dev/null)
    if valid_root "${root}" && [ -d "${final}/${root}" ]; then rel_dir=${final}/${root}; return 0; fi
    die "release: ${final} is not a complete unpacked release (remove it and rerun)"
  fi
  # -P: the member names as stored (GNU tar would otherwise list them with a leading / or ../ already removed)
  names=$(tar -tPzf "${tgz}" 2> /dev/null) || die "release: cannot list $(basename "${tgz}")"
  types=$(tar -tvPzf "${tgz}" 2> /dev/null | cut -c1) || die "release: cannot list $(basename "${tgz}")"
  [ -n "${names}" ] || die "release: $(basename "${tgz}") is empty (no release folder in it)"
  n=$(printf '%s\n' "${names}" | wc -l | tr -d ' ')
  [ "$(printf '%s\n' "${types}" | wc -l | tr -d ' ')" = "${n}" ] || die "release: $(basename "${tgz}"): its listings disagree"
  if printf '%s\n' "${types}" | grep -qv '^[-d]$'; then
    die "release: $(basename "${tgz}") holds a member that is not a file or a folder (a link, device or other)"
  fi
  bad=$(printf '%s\n' "${names}" | grep -Ev '^[A-Za-z0-9._+@=,-]+(/[A-Za-z0-9._+@=, -]+)*/?$' | head -3)
  [ -z "${bad}" ] || die "release: $(basename "${tgz}"): unexpected member path(s): $(printf '%s ' ${bad})"
  bad=$(printf '%s\n' "${names}" | grep -E '(^|/)\.\.?(/|$)' | head -3)
  [ -z "${bad}" ] || die "release: $(basename "${tgz}"): a member path with . or ..: $(printf '%s ' ${bad})"
  root=$(printf '%s\n' "${names}" | cut -d/ -f1 | LC_ALL=C sort -u)
  [ "$(printf '%s\n' "${root}" | wc -l | tr -d ' ')" = 1 ] || die "release: $(basename "${tgz}") has more than one top folder: $(printf '%s ' ${root})"
  valid_root "${root}" || die "release: $(basename "${tgz}")'s top folder '${root}' is not tensorfold-inkling2-tp4-*"
  printf '%s\n' "${names}" | grep -qx "${root}/SHA256SUMS" || die "release: $(basename "${tgz}") has no ${root}/SHA256SUMS"
  mkdir -p "${cache}" || die "release: cannot create ${cache}"
  t=$(mktemp -d "${cache}/.unpack.XXXXXXXX") || die "release: cannot make a folder in ${cache}"
  if ! { tar -xzf "${tgz}" -C "${t}" --no-same-owner && [ "$(ls -A "${t}")" = "${root}" ] &&
         [ -z "$(find "${t}" ! -type f ! -type d | head -1)" ] &&
         (cd "${t}/${root}" && sha -c --quiet SHA256SUMS > /dev/null 2>&1) &&
         printf '%s\n' "${root}" > "${t}/.complete"; }; then
    rm -rf "${t}"
    die "release: $(basename "${tgz}") did not unpack to one folder whose files match its SHA256SUMS"
  fi
  if ! mv "${t}" "${final}" 2> /dev/null || [ -e "${final}/$(basename "${t}")" ]; then
    # another setup.sh published the same release first (or the rename failed): ours is the temporary copy
    rm -rf "${t}" "${final:?}/$(basename "${t}")"
  fi
  [ -f "${final}/.complete" ] && [ -d "${final}/${root}" ] || die "release: cannot publish ${final}"
  rel_dir=${final}/${root}
}

# serve_config MASTER: the server config (the format of config.example.sh) on stdout. Every value is written as a
# shell word (printf %q; array entries one by one), so sourcing it sets exactly the values setup.sh holds, whatever
# spaces or shell characters a path has. Reads conf, short, container, HOSTS, NCCL_PORT, SOCKET_IFNAME, ROOT, MODEL,
# HTTP_HOST, HTTP_PORT, EXTRA_ARGS, REL_VARS (with each def_VAR), FABRIC_*, NCCL_ENV, OPT_ENV.
serve_config() {
  local m=$1 v x name
  name=$(basename "${conf}" | tr -cd '[:print:]')
  echo "# Written by setup.sh from ${name} at $(date -u +%FT%TZ); release commit $(printf '%s' "${short}" | tr -cd '[:alnum:]'). setup.sh rewrites it,"
  echo "# so change ${name} and rerun \`setup.sh ${name} config\`. Format: config.example.sh."
  printf 'HOSTS=('; for x in "${HOSTS[@]}"; do printf ' %q' "${x}"; done; printf ' )\n'
  echo "# NCCL's bootstrap (NCCL_COMM_ID) and the liveness lines (NCCL_PORT + 1) at rank 0's fabric address, and NCCL's"
  echo "# sockets on the fabric port: a firewall on the LAN cannot drop them."
  printf '%s=%q\n' MASTER "${m}" NCCL_PORT "${NCCL_PORT}" SOCKET_IFNAME "${SOCKET_IFNAME}" ROOT "${ROOT}"
  echo 'BIN="$ROOT/bin/tensorfold-native"'; echo 'DECODE_BIN="$ROOT/bin/tf-inkling2-decode"'
  printf '%s=%q\n' MODEL "${MODEL}"
  echo 'LIBS="$ROOT/lib"'; echo 'PLAN="$ROOT/plan"'; echo 'DATA="$ROOT/data"'; echo 'GRAPH=""'
  echo 'RUN="$ROOT/serve"'
  printf '%s=%q\n' HTTP_HOST "${HTTP_HOST}" HTTP_PORT "${HTTP_PORT}" EXTRA_ARGS "${EXTRA_ARGS}"
  echo "# The release settings are release-defaults.sh's (rank.sh reads it first); only HOSTS_CONF's overrides follow."
  for v in "${REL_VARS[@]}"; do eval "[ \"\${${v}}\" = \"\${def_${v}}\" ]" || printf '%s=%q\n' "${v}" "${!v}"; done
  printf 'FABRIC_DEVS=('; for x in "${FABRIC_DEVS[@]}"; do printf ' %q' "${x}"; done; printf ' )\n'
  printf 'FABRIC_HCAS=('; for x in "${FABRIC_HCAS[@]}"; do printf ' %q' "${x}"; done; printf ' )\n'
  printf '%s=%q\n' FABRIC_SPEED "${FABRIC_SPEED}" FABRIC_MTU "${FABRIC_MTU}" FABRIC_GID "${FABRIC_GID}"
  echo "NCCL_ENV=("; for x in "${NCCL_ENV[@]}"; do printf '  %q\n' "${x}"; done; echo ")"
  echo "# Extra environment (OPT_ENV in HOSTS_CONF), the same on all four ranks; it comes after the release's settings."
  echo "EXTRA_ENV=("
  for x in ${OPT_ENV[@]+"${OPT_ENV[@]}"}; do printf '  %q\n' "${x}"; done
  echo ")"
  printf '%s=%q\n' IMAGE "inkling2-sparks:${short}" CONTAINER "${container}"
}

# ---------------------------------------------------------------- weights copy jobs (setup.sh's weights step)
# Each rank's copy runs detached on its host (setsid nohup copy-rank.sh), with RUN/weights-R.pid, RUN/weights-R.log
# and, when it copies from a CHECKPOINT server, RUN/weights-R.src (the server's ADDR:PORT). That server lives only as
# long as the setup.sh run that started it: on any exit setup.sh stops and joins the copies it started against it
# (wjob_stop) before it stops the server, and a later run never waits for a copy whose server is gone: it stops and
# relaunches it (wjob_next says restart). A Hugging Face copy needs nothing from setup.sh and is waited for.

# wjob_state R: running | done | failed | none (R's copy process, else its log's last line)
wjob_state() {
  rq "${HOSTS[$1]}" "p=$(q "${RUN}/weights-$1.pid"); l=$(q "${RUN}/weights-$1.log");
    if [ -f \"\$p\" ] && kill -0 \$(cat \"\$p\") 2>/dev/null; then echo running;
    elif tail -n 1 \"\$l\" 2>/dev/null | grep -q '^== done'; then echo done;
    elif [ -f \"\$l\" ]; then echo failed; else echo none; fi"
}
# wjob_src R: the checkpoint server R's copy was started against (empty: Hugging Face, or no record)
wjob_src() { rq "${HOSTS[$1]}" "cat $(q "${RUN}/weights-$1.src") 2>/dev/null"; }
# wjob_next R STATE STAMPED: done (stamped and not running) | wait (a Hugging Face copy still running) | restart (a
# copy still running from a CHECKPOINT server of an earlier run: that server is gone) | new (start a copy)
wjob_next() {
  if [ "$3" = 1 ] && [ "$2" != running ]; then echo done
  elif [ "$2" = running ] && [ -n "$(wjob_src "$1")" ]; then echo restart
  elif [ "$2" = running ]; then echo wait
  else echo new; fi
}
# wjob_launch R SRC OPTS: start R's copy into MODEL_DIR (SRC empty: Hugging Face, else the CHECKPOINT server
# ADDR:PORT; OPTS the copy-rank.sh options, already quoted) and record its pid, log and source
wjob_launch() {
  # (cd first, on its own: a backgrounded `cd && cmd` list would hold ssh's output open until the copy ends)
  rq "${HOSTS[$1]}" "cd $(q "${ROOT}/recipe/weights") || exit 1; printf '%s\n' $(q "$2") > $(q "${RUN}/weights-$1.src") || exit 1;
    setsid nohup bash copy-rank.sh $1 $(q "${MODEL_DIR}") ${2:+--server $(q "$2")} $3 \
    > $(q "${RUN}/weights-$1.log") 2>&1 < /dev/null & echo \$! > $(q "${RUN}/weights-$1.pid")"
}
# wjob_stop R: stop R's copy (its process group: copy-rank.sh and the copy tool) if it runs, wait until it has
# ended (TERM, then KILL after 30 s), and mark its log, so the next run starts it again (the copy resumes from the
# runs it recorded). Only a process whose command line names copy-rank.sh is signalled.
wjob_stop() {
  rq "${HOSTS[$1]}" "p=$(q "${RUN}/weights-$1.pid"); l=$(q "${RUN}/weights-$1.log");
    [ -f \"\$p\" ] || exit 0; pid=\$(cat \"\$p\"); case \$pid in ''|*[!0-9]*) exit 0;; esac
    kill -0 \$pid 2>/dev/null || exit 0
    ps -o args= -p \$pid 2>/dev/null | grep -q copy-rank.sh || exit 0
    kill -TERM -- -\$pid 2>/dev/null || kill -TERM \$pid 2>/dev/null
    for i in \$(seq 1 60); do kill -0 \$pid 2>/dev/null || break; sleep 0.5; done
    if kill -0 \$pid 2>/dev/null; then kill -KILL -- -\$pid 2>/dev/null || kill -KILL \$pid 2>/dev/null; sleep 1; fi
    kill -0 \$pid 2>/dev/null && exit 1
    echo '== stopped by setup.sh (its checkpoint server ends with it; a rerun starts this copy again)' >> \"\$l\"; exit 0"
}
# wjob_stop_all R...: wjob_stop each R (setup.sh's EXIT trap, before the checkpoint server stops), saying what it did
wjob_stop_all() {
  local r
  for r in "$@"; do
    if wjob_stop "${r}" > /dev/null 2>&1; then echo "setup: rank ${r}'s copy stopped (its checkpoint server ends with this run; a rerun starts it again)" >&2
    else echo "setup: rank ${r}'s copy on ${HOSTS[$r]} could not be stopped; a rerun stops and restarts it" >&2; fi
  done
}
