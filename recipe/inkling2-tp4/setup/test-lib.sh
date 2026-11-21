#!/usr/bin/env bash
# setup/test-lib.sh: fixture tests of setup/lib.sh on this machine (no hosts, no network; python3 builds the crafted
# archives). Release unpacking: an empty archive, two top folders, a link member, a `..` member, an absolute member, a
# foreign top folder, a top folder alone, a SHA256SUMS mismatch (each refused, nothing published, no temporary folder
# left), then a good release and its rerun; a sentinel cache entry beside them must survive every case. Server config:
# ROOT, MODEL, hosts, overrides and arrays with spaces and shell characters, written by serve_config, sourced in a
# clean shell, every value equal and no command in them run. Weights copy jobs (setup/lib.sh wjob_*): a failure after
# one rank's copy started from a CHECKPOINT server stops and joins it before the server ends, and a rerun restarts a
# copy whose server is gone. bench.sh check with a stand-in ssh: a refused, denied, dropped or cut-short host counts as
# failed. In a repository, tools/tests/test_inkling2_bench.py (incomplete replies fail the bench tool's verdicts). Then
# weights/test-checkpoint.sh (the checkpoint drop-in: checkpoint.py check/serve and copy-rank.sh --server on
# 127.0.0.1), and weights/test-vision.sh (VISION's weights step: v0.2's commands while off). Prints PASS/FAIL lines; exits 0 only if all pass.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
w=$(mktemp -d "${TMPDIR:-/tmp}/inkling2-lib-test.XXXXXX") || exit 1
trap 'rm -rf "${w}"' EXIT
sha() { if command -v sha256sum > /dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
die() { echo "  die: $*" >&2; exit 1; }
# shellcheck source=lib.sh
source "${here}/lib.sh"
fails=0
ok() { echo "PASS $*"; }
no() { echo "FAIL $*"; fails=$((fails + 1)); }

# ---------------------------------------------------------------- unpack_release
cache=${w}/cache
mkdir -p "${cache}/other-release"
echo keep > "${cache}/other-release/sentinel"
# craft NAME SPEC...: w/NAME.tar.gz; SPEC is f:PATH:TEXT (a file), d:PATH (a folder), l:PATH:TARGET (a symlink),
# s:ROOT (ROOT/bin/x and a matching ROOT/SHA256SUMS), b:ROOT (ROOT/bin/x and a SHA256SUMS that does not match)
craft() {
  local name=$1; shift
  python3 -I - "${w}/${name}.tar.gz" "$@" << 'EOF'
import hashlib, io, sys, tarfile
out, specs = sys.argv[1], sys.argv[2:]
def add(t, path, data=None, kind="f", target=""):
    i = tarfile.TarInfo(path)
    i.mode = 0o755 if kind == "d" else 0o644
    if kind == "d":
        i.type = tarfile.DIRTYPE; t.addfile(i)
    elif kind == "l":
        i.type = tarfile.SYMTYPE; i.linkname = target; t.addfile(i)
    else:
        i.size = len(data); t.addfile(i, io.BytesIO(data))
with tarfile.open(out, "w:gz", format=tarfile.PAX_FORMAT) as t:
    for s in specs:
        k, rest = s[0], s[2:]
        if k == "f":
            p, text = rest.split(":", 1); add(t, p, text.encode())
        elif k == "d":
            add(t, rest, kind="d")
        elif k == "l":
            p, target = rest.split(":", 1); add(t, p, kind="l", target=target)
        elif k in "sb":
            body = b"#!/bin/sh\necho release\n"
            digest = hashlib.sha256(body if k == "s" else b"other").hexdigest()
            add(t, rest, kind="d"); add(t, rest + "/bin", kind="d"); add(t, rest + "/bin/x", body)
            add(t, rest + "/SHA256SUMS", f"{digest}  bin/x\n".encode())
EOF
}
R=tensorfold-inkling2-tp4-t1
craft empty
craft tworoots "s:${R}" "s:tensorfold-inkling2-tp4-t2"
craft link "s:${R}" "l:${R}/bin/y:/etc/passwd"
craft dotdot "s:${R}" "f:${R}/../escaped:x"
craft absolute "s:${R}" "f:/tmp/inkling2-lib-test-escaped:x"
craft foreign "s:other-root"
craft rootonly "d:${R}"
craft badsums "b:${R}"
craft good "s:${R}" "f:${R}/plan/ranges-w4-r0.tsv:a	b"

intact() { # nothing published, no temporary folder, the sentinel unchanged
  [ "$(cat "${cache}/other-release/sentinel" 2> /dev/null)" = keep ] &&
    [ -z "$(find "${cache}" -mindepth 1 -maxdepth 1 \( -name '.unpack.*' -o -name 'rel-*' \) | head -1)" ] &&
    [ ! -e /tmp/inkling2-lib-test-escaped ] && [ ! -e "${cache}/escaped" ]
}
for c in empty tworoots link dotdot absolute foreign rootonly badsums; do
  tgz=${w}/${c}.tar.gz
  out=$( (unpack_release "${tgz}" "$(sha "${tgz}" | awk '{print $1}')"; echo "rel_dir=${rel_dir}") 2>&1 ); rc=$?
  if [ ${rc} != 0 ] && intact; then ok "unpack ${c}: refused ($(printf '%s\n' "${out}" | grep 'die:' | head -1 | sed 's/^ *die: //' | cut -c1-120))"
  else no "unpack ${c}: rc ${rc}, output: ${out}"; fi
done
tgz=${w}/good.tar.gz; got=$(sha "${tgz}" | awk '{print $1}')
out=$( (unpack_release "${tgz}" "${got}" && echo "${rel_dir}") 2>&1 ); rc=$?
if [ ${rc} = 0 ] && [ "${out}" = "${cache}/rel-${got}/${R}" ] && [ -f "${out}/bin/x" ] && [ -f "${out}/plan/ranges-w4-r0.tsv" ] &&
   [ "$(cat "${cache}/rel-${got}/.complete")" = "${R}" ] &&
   [ "$(cat "${cache}/other-release/sentinel")" = keep ] && [ -z "$(find "${cache}" -maxdepth 1 -name '.unpack.*')" ]; then
  ok "unpack good: ${out#"${w}"/}"
else no "unpack good: rc ${rc}, output: ${out}"; fi
touch "${cache}/rel-${got}/${R}/marker"
out=$( (unpack_release "${tgz}" "${got}" && echo "${rel_dir}") 2>&1 ); rc=$?
if [ ${rc} = 0 ] && [ -f "${out}/marker" ] && [ "$(cat "${cache}/other-release/sentinel")" = keep ]; then ok "unpack good again: the cached folder reused"
else no "unpack good again: rc ${rc}, output: ${out}"; fi
out=$( (unpack_release "${tgz}" "../../etc"; echo "rel_dir=${rel_dir}") 2>&1 ); rc=$?
[ ${rc} != 0 ] && ok "unpack: a checksum that is not 64 hex digits refused" || no "unpack: bad checksum accepted: ${out}"

# ---------------------------------------------------------------- serve_config
weird='/srv/Inkling Models/$(touch PWNED1) `touch PWNED2` ;x&y|z *?[a] "dq" '"'sq'"' \back $HOME ~'
conf="${w}/my sparks.conf"; short=0123456789ab; container=0
HOSTS=(host-a 'host two' "host;\$(touch PWNED3)" host-d)
NCCL_PORT=29551; SOCKET_IFNAME=enp1s0f0np0; ROOT=${weird}; MODEL="${weird}/model split"
HTTP_HOST=127.0.0.1; HTTP_PORT=8890; EXTRA_ARGS='--x "a b" $(touch PWNED4)'
REL_VARS=(MODEL_DIRNAME CONTEXT DSPARK)
def_MODEL_DIRNAME=model-split; MODEL_DIRNAME='dir with space;$(touch PWNED5)'
def_CONTEXT=131072; CONTEXT=131072
def_DSPARK=1; DSPARK='1 && touch PWNED6'
FABRIC_DEVS=(enp1s0f0np0 'dev two'); FABRIC_HCAS=(rocep1s0f0 'hca`touch PWNED7`')
FABRIC_SPEED=200000; FABRIC_MTU=9000; FABRIC_GID=3
NCCL_ENV=(NCCL_IB_DISABLE=0 'NCCL_X=a b $(touch PWNED8)')
OPT_ENV=('TF_X=$(touch PWNED9)' 'TF_Y=two words')
cfg=${w}/serve.conf
serve_config '198.51.100.1 $(touch PWNED0)' > "${cfg}" || no "serve_config failed"
bash -n "${cfg}" && ok "config: bash -n" || no "config: bash -n"
mkdir -p "${w}/iso"
names=(HOSTS MASTER NCCL_PORT SOCKET_IFNAME ROOT MODEL HTTP_HOST HTTP_PORT EXTRA_ARGS MODEL_DIRNAME DSPARK FABRIC_DEVS FABRIC_HCAS FABRIC_SPEED FABRIC_MTU FABRIC_GID NCCL_ENV)
got=$(cd "${w}/iso" && env -i PATH="${PATH}" HOME=/nonexistent "${BASH}" --norc --noprofile -c \
  'source "$1" || exit 9; shift; declare -p "$@" EXTRA_ENV; printf "BIN=%s\nRUN=%s\nCONTEXT=%s\nIMAGE=%s\n" "$BIN" "$RUN" "${CONTEXT-unset}" "$IMAGE"' _ "${cfg}" "${names[@]}") || no "config: sourcing failed"
MASTER='198.51.100.1 $(touch PWNED0)'; EXTRA_ENV=("${OPT_ENV[@]}")
want=$(declare -p "${names[@]}" EXTRA_ENV; printf 'BIN=%s\nRUN=%s\nCONTEXT=%s\nIMAGE=%s\n' "${ROOT}/bin/tensorfold-native" "${ROOT}/serve" unset "inkling2-sparks:${short}")
if [ "${got}" = "${want}" ]; then ok "config: sourced in a clean shell, ${#names[@]} values + EXTRA_ENV, BIN, RUN equal; CONTEXT (= its default) not written"
else no "config: values differ"; diff <(printf '%s\n' "${want}") <(printf '%s\n' "${got}") | head -20; fi
if [ -z "$(find "${w}" "${PWD}" -maxdepth 3 -name 'PWNED*' 2> /dev/null | head -1)" ]; then ok "config: no command in a value ran"
else no "config: a command in a value ran"; fi

# ---------------------------------------------------------------- weights copy jobs (wjob_*, the CHECKPOINT path)
# Four stand-in hosts on this machine (rq runs the command here), a stand-in copy-rank.sh that copies "from" a server
# while its marker file w/jobs/srv/ADDR exists (and fails, "lost its source", when it goes) and ends "== done" when
# w/jobs/finish-R appears. Run 1 starts rank 0 against server A and then fails before rank 1 (setup.sh's EXIT trap
# order: wjob_stop_all, then the server ends): rank 0's copy must be stopped and joined before A goes. A copy left
# running against an earlier run's server must be restarted, a Hugging Face copy waited for. Run 2 then relaunches
# against server B and every copy ends done.
J=${w}/jobs
mkdir -p "${J}/bin" "${J}/srv" "${J}/root/recipe/weights" "${J}/run"
if ! command -v setsid > /dev/null; then
  printf '%s\n' '#!/usr/bin/env python3' 'import os, sys' 'os.setsid()' 'os.execvp(sys.argv[1], sys.argv[1:])' > "${J}/bin/setsid"
  chmod +x "${J}/bin/setsid"
fi
cat > "${J}/root/recipe/weights/copy-rank.sh" << EOF
#!/usr/bin/env bash
r=\$1 srv=""; shift 2
while [ \$# -gt 0 ]; do case \$1 in --server) srv=\$2; shift 2;; *) shift;; esac; done
echo "\$r \${srv:-hf}" >> "${J}/launches"
echo "== fetch from \${srv:-hf}"
for i in \$(seq 1 600); do
  if [ -n "\${srv}" ] && [ ! -e "${J}/srv/\${srv}" ]; then echo "lost its source \${srv}"; exit 2; fi
  [ -e "${J}/finish-\${r}" ] && { echo "== done"; exit 0; }
  sleep 0.1
done
echo "timed out"; exit 9
EOF
jobs_test() (
  PATH=${J}/bin:${PATH}
  HOSTS=(h0 h1 h2 h3); ROOT=${J}/root; RUN=${J}/run; MODEL_DIR=${J}/model
  rq() { bash -c "$2"; }
  q() { printf '%q' "$1"; }
  st_wait() { local i; for i in $(seq 1 100); do [ "$(wjob_state "$1")" = "$2" ] && return 0; sleep 0.1; done; return 1; }
  # run 1: server A up, rank 0 launched, then a failure before rank 1 (a subshell with setup.sh's trap order)
  touch "${J}/srv/source-a:4001"
  ( ck_jobs=()
    trap 'wjob_stop_all ${ck_jobs[@]+"${ck_jobs[@]}"} 2> /dev/null; rm -f "${J}/srv/source-a:4001"' EXIT
    ck_jobs+=(0); wjob_launch 0 source-a:4001 "--drafter" || exit 3
    st_wait 0 running || exit 4
    sleep 0.3; exit 1 ) # "die: cannot start the copy on h1"
  [ "$(wjob_state 0)" = failed ] && ok "jobs: a failure after rank 0's launch stops rank 0's copy (state failed)" || no "jobs: rank 0 after the failure: $(wjob_state 0)"
  grep -q '^lost its source' "${RUN}/weights-0.log" && no "jobs: rank 0's copy outlived its server (log: $(tail -2 "${RUN}/weights-0.log" | tr '\n' ' '))" ||
    ok "jobs: rank 0's copy was joined before its server stopped"
  tail -n 1 "${RUN}/weights-0.log" | grep -q '^== stopped by setup.sh' && ok "jobs: rank 0's log says it was stopped" || no "jobs: rank 0's log: $(tail -1 "${RUN}/weights-0.log")"
  # a copy an earlier run could not stop (still running, its server recorded) and a Hugging Face copy, both running
  touch "${J}/srv/source-0:4000"
  wjob_launch 1 source-0:4000 "" && st_wait 1 running || no "jobs: could not start rank 1's stand-in"
  wjob_launch 3 "" "" && st_wait 3 running || no "jobs: could not start rank 3's stand-in"
  # run 2 (the immediate rerun): what each rank needs, then relaunch against server B
  local r st plan="" new=()
  for r in 0 1 2 3; do
    st=$(wjob_state "${r}"); plan+="$(wjob_next "${r}" "${st}" 0) "
  done
  [ "${plan}" = "new restart new wait " ] && ok "jobs: rerun plan: failed -> new, earlier server's copy -> restart, none -> new, Hugging Face -> wait" ||
    no "jobs: rerun plan: ${plan}"
  wjob_stop 1 && [ "$(wjob_state 1)" = failed ] && ok "jobs: the earlier server's copy stopped" || no "jobs: rank 1 not stopped: $(wjob_state 1)"
  touch "${J}/srv/source-b:4002"
  for r in 0 1 2; do wjob_launch "${r}" source-b:4002 "--drafter" || no "jobs: relaunch of rank ${r} failed"; done
  for r in 0 1 2 3; do touch "${J}/finish-${r}"; done
  for r in 0 1 2 3; do st_wait "${r}" done || no "jobs: rank ${r} not done after the rerun: $(wjob_state "${r}") $(tail -1 "${RUN}/weights-${r}.log")"; done
  [ "$(grep -c 'source-b:4002$' "${J}/launches")" = 3 ] && [ "$(grep -c '^3 ' "${J}/launches")" = 1 ] &&
    ok "jobs: ranks 0-2 copied from the new server, rank 3's Hugging Face copy kept running" || no "jobs: launches: $(tr '\n' ',' < "${J}/launches")"
  [ "$(wjob_next 0 done 1)" = done ] && [ "$(wjob_next 0 done 0)" = new ] && ok "jobs: a stamped finished rank is done, an unstamped one copies again (resumes)" || no "jobs: done/stamp plan"
  for r in 0 1 2 3; do wjob_stop "${r}" > /dev/null 2>&1; done
)
jobs_out=$(jobs_test 2>&1)
printf '%s\n' "${jobs_out}"
fails=$((fails + $(printf '%s\n' "${jobs_out}" | grep -c '^FAIL')))
[ "$(printf '%s\n' "${jobs_out}" | grep -c '^PASS jobs')" = 7 ] || no "jobs: expected 7 PASS lines"

# ---------------------------------------------------------------- bench.sh check: a host whose ssh fails counts as failed
# A stand-in ssh on PATH: good* hosts answer one OK line and the end line, refused/denied fail as ssh does (exit 255),
# cut answers an OK line then drops the connection (exit 255), trunc answers an OK line and exits 0 without the end
# line, local runs bench.sh's real remote half here (it fails its file checks on this machine, but must end with the
# end line). bench.sh and release-defaults.sh run from a copy beside a stand-in tools/inkling2_bench.py.
B=${w}/bench
if [ -f "${here}/../bench.sh" ]; then
  mkdir -p "${B}/recipe" "${B}/tools" "${B}/bin"
  cp "${here}/../bench.sh" "${here}/../release-defaults.sh" "${B}/recipe/"
  echo '# stand-in' > "${B}/tools/inkling2_bench.py"
  cat > "${B}/bin/ssh" << 'EOF'
#!/usr/bin/env bash
while [ $# -gt 0 ]; do case $1 in -o) shift 2;; -*) shift;; *) break;; esac; done
h=$1; shift
r=$(printf '%s\n' "$*" | sed -nE 's/^bash -s -- ([0-9]).*/\1/p')
case ${h} in
  good*) cat > /dev/null; printf '  OK    stand-in\n  END check rank %s\n' "${r}";;
  refused) cat > /dev/null; echo "ssh: connect to host refused port 22: Connection refused" >&2; exit 255;;
  denied) cat > /dev/null; echo "${h}: Permission denied (publickey)." >&2; exit 255;;
  cut) cat > /dev/null; printf '  OK    stand-in\n'; echo "Connection to cut closed by remote host." >&2; exit 255;;
  trunc) cat > /dev/null; printf '  OK    stand-in\n';;
  local) exec bash -c "$*";;
esac
EOF
  chmod +x "${B}/bin/ssh"
  bcheck() { # HOSTS...: bench.sh check with those four hosts; sets bout, brc
    printf 'HOSTS=(%s)\nHTTP_PORT=8890\nROOT=/nonexistent/ink\nBIN=/nonexistent/ink/bin/tensorfold-native\nDECODE_BIN=/nonexistent/ink/bin/tf-inkling2-decode\nMODEL=/nonexistent/ink/model\nLIBS=/nonexistent/ink/lib\nRUN=/nonexistent/ink/run\nFABRIC_HCAS=(hca0)\nAPI_KEY=k\n' "$*" > "${B}/conf.sh"
    bout=$(PATH=${B}/bin:${PATH} bash "${B}/recipe/bench.sh" "${B}/conf.sh" check 2>&1); brc=$?
  }
  bcheck good0 good1 good2 good3
  [ ${brc} = 0 ] && [[ ${bout} == *"check: all OK"* ]] && [[ ${bout} != *"END check"* ]] && ok "bench check: four answering hosts: all OK, exit 0" ||
    no "bench check: four answering hosts: exit ${brc}: $(printf '%s\n' "${bout}" | tail -2 | tr '\n' ' ')"
  for c in "refused:1:ssh exit 255" "denied:2:ssh exit 255" "cut:3:ssh exit 255" "trunc:0:the check ended before its last line"; do
    hs=(good0 good1 good2 good3)
    n=${c#*:}; n=${n%%:*}; hs[${n}]=${c%%:*}
    bcheck "${hs[@]}"
    [ ${brc} = 1 ] && [[ ${bout} == *"check: 1 failed"* ]] && [[ ${bout} == *"rank ${n} (${c%%:*}): ${c##*:}"* ]] &&
      ok "bench check: ${c%%:*} on rank ${n} counts as a failed rank (exit 1)" ||
      no "bench check: ${c%%:*} on rank ${n}: exit ${brc}: $(printf '%s\n' "${bout}" | tail -3 | tr '\n' ' ')"
  done
  bcheck local good1 good2 good3
  [ ${brc} = 1 ] && [[ ${bout} != *"rank 0 (local): "* ]] && [[ ${bout} == *"  FAIL"* ]] &&
    ok "bench check: the real remote half ends with its end line (its own FAIL lines counted, no ssh/end failure)" ||
    no "bench check: real remote half: exit ${brc}: $(printf '%s\n' "${bout}" | grep -E 'rank 0|check:' | tr '\n' ' ')"
fi

# ---------------------------------------------------------------- bench.sh from the release tarball's layout (repository only)
# The paths scripts/release/build-assets.sh packs (--list), laid out from this checkout as the tarball holds them: bench.sh
# must find tools/inkling2_bench.py and the files it copies or reads beside it, and run the tool (ledger --help).
ba=${here}/../../../scripts/release/build-assets.sh
if [ -f "${ba}" ] && [ -f "${here}/../bench.sh" ]; then
  top=$(cd "${here}/../../.." && pwd)
  T=${w}/tarball/tensorfold-inkling2-tp4-test
  mkdir -p "${T}"
  paths=()
  while IFS= read -r p; do paths+=("${p}"); done < <(bash "${ba}" --list)
  (cd "${top}" && tar -cf - "${paths[@]}") | tar -x -C "${T}"
  printf 'HOSTS=(h0 h1 h2 h3)\nHTTP_PORT=8890\nROOT=/nonexistent/ink\nBIN=/nonexistent/ink/bin/tensorfold-native\nDECODE_BIN=/nonexistent/ink/bin/tf-inkling2-decode\nMODEL=/nonexistent/ink/model\nLIBS=/nonexistent/ink/lib\nRUN=/nonexistent/ink/run\nFABRIC_HCAS=(hca0)\nAPI_KEY=k\n' > "${w}/tarball/conf.sh"
  out=$(cd "${w}" && bash "${T}/recipe/inkling2-tp4/bench.sh" "${w}/tarball/conf.sh" ledger --help 2>&1); rc=$?
  miss=""
  for f in inkling2_bench.py inkling2_prompts.json bench_prompts.json inkling2_sheet.json; do
    [ -f "${T}/tools/${f}" ] || miss="${miss} ${f}"
  done
  [ ${rc} = 0 ] && [[ ${out} == *"usage: inkling2_bench.py ledger"* ]] && [ -z "${miss}" ] &&
    ok "bench.sh from the release tarball's layout: tools/ holds the bench tool and its files, ledger --help exit 0" ||
    no "bench.sh from the release tarball's layout: exit ${rc}, missing:${miss:- none}: $(printf '%s\n' "${out}" | tail -1)"
fi

# ---------------------------------------------------------------- tools/inkling2_bench.py: incomplete replies fail (repository only)
bt=${here}/../../../tools/tests/test_inkling2_bench.py
if [ -f "${bt}" ]; then
  out=$(python3 -I "${bt}" 2>&1); rc=$?
  [ ${rc} = 0 ] && ok "bench tool: $(printf '%s\n' "${out}" | grep -E '^Ran ' | head -1) (EOF before [DONE] fails every verdict, a missing sha is never equal)" ||
    no "bench tool: tools/tests/test_inkling2_bench.py exit ${rc}: $(printf '%s\n' "${out}" | grep -E '^(FAIL|ERROR)' | head -3 | tr '\n' ' ')"
fi

# ---------------------------------------------------------------- release-defaults.sh: DSpark at every context
# release_settings leaves DSPARK and the prompt cache as set at any CONTEXT up to 1048576 (DSpark fits beside a 1M
# stream); past it, refused. --vision by default (VISION=1), none with VISION=0. Each case in a clean shell.
rd=${here}/../release-defaults.sh
rel() { env -i PATH="${PATH}" "${BASH}" --norc --noprofile -c 'source "$1" || exit 9; shift; for a in "$@"; do eval "${a}"; done
  release_settings || exit 3; printf "%s %s %s\n" "${CONTEXT}" "${DSPARK}" "$(release_args)"' _ "${rd}" "$@" 2>&1; }
cases=(
  "131072 1 --prompt-cache-gib 1.5 --decode-share 0.5 --vision|"
  "524288 1 --prompt-cache-gib 1.5 --decode-share 0.5 --vision|CONTEXT=524288"
  "1048575 1 --prompt-cache-gib 1.5 --decode-share 0.5 --vision|CONTEXT=1048575"
  "1048576 1 --prompt-cache-gib 1.5 --decode-share 0.5 --vision|CONTEXT=1048576"
  "1048576 0 --prompt-cache-gib 1.5 --decode-share 0.5 --vision|CONTEXT=1048576 DSPARK=0"
  "1048576 1 --prompt-cache-gib 0.5 --decode-share 0.5 --vision|CONTEXT=1048576 PROMPT_CACHE_GIB=0.5"
  "131072 1 --prompt-cache-gib 1.5 --decode-share 0.5|VISION=0"
)
bad=0
for c in "${cases[@]}"; do
  want=${c%%|*}; read -r -a sets <<< "${c#*|}"
  got=$(rel ${sets[@]+"${sets[@]}"})
  [ "${got}" = "${want}" ] || { no "release_settings (${c#*|}): got '${got}', want '${want}'"; bad=1; }
done
rel CONTEXT=1048577 > /dev/null; [ $? = 3 ] || { no "release_settings: CONTEXT=1048577 not refused"; bad=1; }
rel CONTEXT=9999999999999999999999999 > /dev/null; [ $? = 3 ] || { no "release_settings: CONTEXT=9999999999999999999999999 not refused"; bad=1; }
[ ${bad} = 0 ] && ok "release_settings: ${#cases[@]} contexts keep DSPARK and the prompt cache as set (1M with DSpark); past 1048576 refused (a 25-digit one too)"

# ---------------------------------------------------------------- the checkpoint drop-in (weights/test-checkpoint.sh)
out=$(bash "${here}/../weights/test-checkpoint.sh" 2>&1); rc=$?
printf '%s\n' "${out}" | grep -E '^(PASS|FAIL|SKIP)' | sed -E 's/^(PASS|FAIL) /\1 checkpoint: /'
n=$(printf '%s\n' "${out}" | grep -c '^FAIL'); fails=$((fails + n))
[ ${rc} = 0 ] || [ "${n}" -gt 0 ] || no "checkpoint: weights/test-checkpoint.sh exit ${rc}: $(printf '%s\n' "${out}" | tail -1)"

# ---------------------------------------------------------------- VISION's weights step (weights/test-vision.sh)
out=$(bash "${here}/../weights/test-vision.sh" 2>&1); rc=$?
printf '%s\n' "${out}" | grep -E '^(PASS|FAIL)' | sed -E 's/^(PASS|FAIL) /\1 vision weights: /'
n=$(printf '%s\n' "${out}" | grep -c '^FAIL'); fails=$((fails + n))
[ ${rc} = 0 ] || [ "${n}" -gt 0 ] || no "vision weights: weights/test-vision.sh exit ${rc}: $(printf '%s\n' "${out}" | tail -1)"

echo "$([ ${fails} = 0 ] && echo "ALL PASS" || echo "${fails} FAILED")"
exit $((fails > 0))
