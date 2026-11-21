#!/usr/bin/env bash
# weights/test-vision.sh: VISION's weights step on this machine, with stand-ins for ssh, scp and the hosts (no network).
# VISION=0 must keep v0.2's exact files, options and remote commands (weights.sh's scp list and copy-rank.sh command;
# setup.sh's w_prepare, w_opts, weights_key and the container's .env); VISION=1 adds the tower's tool and --vision on
# rank 0 alone, as one argument even in a folder with a space; setup.sh never stamps rank 0's weights as vision when
# its folder holds no checked tower (a text-only copy it waited for). Prints PASS/FAIL lines; exit 0 only if all pass.
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
w=$(mktemp -d "${TMPDIR:-/tmp}/inkling2-tower-test.XXXXXX") || exit 1
trap 'rm -rf "${w}"' EXIT
fails=0
ok() { echo "PASS $*"; }
no() { echo "FAIL $*"; fails=$((fails + 1)); }

# ---------------------------------------------------------------- weights.sh start, ssh and scp stood in
mkdir -p "${w}/bin" "${w}/tools dir" "${w}/plan"
cat > "${w}/bin/ssh" << 'EOF'
#!/usr/bin/env bash
h=$1; shift
printf 'SSH %s|%s\n' "$h" "$*" >> "${STUB_LOG}"
case $* in *"kill -0"*) exit 1;; esac # no copy running
exit 0
EOF
cat > "${w}/bin/scp" << 'EOF'
#!/usr/bin/env bash
[ "$1" = -q ] && shift
printf 'SCP' >> "${STUB_LOG}"; for a in "$@"; do printf '|%s' "$a" >> "${STUB_LOG}"; done; echo >> "${STUB_LOG}"
EOF
chmod +x "${w}/bin/ssh" "${w}/bin/scp"
touch "${w}/tools dir/inkling2_rank_copy.py" "${w}/tools dir/inkling2_vision_copy.py"
for f in ranges-w4-r0.tsv ranges-w4-r1.tsv ranges-w4-r2.tsv ranges-w4-r3.tsv inkling2-digests-w4.tsv; do touch "${w}/plan/${f}"; done
cat > "${w}/conf.sh" << EOF
HOSTS=(h0 h1 h2 h3); ROOT=/srv/x; RUN=/srv/x/run; MODEL=/srv/x/model-split
EOF
run_weights() { # VISION
  : > "${w}/log"
  (echo "VISION=$1" >> "${w}/conf.sh"; PATH="${w}/bin:${PATH}" STUB_LOG="${w}/log" RANK_COPY="${w}/tools dir/inkling2_rank_copy.py" \
    bash "${here}/weights.sh" "${w}/conf.sh" start "${w}/plan" > /dev/null 2>&1)
  sed -i.bak '$d' "${w}/conf.sh"
}
# v0.2's scp list and copy-rank.sh command for rank R (weights.sh at 016e2f4e6, DSPARK=1's options)
v02_scp() { printf 'SCP|%s|%s|%s|%s|%s|%s|%s|h%s:/srv/x/recipe/weights/\n' "${w}/tools dir/inkling2_rank_copy.py" "${here}/weights/copy-rank.sh" \
  "${here}/weights/fold-rank.sh" "${here}/weights/seed_rank_folder.py" "${here}/weights/netrun.py" "${w}/plan/ranges-w4-r$1.tsv" "${w}/plan/inkling2-digests-w4.tsv" "$1"; }
v02_cmd() { printf "SSH h%s|cd '/srv/x/recipe/weights' || exit 1; setsid nohup bash copy-rank.sh %s '/srv/x/model-split' --drafter --folds /srv/x/bin/tf-inkling2-fold %s > '/srv/x/run/weights-%s.log' 2>&1 < /dev/null & echo \$! > '/srv/x/run/weights-%s.pid'; echo \"rank %s on \$(hostname): copy pid \$(cat '/srv/x/run/weights-%s.pid'), log /srv/x/run/weights-%s.log\"\n" "$1" "$1" "$2" "$1" "$1" "$1" "$1" "$1"; }
run_weights 0
for r in 0 1 2 3; do
  grep -qxF "$(v02_scp "${r}")" "${w}/log" && ok "weights.sh VISION=0 rank ${r}: v0.2's scp list" || no "weights.sh VISION=0 rank ${r}: scp $(grep "^SCP.*h${r}:" "${w}/log")"
  grep -qxF "$(v02_cmd "${r}" "")" "${w}/log" && ok "weights.sh VISION=0 rank ${r}: v0.2's copy-rank.sh command" || no "weights.sh VISION=0 rank ${r}: $(grep "^SSH h${r}|cd" "${w}/log")"
done
grep -qE 'inkling2_vision_copy|--vision' "${w}/log" && no "weights.sh VISION=0 names the tower: $(grep -E 'inkling2_vision_copy|--vision' "${w}/log" | head -1)" || ok "weights.sh VISION=0: neither the tower's tool nor --vision reaches a host"
run_weights 1
grep -qxF "$(v02_scp 0 | sed "s#|h0:#|${w}/tools dir/inkling2_vision_copy.py|h0:#")" "${w}/log" &&
  ok "weights.sh VISION=1 rank 0: the tower's tool added, one argument in a folder with a space" || no "weights.sh VISION=1 rank 0: $(grep '^SCP.*h0:' "${w}/log")"
grep -qxF "$(v02_cmd 0 "--vision ")" "${w}/log" && ok "weights.sh VISION=1 rank 0: --vision" || no "weights.sh VISION=1 rank 0: $(grep '^SSH h0|cd' "${w}/log")"
for r in 1 2 3; do
  { grep -qxF "$(v02_scp "${r}")" "${w}/log" && grep -qxF "$(v02_cmd "${r}" "")" "${w}/log"; } && ok "weights.sh VISION=1 rank ${r}: as v0.2" || no "weights.sh VISION=1 rank ${r}: changed"
done

# ---------------------------------------------------------------- setup.sh's weights functions, its hosts stood in
{
  sed -n '/^w_vision()/,/^weights_verify_host()/p' "${here}/setup.sh" | sed '$d'
  sed -n '/^do_weights()/,/^}/p' "${here}/setup.sh"
  sed -n '/^ctr_write()/,/^}/p' "${here}/setup.sh"
} > "${w}/fns.sh"
setup_case() { # VISION CHECK_RC: prints the rq commands, the stamps and the outcome of do_weights for a waited-for job
  ( VISION=$1 HOSTS=(h0 h1 h2 h3) ROOT=/srv/x RUN=/srv/x/run MODEL=/srv/x/model-split MODEL_DIR=/srv/x/model-split PLAN=/srv/x/plan
    DSPARK=1 wdir="" ckpt="" wprefix=() ck_jobs=() HTTP_HOST=127.0.0.1 HTTP_PORT=8890 CONTEXT=131072 PARALLEL=auto
    PROMPT_CACHE_GIB=1.5 DECODE_SHARE=0.5 EXTRA_ARGS="" MODEL=/m PLAN=/srv/x/plan DATA=/d
    q() { printf '%q' "$1"; }
    rq() { shift; printf 'RQ %s\n' "$*" >> "${w}/rq"; case $* in *"--vision-check"*) return "${CHECK_RC}";; *sha256sum*) echo abc;; esac; return 0; }
    img() { echo image; }
    die() { echo "DIE $*"; exit 1; }
    installed() { :; }
    forced() { return 1; }
    sleep() { :; }
    stamp_get() { echo stamped-before; }
    stamp_put() { echo "STAMP $1 $3"; }
    n=0; wjob_state() { if [ "$1" = 0 ]; then echo running; else echo done; fi; }
    wjob_next() { if [ "$1" = 0 ]; then echo wait; else echo done; fi; }
    ck_stop() { :; }; ck_start() { :; }
    # shellcheck source=/dev/null
    source "${w}/fns.sh"
    : > "${w}/rq"
    if [ "${2:-}" = prepare ]; then for r in 0 1 2 3; do w_prepare "${r}"; done; echo "OPTS0 $(w_opts 0)|"; echo "KEY0 $(weights_key 0)"; ctr_write 0; cat "${w}/rq"; exit 0; fi
    # rank 0's copy ran this run without --vision: let it end, then judge the stamp
    wjob_state() { echo done; }
    do_weights; rc=$?; cat "${w}/rq"; exit ${rc} )
}
CHECK_RC=0 out=$(setup_case 0 prepare)
want="RQ cd /srv/x && cp -p tools/inkling2_rank_copy.py plan/ranges-w4-r0.tsv plan/inkling2-digests-w4.tsv recipe/weights/ && mkdir -p /srv/x/run"
printf '%s\n' "${out}" | grep -qxF "${want}" && ok "setup.sh VISION=0 w_prepare: v0.2's command" || no "setup.sh VISION=0 w_prepare: $(printf '%s\n' "${out}" | grep 'cp -p' | head -1)"
printf '%s\n' "${out}" | grep -qxF "OPTS0 --drafter |" && ok "setup.sh VISION=0 w_opts: v0.2's" || no "setup.sh VISION=0 w_opts: $(printf '%s\n' "${out}" | grep OPTS0)"
printf '%s\n' "${out}" | grep -qxF "KEY0 abc /srv/x/model-split model.llm. drafter 1" && ok "setup.sh VISION=0 weights_key: v0.2's" || no "setup.sh VISION=0 weights_key: $(printf '%s\n' "${out}" | grep KEY0)"
printf '%s\n' "${out}" | grep -qF "'DECODE_SHARE=0.5' \"EXTRA_ARGS=''\"" && ! printf '%s\n' "${out}" | grep -q VISION_ARGS &&
  ok "setup.sh VISION=0 container .env: v0.2's lines" || no "setup.sh VISION=0 container .env: $(printf '%s\n' "${out}" | grep -o "'DECODE_SHARE.\{0,60\}")"
CHECK_RC=0 out=$(setup_case 1 prepare)
printf '%s\n' "${out}" | grep -qxF "RQ cd /srv/x && cp -p tools/inkling2_rank_copy.py plan/ranges-w4-r0.tsv plan/inkling2-digests-w4.tsv tools/inkling2_vision_copy.py recipe/weights/ && mkdir -p /srv/x/run" &&
  printf '%s\n' "${out}" | grep -qxF "RQ cd /srv/x && cp -p tools/inkling2_rank_copy.py plan/ranges-w4-r1.tsv plan/inkling2-digests-w4.tsv recipe/weights/ && mkdir -p /srv/x/run" &&
  ok "setup.sh VISION=1 w_prepare: the tower's tool on rank 0 alone" || no "setup.sh VISION=1 w_prepare"
printf '%s\n' "${out}" | grep -qxF "OPTS0 --drafter --vision |" && printf '%s\n' "${out}" | grep -qF "'VISION_ARGS=--vision' \"EXTRA_ARGS=" &&
  ok "setup.sh VISION=1: --vision for rank 0's copy and the container" || no "setup.sh VISION=1 options: $(printf '%s\n' "${out}" | grep OPTS0)"
CHECK_RC=1 out=$(setup_case 1)
{ printf '%s\n' "${out}" | grep -q '^DIE ' && ! printf '%s\n' "${out}" | grep -q '^STAMP 0 '; } &&
  ok "setup.sh VISION=1: a waited-for copy without the tower is not stamped (refused, rerun copies it)" || no "setup.sh VISION=1 stale stamp: $(printf '%s\n' "${out}" | grep -E '^(DIE|STAMP)' | tr '\n' ' ')"
CHECK_RC=0 out=$(setup_case 1)
printf '%s\n' "${out}" | grep -qxF "STAMP 0 abc /srv/x/model-split model.llm. drafter 1 vision" &&
  ok "setup.sh VISION=1: with its checked tower rank 0 is stamped" || no "setup.sh VISION=1 stamp: $(printf '%s\n' "${out}" | grep -E '^(DIE|STAMP)' | tr '\n' ' ')"

# a stamp written with the tower, whose file has since gone: checked again, rank 0 copied again (with --vision)
stamped_case() { # MODE: copy | verify
  ( VISION=1 HOSTS=(h0 h1 h2 h3) ROOT=/srv/x RUN=/srv/x/run MODEL_DIR=/srv/x/model-split PLAN=/srv/x/plan DSPARK=1
    wdir=""; [ "$1" = verify ] && wdir=/srv/x/model-split
    ckpt="" wprefix=() ck_jobs=() launched=0
    q() { printf '%q' "$1"; }
    rq() { shift; case $* in *"--vision-check"*) [ -e "${w}/launched" ];; *sha256sum*) echo abc;; *"--verify-only"*) echo VERIFY >> "${w}/rq";; *) return 0;; esac; }
    die() { echo "DIE $*"; exit 1; }
    installed() { :; }; forced() { return 1; }; sleep() { :; }; ck_stop() { :; }; ck_start() { :; }
    stamp_get() { echo "abc /srv/x/model-split model.llm. drafter 1$([ "$1" = 0 ] && printf ' vision')"; }
    stamp_put() { echo "STAMP $1 $3"; }
    wjob_state() { if [ -e "${w}/launched" ]; then echo done; else echo done; fi; }
    wjob_next() { if [ "$3" = 1 ] && [ "$2" != running ]; then echo done; elif [ "$2" = running ]; then echo wait; else echo new; fi; }
    wjob_launch() { echo "LAUNCH $1 $3"; touch "${w}/launched"; }
    each_rank() { local r; for r in 0 1 2 3; do "$1" "${r}" || return 1; done; }
    # shellcheck source=/dev/null
    source "${w}/fns.sh"
    sed -n '/^weights_verify_host()/,/^}/p' "${here}/setup.sh" > "${w}/verify.sh"
    # shellcheck source=/dev/null
    source "${w}/verify.sh"
    : > "${w}/rq"; rm -f "${w}/launched"
    do_weights; echo "RC $?"; cat "${w}/rq" )
}
out=$(stamped_case copy)
{ printf '%s\n' "${out}" | grep -qxF "LAUNCH 0 --drafter --vision " && ! printf '%s\n' "${out}" | grep -q '^LAUNCH [123]' &&
  printf '%s\n' "${out}" | grep -qxF "STAMP 0 abc /srv/x/model-split model.llm. drafter 1 vision"; } &&
  ok "setup.sh VISION=1: a stamped rank 0 whose tower file is gone is copied again, then stamped" || no "setup.sh VISION=1 stale stamp, copy: $(printf '%s\n' "${out}" | grep -E '^(LAUNCH|STAMP|DIE)' | tr '\n' ' ')"
out=$(stamped_case verify)
{ ! printf '%s\n' "${out}" | grep -q 'h0: /srv/x/model-split already verified' && printf '%s\n' "${out}" | grep -q '^DIE weights: verify failed'; } &&
  ok "setup.sh VISION=1: an existing folder's stamp is not trusted without its tower (verified again, refused)" || no "setup.sh VISION=1 stale stamp, verify: $(printf '%s\n' "${out}" | tr '\n' ' ' | cut -c1-200)"

echo "$([ ${fails} = 0 ] && echo "ALL PASS" || echo "${fails} FAILED")"
exit $((fails > 0))
