#!/usr/bin/env bash
# setup.sh HOSTS_CONF [STEP...] [OPTIONS]: install and start Inkling-2-Small at TP4 on four GB10 hosts, from any
# machine with key-based ssh to all four (it can be one of them). Copy hosts.conf.example, fill it in, then:
#
#   ./setup.sh my-sparks.conf --release URL|TARBALL|DIR              host install: binaries run as host processes
#   ./setup.sh my-sparks.conf --release URL|TARBALL|DIR --container  the same in a container a rank (Docker Compose)
#
# With no STEP it runs, in order (each step skips what is already done, so a rerun resumes where it stopped):
#   preflight  every host: GB10 / sm_121, driver and CUDA version, libcuda and libibverbs, python3, free disk,
#              MemAvailable, other inkling2 processes, both fabric ports (speed, MTU, carrier, address, HCA), the NIC
#              re-probe detector, firewall rules on the bootstrap interface; across hosts: a TCP probe of rank 0's
#              bootstrap and liveness ports and of ephemeral ports on every host's fabric address (NCCL's listeners).
#              Stops on a FAIL (--ignore-preflight goes on anyway); a WARN is reported and setup goes on.
#   fabric     only with --fabric-install: fabric/host/install.sh RANK on every host, as root (sudo -n)
#   assets     the release (a URL, a .tar.gz or an unpacked folder), checked against its sha256 and SHA256SUMS,
#              to ROOT/{bin,plan,data,tools} on every host and checked again there; this recipe to ROOT/recipe
#   libs       ROOT/lib: NCCL 2.30.7 and cuBLAS 13.1.1.3 (the release's lib/, --libs DIR, or NVIDIA's pinned
#              wheels from PyPI, fetched on each host)
#   weights    each rank's folder: fetched from the pinned Hugging Face revision (inkling2_rank_copy.py fetch
#              --http, resumable, then verify and a sample check), or with --weights-dir DIR an existing folder,
#              verified in place (coverage, headers, every run re-read; the folder is only read), or with
#              --checkpoint SRC copied from a full checkpoint already on a host over the fabric: SRC is checked
#              first against the pinned revision's file list, sizes and Hugging Face hashes
#              (weights/checkpoint-files.tsv; refused on any difference), then served read only on its fabric
#              address(es) to the four ranks' fabric addresses for the length of the step (weights/checkpoint.py),
#              and each rank fetches its runs from it (fetch --server, resumable, then verify and a sample check)
#   folds      DSPARK=1 only: each rank's DSpark folds file MODEL/inkling2-dspark-fold.safetensors, built on its host
#              by the release's tf-inkling2-fold from the rank folder's own bytes (weights/fold-rank.sh, about a
#              second); kept when present and its recorded digest, or its header and size, check out
#   manifest   the content-digest manifest at ROOT/plan/inkling2-digests-w4.tsv on every host, checked against the
#              release's SHA256SUMS and equal on all four (TF_INK2_DIGESTS points at it)
#   config     writes the server config (HOSTS_CONF's name with .serve.conf, beside it; the format of
#              config.example.sh) and copies it to ROOT/recipe/config.sh: NCCL bootstraps over the fabric (rank 0's
#              fabric address), the switch's NCCL settings, the release settings (release-defaults.sh:
#              checkpoint mode, split experts, DSpark, the RoCE wire, CONTEXT 131072, --parallel auto, reserve 8 GiB,
#              --prompt-cache-gib, the digest manifest); with
#              --container also ROOT/container/{compose.yaml,.env,rank.env} on every host
#   image      only with --container: builds inkling2-sparks:<release commit> on rank 0's host (container/
#              Dockerfile) and copies it to the other hosts (docker save | docker load through this machine)
#   start      refuses while another inkling2 process runs on a host; rank 0, then 1-3; waits until all have loaded
#   status     each rank's state, MemAvailable, last log lines, and the fabric check
#   smoke      on rank 0's host: smoke.py smoke (chat, thinking, tool call, stream = whole, a long prompt), and the
#              oracle check (greedy replies token for token) when the release ships data/decode.json
# Steps run on their own too: setup.sh my-sparks.conf status. Also: stop, and check (the W=4 decode check: every
# decode step bit-equal to the whole-prompt forward; it needs the server stopped).
#
# Options:
#   --release SRC          http(s) or file URL of the release .tar.gz, a local .tar.gz, or an unpacked release
#                          folder (default RELEASE in HOSTS_CONF). A URL or tarball needs its sha256: SRC.sha256
#                          next to it, or --sha256 HEX.
#   --sha256 HEX           the release tarball's sha256
#   --container            run the ranks as containers (image + compose.yaml) instead of host processes
#   --weights-dir DIR      reuse a rank folder already on every host at DIR (its weights verified, never written;
#                          the folds step adds the folds file there when it is missing)
#   --weights-prefix P     only the tensors under P (repeatable; tests the download path; such a folder does not
#                          serve)
#   --checkpoint SRC       build the rank folders from a full Inkling-2-Small checkpoint instead of Hugging Face:
#                          HOST:/path (HOST an ssh name, e.g. one of the four) or /path on this machine
#                          (default CHECKPOINT in HOSTS_CONF; the result is byte for byte the download's)
#   --libs DIR             libnccl.so.2, libcublas.so.13 and libcublasLt.so.13 from DIR on this machine
#   --fabric-install       the fabric step (root on each host through sudo -n); --fabric-dry-run prints it only
#   --force STEP           redo STEP even if its stamp says done (repeatable)
#   --ignore-preflight     go on after a preflight FAIL
# Settings: see hosts.conf.example. State: ROOT/.setup/ on each host; the release cache: SETUP_CACHE (default
# ~/.cache/inkling2-setup) on this machine.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
[ $# -ge 1 ] || { usage; exit 2; }
case $1 in -h|--help|help) usage; exit 0;; esac
conf=$1; shift
[ -f "${conf}" ] || { echo "no such file: ${conf}" >&2; exit 2; }
conf=$(cd "$(dirname "${conf}")" && pwd)/$(basename "${conf}")

steps=(); release=""; rsha=""; container=0; wdir=""; wprefix=(); libs_dir=""; fabric=0; fabric_dry=0; ckpt=""
force=" "; ignore_pre=0
while [ $# -gt 0 ]; do
  case $1 in
    --release) release=$2; shift 2;;
    --sha256) rsha=$2; shift 2;;
    --container) container=1; shift;;
    --weights-dir) wdir=$2; shift 2;;
    --weights-prefix) wprefix+=("$2"); shift 2;;
    --checkpoint) ckpt=$2; shift 2;;
    --libs) libs_dir=$2; shift 2;;
    --fabric-install) fabric=1; shift;;
    --fabric-dry-run) fabric=1; fabric_dry=1; shift;;
    --force) force+="$2 "; shift 2;;
    --ignore-preflight) ignore_pre=1; shift;;
    -h|--help) usage; exit 0;;
    -*) echo "unknown option $1 (setup.sh --help)" >&2; exit 2;;
    *) steps+=("$1"); shift;;
  esac
done

# ---------------------------------------------------------------- settings (hosts.conf.example documents them)
FABRIC_DEVS=(enp1s0f0np0 enP2p1s0f0np0)
FABRIC_HCAS=(rocep1s0f0 roceP2p1s0f0)
FABRIC_SPEED=200000; FABRIC_MTU=9000; FABRIC_GID=3; FABRIC_NETS=""
MASTER=""; NCCL_PORT=29551; SOCKET_IFNAME=""
HTTP_HOST=127.0.0.1; HTTP_PORT=8890; EXTRA_ARGS=""
OPT_ENV=(); RELEASE=""; RELEASE_SHA256=""; WEIGHTS_DIR=""; MODEL_DIR=""; LIBS_DIR=""
CHECKPOINT=""; CHECKPOINT_ADDR=""; CHECKPOINT_PORT=0
BASE_IMAGE=nvcr.io/nvidia/cuda:13.3.1-runtime-ubuntu24.04@sha256:585155539dd4ff2ab653ee59d812aae81a9406a5804596d516c06e3a5719eb26 # the gated arm64 manifest (container/Dockerfile)
SMOKE_LONG_TOKENS=8000; READY_WAIT=900; CHECK_PROMPT=3040; CHECK_STEPS=80; PROBE_PORTS=10
# The release's settings (CONTEXT, PARALLEL, the reserve, the prompt cache, DSpark, XROWS, the RoCE wire, the folder
# name): release-defaults.sh. HOSTS_CONF may override any of them; the config written below carries only the overrides.
REL_VARS=(MODEL_DIRNAME EXPERTS CONTEXT PARALLEL MAX_STREAMS MEMORY_RESERVE_GIB PROMPT_CACHE_GIB DECODE_SHARE DSPARK XROWS ROCE_MULTI MAILBOX_MAX_KB TP_PORT)
# shellcheck source=/dev/null
source "${here}/release-defaults.sh"
# shellcheck source=setup/lib.sh
source "${here}/setup/lib.sh" # unpack_release, serve_config, the weights copy jobs (wjob_*)
for v in "${REL_VARS[@]}"; do eval "def_${v}=\${${v}}"; done
# shellcheck source=/dev/null
source "${conf}"
release_settings || exit 2
# NCCL on the switch: both PCIe halves of each host's 200G port, RoCE v2, no graph file (HOSTS_CONF may set NCCL_ENV)
if [ -z "${NCCL_ENV+x}" ]; then
  NCCL_ENV=(
    NCCL_IB_DISABLE=0
    "NCCL_IB_HCA=$(IFS=,; echo "${FABRIC_HCAS[*]}")"
    "NCCL_IB_GID_INDEX=${FABRIC_GID}" NCCL_IB_MERGE_NICS=0 NCCL_CROSS_NIC=1
    NCCL_IB_SUBNET_AWARE_ROUTING=1 NCCL_IB_SUBNET_PREFIX_LEN=24
    NCCL_ALGO=Ring NCCL_CUMEM_ENABLE=1 NCCL_RUNTIME_CONNECT=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
    NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4
    NCCL_DEBUG=WARN
  )
fi
[ ${#HOSTS[@]} = 4 ] || { echo "HOSTS must name four hosts in rank order" >&2; exit 2; }
[ -n "${ROOT:-}" ] || { echo "set ROOT (the install folder on every host)" >&2; exit 2; }
[ -n "${release}" ] || release=${RELEASE}
[ -n "${rsha}" ] || rsha=${RELEASE_SHA256}
[ -n "${wdir}" ] || wdir=${WEIGHTS_DIR}
[ -n "${ckpt}" ] || ckpt=${CHECKPOINT}
ck_host=""; ck_path=""
if [ -n "${ckpt}" ]; then
  [ -z "${wdir}" ] || { echo "set one of WEIGHTS_DIR (--weights-dir) and CHECKPOINT (--checkpoint)" >&2; exit 2; }
  case ${ckpt} in
    /*) ck_path=${ckpt};;
    ?*:/*) ck_host=${ckpt%%:*}; ck_path=${ckpt#*:};;
    *) echo "CHECKPOINT=${ckpt}: expected HOST:/absolute/path (HOST an ssh name) or /absolute/path on this machine" >&2; exit 2;;
  esac
fi
[ -n "${libs_dir}" ] || libs_dir=${LIBS_DIR}
[ -n "${MODEL_DIR}" ] || MODEL_DIR=${ROOT}/${MODEL_DIRNAME}
MODEL=${MODEL_DIR}; [ -n "${wdir}" ] && MODEL=${wdir}
[ -n "${SOCKET_IFNAME}" ] || SOCKET_IFNAME=${FABRIC_DEVS[0]}
PLAN=${ROOT}/plan; DATA=${ROOT}/data; RUN=${ROOT}/serve
case ${conf} in *.conf) serve_conf=${conf%.conf}.serve.conf;; *) serve_conf=${conf}.serve.conf;; esac
cache=${SETUP_CACHE:-${HOME}/.cache/inkling2-setup}
if [ ${#steps[@]} = 0 ]; then
  steps=(preflight)
  [ ${fabric} = 1 ] && steps+=(fabric)
  steps+=(assets libs weights folds manifest config)
  [ ${container} = 1 ] && steps+=(image)
  steps+=(start status smoke)
fi
for s in "${steps[@]}"; do
  case ${s} in preflight|fabric|assets|libs|weights|folds|manifest|config|image|start|stop|status|smoke|check) ;;
    *) echo "unknown step ${s}" >&2; exit 2;; esac
done

B=$'\033[1m'; D=$'\033[2m'; Z=$'\033[0m'; [ -t 1 ] || { B=; D=; Z=; }
n_step=0
step() { n_step=$((n_step + 1)); printf '\n%s== [%s/%s] %s%s %s(%s)%s\n' "${B}" "${n_step}" "${#steps[@]}" "$1" "${Z}" "${D}" "$(date -u +%T)" "${Z}"; }
die() { echo "setup: $*" >&2; exit 1; }
sha() { if command -v sha256sum > /dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
rq() { local h=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 "${h}" "$@"; }
q() { printf '%q' "$1"; }
forced() { case ${force} in *" $1 "*) return 0;; esac; return 1; }
# tar on this machine without macOS metadata (bsdtar), and an untar on the hosts that ignores such headers
tarc() { if tar --version 2> /dev/null | grep bsdtar > /dev/null; then COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs "$@"; else tar "$@"; fi; }
UNTAR='tar --warning=no-unknown-keyword -xf -'
# a remote prefix that picks docker or sudo -n docker
DK='dk=docker; docker info > /dev/null 2>&1 || dk="sudo -n docker";'
tmp=$(mktemp -d "${TMPDIR:-/tmp}/inkling2-setup.XXXXXX")
# CHECKPOINT: commands on the checkpoint's host (ck_sh), and ck_stop: the end of its server (ck_start, weights step).
# ck_jobs: the ranks whose copies this run started against that server; on any exit they are stopped and joined
# before the server ends (wjob_stop_all), so none is left running without its source (setup/lib.sh, wjob_*).
ck_dir=""; ck_srv=(); ck_jobs=()
ck_sh() { if [ -n "${ck_host}" ]; then rq "${ck_host}" "$1"; else bash -c "$1"; fi; }
ck_where() { echo "${ck_host:-this machine}:${ck_path}"; }
ck_stop() {
  [ -n "${ck_dir}" ] || return 0
  ck_sh "p=$(q "${ck_dir}/serve.pid"); [ -f \"\$p\" ] && kill \$(cat \"\$p\") 2> /dev/null; rm -rf $(q "${ck_dir}")" > /dev/null 2>&1
  ck_dir=""
}
trap 'wjob_stop_all ${ck_jobs[@]+"${ck_jobs[@]}"}; ck_stop; rm -rf "${tmp}"' EXIT
# each_rank FN: FN RANK on all four hosts at once; prints each rank's output in order; returns 1 if any failed
each_rank() {
  local fn=$1 r rc=0
  for r in 0 1 2 3; do ( "${fn}" "${r}" > "${tmp}/${fn}.${r}.out" 2>&1; echo $? > "${tmp}/${fn}.${r}.rc" ) & done
  wait
  for r in 0 1 2 3; do
    cat "${tmp}/${fn}.${r}.out"
    [ "$(cat "${tmp}/${fn}.${r}.rc")" = 0 ] || rc=1
  done
  return ${rc}
}
stamp_get() { rq "${HOSTS[$1]}" "cat $(q "${ROOT}/.setup/$2") 2>/dev/null"; }
stamp_put() { rq "${HOSTS[$1]}" "mkdir -p $(q "${ROOT}/.setup") && printf '%s\n' $(q "$3") > $(q "${ROOT}/.setup/$2")"; }

# ---------------------------------------------------------------- preflight
pf_host() {
  local r=$1 need=4
  if [ -z "${wdir}" ]; then need=100; [ ${#wprefix[@]} -gt 0 ] && need=4; fi
  [ ${container} = 1 ] && need=$((need + 8))
  rq "${HOSTS[$r]}" "env RANK=${r} ROOT=$(q "${ROOT}") NEED_GIB=${need} MODE=$([ ${container} = 1 ] && echo container || echo host) \
    FABRIC_DEVS='${FABRIC_DEVS[*]}' FABRIC_HCAS='${FABRIC_HCAS[*]}' FABRIC_SPEED=${FABRIC_SPEED} FABRIC_MTU=${FABRIC_MTU} \
    SOCKET_IFNAME=${SOCKET_IFNAME} $( [ -z "${wdir}" ] && [ "${MODEL_DIR}" != "${ROOT}/${MODEL_DIRNAME}" ] && echo "MODEL_DIR=$(q "${MODEL_DIR}")") \
    bash -s" < "${here}/setup/host-check.sh"
}
# probe_ports: rank r listens on its SOCKET_IFNAME address, rank r+1 connects (rank 0's set includes the
# bootstrap and liveness ports); prints a line a pair; returns 1 if any port is blocked or the bootstrap port is busy
probe_ports() {
  local ports addrs=() r n rc=0
  for r in 0 1 2 3; do addrs[r]=$(grep -h '^INFO sockaddr=' "${tmp}/pf_host.${r}.out" | head -1 | cut -d= -f2 | cut -d/ -f1); done
  ports=$(rq "${HOSTS[0]}" "python3 - ports ${NCCL_PORT} ${PROBE_PORTS}" < "${here}/setup/netprobe.py")
  [ -n "${ports}" ] || { echo "  port probe: could not run python3 on ${HOSTS[0]}"; return 1; }
  # the RoCE wire's rendezvous (ROCE_MULTI) at rank 0: TF_TP_PORT and TF_TP_PORT + 7, on the fabric address too
  [ "${ROCE_MULTI}" = 1 ] && ports="${ports} ${TP_PORT} $((TP_PORT + 7))"
  for r in 0 1 2 3; do
    [ -n "${addrs[r]}" ] || { echo "  port probe: rank ${r} has no address on ${SOCKET_IFNAME}"; rc=1; continue; }
    rq "${HOSTS[$r]}" "python3 - listen ${addrs[r]} 25 ${ports}" < "${here}/setup/netprobe.py" > "${tmp}/listen.${r}" 2>&1 &
  done
  for _ in $(seq 1 40); do
    local ready=0; for r in 0 1 2 3; do grep -q '^ready' "${tmp}/listen.${r}" 2>/dev/null && ready=$((ready + 1)); done
    [ ${ready} = 4 ] && break; sleep 0.5
  done
  for r in 0 1 2 3; do
    n=$(( (r + 1) % 4 ))
    [ -n "${addrs[r]}" ] || continue
    local busy want=""
    busy=$(awk '/^busy/{print $2}' "${tmp}/listen.${r}" | tr '\n' ' ')
    for p in ${ports}; do case " ${busy} " in *" ${p} "*) ;; *) want+="${p} ";; esac; done
    if [ ${r} = 0 ]; then
      case " ${busy} " in *" ${NCCL_PORT} "*|*" $((NCCL_PORT + 1)) "*)
        echo "  port probe FAIL: NCCL_PORT ${NCCL_PORT} or ${NCCL_PORT}+1 is already in use on ${HOSTS[0]} (another server?)"; rc=1;; esac
    fi
    local out; out=$(rq "${HOSTS[$n]}" "python3 - connect ${addrs[r]} ${want}" < "${here}/setup/netprobe.py" 2>&1)
    if printf '%s\n' "${out}" | grep '^blocked' > /dev/null; then
      echo "  port probe FAIL: rank ${n} -> rank ${r} ${addrs[r]} (${SOCKET_IFNAME}): $(printf '%s\n' "${out}" | grep '^blocked' | tr '\n' ' ')"; rc=1
    else
      echo "  port probe OK: rank ${n} -> rank ${r} ${addrs[r]} (${SOCKET_IFNAME}), TCP ${want}"
    fi
  done
  wait
  return ${rc}
}
do_preflight() {
  each_rank pf_host | grep -v '^INFO '
  local rc=0 r
  for r in 0 1 2 3; do [ "$(cat "${tmp}/pf_host.${r}.rc")" = 2 ] && rc=1; done
  local drivers; drivers=$(cat "${tmp}"/pf_host.?.out | grep '^INFO driver=' | sort -u | wc -l | tr -d ' ')
  [ "${drivers}" -le 1 ] || echo "  WARN: the hosts run different drivers: $(cat "${tmp}"/pf_host.?.out | grep '^INFO driver=' | cut -d= -f2 | tr '\n' ' ')"
  local a0 net0 a
  a0=$(grep -h '^INFO addr0=' "${tmp}/pf_host.0.out" | cut -d= -f2 | cut -d/ -f1)
  net0=${a0%.*}
  for r in 1 2 3; do
    a=$(grep -h '^INFO addr0=' "${tmp}/pf_host.${r}.out" | cut -d= -f2 | cut -d/ -f1)
    [ -z "${a}" ] || [ "${a%.*}" = "${net0}" ] || echo "  WARN: rank ${r}'s ${FABRIC_DEVS[0]} address ${a} is not in rank 0's /24 (${net0}.0)"
  done
  local s0; s0=$(grep -h '^INFO sockaddr=' "${tmp}/pf_host.0.out" | cut -d= -f2 | cut -d/ -f1)
  if [ -n "${MASTER}" ] && [ "${MASTER}" != "${s0}" ]; then echo "  note: MASTER=${MASTER} is set; rank 0's ${SOCKET_IFNAME} address is ${s0:-none}"; fi
  probe_ports || rc=1
  if [ ${rc} != 0 ]; then
    if [ ${ignore_pre} = 1 ]; then echo "preflight: FAIL above; going on (--ignore-preflight)"; return 0; fi
    die "preflight failed (FAIL lines above); fix them, or rerun with --ignore-preflight"
  fi
  echo "preflight: OK"
}

# ---------------------------------------------------------------- fabric
fab_host() {
  local r=$1
  rq "${HOSTS[$r]}" "mkdir -p $(q "${ROOT}/recipe")" && push_recipe_to "${r}" || return 1
  rq "${HOSTS[$r]}" "cd $(q "${ROOT}/recipe/fabric/host") && $([ ${fabric_dry} = 1 ] || echo 'sudo -n') bash install.sh ${r} \
    ${FABRIC_NETS:+--nets '${FABRIC_NETS}'} --devs '${FABRIC_DEVS[*]}' --mtu ${FABRIC_MTU} $([ ${fabric_dry} = 1 ] && echo --dry-run)"
}
do_fabric() {
  [ -n "${FABRIC_NETS}" ] || die "set FABRIC_NETS (the two /24s, e.g. \"198.18.0 198.18.1\") for --fabric-install"
  each_rank fab_host || die "fabric install failed on a host (it needs root through sudo -n; or run fabric/host/install.sh RANK as root by hand)"
  [ ${fabric_dry} = 1 ] && echo "fabric: dry run only" || echo "fabric: installed; reboot a host whose cables changed, then rerun setup.sh"
}

# ---------------------------------------------------------------- assets
push_recipe_to() {
  local r=$1
  tarc -C "${here}" --exclude '*.serve.conf' --exclude '__pycache__' -cf - \
    release-defaults.sh rank.sh start.sh stop.sh warmup.sh status.sh check.sh smoke.py prompt_ids.py config.example.sh hosts.conf.example setup.sh \
    setup weights weights.sh container fabric README.md |
    rq "${HOSTS[$r]}" "mkdir -p $(q "${ROOT}/recipe") && cd $(q "${ROOT}/recipe") && ${UNTAR}"
}
# fetch_release: sets rel_dir to an unpacked, checked release folder
fetch_release() {
  local src=${release} tgz="" want="" name
  [ -n "${src}" ] || die "no release: pass --release URL|TARBALL|DIR (or RELEASE in ${conf})"
  mkdir -p "${cache}"
  case ${src} in
    http://*|https://*|file://*)
      name=$(basename "${src%%\?*}"); tgz=${cache}/${name}
      [[ ${name} =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*\.(tar\.gz|tgz)$ ]] || die "release URL ${src}: expected a .tar.gz file name"
      want=${rsha}
      if [ -z "${want}" ]; then
        want=$(curl -fsSL "${src}.sha256" 2>/dev/null | awk '{print $1}')
        [ -n "${want}" ] || die "no ${src}.sha256; pass --sha256 HEX"
      fi
      if [ -f "${tgz}" ] && [ "$(sha "${tgz}" | awk '{print $1}')" = "${want}" ]; then echo "release: ${name} already downloaded"
      else
        echo "release: downloading ${src}"
        curl -fL --retry 3 -o "${tgz}.part" "${src}" || die "download failed: ${src}"
        mv -f "${tgz}.part" "${tgz}"
      fi;;
    *.tar.gz|*.tgz)
      [ -f "${src}" ] || die "no such file: ${src}"
      tgz=$(cd "$(dirname "${src}")" && pwd)/$(basename "${src}")
      want=${rsha}
      [ -n "${want}" ] || { [ -f "${tgz}.sha256" ] && want=$(awk '{print $1}' "${tgz}.sha256"); }
      [ -n "${want}" ] || die "no ${tgz}.sha256; pass --sha256 HEX";;
    *)
      [ -d "${src}" ] || die "no such release folder: ${src}"
      rel_dir=$(cd "${src}" && pwd);;
  esac
  if [ -n "${tgz}" ]; then
    local got; got=$(sha "${tgz}" | awk '{print $1}')
    [ "${got}" = "${want}" ] || die "$(basename "${tgz}"): sha256 ${got}, expected ${want}"
    echo "release: $(basename "${tgz}") sha256 OK (${got:0:16}...)"
    unpack_release "${tgz}" "${got}" # sets rel_dir (setup/lib.sh: checked listing, temporary folder, one rename)
  fi
  [ -f "${rel_dir}/SHA256SUMS" ] || die "${rel_dir} has no SHA256SUMS"
  (cd "${rel_dir}" && sha -c --quiet SHA256SUMS) > /dev/null || die "${rel_dir}: files differ from its SHA256SUMS"
  echo "release: ${rel_dir}: $(wc -l < "${rel_dir}/SHA256SUMS" | tr -d ' ') files match SHA256SUMS"
}
# stage_release: the release in this install's layout at ${stage}: bin/ tools/ plan/ data/ [lib/] VERSION SHA256SUMS.
# Takes the tarball layout (bin/, plan/, data/, tools/zig/) or a build's release-dist layout (bin/ with the copy tool,
# ranges/ranges-w4-split-rR.tsv and inkling2-digests-w4-split.tsv, REVISION).
stage_release() {
  local d=${rel_dir} f r commit
  stage=${tmp}/stage; rm -rf "${stage}"; mkdir -p "${stage}/bin" "${stage}/tools" "${stage}/plan"
  for f in tensorfold-native tf-inkling2-decode; do [ -f "${d}/bin/${f}" ] || die "the release has no bin/${f}"; done
  for f in "${d}"/bin/*; do case ${f} in *.py) cp -p "${f}" "${stage}/tools/";; *) cp -p "${f}" "${stage}/bin/";; esac; done
  [ -f "${stage}/tools/inkling2_rank_copy.py" ] || cp -p "${d}/tools/zig/inkling2_rank_copy.py" "${stage}/tools/" 2> /dev/null ||
    die "the release has no inkling2_rank_copy.py"
  for f in "${d}"/tools/*.py; do [ -f "${f}" ] && cp -p "${f}" "${stage}/tools/"; done
  for r in 0 1 2 3; do
    if [ -f "${d}/plan/ranges-w4-r${r}.tsv" ]; then cp -p "${d}/plan/ranges-w4-r${r}.tsv" "${stage}/plan/"
    elif [ -f "${d}/ranges/ranges-w4-split-r${r}.tsv" ]; then cp -p "${d}/ranges/ranges-w4-split-r${r}.tsv" "${stage}/plan/ranges-w4-r${r}.tsv"
    else die "the release has no ranges for rank ${r} (plan/ranges-w4-r${r}.tsv)"; fi
  done
  if [ -f "${d}/plan/inkling2-digests-w4.tsv" ]; then cp -p "${d}/plan/inkling2-digests-w4.tsv" "${stage}/plan/"
  elif [ -f "${d}/ranges/inkling2-digests-w4-split.tsv" ]; then cp -p "${d}/ranges/inkling2-digests-w4-split.tsv" "${stage}/plan/inkling2-digests-w4.tsv"
  else die "the release has no digest manifest (plan/inkling2-digests-w4.tsv)"; fi
  [ -d "${d}/data" ] && { mkdir -p "${stage}/data"; cp -p "${d}"/data/* "${stage}/data/"; }
  [ -d "${d}/lib" ] && { mkdir -p "${stage}/lib"; cp -p "${d}"/lib/* "${stage}/lib/"; }
  mkdir -p "${stage}/doc"
  for f in LICENSE NOTICE THIRD_PARTY_NOTICES.md LICENSES; do
    if [ -e "${d}/${f}" ]; then cp -pR "${d}/${f}" "${stage}/doc/"
    elif [ -e "${here}/../../${f}" ]; then cp -pR "${here}/../../${f}" "${stage}/doc/"; fi
  done
  # the release commit: the binaries' build (REVISION, from the build), else the tarball's source commit
  rev=""; for f in "${d}/REVISION" "${d}/bin/REVISION"; do [ -f "${f}" ] && { rev=${f}; break; }; done
  commit=""; [ -n "${rev}" ] && commit=$(awk 'NR == 1 {print $1}' "${rev}")
  [ -n "${commit}" ] || commit=$(awk '/^source commit/{print $3}' "${d}/VERSION" 2>/dev/null)
  [ -n "${commit}" ] || commit=$(sha "${d}/SHA256SUMS" | cut -c1-12)
  short=${commit:0:9}
  { [ -f "${d}/VERSION" ] && cat "${d}/VERSION"; [ -n "${rev}" ] && cat "${rev}"
    echo "release commit ${short}"; } > "${stage}/VERSION"
  (cd "${stage}" && find . -type f | LC_ALL=C sort | sed 's|^\./||' | while read -r f; do sha "${f}"; done > "${tmp}/assets.sums")
  mv "${tmp}/assets.sums" "${stage}/SHA256SUMS"
  assets_key=$(sha "${stage}/SHA256SUMS" | cut -c1-16)
}
assets_host() {
  local r=$1 h=${HOSTS[$1]}
  push_recipe_to "${r}" || { echo "rank ${r}: copy of the recipe to ${h} failed"; return 1; }
  if [ "$(stamp_get "${r}" assets)" = "${assets_key}" ] && ! forced assets &&
     rq "${h}" "cd $(q "${ROOT}") && sha256sum -c --quiet .setup/assets.SHA256SUMS > /dev/null 2>&1"; then
    echo "rank ${r} ${h}: release ${short} already installed in ${ROOT} (checked)"; return 0
  fi
  tarc -C "${stage}" -cf - . | rq "${h}" "mkdir -p $(q "${ROOT}/.setup") && cd $(q "${ROOT}") && ${UNTAR} &&
    mv -f SHA256SUMS .setup/assets.SHA256SUMS && printf '%s\n' $(q "${short}") > .setup/release && chmod 0755 bin/* && sha256sum -c --quiet .setup/assets.SHA256SUMS" ||
    { echo "rank ${r} ${h}: copy or check failed"; return 1; }
  stamp_put "${r}" assets "${assets_key}"
  echo "rank ${r} ${h}: release ${short} installed in ${ROOT}, $(wc -l < "${stage}/SHA256SUMS" | tr -d ' ') files checked"
}
do_assets() {
  fetch_release
  stage_release
  echo "release commit ${short}; assets ${assets_key}"
  each_rank assets_host || die "assets: a host failed"
}
# installed: the release commit and assets key of the install on rank 0's host (the steps after assets use only that)
installed() {
  [ -n "${short:-}" ] && [ -n "${assets_key:-}" ] && return 0
  short=$(rq "${HOSTS[0]}" "cat $(q "${ROOT}/.setup/release") 2>/dev/null")
  assets_key=$(rq "${HOSTS[0]}" "cat $(q "${ROOT}/.setup/assets") 2>/dev/null")
  [ -n "${short}" ] && [ -n "${assets_key}" ] || die "no release installed in ${ROOT} on ${HOSTS[0]}: run the assets step (--release SRC)"
}

# ---------------------------------------------------------------- libs
libs_host() {
  local r=$1 h=${HOSTS[$1]}
  if rq "${h}" "grep -q ' lib/libnccl.so.2\$' $(q "${ROOT}/.setup/assets.SHA256SUMS")"; then
    # libtfgrammar.so (structured output) is the release's own build: listed when the release carries it
    rq "${h}" "cd $(q "${ROOT}/lib") && sha256sum libnccl.so.2 libcublas.so.13 libcublasLt.so.13 \$(ls libtfgrammar.so 2>/dev/null) > SHA256SUMS && echo \"rank ${r} ${h}: lib/ from the release\""
  elif [ -n "${libs_dir}" ]; then
    tarc -C "${libs_dir}" -cf - libnccl.so.2 libcublas.so.13 libcublasLt.so.13 |
      rq "${h}" "mkdir -p $(q "${ROOT}/lib") && cd $(q "${ROOT}/lib") && ${UNTAR} && sha256sum libnccl.so.2 libcublas.so.13 libcublasLt.so.13 > SHA256SUMS &&
        echo \"rank ${r} ${h}: lib/ from ${libs_dir}\""
  else
    rq "${h}" "bash $(q "${ROOT}/recipe/setup/libs.sh") $(q "${ROOT}")" | sed "s/^/rank ${r} ${h}: /"
  fi
}
do_libs() {
  installed
  each_rank libs_host || die "libs: a host failed"
  local sums want; sums=$(for r in 0 1 2 3; do rq "${HOSTS[$r]}" "cat $(q "${ROOT}/lib/SHA256SUMS")"; done | sort -u | wc -l | tr -d ' ')
  want=$(rq "${HOSTS[0]}" "wc -l < $(q "${ROOT}/lib/SHA256SUMS")" | tr -d ' ')
  [ "${want}" -ge 3 ] && [ "${sums}" = "${want}" ] || die "libs: the hosts' ROOT/lib differ (each should hold the same libraries)"
  rq "${HOSTS[0]}" "grep -q ' libtfgrammar.so\$' $(q "${ROOT}/lib/SHA256SUMS")" ||
    echo "libs: note: no libtfgrammar.so in ROOT/lib: json_object, json_schema and required tool calls get HTTP 400 (use a release that carries it)"
  echo "libs: the same ${want} libraries on all four hosts"
}

# ---------------------------------------------------------------- weights
weights_key() { echo "$(rq "${HOSTS[$1]}" "sha256sum $(q "${PLAN}/ranges-w4-r$1.tsv")" | cut -c1-16) ${wdir:-${MODEL_DIR}} ${wprefix[*]:-model.llm.} drafter ${DSPARK}"; }
w_opts() { local p; for p in ${wprefix[@]+"${wprefix[@]}"}; do printf -- '--prefix %q ' "${p}"; done; [ "${DSPARK}" = 1 ] && printf -- '--drafter '; }
w_prepare() { # copy-rank.sh runs in ROOT/recipe/weights with the tool, the rank's ranges and the manifest beside it
  rq "${HOSTS[$1]}" "cd $(q "${ROOT}") && cp -p tools/inkling2_rank_copy.py plan/ranges-w4-r$1.tsv plan/inkling2-digests-w4.tsv recipe/weights/ && mkdir -p $(q "${RUN}")"
}
weights_verify_host() {
  local r=$1 h=${HOSTS[$1]} key; key=$(weights_key "$1")
  if [ "$(stamp_get "${r}" weights)" = "${key}" ] && ! forced weights; then echo "rank ${r} ${h}: ${wdir} already verified"; return 0; fi
  w_prepare "${r}" || return 1
  rq "${h}" "nice ionice -c3 bash $(q "${ROOT}/recipe/weights/copy-rank.sh") ${r} $(q "${wdir}") --verify-only $(w_opts) > $(q "${RUN}/weights-${r}.log") 2>&1; rc=\$?;
    grep -E '^(coverage|headers|verify)' $(q "${RUN}/weights-${r}.log") | sed 's/^/rank ${r} ${h}: /'; exit \${rc}" || { echo "rank ${r} ${h}: verify FAILED (log ${RUN}/weights-${r}.log)"; return 1; }
  stamp_put "${r}" weights "${key}"
}
do_weights() {
  installed
  if [ -n "${wdir}" ]; then
    echo "weights: verifying the existing rank folders at ${wdir} (read only)"
    each_rank weights_verify_host || die "weights: verify failed on a host (logs ${RUN}/weights-R.log)"
    return 0
  fi
  local r st key stamped todo=() new=() src
  for r in 0 1 2 3; do
    key=$(weights_key "${r}"); st=$(wjob_state "${r}")
    stamped=0; [ "$(stamp_get "${r}" weights)" = "${key}" ] && ! forced weights && stamped=1
    case $(wjob_next "${r}" "${st}" "${stamped}") in
      done) echo "rank ${r} ${HOSTS[$r]}: ${MODEL_DIR} done";;
      wait) echo "rank ${r} ${HOSTS[$r]}: a copy is already running; waiting for it"; todo+=("${r}");;
      restart) echo "rank ${r} ${HOSTS[$r]}: a copy from an earlier run's checkpoint server is still running, and that server is gone: stopping it to start again"
        wjob_stop "${r}" || die "weights: cannot stop rank ${r}'s earlier copy on ${HOSTS[$r]} (${RUN}/weights-${r}.pid)"
        new+=("${r}");;
      *) new+=("${r}");;
    esac
  done
  [ ${#new[@]} -gt 0 ] && [ -n "${ckpt}" ] && ck_start "${new[@]}"
  for r in ${new[@]+"${new[@]}"}; do
    src=""; [ -n "${ckpt}" ] && src=${ck_srv[$((r % ${#ck_srv[@]}))]}
    w_prepare "${r}" || die "weights: cannot prepare rank ${r}"
    [ -n "${src}" ] && ck_jobs+=("${r}") # stopped on any exit before the server (wjob_stop_all), also if the launch fails half way
    wjob_launch "${r}" "${src}" "$(w_opts)" || die "weights: cannot start the copy on ${HOSTS[$r]}"
    echo "rank ${r} ${HOSTS[$r]}: copy started into ${MODEL_DIR}${src:+ from ${src}} (log ${RUN}/weights-${r}.log)"
    todo+=("${r}")
  done
  [ ${#todo[@]} = 0 ] && return 0
  if [ -n "${ckpt}" ]; then echo "weights: copying from the checkpoint over the fabric (resumable: the copies stop with this run, since the checkpoint server does; rerun setup.sh and each copy continues from the runs it already holds)"
  else echo "weights: fetching from Hugging Face (resumable: if this machine disconnects, rerun setup.sh; it waits for the running copies)"; fi
  local left=${#todo[@]} t0=${SECONDS} line
  while [ "${left}" -gt 0 ]; do
    sleep 30; left=0
    for r in "${todo[@]}"; do
      st=$(wjob_state "${r}")
      line=$(rq "${HOSTS[$r]}" "du -sh $(q "${MODEL_DIR}") 2>/dev/null | cut -f1; tail -n 1 $(q "${RUN}/weights-${r}.log") | cut -c1-120" | tr '\n' ' ')
      printf '  %4d s rank %s: %-8s %s\n' $((SECONDS - t0)) "${r}" "${st}" "${line}"
      case ${st} in running) left=$((left + 1));; failed|none) die "weights: rank ${r}'s copy failed: ${RUN}/weights-${r}.log on ${HOSTS[$r]}";; esac
    done
  done
  ck_jobs=() # every copy ended: nothing left to stop
  ck_stop
  for r in "${todo[@]}"; do stamp_put "${r}" weights "$(weights_key "${r}")"; done
  echo "weights: all four ranks copied, verified and checked"
}

# ---------------------------------------------------------------- weights from a full checkpoint (CHECKPOINT)
# ck_start RANK...: on the checkpoint's host, in a new temporary folder: checkpoint.py check (refuses a folder that is
# not the pinned revision: file list, sizes, Hugging Face hashes), then checkpoint.py serve on its fabric address(es)
# (those in the ranks' fabric subnets, or CHECKPOINT_ADDR), a free ephemeral port unless CHECKPOINT_PORT, only to the
# four ranks' fabric addresses; each RANK must reach it. Sets ck_srv (ADDR:PORT, one per address; rank r takes
# r mod their count). ck_stop (beside the EXIT trap above) ends the server and removes the folder.
ck_start() {
  local r d a allow="" addrs out
  echo "weights: CHECKPOINT $(ck_where): checking it against Inkling-2-Small at the pinned revision (weights/checkpoint-files.tsv)"
  d=$(ck_sh 'mktemp -d "${TMPDIR:-/tmp}/inkling2-checkpoint.XXXXXX"') && [ -n "${d}" ] || die "weights: cannot make a temporary folder on ${ck_host:-this machine}"
  ck_dir=${d}
  { ck_sh "cat > $(q "${d}/checkpoint.py")" < "${here}/weights/checkpoint.py" &&
    ck_sh "cat > $(q "${d}/checkpoint-files.tsv")" < "${here}/weights/checkpoint-files.tsv" &&
    rq "${HOSTS[0]}" "cat $(q "${ROOT}/tools/inkling2_rank_copy.py")" | ck_sh "cat > $(q "${d}/inkling2_rank_copy.py")"; } ||
    die "weights: cannot copy the checkpoint tools to ${ck_host:-this machine}"
  ck_sh "python3 $(q "${d}/checkpoint.py") check $(q "${ck_path}") --cache \"\${HOME}/.cache/inkling2-setup/checkpoint-hashes.json\"" |
    sed 's/^/  /'
  [ "${PIPESTATUS[0]}" = 0 ] || die "weights: CHECKPOINT $(ck_where) refused (above); nothing was copied. Fix the folder, or leave CHECKPOINT empty to fetch from Hugging Face"
  for r in 0 1 2 3; do
    a=$(rq "${HOSTS[$r]}" "for d in ${FABRIC_DEVS[*]}; do ip -4 -o address show dev \$d; done" | awk '{print $4}' | cut -d/ -f1 | tr '\n' ',')
    [ -n "${a}" ] || die "weights: rank ${r} (${HOSTS[$r]}) has no IPv4 address on ${FABRIC_DEVS[*]}"
    allow+=${a}
  done
  allow=${allow%,}
  if [ -n "${CHECKPOINT_ADDR}" ]; then addrs=${CHECKPOINT_ADDR}
  else addrs=$(ck_sh "python3 $(q "${d}/checkpoint.py") addrs --peers ${allow}" | tr '\n' ','); addrs=${addrs%,}; fi
  [ -n "${addrs}" ] || die "weights: ${ck_host:-this machine} has no IPv4 address in the ranks' fabric subnets (ranks: ${allow}); set CHECKPOINT_ADDR"
  ck_sh "cd $(q "${d}") || exit 1; setsid nohup python3 checkpoint.py serve $(q "${ck_path}") --bind $(q "${addrs}") --allow $(q "${allow}") \
    --port $(q "${CHECKPOINT_PORT}") > serve.log 2>&1 < /dev/null & echo \$! > serve.pid" || die "weights: cannot start the checkpoint server on ${ck_host:-this machine}"
  for _ in $(seq 1 40); do
    out=$(ck_sh "cat $(q "${d}/serve.log")")
    case ${out} in *$'\n'ready*|ready*) break;; esac
    sleep 0.5
  done
  printf '%s\n' "${out}" | grep '^ready' > /dev/null || die "weights: the checkpoint server did not start on ${ck_host:-this machine}: $(printf '%s' "${out}" | tail -3)"
  read -r -a ck_srv <<< "$(printf '%s\n' "${out}" | awk '/^listening /{print $2}' | tr '\n' ' ')"
  [ ${#ck_srv[@]} -gt 0 ] || die "weights: the checkpoint server names no address: ${out}"
  for r in "$@"; do
    a=${ck_srv[$((r % ${#ck_srv[@]}))]}
    out=$(rq "${HOSTS[$r]}" "python3 - connect ${a%:*} ${a##*:}" < "${here}/setup/netprobe.py" 2>&1) ||
      die "weights: rank ${r} (${HOSTS[$r]}) cannot reach the checkpoint server at ${a}: ${out} (a firewall? CHECKPOINT_PORT picks a fixed port)"
  done
  echo "weights: serving $(ck_where) read only on ${ck_srv[*]} to the ranks' fabric addresses until the copies end"
}
# ---------------------------------------------------------------- folds
folds_host() {
  local r=$1 f=""; forced folds && f=--force
  rq "${HOSTS[$r]}" "bash $(q "${ROOT}/recipe/weights/fold-rank.sh") ${r} $(q "${MODEL}") $(q "${ROOT}/bin/tf-inkling2-fold") ${f}"
}
do_folds() {
  if [ "${DSPARK}" != 1 ]; then echo "folds: DSPARK=${DSPARK}, nothing to build"; return 0; fi
  installed
  each_rank folds_host || die "folds: a host failed (the rank folder needs the drafter's tensors: the weights step with DSPARK=1)"
  echo "folds: ${MODEL}/inkling2-dspark-fold.safetensors on all four hosts (built locally from the weights; never shipped)"
}

# ---------------------------------------------------------------- manifest
do_manifest() {
  installed
  local want; want=$(rq "${HOSTS[0]}" "awk '\$2 == \"plan/inkling2-digests-w4.tsv\" {print \$1}' $(q "${ROOT}/.setup/assets.SHA256SUMS")")
  [ -n "${want}" ] || die "manifest: the release's SHA256SUMS on ${HOSTS[0]} names no plan/inkling2-digests-w4.tsv"
  local r got bad=0
  for r in 0 1 2 3; do
    got=$(rq "${HOSTS[$r]}" "sha256sum $(q "${PLAN}/inkling2-digests-w4.tsv") 2>/dev/null" | awk '{print $1}')
    if [ "${got}" = "${want}" ]; then echo "rank ${r} ${HOSTS[$r]}: ${PLAN}/inkling2-digests-w4.tsv ${got:0:16}... OK"
    else echo "rank ${r} ${HOSTS[$r]}: manifest ${got:-missing}, expected ${want}"; bad=1; fi
  done
  [ ${bad} = 0 ] || die "manifest: rerun the assets step"
  echo "manifest: the release's digest manifest on all four hosts (TF_INK2_DIGESTS=${PLAN}/inkling2-digests-w4.tsv)"
}

# ---------------------------------------------------------------- config
master_addr() {
  if [ -n "${MASTER}" ]; then echo "${MASTER}"; return; fi
  rq "${HOSTS[0]}" "ip -4 -o address show dev ${SOCKET_IFNAME}" | awk '{print $4}' | cut -d/ -f1 | head -1
}
do_config() {
  installed
  local m; m=$(master_addr)
  [ -n "${m}" ] || die "config: rank 0 (${HOSTS[0]}) has no IPv4 address on ${SOCKET_IFNAME}; set MASTER"
  serve_config "${m}" > "${serve_conf}.new" || die "config: cannot write ${serve_conf}.new" # setup/lib.sh: every value %q
  bash -n "${serve_conf}.new" || die "config: ${serve_conf}.new does not parse"
  mv -f "${serve_conf}.new" "${serve_conf}" || die "config: cannot replace ${serve_conf}"
  local r
  for r in 0 1 2 3; do rq "${HOSTS[$r]}" "cat > $(q "${ROOT}/recipe/config.sh")" < "${serve_conf}" || die "config: copy to ${HOSTS[$r]} failed"; done
  echo "config: ${serve_conf} (MASTER ${m}:${NCCL_PORT} on ${SOCKET_IFNAME}; release settings from release-defaults.sh: CONTEXT ${CONTEXT}, --parallel ${PARALLEL} (MAX_STREAMS ${MAX_STREAMS}), DSpark ${DSPARK}, RoCE wire ${ROCE_MULTI})"
  echo "        copied to ${ROOT}/recipe/config.sh on every host"
  if [ ${container} = 1 ]; then
    for r in 0 1 2 3; do ctr_write "${r}" || die "config: cannot write ${ROOT}/container on ${HOSTS[$r]}"; done
    echo "config: ${ROOT}/container/{compose.yaml,.env,rank.env} on every host (image $(img))"
  fi
}

# ---------------------------------------------------------------- image
img() { echo "inkling2-sparks:${short}"; }
do_image() {
  installed
  local h0=${HOSTS[0]} image; image=$(img)
  local have; have=$(rq "${h0}" "${DK} \$dk image inspect -f '{{index .Config.Labels \"inkling2.assets\"}} {{index .Config.Labels \"inkling2.base\"}}' ${image} 2>/dev/null")
  if [ "${have}" = "${assets_key} ${BASE_IMAGE}" ] && ! forced image; then echo "image: ${image} already built on ${h0} (assets ${assets_key})"
  else
    echo "image: building ${image} on ${h0} (base ${BASE_IMAGE})"
    rq "${h0}" "set -e; ${DK} c=$(q "${ROOT}/.setup/image-ctx"); rm -rf \"\$c\"; mkdir -p \"\$c\"; cd $(q "${ROOT}");
      cp -a bin lib doc VERSION \"\$c\"/; cp recipe/container/Dockerfile \"\$c\"/;
      cd \"\$c\" && find bin lib doc VERSION -type f | LC_ALL=C sort | xargs sha256sum > SHA256SUMS;
      \$dk build --platform linux/arm64 --build-arg BASE_IMAGE=${BASE_IMAGE} --build-arg RELEASE=${short} \
        --build-arg ASSETS=${assets_key} -t ${image} . > build.log 2>&1 || { tail -n 30 build.log; exit 1; }; tail -n 4 build.log; cd /; rm -rf \"\$c\"" ||
      die "image: build failed on ${h0}"
  fi
  # the same image on every host: the same layer contents and labels (image ids differ between Docker's classic
  # store and the containerd store, so they are not compared)
  local sig='{{json .RootFS.Layers}} {{index .Config.Labels "inkling2.assets"}}'
  local id0 r id; id0=$(rq "${h0}" "${DK} \$dk image inspect -f '${sig}' ${image} | sha256sum | cut -c1-16") || die "image: ${image} missing on ${h0}"
  for r in 1 2 3; do
    id=$(rq "${HOSTS[$r]}" "${DK} \$dk image inspect -f '${sig}' ${image} 2>/dev/null | sha256sum | cut -c1-16")
    if [ "${id}" = "${id0}" ]; then echo "image: ${HOSTS[$r]} already has ${image}"; continue; fi
    echo "image: copying ${image} to ${HOSTS[$r]} (docker save | docker load through this machine)"
    rq "${h0}" "${DK} \$dk save ${image} | gzip -1" | rq "${HOSTS[$r]}" "${DK} gunzip | \$dk load" | tail -n 1
    id=$(rq "${HOSTS[$r]}" "${DK} \$dk image inspect -f '${sig}' ${image} 2>/dev/null | sha256sum | cut -c1-16")
    [ "${id}" = "${id0}" ] || die "image: ${HOSTS[$r]}'s ${image} differs from rank 0's (layers and labels ${id} against ${id0})"
  done
  echo "image: ${image} on all four hosts (layers and labels ${id0})"
}

# ---------------------------------------------------------------- start / stop / status / smoke / check
busy_host() { # prints this host's inkling2 processes that are not this install's own server
  local r=$1
  rq "${HOSTS[$r]}" "own=\$(cat $(q "${RUN}/rank-${r}.pid") 2>/dev/null);
    pgrep -a -f '(^|/)(tensorfold-native|tf-inkling2-[a-z]+)( |\$)' | grep -v -e pgrep -e 'bash -c' | while read -r p rest; do
      [ \"\$p\" = \"\$own\" ] && continue
      case \$rest in /opt/inkling2/bin/*) [ ${container} = 1 ] && continue;; esac
      echo \"  \${p} \${rest}\" | cut -c1-160; done"
}
serving() { # 0 if rank 0 answers HTTP (401: started with an API key, and up)
  local probe=${HTTP_HOST}; [ "${probe}" = 0.0.0.0 ] && probe=127.0.0.1
  local code; code=$(rq "${HOSTS[0]}" "curl -s -o /dev/null -w '%{http_code}' -m 5 http://${probe}:${HTTP_PORT}/v1/models")
  [ "${code}" = 200 ] || [ "${code}" = 401 ]
}
ctr_running() { rq "${HOSTS[$1]}" "${DK} [ \"\$(\$dk inspect -f '{{.State.Running}}' inkling2-rank$1 2>/dev/null)\" = true ]"; }
ctr_write() {
  local r=$1 uid gid
  uid=$(rq "${HOSTS[$r]}" "id -u"); gid=$(rq "${HOSTS[$r]}" "id -g")
  rq "${HOSTS[$r]}" "set -e; d=$(q "${ROOT}/container"); mkdir -p \"\$d\" $(q "${RUN}"); cp $(q "${ROOT}/recipe/container/compose.yaml") \"\$d\"/;
    ENV_LIBS=/opt/inkling2/lib ENV_BIN_DIR=/opt/inkling2/bin bash $(q "${ROOT}/recipe/rank.sh") $(q "${ROOT}/recipe/config.sh") env ${r} > \"\$d/rank.env\";
    printf '%s\n' 'RANK=${r}' 'IMAGE=$(img)' 'MODEL=${MODEL}' 'PLAN=${PLAN}' 'DATA=${DATA}' 'RUN=${RUN}' \
      'HTTP_HOST=${HTTP_HOST}' 'HTTP_PORT=${HTTP_PORT}' 'CONTEXT=${CONTEXT}' 'PARALLEL=${PARALLEL}' \
      'PROMPT_CACHE_GIB=${PROMPT_CACHE_GIB}' 'DECODE_SHARE=${DECODE_SHARE}' \"EXTRA_ARGS='${EXTRA_ARGS}'\" 'RUN_UID=${uid}' 'RUN_GID=${gid}' > \"\$d/.env\"; mkdir -p $(q "${DATA}")"
}
do_start() {
  [ ${container} = 1 ] && installed
  local r b busy=0
  if serving; then
    local all=1
    for r in 0 1 2 3; do
      if [ ${container} = 1 ]; then ctr_running "${r}" || all=0
      else rq "${HOSTS[$r]}" "test -f $(q "${RUN}/rank-${r}.pid") && kill -0 \$(cat $(q "${RUN}/rank-${r}.pid")) 2>/dev/null" || all=0; fi
    done
    if [ ${all} = 1 ] && ! forced start; then echo "start: already serving on ${HOSTS[0]} ${HTTP_HOST}:${HTTP_PORT}"; return 0; fi
    die "start: something already answers on ${HOSTS[0]} ${HTTP_HOST}:${HTTP_PORT}; stop it first"
  fi
  for r in 0 1 2 3; do
    b=$(busy_host "${r}")
    if [ -n "${b}" ]; then echo "rank ${r} ${HOSTS[$r]}: another inkling2 process holds the GPU:"; echo "${b}"; busy=1; fi
  done
  [ ${busy} = 0 ] || die "start: stop the processes above first (they are not this install's)"
  if [ "${DSPARK}" = 1 ]; then # DSpark needs the folds file beside the weights (TF_INK2_DSPARK_FOLD overrides the path)
    local fold=${MODEL}/inkling2-dspark-fold.safetensors miss=0
    for r in 0 1 2 3; do
      rq "${HOSTS[$r]}" "test -s $(q "${fold}")" || { echo "rank ${r} ${HOSTS[$r]}: no ${fold} (DSPARK=1)"; miss=1; }
    done
    [ ${miss} = 0 ] || die "start: DSpark needs the folds file on every rank: ./setup.sh ${conf##*/} folds; or set DSPARK=0 in ${conf##*/} and rerun the config step"
  fi
  if [ ${container} = 0 ]; then
    bash "${here}/start.sh" "${serve_conf}" || die "start: see the rank logs (${RUN}/rank-R.log); ./setup.sh $(basename "${conf}") stop"
    return 0
  fi
  # shellcheck source=/dev/null
  (source "${serve_conf}"; http_open_warning) # the container path's warning (start.sh warns on the host path)
  for r in 0 1 2 3; do ctr_write "${r}" || die "start: cannot write ${ROOT}/container on ${HOSTS[$r]}"; done
  for r in 0 1 2 3; do
    rq "${HOSTS[$r]}" "${DK} cd $(q "${ROOT}/container") && \$dk rm -f inkling2-rank${r} > /dev/null 2>&1; \$dk compose up -d 2>&1 | tail -n 2" ||
      die "start: rank ${r}'s container did not start on ${HOSTS[$r]}"
  done
  local t0=${SECONDS} following
  while :; do
    if serving; then
      following=0
      for r in 1 2 3; do rq "${HOSTS[$r]}" "${DK} \$dk logs inkling2-rank${r} 2>&1 | grep -q 'following rank 0'" && following=$((following + 1)); done
      if [ ${following} = 3 ]; then echo "ready after $((SECONDS - t0)) s: http://${HTTP_HOST}:${HTTP_PORT} on ${HOSTS[0]} (containers inkling2-rank0..3, $(img))"; bash "${here}/warmup.sh" "${serve_conf}"; break; fi
    fi
    for r in 0 1 2 3; do
      if ! ctr_running "${r}"; then
        echo "rank ${r}'s container ended before the server was ready:" >&2
        rq "${HOSTS[$r]}" "${DK} \$dk logs --tail 20 inkling2-rank${r}" >&2
        die "start failed; ./setup.sh $(basename "${conf}") --container stop"
      fi
    done
    (( SECONDS - t0 > READY_WAIT )) && die "start: not ready after ${READY_WAIT} s"
    sleep 10
  done
}
do_stop() {
  local r
  if [ ${container} = 0 ]; then bash "${here}/stop.sh" "${serve_conf}"; return; fi
  for r in 0 1 2 3; do
    rq "${HOSTS[$r]}" "${DK} cd $(q "${ROOT}/container") 2>/dev/null && \$dk compose down --timeout 60 2>&1 | tail -n 1; echo \"rank ${r} ${HOSTS[$r]}: stopped\""
  done
}
do_status() {
  local r rc=0
  if [ ${container} = 0 ]; then bash "${here}/status.sh" "${serve_conf}" || echo "status: a check warned (above)"; return 0; fi
  for r in 0 1 2 3; do
    rq "${HOSTS[$r]}" "${DK} s=\$(\$dk ps -a --filter name=^inkling2-rank${r}\$ --format '{{.Status}} ({{.Image}})'); echo \"rank ${r} on \$(hostname): container \${s:-none}; MemAvailable \$(awk '/MemAvailable/{printf \"%.1f GiB\", \$2 / 1048576}' /proc/meminfo)\";
      \$dk logs --tail 3 inkling2-rank${r} 2>&1 | cut -c1-240"
  done
  echo "fabric:"
  for r in 0 1 2 3; do rq "${HOSTS[$r]}" "bash $(q "${ROOT}/recipe/rank.sh") $(q "${ROOT}/recipe/config.sh") fabric ${r}" || rc=1; done
  [ ${rc} = 0 ] || echo "status: a check warned (above)"
  return 0
}
do_smoke() {
  local probe=${HTTP_HOST} url rc=0; [ "${probe}" = 0.0.0.0 ] && probe=127.0.0.1
  url=http://${probe}:${HTTP_PORT}
  serving || die "smoke: nothing answers at ${url} on ${HOSTS[0]}; run the start step"
  rq "${HOSTS[0]}" "cd $(q "${ROOT}") && python3 recipe/smoke.py ${url} smoke --long-tokens ${SMOKE_LONG_TOKENS} 2>&1 | tee serve/smoke.txt | cut -c1-300; exit \${PIPESTATUS[0]}" || rc=1
  if rq "${HOSTS[0]}" "test -f $(q "${DATA}/decode.json") && test -f $(q "${DATA}/chat_short.ids.json")"; then
    echo "--- oracle: greedy replies against the oracle's, token for token"
    rq "${HOSTS[0]}" "cd $(q "${ROOT}") && python3 recipe/smoke.py ${url} oracle data/decode.json chat_short=data/chat_short.ids.json \
      \$(test -f data/tool_call.ids.json && echo tool_call=data/tool_call.ids.json) 2>&1 | tee serve/oracle.txt | cut -c1-300; exit \${PIPESTATUS[0]}" || rc=1
  else
    echo "--- oracle: skipped (the release has no data/decode.json)"
  fi
  [ ${rc} = 0 ] || die "smoke: a check failed (${RUN}/smoke.txt, ${RUN}/oracle.txt on ${HOSTS[0]})"
  echo "smoke: PASS"
}
check_ctr_rank() {
  local r=$1
  rq "${HOSTS[$r]}" "${DK} cd $(q "${ROOT}/container") && timeout ${CHECK_SECS:-1800} \$dk compose run --rm --no-deps -T --entrypoint /opt/inkling2/bin/tf-inkling2-decode rank \
    $(q "${MODEL}") --ids $(q "${DATA}/long_4000.safetensors") --prompt ${CHECK_PROMPT} --steps ${CHECK_STEPS} --world 4 --rank ${r} \
    > $(q "${RUN}/check-decode-${r}.log") 2>&1; rc=\$?; echo \"rank ${r} on \$(hostname): exit \${rc}, log ${RUN}/check-decode-${r}.log\"; exit \${rc}"
}
do_check() {
  [ ${container} = 1 ] && installed
  rq "${HOSTS[0]}" "test -f $(q "${DATA}/long_4000.safetensors")" || die "check: no ${DATA}/long_4000.safetensors (the release's data/)"
  if serving; then die "check: the server is running; stop it first (setup.sh $(basename "${conf}") $([ ${container} = 1 ] && echo '--container ')stop)"; fi
  local r b busy=0
  for r in 0 1 2 3; do b=$(busy_host "${r}"); [ -z "${b}" ] || { echo "rank ${r}: busy:"; echo "${b}"; busy=1; }; done
  [ ${busy} = 0 ] || die "check: another inkling2 process holds a GPU"
  if [ ${container} = 0 ]; then
    bash "${here}/check.sh" "${serve_conf}" "${DATA}/long_4000.safetensors" "${CHECK_PROMPT}" "${CHECK_STEPS}"; return
  fi
  for r in 0 1 2 3; do ctr_write "${r}" || die "check: cannot write ${ROOT}/container on ${HOSTS[$r]}"; done
  local t0=${SECONDS} rc=0
  each_rank check_ctr_rank || rc=1
  echo "decode check: all ranks done in $((SECONDS - t0)) s"
  rq "${HOSTS[0]}" "grep -hE 'PASS|FAIL' $(q "${RUN}/check-decode-0.log") | tail -3"
  [ ${rc} = 0 ] && rq "${HOSTS[0]}" "grep -q PASS $(q "${RUN}/check-decode-0.log")"
}

# ---------------------------------------------------------------- run
mode_word=host; [ ${container} = 1 ] && mode_word=container
printf '%sInkling-2-Small TP4 setup%s: %s (%s), %s install in %s, steps: %s\n' "${B}" "${Z}" "$(basename "${conf}")" \
  "${HOSTS[*]}" "${mode_word}" "${ROOT}" "${steps[*]}"
for s in "${steps[@]}"; do
  step "${s}"
  "do_${s}" || die "${s} failed"
done
printf '\n%sdone%s (%s)\n' "${B}" "${Z}" "$(date -u +%T)"
