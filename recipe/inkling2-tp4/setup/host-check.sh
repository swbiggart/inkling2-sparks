#!/usr/bin/env bash
# host-check.sh: setup.sh's preflight on one host (run over ssh as `bash -s`; settings come in the environment).
#   RANK ROOT NEED_GIB MODE(host|container) FABRIC_DEVS FABRIC_HCAS FABRIC_SPEED FABRIC_MTU SOCKET_IFNAME MODEL_DIR
# Prints one line a check: "OK", "WARN" or "FAIL", then the detail, and "INFO key=value" lines setup.sh compares
# across hosts. Exit 2 if any check failed, 1 if any warned, else 0. Changes nothing on the host.
set -uo pipefail
r=${RANK:?}; root=${ROOT:?}; need=${NEED_GIB:-100}; mode=${MODE:-host}
read -r -a devs <<< "${FABRIC_DEVS:?}"
read -r -a hcas <<< "${FABRIC_HCAS:?}"
speed_want=${FABRIC_SPEED:-200000}; mtu_want=${FABRIC_MTU:-9000}; sockif=${SOCKET_IFNAME:-${devs[0]}}
fails=0; warns=0
say() { # LEVEL CHECK DETAIL
  printf '  rank %s %-10s %-4s %s\n' "${r}" "$2" "$1" "$3"
  case $1 in FAIL) fails=$((fails + 1));; WARN) warns=$((warns + 1));; esac
}
info() { echo "INFO $1"; }

# GPU, driver, CUDA
if q=$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null | head -1) && [ -n "${q}" ]; then
  IFS=',' read -r gname gcap gdrv <<< "${q}"
  gname=$(echo "${gname}" | xargs); gcap=$(echo "${gcap}" | xargs); gdrv=$(echo "${gdrv}" | xargs)
  cudav=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')
  info "driver=${gdrv}"; info "cuda=${cudav}"
  case ${gname} in *GB10*) say OK gpu "${gname}, compute ${gcap}";; *) say FAIL gpu "${gname} (compute ${gcap}): the release binaries are built for GB10 (sm_121)";; esac
  [ "${gcap}" = 12.1 ] || say FAIL gpu "compute capability ${gcap}, the binaries need 12.1 (sm_121)"
  if [ "${gdrv%%.*}" -ge 580 ] 2>/dev/null && [ "${cudav%%.*}" -ge 13 ] 2>/dev/null; then
    say OK driver "${gdrv} (CUDA ${cudav} driver API; the build's CUDA 13.3 user space runs on any 580+ driver)"
  else
    say FAIL driver "${gdrv} (CUDA ${cudav}): needs driver 580 or later (CUDA 13)"
  fi
else
  say FAIL gpu "nvidia-smi found no GPU"
fi
if ldconfig -p 2>/dev/null | grep 'libcuda.so.1 ' > /dev/null; then say OK libcuda "$(ldconfig -p | awk '/libcuda.so.1 /{print $NF; exit}')"
else say FAIL libcuda "libcuda.so.1 not in the loader cache (the NVIDIA driver's user-space library)"; fi
if [ "${mode}" = host ]; then
  if ldconfig -p 2>/dev/null | grep 'libibverbs.so.1 ' > /dev/null; then say OK ibverbs "libibverbs.so.1 present (NCCL's RoCE transport)"
  else say FAIL ibverbs "libibverbs.so.1 missing: install rdma-core's libibverbs1 and ibverbs-providers"; fi
fi
if [ -f "${root}/lib/SHA256SUMS" ]; then
  if (cd "${root}/lib" && sha256sum -c --quiet SHA256SUMS > /dev/null 2>&1); then
    nccl=$(tr -d '\000' < "${root}/lib/libnccl.so.2" | grep -a -o -m1 'NCCL version [0-9.]*+cuda[0-9.]*' | head -1)
    say OK libs "${root}/lib: ${nccl:-libnccl.so.2}, libcublas 13 (checked against lib/SHA256SUMS)"
  else say WARN libs "${root}/lib does not match its SHA256SUMS (the libs step fetches them again)"; fi
fi

# python3 (the weight copy tool and smoke.py; standard library only)
if pv=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null); then
  if python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))'; then say OK python "python3 ${pv}"; else say FAIL python "python3 ${pv}, needs 3.9 or later"; fi
else say FAIL python "no python3"; fi

# disk: the nearest existing folder of ROOT (and of the weights folder when it is fetched elsewhere)
for d in "${root}" ${MODEL_DIR:+"${MODEL_DIR}"}; do
  p=${d}; while [ ! -d "${p}" ] && [ "${p}" != / ]; do p=$(dirname "${p}"); done
  avail=$(df -Pk "${p}" | awk 'NR == 2 {printf "%d", $4 / 1048576}')
  mnt=$(df -Pk "${p}" | awk 'NR == 2 {print $6}')
  info "disk_gib=${avail}"
  if [ "${avail}" -ge "${need}" ]; then say OK disk "${avail} GiB free on ${mnt} (need ${need} GiB for ${d})"
  else say FAIL disk "${avail} GiB free on ${mnt}, need ${need} GiB for ${d}"; fi
  [ -n "${MODEL_DIR:-}" ] || break
done

# memory
ma=$(awk '/MemAvailable/{printf "%d", $2 / 1048576}' /proc/meminfo)
info "memavail_gib=${ma}"
if [ "${ma}" -ge 100 ]; then say OK memory "MemAvailable ${ma} GiB"
else say WARN memory "MemAvailable ${ma} GiB: a rank needs about 95 GiB (weights plus caches); stop other GPU work before start"; fi
others=$(pgrep -a -f '(^|/)(tensorfold-native|tf-inkling2-[a-z]+)( |$)' 2>/dev/null | grep -v -e pgrep -e host-check | cut -c1-160)
if [ -n "${others}" ]; then
  say WARN gpu-busy "another inkling2 process runs here (start and check refuse until it ends):"
  printf '%s\n' "${others}" | sed 's/^/      /'
fi

# fabric ports: speed, MTU, carrier, address, HCA state
for i in "${!devs[@]}"; do
  d=${devs[$i]}; n=/sys/class/net/${d}; hca=${hcas[$i]:-}
  if [ ! -e "${n}" ]; then say FAIL fabric "${d}: no such netdev (ip -br link)"; continue; fi
  carrier=$(cat "${n}/carrier" 2>/dev/null || echo 0); speed=$(cat "${n}/speed" 2>/dev/null || echo unknown); mtu=$(cat "${n}/mtu")
  addr=$(ip -4 -o address show dev "${d}" 2>/dev/null | awk '{print $4}' | head -1)
  state=$(awk '{print $2}' "/sys/class/infiniband/${hca}/ports/1/state" 2>/dev/null)
  info "addr${i}=${addr}"
  notes=()
  [ "${carrier}" = 1 ] || notes+=("no carrier")
  [ "${speed}" = "${speed_want}" ] || notes+=("speed ${speed} Mb/s, want ${speed_want}")
  [ "${mtu}" = "${mtu_want}" ] || notes+=("MTU ${mtu}, want ${mtu_want}")
  [ -n "${addr}" ] || notes+=("no IPv4 address (fabric/host/install.sh, or setup.sh --fabric-install)")
  [ "${state}" = ACTIVE ] || notes+=("HCA ${hca:-?} ${state:-not found}")
  line="${d} (${hca:-?}): ${speed} Mb/s, MTU ${mtu}, ${addr:-no address}, HCA ${state:-?}"
  if [ ${#notes[@]} = 0 ]; then say OK fabric "${line}"; else say FAIL fabric "${line}: $(IFS=';'; echo "${notes[*]}")"; fi
done
sa=$(ip -4 -o address show dev "${sockif}" 2>/dev/null | awk '{print $4}' | head -1)
info "sockaddr=${sa}"
[ -n "${sa}" ] || say FAIL bootstrap "${sockif} (SOCKET_IFNAME, NCCL's bootstrap interface) has no IPv4 address"
# the NIC re-probe detector: a runtime re-probe (after a cable change) caps transmit at ~13 Gb/s a PCIe half until reboot
# (the ConnectX driver logs "insufficient power on the PCIe slot" for every port at its boot probe, seconds after boot;
# the same line later than RE_PROBE_AFTER seconds of uptime means the port was probed again at run time)
kl=$(journalctl -k -b -q --no-pager -o short-monotonic 2>/dev/null) && [ -n "${kl}" ] || kl=$(dmesg 2>/dev/null) || kl=""
if [ -n "${kl}" ]; then
  pat='PCIe slot power capability was not advertised|insufficient power on the PCIe slot'
  boot=$(printf '%s\n' "${kl}" | grep -c -E "${pat}")
  late=$(printf '%s\n' "${kl}" | grep -E "${pat}" | awk -v t="${RE_PROBE_AFTER:-120}" '{ s = $0; sub(/^[^[]*\[ */, "", s); sub(/\].*/, "", s); if (s + 0 > t) n++ } END { print n + 0 }')
  if [ "${late}" -gt 0 ]; then say WARN re-probe "the NIC was re-probed after boot (${late} 'PCIe slot power' kernel lines later than ${RE_PROBE_AFTER:-120} s of uptime): transmit is capped at ~13 Gb/s a half until this host reboots"
  else say OK re-probe "no NIC re-probe since boot (${boot} 'PCIe slot power' lines, all at the boot probe)"; fi
else
  say WARN re-probe "cannot read the kernel log (add the user to adm or systemd-journal); check by hand: journalctl -k -b | grep 'PCIe slot power'"
fi
# firewall rules that drop TCP arriving on the bootstrap interface (NCCL's bootstrap and its listeners' random ports)
fw=""
for f in /etc/nftables.conf /etc/nftables.d/*.nft; do
  [ -r "${f}" ] || continue
  hit=$(grep -n -E "iifname.*\"?${sockif}\"?.*drop" "${f}" 2>/dev/null | head -3)
  [ -n "${hit}" ] && fw+="${f}: ${hit}"$'\n'
done
if [ -n "${fw}" ]; then
  say WARN firewall "rules drop TCP on ${sockif}, the interface NCCL bootstraps over (the port probe below shows whether they hit; bootstrap over the fabric port, the default, avoids them):"
  printf '%s' "${fw}" | sed 's/^/      /'
else
  say OK firewall "no readable nftables rule drops TCP on ${sockif} (the port probe checks it end to end)"
fi
if [ "${mode}" = container ]; then
  if docker info > /dev/null 2>&1; then dk=docker; elif sudo -n docker info > /dev/null 2>&1; then dk="sudo -n docker"; else dk=""; fi
  if [ -z "${dk}" ]; then say FAIL docker "cannot reach the Docker daemon (as this user or with sudo -n)"
  else
    info "docker=${dk}"
    say OK docker "$(${dk} version --format '{{.Server.Version}}' 2>/dev/null) (${dk})"
    cv=$(${dk} compose version --short 2>/dev/null) && say OK compose "Compose ${cv}" || say FAIL compose "docker compose (v2) is missing"
    if command -v nvidia-container-runtime-hook > /dev/null || command -v nvidia-ctk > /dev/null; then say OK gpu-ctr "NVIDIA Container Toolkit present"
    else say FAIL gpu-ctr "the NVIDIA Container Toolkit is missing (containers cannot see the GPU)"; fi
    [ -d /dev/infiniband ] && say OK rdma-dev "/dev/infiniband present" || say FAIL rdma-dev "/dev/infiniband missing"
  fi
fi
command -v ib_write_bw > /dev/null && say OK perftest "ib_write_bw present (status --bw)" || say OK perftest "ib_write_bw not installed (only status --bw needs it)"
[ ${fails} = 0 ] || exit 2
[ ${warns} = 0 ] || exit 1
exit 0
