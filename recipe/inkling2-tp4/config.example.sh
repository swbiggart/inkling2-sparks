# Inkling-2-Small at TP4 on four GB10s: copy to a private file, fill in, and pass it to start.sh / stop.sh /
# status.sh / check.sh / weights.sh. Sourced by bash on this machine (the one running the scripts) and on every rank
# host, after release-defaults.sh. That file holds the release's settings (CONTEXT, PARALLEL, the memory reserve, the
# prompt cache, DSpark, XROWS, the RoCE wire, the folder name); this one holds the site's own values and may override
# any of them by setting the variable again. Every TF_INK2_* setting and every serve flag must be the same on all
# four ranks: they go into the rank fingerprint, and ranks that differ refuse to start.

# ssh names of the four hosts, in rank order (rank 0 serves HTTP).
HOSTS=(rank0-host rank1-host rank2-host rank3-host)

# Rank 0's address on the fabric's first port and the NCCL bootstrap port (NCCL_COMM_ID). The liveness lines use
# NCCL_PORT + 1, so both ports must reach rank 0 from the other ranks. Bootstrapping over the fabric keeps NCCL clear
# of LAN firewalls: besides these two ports its listeners bind random ephemeral ports on SOCKET_IFNAME, and a rule
# that drops some of them on the LAN hangs a start at random. (setup.sh writes this file from hosts.conf.)
MASTER=198.18.0.1
NCCL_PORT=29551
# The interface NCCL bootstraps over (rank 0's MASTER is on it); data goes over the RoCE HCAs below.
SOCKET_IFNAME=enp1s0f0np0

# The folder on every host that holds the files below (same path on all four).
ROOT=/path/on/hosts/inkling2
BIN=$ROOT/bin/tensorfold-native     # the same aarch64 sm_121 build on all four (the ranks compare executable hashes)
DECODE_BIN=$ROOT/bin/tf-inkling2-decode   # the decode check (check.sh decode)
PLAN=$ROOT/plan                     # the release's plan files; TF_INK2_DIGESTS is PLAN/inkling2-digests-w4.tsv
MODEL=$ROOT/$MODEL_DIRNAME          # this rank's folder (weights.sh fills it): split experts, model-split by default
LIBS=$ROOT/lib                      # libnccl.so.2, libcublas.so.13, libcublasLt.so.13 (the release's lib/)
GRAPH=""                            # per-rank NCCL graph file ({R} becomes the rank); empty on the switch
RUN=$ROOT/serve                     # logs and pid files

# HTTP on rank 0. Keep 127.0.0.1 unless the endpoint should be reachable from the network.
HTTP_HOST=127.0.0.1
HTTP_PORT=8890
# CONTEXT=, PARALLEL=, MAX_STREAMS=, PROMPT_CACHE_GIB=, MEMORY_RESERVE_GIB=: release-defaults.sh (set one here only to differ).
# CONTEXT is the one knob: the streams that fit at it run at once (PARALLEL=auto), at most MAX_STREAMS (4; up to 8);
# CONTEXT is also the largest request: CONTEXT=524288 serves requests of up to 524,288 tokens, and each such request
# takes one stream (DSpark by depth); CONTEXT=1048576 (1M) serves one stream, DSpark by depth too.
EXTRA_ARGS=""                       # more serve flags, the same on every rank

# The fabric ports status.sh checks on every host (fabric/host/): the switch-facing port's two PCIe halves.
FABRIC_DEVS=(enp1s0f0np0 enP2p1s0f0np0)
FABRIC_HCAS=(rocep1s0f0 roceP2p1s0f0)
FABRIC_SPEED=200000                 # Mb/s each netdev should report (a 2x100G splitter shows 100000)
FABRIC_MTU=9000
FABRIC_GID=3

# NCCL over the switch: both PCIe halves of each host's one 200G port, RoCE v2 (GID 3), no graph file. The settings
# follow MiaAI-Lab's four-Spark DeepSeek recipe (scripts/nodes.sh) and, for cuMem and runtime connect, upstream's
# docs/recipes/glm-5.3-flash-cuda.md; see THIRD_PARTY_NOTICES.md.
NCCL_ENV=(
  NCCL_IB_DISABLE=0
  NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
  NCCL_IB_GID_INDEX=3 NCCL_IB_MERGE_NICS=0 NCCL_CROSS_NIC=1
  NCCL_IB_SUBNET_AWARE_ROUTING=1 NCCL_IB_SUBNET_PREFIX_LEN=24
  NCCL_ALGO=Ring NCCL_CUMEM_ENABLE=1 NCCL_RUNTIME_CONNECT=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
  NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4
  NCCL_DEBUG=WARN
)

# EXPERIMENTAL, not validated for this release: the switchless ring (fabric/ring-experimental/). Hosts in ring
# order, all four ports, a per-rank graph file. Replace the NCCL_ENV above and GRAPH with:
# GRAPH=$ROOT/nccl/graph-rank{R}.xml
# NCCL_ENV=(
#   NCCL_IB_DISABLE=0
#   NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1
#   NCCL_IB_GID_INDEX=3 NCCL_IB_MERGE_NICS=0 NCCL_CROSS_NIC=1
#   NCCL_IB_SUBNET_AWARE_ROUTING=1 NCCL_IB_SUBNET_PREFIX_LEN=24
#   NCCL_ALGO=Ring NCCL_CUMEM_ENABLE=1 NCCL_RUNTIME_CONNECT=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
#   NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4
#   NCCL_DEBUG=WARN
# )
# FABRIC_DEVS=(enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1)
# FABRIC_HCAS=(rocep1s0f0 rocep1s0f1 roceP2p1s0f0 roceP2p1s0f1)

# The release's engine settings (checkpoint mode, split experts, the digest manifest, DSpark on, the RoCE mesh wire
# with tp_mailbox.fatbin, the memory reserve, XROWS) are release-defaults.sh's, passed to every rank by rank.sh.
# EXTRA_ENV adds to them, or overrides one (the last entry for a name wins); the same on all four ranks.
EXTRA_ENV=(
)

# Other switches, off by default, to add to EXTRA_ENV:
#   TF_ROCE=1                     upstream's RoCE gather instead of the mesh wire (set ROCE_MULTI=0 as well: use one of
#                                 the two); gathers up to TF_ROCE_MAX_KB (default 512) a rank; HCAs from NCCL_IB_HCA
#   (on the experimental ring, instead of the mesh wire: ROCE_MULTI=0 and TF_INK2_ROCE_MULTI=1 TF_TP_ROCE_RELAY=1
#    TF_TP_ROCE_HCA=rocep1s0f0:3,rocep1s0f1:3 TF_TP_MASTER=$MASTER TF_TP_PORT=29640 TF_TP_KERNEL_IMAGE=$ROOT/bin/tp_mailbox.fatbin)
#   DSpark off: DSPARK=0 in release-defaults.sh or here.
