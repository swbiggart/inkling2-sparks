# The release's settings, in one place. Sourced (before the config) by rank.sh on every host and by setup.sh,
# start.sh, stop.sh, status.sh, check.sh and weights.sh on the machine that runs them; setup.sh and start.sh copy this
# file next to rank.sh on every host. A private config (config.example.sh, hosts.conf.example) only needs the
# site's own values (hosts, folder, fabric, ports); it can override any variable below by setting it again.
# Every TF_INK2_*, TF_TP_* and serve setting here must be the same on all four ranks: they go into the rank
# fingerprint, and ranks that differ refuse to start.

# --- Weights ----------------------------------------------------------------------------------------------------
# The rank folders (split experts: every rank holds a quarter of each expert) sit in ROOT/MODEL_DIRNAME.
MODEL_DIRNAME=model-split
EXPERTS=split                       # TF_INK2_EXPERTS

# --- Serving ----------------------------------------------------------------------------------------------------
CONTEXT=131072                      # --context: tokens a stream can hold (128k; at most 1048576, the model's)
PARALLEL=auto                       # --parallel: streams at once; auto = as many as fit at CONTEXT, at most MAX_STREAMS,
                                    # counted at start-up from every rank's memory after the reserve, the prompt cache
                                    # and the graph cache (the log says "context C -> N streams (cap M)"); a number is
                                    # served as asked (MAX_STREAMS does not apply), refused if it does not fit.
                                    # CONTEXT -> streams: README, "Context and streams".
MAX_STREAMS=4                       # TF_INK2_MAX_STREAMS: the most streams PARALLEL=auto serves, 1-8 (8: the most the
                                    # exact gates ran together). At CONTEXT=131072 four streams fit with every rank above
                                    # the reserve (docs/BENCHMARKS.md, memory while serving).
MEMORY_RESERVE_GIB=6                # TENSORFOLD_MEMORY_RESERVE_GIB: memory left to the host and other users, each rank
# --prompt-cache-gib, passed explicitly: omitted, the server sizes the cache itself, up to 16 GiB a rank. The cache
# keeps prompt states (bit-exact: a warm resume gives the same output as a cold prefill); 0 turns it off.
PROMPT_CACHE_GIB=1.5                # a 1.5 GiB arena on every rank: about four 32k DSpark states or one 128k (fits beside
                                    # the streams at CONTEXT: admission counts it before the streams)
# Long requests: CONTEXT is the largest request (prompt plus reply) the server takes; the streams share one KV page pool
# and DSpark prices each stream's rounds at its own depth (deep streams draft only while drafts pay). CONTEXT=524288
# serves requests of up to 524,288 tokens, and each such request takes one stream; CONTEXT=1048576 (the model's 1M)
# serves one stream, DSpark by depth too.
# --decode-share: while other streams decode, a long prompt's chunks (512 rows) take this share of the time and
# the decoders the rest (0.5: decoders keep about half their rate, the prompt's time to first token grows ~2.5x).
DECODE_SHARE=0.5

# --- Engine settings --------------------------------------------------------------------------------------------
DSPARK=1                            # TF_INK2_DSPARK: drafting with the checkpoint's drafter (needs weights.sh --drafter)
XROWS=1                             # TF_INK2_XROWS: by-rows layer sums in prompt chunks (the same bits; cold prefill 3-6 %
                                    # faster at 2k-32k); start.sh's warm-up (warmup.sh) takes its first-use cost

# --- Image input ------------------------------------------------------------------------------------------------
VISION=1                            # --vision on every rank: image parts (PNG, JPEG, WebP; data URLs, not image URLs) up
                                    # to 4096 pixels a side and 10,816 image tokens a request, refused above them. 1
                                    # needs rank 0's inkling2-vision.safetensors (weights.sh adds it while VISION=1) and
                                    # the release's lib/ decoders (libjpeg-turbo, libwebp; PNG needs none). 0: text only,
                                    # v0.2's launches.

# --- The RoCE mesh wire (small decode exchanges over RoCE instead of NCCL, on the switch) ----------------------------
ROCE_MULTI=1                        # TF_INK2_ROCE_MULTI: 1 turns the wire on (bin/tp_mailbox.fatbin ships with the assets)
MAILBOX_MAX_KB=640                  # TF_TP_MAILBOX_MAX_KB: gathers up to this many KiB a rank stay on the wire (>= 640)
TP_PORT=29640                       # TF_TP_PORT; TF_TP_PORT + 7 is used too. Both must be free and reach rank 0.
# TF_TP_ROCE_HCA is FABRIC_HCAS with FABRIC_GID, TF_TP_MASTER is MASTER (the fabric address), and
# TF_TP_KERNEL_IMAGE is TP_FATBIN (default BIN's folder / tp_mailbox.fatbin).

# release_settings: checks CONTEXT and MAX_STREAMS. Every script that reads them calls it right after the config. DSPARK
# applies as set at every context: at the model's full 1,048,576 the drafter's caches (about 1 GiB a rank) fit beside
# the stream and the prompt cache, taking pool room kept prompts would have had (the rank with the least room keeps
# about 1.6 GiB spare).
release_settings() {
  case ${CONTEXT} in ''|*[!0-9]*) echo "CONTEXT=${CONTEXT}: a token count (at most 1048576)" >&2; return 1 ;; esac
  # more than 7 digits is past 1048576 (and may overflow the shell's integer test); leading zeros count as digits here
  if [ "${#CONTEXT}" -gt 7 ] || [ "${CONTEXT}" -gt 1048576 ]; then echo "CONTEXT=${CONTEXT}: past the model's 1048576 tokens" >&2; return 1; fi
  case ${MAX_STREAMS} in [1-8]) ;; *) echo "MAX_STREAMS=${MAX_STREAMS}: a stream count from 1 to 8" >&2; return 1 ;; esac
  case ${VISION:-0} in 0|1) ;; *) echo "VISION=${VISION}: 0 (text only) or 1 (image input)" >&2; return 1 ;; esac
}

# http_open_warning: prints a warning (stderr) when HTTP_HOST is not loopback (127.x, ::1, localhost) and no API key is
# configured (EXTRA_ARGS --api-key / --api-key-file, or TENSORFOLD_API_KEY in EXTRA_ENV): the endpoint is then open to
# anyone who can reach it. A warning, not a refusal; rank 0's log has the server's own. start.sh and setup.sh's start
# call it after the config.
http_open_warning() {
  local h=${HTTP_HOST:-127.0.0.1} e
  case $(printf '%s' "${h}" | tr '[:upper:]' '[:lower:]') in localhost|127.*|::1|::ffff:127.*) return 0 ;; esac
  case " ${EXTRA_ARGS:-} " in
    *' --api-key '*|*' --api-key='*|*' --api-key-file '*|*' --api-key-file='*) return 0 ;;
  esac
  for e in ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}; do
    case ${e} in TENSORFOLD_API_KEY=*[![:space:],]*) return 0 ;; esac
  done
  {
    echo "WARNING: HTTP_HOST=${h} is not loopback and no API key is configured: anyone who can reach"
    echo "WARNING: ${HOSTS[0]:-rank 0}:${HTTP_PORT:-8890} can use the model (there is no TLS). Set a key"
    echo "WARNING: (EXTRA_ARGS=\"--api-key-file FILE\"; README, API keys) or keep HTTP_HOST=127.0.0.1. Starting anyway."
  } >&2
}

# release_args: the serve flags the release sets, after --context and --parallel (rank.sh start); VISION=1 adds
# --vision (every rank, so the ranks agree). compose.yaml spells them out (VISION_ARGS in the container's .env).
release_args() { printf '%s\n' "--prompt-cache-gib ${PROMPT_CACHE_GIB} --decode-share ${DECODE_SHARE}$(vision_args)"; }
vision_args() { [ "${VISION:-0}" = 1 ] && printf ' --vision'; return 0; }

# release_env: the release's environment, one VAR=value a line, into the array release_envs. Call it after the
# config (it reads ROOT, PLAN, MASTER, FABRIC_HCAS, FABRIC_GID, BIN). EXTRA_ENV comes after it, so an entry there wins.
release_env() {
  local hca="" h plan=${PLAN:-$ROOT/plan} fatbin=${TP_FATBIN:-$(dirname "$BIN")/tp_mailbox.fatbin}
  for h in "${FABRIC_HCAS[@]}"; do hca+=${hca:+,}${h}:${FABRIC_GID:-3}; done
  release_envs=(
    TF_INK2_MODE=checkpoint
    TF_INK2_EXPERTS="$EXPERTS"
    TF_INK2_DIGESTS="$plan/inkling2-digests-w4.tsv"
    TF_INK2_DSPARK="$DSPARK"
    TF_INK2_XROWS="$XROWS"
    TF_INK2_MAX_STREAMS="$MAX_STREAMS"
    TENSORFOLD_MEMORY_RESERVE_GIB="$MEMORY_RESERVE_GIB"
    # every stream's convolutions and their commits in one launch a site in shared rounds (segs): the same bits
    TF_INK2_FUSE=attn,mlp,route,proj,conv,segs
    # the experts' unit shapes for calls of 9-16 rows (shape 3 elsewhere): measured faster at those rows, the same bits
    TF_INK2_CK_SHAPEW='4,3,3,3,3;5,3,3,3,3'
    # prompt chunks of 576 rows or more: each half's by-rows attention sum crosses the wire under the other half's
    # compute (needs TF_INK2_XROWS=1), and MoE-A carries each rank's quantized columns in place of bf16 ones: the same
    # bits, prompt fill measured faster at 2k-32k
    TF_INK2_XOVERLAP=1
    TF_INK2_XCODES=1
  )
  if [ "$ROCE_MULTI" = 1 ]; then
    release_envs+=(TF_INK2_ROCE_MULTI=1 TF_TP_ROCE_HCA="$hca" TF_TP_PORT="$TP_PORT" TF_TP_KERNEL_IMAGE="$fatbin"
                   TF_TP_MAILBOX_MAX_KB="$MAILBOX_MAX_KB")
  fi
}
