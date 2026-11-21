#!/usr/bin/env bash
# copy-rank.sh RANK DEST [--seed-from DIR] [--server ADDR:PORT] [--drafter] [--folds BIN] [--jobs J] [--check-gib G]
# [--prefix P]... [--verify-only] [--no-manifest]: build one rank's weight folder on this host from the pinned Hugging
# Face revision (or from a checked full checkpoint, --server), then check it. Run from the folder weights.sh fills (it
# holds inkling2_rank_copy.py, seed_rank_folder.py, ranges-w4-rR.tsv and inkling2-digests-w4.tsv).
#   1. seed (optional, --seed-from DIR): copy the runs DIR already holds (a checked rank folder on this host)
#   2. fetch: every other run of this rank by HTTP range requests (resumable: a rerun skips recorded runs); with
#      --server ADDR:PORT from a full checkpoint served by weights/checkpoint.py serve (setup.sh CHECKPOINT) instead,
#      each run hashed by the server as it reads the checkpoint and again here as it arrives
#   3. the content-digest manifest into DEST/inkling2-digests-w4.tsv
#   4. verify: coverage (every planned run is held), the shard headers, every run re-read from disk
#   5. check --http: shard headers and small files against Hugging Face, and a sample of G GiB of runs read again
#      from Hugging Face (default 3; 0: headers, small files and coverage only; `all`: every run); with --server
#      against the checkpoint's disk (the server's sha256 of each sampled run)
#   6. folds (--folds BIN): the DSpark folds file from the folder's own bytes (weights/fold-rank.sh)
# --drafter also copies the DSpark drafter's tensors (needed only with TF_INK2_DSPARK=1).
# --vision (rank 0, VISION=1) then adds the vision tower as one NEW file, DEST/inkling2-vision.safetensors: its 422
#   tensors by range requests from the pinned revision (also with --server), written beside the shards and renamed
#   into place once its sha256 is VISION_SHA256 (inkling2_vision_copy.py; no existing file is written). With
#   --verify-only the file's sha256 is checked. --vision-check checks only that file (a read) and exits.
# --folds BIN then builds the drafter's folds file DEST/inkling2-dspark-fold.safetensors with BIN (tf-inkling2-fold)
#   through fold-rank.sh (step 6; skipped with a note when BIN is not there yet: setup.sh's folds step builds it).
# --prefix P (repeatable) copies only the tensors under P instead of all of model.llm. (a small download to test the
# path; such a folder does not serve). --verify-only skips 1-3 and 5: step 4 alone, for a folder that already holds
# the rank (it is only read). --no-manifest leaves step 3 out (setup.sh points TF_INK2_DIGESTS at its own copy).
set -uo pipefail
cd "$(dirname "$0")"
r=${1:?usage: copy-rank.sh RANK DEST [options]}
dest=${2:?usage: copy-rank.sh RANK DEST [options]}
shift 2
REVISION=aac974122acd33ddb0d3e1f54433a9c5dd2249ce
HF=${HF_BASE:-https://huggingface.co/thinkingmachines/Inkling-2-Small/resolve/${REVISION}}
# inkling2-vision.safetensors made from that revision's model-00024-of-00024.safetensors (the tower's 422 tensors;
# each also equals the digest manifest's rank-0 line for it)
VISION_SHA256=b8e7df24b5bcb0d69f246cd5c703912b81b4f752c2316850b1225959c3d79a71
seed=""; jobs=8; gib=3; only=0; manifest=1; folds=""; server=""; vision=0
P=(--prefix model.llm.)
custom=()
while [ $# -gt 0 ]; do
  case $1 in
    --seed-from) seed=$2; shift 2;;
    --server) server=$2; shift 2;;
    --drafter) P+=(--prefix fuse_weight --prefix layers. --prefix markov_head --prefix mask_embed --prefix norm.weight); shift;;
    --folds) folds=$2; shift 2;;
    --vision) vision=1; shift;;
    --vision-check) exec python3 -I inkling2_vision_copy.py check --dest "${dest}" --sha256 "${VISION_SHA256}";;
    --jobs) jobs=$2; shift 2;;
    --check-gib) gib=$2; shift 2;;
    --prefix) custom+=(--prefix "$2"); shift 2;;
    --verify-only) only=1; shift;;
    --no-manifest) manifest=0; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[ ${#custom[@]} -gt 0 ] && P=("${custom[@]}")
# the source: the pinned Hugging Face revision, or a full checkpoint's server (weights/checkpoint.py serve)
SRC=(--http "${HF}"); from="Hugging Face (${REVISION})"
[ -n "${server}" ] && { SRC=(--server "${server}"); from="the checkpoint at ${server}"; }
# the copy tool, through netrun.py (a dead CDN address costs INKLING2_CONNECT_TIMEOUT seconds, not the 120 s request
# timeout; see netrun.py), unless INKLING2_NETRUN=0
rc_py() { if [ "${INKLING2_NETRUN:-1}" != 0 ] && [ -f netrun.py ]; then python3 netrun.py inkling2_rank_copy.py "$@"; else python3 inkling2_rank_copy.py "$@"; fi; }
ranges=ranges-w4-r${r}.tsv
for f in inkling2_rank_copy.py "${ranges}" inkling2-digests-w4.tsv; do
  [ -f "${f}" ] || { echo "missing ${f} in $(pwd)" >&2; exit 2; }
done
[ ${vision} = 0 ] || [ -f inkling2_vision_copy.py ] || { echo "missing inkling2_vision_copy.py in $(pwd)" >&2; exit 2; }
if [ ${only} = 1 ]; then
  [ -f "${dest}/.rank_copy.json" ] || { echo "${dest} has no copy state (.rank_copy.json): not a rank folder made by this tool" >&2; exit 3; }
  echo "== verify (existing folder, read only) $(date -u +%T)"
  rc_py verify --ranges "${ranges}" --dest "${dest}" "${P[@]}" || exit 3
  if [ ${vision} = 1 ]; then
    echo "== vision tower (read only) $(date -u +%T)"
    python3 -I inkling2_vision_copy.py check --dest "${dest}" --sha256 "${VISION_SHA256}" || exit 7
  fi
  echo "== done $(date -u +%T)"
  exit 0
fi
mkdir -p "${dest}"
if [ -n "${seed}" ]; then
  if [ -f "${dest}/.rank_copy.json" ]; then
    echo "== seed skipped: ${dest} already has a copy state (seed only into a new folder)"
  else
    echo "== seed from ${seed} $(date -u +%T)"
    python3 seed_rank_folder.py --old "${seed}" --ranges "${ranges}" --dest "${dest}" "${P[@]}" || exit 1
  fi
fi
echo "== fetch from ${from} $(date -u +%T)"
rc_py fetch "${SRC[@]}" --ranges "${ranges}" --dest "${dest}" --jobs "${jobs}" "${P[@]}" || exit 2
[ ${manifest} = 1 ] && cp inkling2-digests-w4.tsv "${dest}/inkling2-digests-w4.tsv"
echo "== verify $(date -u +%T)"
rc_py verify --ranges "${ranges}" --dest "${dest}" "${P[@]}" || exit 3
case ${gib} in
  all) echo "== check (every run) $(date -u +%T)"
       rc_py check "${SRC[@]}" --ranges "${ranges}" --dest "${dest}" --jobs 4 "${P[@]}" || exit 4;;
  *) echo "== check (${gib} GiB sample) $(date -u +%T)"
     rc_py check "${SRC[@]}" --ranges "${ranges}" --dest "${dest}" --jobs 4 --max-gib "${gib}" "${P[@]}" || exit 4;;
esac
if [ ${vision} = 1 ]; then
  echo "== vision tower $(date -u +%T)"
  python3 -I inkling2_vision_copy.py make --http "${HF}" --dest "${dest}" --sha256 "${VISION_SHA256}" || exit 7
fi
if [ -n "${folds}" ]; then
  echo "== folds $(date -u +%T)"
  if [ -x "${folds}" ]; then bash fold-rank.sh "${r}" "${dest}" "${folds}" || exit 6
  else echo "folds: ${folds} not there yet; setup.sh's folds step (or weights/fold-rank.sh) builds the folds file later"; fi
fi
echo "== done $(date -u +%T)"
