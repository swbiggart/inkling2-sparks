#!/usr/bin/env bash
# fold-rank.sh RANK MODEL BIN [--force]: the DSpark drafter's folds file MODEL/inkling2-dspark-fold.safetensors for
# rank RANK of 4, built on this host by BIN (tf-inkling2-fold, the release's bin/) from the rank folder itself: the
# rank's unembed rows and mtp.safetensors' hidden_out (both in the rank's ranges file with --drafter). About a second
# on a GB10; no PyTorch, nothing downloaded. The file is derived from the weights and never shipped.
# Idempotent: an existing file is kept only when MODEL/.inkling2-dspark-fold.sha256 records its digest for this rank
# and its header names this rank's rows. Any other file (made another way, another rank's, stale) is built again,
# which takes about a second. --force builds it again.
# Writes only the folds file and its digest record into MODEL (the weights are only read), each published atomically
# (complete or not at all) by BIN: a fresh temporary name created exclusively, written and synced through that one
# descriptor, renamed; no name is opened again, so no link is followed. A failed write or rename exits nonzero.
set -uo pipefail
r=${1:?usage: fold-rank.sh RANK MODEL BIN [--force]}
m=${2:?usage: fold-rank.sh RANK MODEL BIN [--force]}
bin=${3:?usage: fold-rank.sh RANK MODEL BIN [--force]}
force=${4:-}
f=${m}/inkling2-dspark-fold.safetensors
rec=${m}/.inkling2-dspark-fold.sha256
# a folds file's header: fold_b [V] and fold_w [V, 256], BF16, the size they imply; rows (if named) this rank's
header_ok() {
  python3 -I - "${f}" "${r}" << 'EOF'
import json, os, struct, sys
path, r, W = sys.argv[1], int(sys.argv[2]), 4
with open(path, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    h = json.loads(fh.read(n))
m = h.pop("__metadata__", {})
b, w = h.get("fold_b", {}), h.get("fold_w", {})
V = (b.get("shape") or [0])[0]
ok = (sorted(h) == ["fold_b", "fold_w"] and b.get("dtype") == w.get("dtype") == "BF16" and w.get("shape") == [V, 256]
      and V % W == 0 and os.path.getsize(path) == 8 + n + V * 2 * 257)
if ok and "rows" in m:
    ok = m["rows"] == f"{r * V // W}:{(r + 1) * V // W}"
sys.exit(0 if ok else 1)
EOF
}
digest() { sha256sum "${f}" | cut -c1-64; }
if [ "${force}" != --force ] && [ -s "${f}" ]; then
  if [ -f "${rec}" ] && [ "$(cat "${rec}")" = "$(digest)  rank ${r}" ] && header_ok; then
    echo "folds: rank ${r}: ${f} kept (digest $(cut -c1-16 "${rec}")... recorded for rank ${r})"; exit 0
  fi
  echo "folds: rank ${r}: ${f} not recorded for rank ${r}; building it again"
fi
[ -x "${bin}" ] || { echo "folds: rank ${r}: no ${bin} (the release's tf-inkling2-fold)" >&2; exit 2; }
rm -f "${rec}" || { echo "folds: rank ${r}: cannot remove ${rec}" >&2; exit 1; }
# the tool publishes the file itself (a fresh temporary file created exclusively beside it, synced, renamed): no
# temporary name is made or followed here
nice "${bin}" "${m}" "${f}" --rank "${r}" --world 4 || { echo "folds: rank ${r}: tf-inkling2-fold failed" >&2; exit 1; }
header_ok || { echo "folds: rank ${r}: ${f}: unexpected header" >&2; exit 1; }
d=$(digest) && [ ${#d} = 64 ] || { echo "folds: rank ${r}: cannot hash ${f}" >&2; exit 1; }
# the record the same way, by the tool (no temporary name is made, reopened or followed here), then read back
if ! { nice "${bin}" --record "${rec}" "${d}  rank ${r}" && [ "$(cat "${rec}")" = "${d}  rank ${r}" ]; }; then
  echo "folds: rank ${r}: cannot record the digest in ${rec}" >&2; exit 1
fi
echo "folds: rank ${r}: ${f} built (digest ${d:0:16}...)"
