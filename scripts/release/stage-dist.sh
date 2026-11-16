#!/usr/bin/env bash
# stage-dist.sh RELEASE DIST: copy lib/, plan/ and data/ of a published release (its .tar.gz or an unpacked folder)
# into DIST, beside the bin/ that scripts/build-engine.sh --out DIST wrote, so that build-assets.sh can rebuild the
# tarball from source. These parts do not depend on the engine build: lib/ is NVIDIA's NCCL and cuBLAS, unmodified
# (recipe/inkling2-tp4/setup/libs.sh fetches the same pinned files), plan/ and data/ are computed from the checkpoint
# (tf-inkling2-plan with tools/zig/inkling2_dspark_ranges.py, and the oracle's files). Every copied file is checked
# against the release's SHA256SUMS. Existing DIST/plan or DIST/data are refused, and so is a DIST/lib holding anything
# but the libtfgrammar.so that scripts/build-engine.sh builds (kept: the release's copy is not staged over it).
set -euo pipefail
rel=${1:?usage: stage-dist.sh RELEASE DIST}; dist=${2:?usage: stage-dist.sh RELEASE DIST}
sha() { if command -v sha256sum > /dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
for d in plan data; do [ ! -e "${dist}/${d}" ] || { echo "${dist}/${d} exists" >&2; exit 1; }; done
[ -z "$(ls -A "${dist}/lib" 2> /dev/null | grep -vx libtfgrammar.so)" ] || { echo "${dist}/lib holds more than libtfgrammar.so" >&2; exit 1; }
tmp=$(mktemp -d); trap 'rm -rf "${tmp}"' EXIT
if [ -f "${rel}" ]; then
  tar -xzf "${rel}" -C "${tmp}"
  src=$(find "${tmp}" -mindepth 1 -maxdepth 1 -type d | head -1)
else src=${rel}; fi
[ -f "${src}/SHA256SUMS" ] || { echo "${rel}: no SHA256SUMS (not a release)" >&2; exit 1; }
(cd "${src}" && grep -E '  (lib|plan|data)/' SHA256SUMS | sha -c --quiet -) || { echo "${rel}: lib/plan/data differ from SHA256SUMS" >&2; exit 1; }
mkdir -p "${dist}"
for d in lib plan data; do
  [ -d "${src}/${d}" ] || { echo "note: ${rel} has no ${d}/" >&2; continue; }
  mkdir -p "${dist}/${d}"
  for f in "${src}/${d}/"*; do
    case $(basename "${f}") in LICENSE-*|NOTICE-*) continue;; esac  # build-assets.sh writes lib/'s licence copies itself
    [ -e "${dist}/${d}/$(basename "${f}")" ] && continue          # a libtfgrammar.so build-engine.sh made
    cp -p "${f}" "${dist}/${d}/"
  done
done
echo "staged lib/ plan/ data/ from ${rel} into ${dist}"
