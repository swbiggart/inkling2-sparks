#!/usr/bin/env bash
# build-assets.sh DIST --engine-src DIR [--version V] [--out DIR] [--ref REF]: the Inkling-2 TP4 release tarball.
# build-assets.sh --list: the repository paths it packs from REF, one a line (setup/test-lib.sh lays them out as a tarball).
#
# DIST holds what the engine build and the release run produced:
#   DIST/bin/   tensorfold-native, tf-inkling2-decode, tf-inkling2-plan, tf-inkling2-fold, tf-tp-test (aarch64,
#               sm_121; required) and tp_mailbox.fatbin (required: the release runs the RoCE mesh wire,
#               TF_INK2_ROCE_MULTI=1): scripts/build-engine.sh --out DIST writes exactly these. A copy of
#               tools/zig/inkling2_rank_copy.py in DIST/bin is allowed if it equals this repository's at REF; the
#               tarball's bin/inkling2_rank_copy.py is always taken from REF.
#   DIST/lib/   libnccl.so.2 (NCCL 2.30.7), libcublas.so.13 and libcublasLt.so.13 (cuBLAS 13.1.1.3), unmodified, aarch64
#               (required: bundled with their licence texts, LICENSES/NCCL-LICENSE.txt and LICENSES/NVIDIA-CUDA-EULA.txt,
#               which are also copied into lib/ beside them; NOTICE and THIRD_PARTY_NOTICES.md credit them), and
#               libtfgrammar.so (required: structured output, built by scripts/build-engine.sh from the engine source and
#               xgrammar v0.2.8; its licences LICENSES/xgrammar-Apache-2.0.txt, xgrammar-NOTICE.txt, dlpack-Apache-2.0.txt
#               and picojson-BSD-2-Clause.txt are copied into lib/ as LICENSE-xgrammar.txt, NOTICE-xgrammar.txt,
#               LICENSE-dlpack.txt and LICENSE-picojson.txt)
#   DIST/plan/  ranges-w4-r0.tsv .. ranges-w4-r3.tsv, inkling2-digests-w4.tsv (required). Each ranges file is
#               tf-inkling2-plan --ranges R with the DSpark drafter's lines added (tools/zig/inkling2_dspark_ranges.py
#               --plan): every rank's drafter slices and markov_head.hidden_out, which tf-inkling2-fold needs on every
#               rank. The folds file itself (inkling2-dspark-fold.safetensors) is derived weights and never shipped.
#   DIST/data/  long_4000.safetensors, decode.json, chat_short.ids.json, tool_call.ids.json, and the folders ext/
#               (the extended prompt set smoke.py yardstick reads) and so-cases/ (structured-output cases); copied
#               whole with its folders (optional)
#   NCCL, cuBLAS, plan/ and data/ do not depend on the engine build: scripts/release/stage-dist.sh copies them from a
#   published release tarball (and libtfgrammar.so too, unless scripts/build-engine.sh built one).
# --engine-src DIR: the engine source the binaries were built from (scripts/build-engine.sh's checkout: upstream
#   TensorFold at ENGINE_COMMIT with patches/ applied; its tree must equal patches/APPLIED_TREE). The tarball's
#   LICENSE, NOTICE, THIRD_PARTY_NOTICES.md and LICENSES/ are that tree's: they cover all the code compiled into the
#   binaries (upstream's and ours) and the bundled NVIDIA libraries.
# The recipe, the copy tool and the benchmark tools are taken from REF (default HEAD) of this repository with git
# archive, so only committed content ships. bench.sh finds tools/inkling2_bench.py and the files it reads
# (inkling2_prompts.json, bench_prompts.json, inkling2_sheet.json) in the tarball's tools/. Writes
# OUT/tensorfold-inkling2-tp4-V-aarch64-sm121.tar.gz (OUT default: DIST), with SHA256SUMS of every file inside it and a
# .sha256 file next to it.
set -euo pipefail
usage="usage: build-assets.sh DIST --engine-src DIR [--version V] [--out DIR] [--ref REF] | --list"
repo_files=(recipe/inkling2-tp4 tools/zig/inkling2_rank_copy.py
  tools/inkling2_bench.py tools/inkling2_prompts.json tools/bench_prompts.json tools/inkling2_sheet.json
  tools/bench_openai.py tools/bench_cache.py tools/bench_concurrent.py tools/bench_args.py tools/bench_auth.py
  tools/bench_evidence.py tools/bench_metrics.py tools/bench_stream.py tools/bench_thinking.py tools/bench_timing.py)
if [ "${1:-}" = --list ]; then printf '%s\n' "${repo_files[@]}"; exit 0; fi
dist=${1:?${usage}}
shift
ver=""; out=$dist; ref=HEAD; esrc=""
while [ $# -gt 0 ]; do
  case $1 in
    --version) ver=$2; shift 2;;
    --out) out=$2; shift 2;;
    --ref) ref=$2; shift 2;;
    --engine-src) esrc=$2; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[ -n "${esrc}" ] || { echo "${usage}" >&2; exit 2; }
top=$(cd "$(dirname "$0")/../.." && pwd)
commit=$(git -C "${top}" rev-parse --short=12 "${ref}^{commit}")
[ -n "${ver}" ] || ver=$(git -C "${top}" describe --tags --always "${ref}" 2>/dev/null || echo "${commit}")
name=tensorfold-inkling2-tp4-${ver}-aarch64-sm121
sha() { if command -v sha256sum > /dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

# the engine source: its committed tree must be the one patches/ produce
want_tree=$(git -C "${top}" show "${ref}:patches/APPLIED_TREE" | tr -d '[:space:]')
engine=$(git -C "${top}" show "${ref}:ENGINE_COMMIT" | sed -n 1p | tr -d '[:space:]')
got_tree=$(git -C "${esrc}" rev-parse "HEAD^{tree}" 2> /dev/null) || { echo "${esrc}: not a git checkout" >&2; exit 1; }
[ "${got_tree}" = "${want_tree}" ] || { echo "${esrc}: HEAD's tree ${got_tree} is not patches/APPLIED_TREE ${want_tree} (upstream ${engine} + patches/)" >&2; exit 1; }
git -C "${esrc}" merge-base --is-ancestor "${engine}" HEAD || { echo "${esrc}: HEAD does not descend from ENGINE_COMMIT ${engine}" >&2; exit 1; }

need_bin=(tensorfold-native tf-inkling2-decode tf-inkling2-plan tf-inkling2-fold tf-tp-test)
for b in "${need_bin[@]}"; do
  [ -f "${dist}/bin/${b}" ] || { echo "missing ${dist}/bin/${b}" >&2; exit 1; }
  if command -v file > /dev/null; then
    file -b "${dist}/bin/${b}" | grep -q 'ELF 64-bit.*aarch64' || { echo "${b}: not an aarch64 ELF ($(file -b "${dist}/bin/${b}"))" >&2; exit 1; }
  fi
done
[ -f "${dist}/bin/tp_mailbox.fatbin" ] || { echo "missing ${dist}/bin/tp_mailbox.fatbin (the RoCE wire's kernel image)" >&2; exit 1; }
for f in "${dist}/bin/"*; do
  case $(basename "${f}") in
    tensorfold-native|tf-inkling2-decode|tf-inkling2-plan|tf-inkling2-fold|tf-tp-test|tp_mailbox.fatbin) ;;
    inkling2_rank_copy.py)
      git -C "${top}" show "${ref}:tools/zig/inkling2_rank_copy.py" | cmp -s - "${f}" ||
        { echo "${f} differs from tools/zig/inkling2_rank_copy.py at ${ref}" >&2; exit 1; };;
    *) echo "${f}: not a release binary (bin/ ships ${need_bin[*]} tp_mailbox.fatbin)" >&2; exit 1;;
  esac
done
need_lib=(libnccl.so.2 libcublas.so.13 libcublasLt.so.13 libtfgrammar.so)
for b in "${need_lib[@]}"; do
  [ -f "${dist}/lib/${b}" ] || { echo "missing ${dist}/lib/${b} (the release bundles NCCL, cuBLAS and libtfgrammar)" >&2; exit 1; }
  if command -v file > /dev/null; then
    file -L -b "${dist}/lib/${b}" | grep -q 'ELF 64-bit.*aarch64' || { echo "${b}: not an aarch64 ELF" >&2; exit 1; }
  fi
done
for f in ranges-w4-r0.tsv ranges-w4-r1.tsv ranges-w4-r2.tsv ranges-w4-r3.tsv inkling2-digests-w4.tsv; do
  [ -f "${dist}/plan/${f}" ] || { echo "missing ${dist}/plan/${f}" >&2; exit 1; }
done
for r in 0 1 2 3; do # weights.sh --drafter fetches the drafter by these lines; tf-inkling2-fold reads hidden_out
  for t in markov_head.hidden_out.weight markov_head.hidden_out.bias layers.0.self_attn.q_proj.weight; do
    awk -F'\t' -v t="${t}" '$7 == t {f = 1} END {exit !f}' "${dist}/plan/ranges-w4-r${r}.tsv" || { echo "${dist}/plan/ranges-w4-r${r}.tsv: no ${t} (add the drafter's lines: tools/zig/inkling2_dspark_ranges.py --plan)" >&2; exit 1; }
  done
done
[ -z "$(find "${dist}" -name 'inkling2-dspark-fold*')" ] || { echo "${dist} holds a DSpark folds file: derived weights, never shipped" >&2; exit 1; }

stage=$(mktemp -d)
trap 'rm -rf "${stage}"' EXIT
root=${stage}/${name}
mkdir -p "${root}"
git -C "${top}" archive --format=tar "${ref}" -- "${repo_files[@]}" | tar -x -C "${root}"
git -C "${esrc}" archive --format=tar HEAD -- LICENSE NOTICE THIRD_PARTY_NOTICES.md LICENSES | tar -x -C "${root}"
for l in NCCL-LICENSE.txt NVIDIA-CUDA-EULA.txt xgrammar-Apache-2.0.txt xgrammar-NOTICE.txt dlpack-Apache-2.0.txt picojson-BSD-2-Clause.txt; do
  [ -s "${root}/LICENSES/${l}" ] || { echo "LICENSES/${l} is not in the engine source: the bundled libraries need it" >&2; exit 1; }
done
mkdir -p "${root}/bin" "${root}/lib" "${root}/plan"
for b in "${need_bin[@]}" tp_mailbox.fatbin; do cp -p "${dist}/bin/${b}" "${root}/bin/${b}"; done
cp -p "${root}/tools/zig/inkling2_rank_copy.py" "${root}/bin/inkling2_rank_copy.py"
chmod 0755 "${root}/bin/"*
for b in "${need_lib[@]}"; do cp -pL "${dist}/lib/${b}" "${root}/lib/${b}"; done
cp -p "${root}/LICENSES/NCCL-LICENSE.txt" "${root}/lib/LICENSE-NCCL.txt"
cp -p "${root}/LICENSES/NVIDIA-CUDA-EULA.txt" "${root}/lib/LICENSE-NVIDIA-CUDA-EULA.txt"
cp -p "${root}/LICENSES/xgrammar-Apache-2.0.txt" "${root}/lib/LICENSE-xgrammar.txt"
cp -p "${root}/LICENSES/xgrammar-NOTICE.txt" "${root}/lib/NOTICE-xgrammar.txt"
cp -p "${root}/LICENSES/dlpack-Apache-2.0.txt" "${root}/lib/LICENSE-dlpack.txt"
cp -p "${root}/LICENSES/picojson-BSD-2-Clause.txt" "${root}/lib/LICENSE-picojson.txt"
cp -p "${dist}/plan/"* "${root}/plan/"
if [ -d "${dist}/data" ]; then mkdir -p "${root}/data" && cp -pR "${dist}/data/." "${root}/data/"; else
  echo "note: no ${dist}/data (the decode check prompt and oracle ids); the tarball ships without data/" >&2; fi
printf 'tensorfold-inkling2-tp4 %s\nsource commit %s\nengine TensorFold %s + patches/ (tree %s)\nmodel thinkingmachines/Inkling-2-Small revision aac974122acd33ddb0d3e1f54433a9c5dd2249ce\n' \
  "${ver}" "${commit}" "${engine}" "${want_tree}" > "${root}/VERSION"
(cd "${root}" && find . -type f ! -name SHA256SUMS | LC_ALL=C sort | sed 's|^\./||' | while read -r f; do sha "${f}"; done > SHA256SUMS)
mkdir -p "${out}"
tgz=$(cd "${out}" && pwd)/${name}.tar.gz
COPYFILE_DISABLE=1 tar -C "${stage}" -czf "${tgz}" "${name}"
(cd "$(dirname "${tgz}")" && sha "$(basename "${tgz}")" > "$(basename "${tgz}").sha256")
echo "${tgz}"
cat "${tgz}.sha256"
echo "$(wc -l < "${root}/SHA256SUMS" | tr -d ' ') files; source commit ${commit}; engine ${engine} + patches/"
