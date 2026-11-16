#!/usr/bin/env bash
# build-engine.sh [--src DIR] [--xgrammar DIR] [--out DIR] [--image TAG] [--no-image-build] [--cache DIR]: build the
# release binaries (aarch64, sm_121) from upstream TensorFold at ENGINE_COMMIT with patches/ applied, in the pinned
# toolchain image.
#
#   --src DIR         the engine checkout (default build/engine-src). Missing: cloned from ENGINE_COMMIT's URL at
#                     ENGINE_COMMIT. At ENGINE_COMMIT and clean: patches/*.patch applied with git am. Already patched
#                     (its tree is patches/APPLIED_TREE): used as is. Anything else is refused.
#   --xgrammar DIR    xgrammar v0.2.8 for structured output's libtfgrammar.so (default build/xgrammar). Missing:
#                     cloned from https://github.com/mlc-ai/xgrammar at XGRAMMAR_COMMIT below, with its dlpack submodule.
#   --out DIR         where bin/, lib/ and BUILD-INFO go (default build/dist): bin/ holds tensorfold-native,
#                     tf-inkling2-decode, tf-inkling2-plan, tf-inkling2-fold, tf-tp-test and tp_mailbox.fatbin, lib/
#                     libtfgrammar.so: the engine's part of what scripts/release/build-assets.sh needs.
#   --image TAG       the toolchain image (default inkling2-engine-build:zig0.17.0-cuda13.3.1, built from
#                     scripts/build-engine.Dockerfile for this machine's architecture unless --no-image-build)
#   --cache DIR       Zig's global cache (default build/zig-cache; delete it for a build from nothing)
#
# Needs git and Docker (no GPU). The build runs as the calling user with the checkout mounted at /src/tf and Zig's
# cache at /tmp/.cache/zig, the paths of the release build. Checked against the v0.1 release on an x86-64 build
# machine: tf-inkling2-fold, tf-inkling2-plan, tf-tp-test and tp_mailbox.fatbin byte for byte; tensorfold-native and
# tf-inkling2-decode up to symbol names that change from build to build (docs/BUILD.md). The patches are applied with a
# fixed committer and dates, so the patched commit is the same everywhere.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
src=${here}/build/engine-src; out=${here}/build/dist; cache=${here}/build/zig-cache; xg=${here}/build/xgrammar
XGRAMMAR_COMMIT=97787376faee5ed8466cfad57c99855e4ce2f6aa  # xgrammar v0.2.8
XGRAMMAR_URL=https://github.com/mlc-ai/xgrammar
image=inkling2-engine-build:zig0.17.0-cuda13.3.1; build_image=1
while [ $# -gt 0 ]; do
  case $1 in
    --src) src=$2; shift 2;;
    --xgrammar) xg=$2; shift 2;;
    --out) out=$2; shift 2;;
    --image) image=$2; shift 2;;
    --no-image-build) build_image=0; shift;;
    --cache) cache=$2; shift 2;;
    -h|--help) sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
engine=$(sed -n 1p "${here}/ENGINE_COMMIT" | tr -d '[:space:]')
url=$(sed -n 2p "${here}/ENGINE_COMMIT" | tr -d '[:space:]')
want_tree=$(tr -d '[:space:]' < "${here}/patches/APPLIED_TREE")
[[ ${engine} =~ ^[0-9a-f]{40}$ ]] || { echo "ENGINE_COMMIT: no 40-hex commit on line 1" >&2; exit 1; }
patches=("${here}"/patches/*.patch)
[ -f "${patches[0]}" ] || { echo "no patches/*.patch" >&2; exit 1; }

# 1. the source: upstream at ENGINE_COMMIT + patches/
if [ ! -e "${src}" ]; then
  echo "cloning ${url} at ${engine} into ${src}"
  git init -q "${src}"
  git -C "${src}" remote add origin "${url}"
  git -C "${src}" fetch -q --depth 1 origin "${engine}"
  git -C "${src}" checkout -q --detach FETCH_HEAD
fi
src=$(cd "${src}" && pwd)
tree=$(git -C "${src}" rev-parse "HEAD^{tree}")
if [ "${tree}" != "${want_tree}" ]; then
  [ "$(git -C "${src}" rev-parse HEAD)" = "${engine}" ] || { echo "${src}: HEAD is neither ENGINE_COMMIT ${engine} nor the patched tree" >&2; exit 1; }
  [ -z "$(git -C "${src}" status --porcelain --untracked-files=no)" ] || { echo "${src}: local changes" >&2; exit 1; }
  echo "applying ${#patches[@]} patches"
  GIT_COMMITTER_NAME="Scott Biggart" GIT_COMMITTER_EMAIL="297695+swbiggart@users.noreply.github.com" \
    git -C "${src}" -c core.whitespace=-blank-at-eof am -q --committer-date-is-author-date "${patches[@]}"
  tree=$(git -C "${src}" rev-parse "HEAD^{tree}")
  [ "${tree}" = "${want_tree}" ] || { echo "patched tree ${tree} is not patches/APPLIED_TREE ${want_tree}" >&2; exit 1; }
fi
[ -z "$(git -C "${src}" status --porcelain --untracked-files=no)" ] || { echo "${src}: local changes" >&2; exit 1; }
patched=$(git -C "${src}" rev-parse HEAD)
echo "engine source ${src}: ${patched} (tree ${tree})"

# 1b. xgrammar at its pinned commit (structured output's library is built from it and the engine's tf_grammar.cc)
if [ ! -e "${xg}" ]; then
  echo "cloning ${XGRAMMAR_URL} at ${XGRAMMAR_COMMIT} into ${xg}"
  git init -q "${xg}"
  git -C "${xg}" remote add origin "${XGRAMMAR_URL}"
  git -C "${xg}" fetch -q --depth 1 origin "${XGRAMMAR_COMMIT}"
  git -C "${xg}" checkout -q --detach FETCH_HEAD
  git -C "${xg}" submodule update -q --init --depth 1 3rdparty/dlpack
fi
xg=$(cd "${xg}" && pwd)
[ "$(git -C "${xg}" rev-parse HEAD)" = "${XGRAMMAR_COMMIT}" ] || { echo "${xg}: not xgrammar ${XGRAMMAR_COMMIT}" >&2; exit 1; }
# the engine's build checks the checkout by its tag (git describe --tags --exact-match = v0.2.8)
if [ "$(git -C "${xg}" describe --tags --exact-match 2> /dev/null)" != v0.2.8 ]; then
  git -C "${xg}" fetch -q --depth 1 origin "+refs/tags/v0.2.8:refs/tags/v0.2.8"
  [ "$(git -C "${xg}" rev-parse 'v0.2.8^{commit}')" = "${XGRAMMAR_COMMIT}" ] || { echo "${xg}: tag v0.2.8 is not ${XGRAMMAR_COMMIT}" >&2; exit 1; }
fi
[ -f "${xg}/3rdparty/dlpack/include/dlpack/dlpack.h" ] || { echo "${xg}: no 3rdparty/dlpack (git submodule update --init 3rdparty/dlpack)" >&2; exit 1; }

# 2. the toolchain image
if [ "${build_image}" = 1 ]; then
  case $(uname -m) in
    x86_64|amd64) zarch=x86_64; zsha=1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026;;
    aarch64|arm64) zarch=aarch64; zsha=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8;;
    *) echo "no Zig pin for $(uname -m)" >&2; exit 1;;
  esac
  ctx=$(mktemp -d); trap 'rm -rf "${ctx}"' EXIT
  cp "${here}/scripts/build-engine.Dockerfile" "${ctx}/Dockerfile"
  docker build -q --build-arg ZIG_ARCH="${zarch}" --build-arg ZIG_SHA256="${zsha}" -t "${image}" "${ctx}" > /dev/null
fi
image_id=$(docker image inspect "${image}" --format '{{.Id}}')

# 3. the build (the release build's command lines)
mkdir -p "${cache}" "${out}/bin" "${out}/lib"
cache=$(cd "${cache}" && pwd); out=$(cd "${out}" && pwd)
start=${SECONDS}
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "${src}:/src/tf" -v "${xg}:/xg:ro" -v "${cache}:/tmp/.cache/zig" -w /src/tf "${image}" \
  bash -c 'set -euo pipefail
    zig build native tf-inkling2-decode tf-tp-test tf-inkling2-fold tf-inkling2-plan -Doptimize=ReleaseFast \
      -Dtarget=aarch64-linux-gnu.2.28 -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dtp-nvcc=/usr/local/cuda/bin/nvcc \
      --prefix zig-out-arm --summary failures
    zig build grammar -Dxgrammar=/xg -Doptimize=ReleaseFast -Dtarget=aarch64-linux-gnu.2.28 --prefix zig-out-arm \
      --summary failures
    /usr/local/cuda/bin/nvcc -fatbin -O3 -std=c++17 -gencode=arch=compute_120,code=sm_120 \
      -gencode=arch=compute_121,code=sm_121 -gencode=arch=compute_90,code=compute_90 \
      -o zig-out-arm/tp_mailbox.fatbin zig/src/fabric/tp/mailbox.cu
    nvcc_version=$(/usr/local/cuda/bin/nvcc --version | grep -o "V[0-9][0-9.]*" | head -1)
    echo "zig $(zig version), nvcc ${nvcc_version}" > zig-out-arm/TOOLCHAIN'
echo "built in $((SECONDS - start)) s"

# 4. the release set
rm -f "${out}/bin/"* "${out}/lib/libtfgrammar.so"
cp -p "${src}/zig-out-arm/native/bin/tensorfold-native" "${out}/bin/"
for b in tf-inkling2-decode tf-tp-test tf-inkling2-fold tf-inkling2-plan; do cp -p "${src}/zig-out-arm/bin/${b}" "${out}/bin/"; done
cp -p "${src}/zig-out-arm/tp_mailbox.fatbin" "${out}/bin/"
cp -p "${src}/zig-out-arm/native/lib/libtfgrammar.so" "${out}/lib/"
sha() { if command -v sha256sum > /dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
{
  echo "engine ${url} ${engine} + ${#patches[@]} patches -> ${patched} (tree ${tree}); xgrammar ${XGRAMMAR_COMMIT}"
  echo "target aarch64-linux-gnu.2.28 sm_121 ReleaseFast; image ${image} ${image_id}; $(cat "${src}/zig-out-arm/TOOLCHAIN")"
  (cd "${out}" && sha bin/* lib/libtfgrammar.so)
} > "${out}/BUILD-INFO"
cat "${out}/BUILD-INFO"
