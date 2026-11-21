#!/usr/bin/env bash
# build-codecs.sh OUT [--src DIR]: the JPEG and WebP decoders the Inkling-2 TP4 release ships in lib/ for --vision
# (rank 0): OUT/libturbojpeg.so.0 (libjpeg-turbo 3.1.4.1) and OUT/libwebpdecoder.so.3 (libwebp 1.6.0), aarch64 Linux,
# glibc 2.28 or later. These are the releases Pillow 12.3.0 bundles, the decoders core/picture pins (jpeg.zig,
# webp.zig refuse any other release): their pixels are Pillow's (gate V1, zig build test-inkling2-image).
#
# The sources are the projects' release tarballs, checked against the sha256 below (the tarballs' signatures were
# checked once against the projects' release keys: libjpeg-turbo 7EC2DBB6F4DBF434, WebP 6B0E6B70976DE303EDF2F601F9C3D6BDB8232B5D).
# --src DIR takes the two tarballs from DIR instead of downloading them. Built unmodified with CMake (Release, the
# projects' default SIMD: Neon) and `zig cc -target aarch64-linux-gnu.2.28` (the release's glibc floor), on macOS or
# Linux; needs zig 0.17.0, cmake and ninja on PATH (pip install cmake ninja). Licences: LICENSES/libjpeg-turbo-*,
# LICENSES/libwebp-* (build-assets.sh copies them into lib/ beside the libraries).
set -euo pipefail
out=${1:?usage: build-codecs.sh OUT [--src DIR]}
src=""
[ "${2:-}" = --src ] && src=${3:?--src DIR}
JPEG=libjpeg-turbo-3.1.4.1
JPEG_URL=https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/${JPEG}.tar.gz
JPEG_SHA=ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022
WEBP=libwebp-1.6.0
WEBP_URL=https://storage.googleapis.com/downloads.webmproject.org/releases/webp/${WEBP}.tar.gz
WEBP_SHA=e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564
sha() { if command -v sha256sum > /dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
for t in zig cmake ninja; do command -v "${t}" > /dev/null || { echo "build-codecs.sh needs ${t} on PATH" >&2; exit 2; }; done
work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT
mkdir -p "${out}" "${work}/tar" "${work}/tc"
out=$(cd "${out}" && pwd)
for p in "${JPEG} ${JPEG_URL} ${JPEG_SHA}" "${WEBP} ${WEBP_URL} ${WEBP_SHA}"; do
  set -- ${p}
  if [ -n "${src}" ]; then cp "${src}/$1.tar.gz" "${work}/tar/"; else curl -fsSL -o "${work}/tar/$1.tar.gz" "$2"; fi
  [ "$(sha "${work}/tar/$1.tar.gz" | cut -d' ' -f1)" = "$3" ] || { echo "$1.tar.gz: sha256 is not $3" >&2; exit 1; }
  tar -xzf "${work}/tar/$1.tar.gz" -C "${work}"
done
# the cross compiler: zig cc for aarch64 Linux at glibc 2.28, no sanitizers, no debug info or build paths (so a rebuild
# is byte-equal)
printf '#!/bin/sh\nexec zig cc -target aarch64-linux-gnu.2.28 -fno-sanitize=all -g0 -ffile-prefix-map=%s=. "$@"\n' "${work}" > "${work}/tc/cc"
printf '#!/bin/sh\nexec zig ar "$@"\n' > "${work}/tc/ar"
printf '#!/bin/sh\nexec zig ranlib "$@"\n' > "${work}/tc/ranlib"
chmod +x "${work}/tc/"*
cat > "${work}/tc/aarch64-linux.cmake" << EOF
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER ${work}/tc/cc)
set(CMAKE_AR ${work}/tc/ar)
set(CMAKE_RANLIB ${work}/tc/ranlib)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
EOF
common=(-G Ninja -DCMAKE_TOOLCHAIN_FILE="${work}/tc/aarch64-linux.cmake" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${work}/install")
# BUILD: libjpeg-turbo's build string (its version line reads "3.1.4.1 (build B)"); fixed so rebuilds are byte-equal
cmake -S "${work}/${JPEG}" -B "${work}/jpeg" "${common[@]}" -DENABLE_STATIC=OFF -DWITH_JAVA=OFF -DBUILD=20260327 > "${work}/jpeg.log" 2>&1 ||
  { tail -20 "${work}/jpeg.log" >&2; exit 1; }
ninja -C "${work}/jpeg" turbojpeg >> "${work}/jpeg.log" 2>&1 || { tail -20 "${work}/jpeg.log" >&2; exit 1; }
cmake -S "${work}/${WEBP}" -B "${work}/webp" "${common[@]}" -DBUILD_SHARED_LIBS=ON -DWEBP_BUILD_ANIM_UTILS=OFF \
  -DWEBP_BUILD_CWEBP=OFF -DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_GIF2WEBP=OFF -DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_VWEBP=OFF \
  -DWEBP_BUILD_WEBPINFO=OFF -DWEBP_BUILD_WEBPMUX=OFF -DWEBP_BUILD_EXTRAS=OFF > "${work}/webp.log" 2>&1 || { tail -20 "${work}/webp.log" >&2; exit 1; }
ninja -C "${work}/webp" webpdecoder >> "${work}/webp.log" 2>&1 || { tail -20 "${work}/webp.log" >&2; exit 1; }
cp -L "${work}/jpeg/libturbojpeg.so.0" "${out}/libturbojpeg.so.0"
cp -L "${work}/webp/libwebpdecoder.so.3" "${out}/libwebpdecoder.so.3"
# the pins core/picture checks at run time
grep -aq 'libjpeg-turbo version 3.1.4.1 ' "${out}/libturbojpeg.so.0" || { echo "libturbojpeg.so.0: no 3.1.4.1 version string" >&2; exit 1; }
(cd "${out}" && sha libturbojpeg.so.0 libwebpdecoder.so.3)
