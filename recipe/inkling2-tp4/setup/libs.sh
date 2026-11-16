#!/usr/bin/env bash
# libs.sh ROOT: the CUDA user-space libraries the release binaries load at run time, into ROOT/lib on this host:
# libnccl.so.2 (NCCL 2.30.7 for CUDA 13) and libcublas.so.13 / libcublasLt.so.13 (cuBLAS 13.1.1.3), from NVIDIA's
# wheels on PyPI at pinned versions and sha256s (the same builds the release was measured with). Writes the pinned
# ROOT/lib/SHA256SUMS and checks the files against it. Skips the download when ROOT/lib already matches it.
# The binaries load them with dlopen (LD_LIBRARY_PATH=ROOT/lib, set by rank.sh); nothing is installed system-wide.
set -uo pipefail
root=${1:?usage: libs.sh ROOT}
lib=${root}/lib
want=(libnccl.so.2 libcublas.so.13 libcublasLt.so.13)
have_all() {
  [ -f "${lib}/SHA256SUMS" ] || return 1
  for f in "${want[@]}"; do grep -q " ${f}\$" "${lib}/SHA256SUMS" || return 1; done
  (cd "${lib}" && sha256sum -c --quiet SHA256SUMS > /dev/null 2>&1)
}
if have_all; then echo "libs: ${lib} already holds ${want[*]} (SHA256SUMS OK)"; exit 0; fi
# name|url|sha256 of each wheel
WHEELS=(
  "nvidia_nccl_cu13-2.30.7|https://files.pythonhosted.org/packages/d1/21/a73174c6157101bdf1ffc22b517f76ff0082613989dd9bc8f43e8034caac/nvidia_nccl_cu13-2.30.7-py3-none-manylinux_2_18_aarch64.whl|ca786ffa5a647c75d4d1f5cc72a6c4f537947e2ba8823d7c8aaf768e7a7b9f77"
  "nvidia_cublas-13.1.1.3|https://files.pythonhosted.org/packages/a7/a1/0bd24ee8c8d03adac032fd2909426a00c88f8c57961b1277ded97f91119f/nvidia_cublas-13.1.1.3-py3-none-manylinux_2_27_aarch64.whl|b7a210458267ac818974c53038fbec2e969d5c99f305ab15c72522fa9f001dd5"
)
cache=${root}/.setup/wheels
mkdir -p "${cache}" "${lib}"
for w in "${WHEELS[@]}"; do
  IFS='|' read -r name url sum <<< "${w}"
  f=${cache}/${name}.whl
  if ! echo "${sum}  ${f}" | sha256sum -c --quiet - > /dev/null 2>&1; then
    echo "libs: downloading ${name} ($(date -u +%T))"
    curl -fsSL --retry 3 -C - -o "${f}" "${url}" || curl -fsSL --retry 3 -o "${f}" "${url}" || { echo "libs: download of ${url} failed" >&2; exit 1; }
    echo "${sum}  ${f}" | sha256sum -c --quiet - || { echo "libs: ${name}.whl sha256 differs from the pinned ${sum}" >&2; rm -f "${f}"; exit 1; }
  fi
  python3 - "${f}" "${lib}" "${want[@]}" <<'PY' || exit 1
import os, sys, zipfile
whl, dest, want = sys.argv[1], sys.argv[2], set(sys.argv[3:])
with zipfile.ZipFile(whl) as z:
    for n in z.namelist():
        b = os.path.basename(n)
        if b in want:
            tmp = os.path.join(dest, b + ".part")
            with z.open(n) as src, open(tmp, "wb") as out:
                while True:
                    chunk = src.read(1 << 22)
                    if not chunk:
                        break
                    out.write(chunk)
            os.replace(tmp, os.path.join(dest, b))
            print(f"libs: {b} from {os.path.basename(whl)}")
PY
done
for f in "${want[@]}"; do [ -f "${lib}/${f}" ] || { echo "libs: ${f} not found in the wheels" >&2; exit 1; }; done
# the files the release was measured with (the same bytes as NVIDIA's arm64 NCCL deb and cuBLAS 13.1.1.3)
printf '%s\n' "fc7ea66334edbc934aa25959b9907dbb2b91a1d2485beff18839afc45cbc08d0  libnccl.so.2" \
  "4e346a7b60a01552bd9bf4bbc6c862e11eb851559c3da3572293345de6aaed75  libcublas.so.13" \
  "930418840b747413584867cb7167dd5911dcf4184135589e0334527e9abbac9f  libcublasLt.so.13" > "${lib}/SHA256SUMS"
(cd "${lib}" && sha256sum -c --quiet SHA256SUMS) || { echo "libs: the extracted libraries differ from the pinned sha256s" >&2; exit 1; }
echo "libs: ${lib}: $(tr -d '\000' < "${lib}/libnccl.so.2" | grep -a -o -m1 'NCCL version [0-9.]*+cuda[0-9.]*' | head -1); SHA256SUMS written"
rm -rf "${cache}"
