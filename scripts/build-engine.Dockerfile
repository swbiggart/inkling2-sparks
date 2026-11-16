# The engine's build toolchain (no sources inside): NVIDIA's CUDA 13.3.1 devel image (nvcc 13.3, g++ 13 as nvcc's host
# compiler), pinned by digest, plus Zig 0.17.0 (TensorFold's .zig-version), checked by sha256. Zig cross-compiles the
# aarch64 binaries, so the image can be built for the build machine's own architecture. scripts/build-engine.sh builds
# it (context: this file only) and runs the build in it with the source mounted at /src/tf.
# This is the toolchain the v0.1 binaries were built with: the same base layers and Zig as the image used for the
# release build (its apt layer added ca-certificates, curl, git, xz-utils, cmake, nasm and python3, as below).
#
#   ZIG_ARCH=x86_64  ZIG_SHA256=1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026 (the release build's)
#   ZIG_ARCH=aarch64 ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
ARG CUDA_IMAGE=nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87
FROM ${CUDA_IMAGE}
ARG ZIG_VERSION=0.17.0
ARG ZIG_ARCH=x86_64
ARG ZIG_SHA256=1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git xz-utils cmake nasm python3 \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL --retry 3 -o /tmp/zig.tar.xz https://ziglang.org/download/${ZIG_VERSION}/zig-${ZIG_ARCH}-linux-${ZIG_VERSION}.tar.xz \
 && echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 \
 && rm /tmp/zig.tar.xz && ln -s /opt/zig/zig /usr/local/bin/zig && test "$(zig version)" = "${ZIG_VERSION}" \
 && /usr/local/cuda/bin/nvcc --version | grep -q 'release 13\.3,'
WORKDIR /src/tf
