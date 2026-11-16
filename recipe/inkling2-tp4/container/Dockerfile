# The Inkling-2-Small TP4 serving image (linux/arm64, GB10): the release's aarch64 sm_121 binaries and the CUDA
# libraries they load (lib/: NCCL 2.30.7, cuBLAS 13, libtfgrammar for structured output) on NVIDIA's CUDA 13.3 runtime image, which matches the build's
# CUDA user space, plus rdma-core's verbs library and providers for NCCL's RoCE transport, and the licence files
# (doc/). No weights: every rank mounts its own rank folder (compose.yaml), with the DSpark folds file that
# setup.sh's folds step builds there on the host (bin/tf-inkling2-fold, also in the image) before the ranks start.
#
# setup.sh --container builds it on rank 0's host from a context of bin/, lib/, doc/, VERSION and SHA256SUMS (the release
# assets as installed there), tags it inkling2-sparks:<release commit>, and copies it to the other hosts with
# docker save / docker load. It is never pushed to a registry. By hand, from such a folder:
#   docker build --platform linux/arm64 -t inkling2-sparks:<commit> -f Dockerfile .
#
# Pinned: the base image by the digest of its linux/arm64 manifest (nvcr.io/nvidia/cuda:13.3.1-runtime-ubuntu24.04 as
# gated, 14 November 2026; the tag alone can move), and the two Ubuntu packages by exact version (the ones in the gated
# image). If Ubuntu's archive no longer lists a pinned version (noble-updates keeps only the newest), the build stops
# and says so rather than installing another one: set VERBS_VERSION to the version you want to run (untested), or
# build from an image that has it.
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.3.1-runtime-ubuntu24.04@sha256:585155539dd4ff2ab653ee59d812aae81a9406a5804596d516c06e3a5719eb26
FROM ${BASE_IMAGE}
# (redeclared: an ARG before FROM is not in scope after it; the label below names it)
ARG BASE_IMAGE
ARG VERBS_VERSION=50.0-2ubuntu0.2
RUN apt-get update && \
    for p in libibverbs1 ibverbs-providers; do \
      apt-cache madison "$p" | awk -v v="${VERBS_VERSION}" '$3 == v { f = 1 } END { exit !f }' || { \
        echo "refused: $p ${VERBS_VERSION} (pinned) is not in this image's apt sources; listed: $(apt-cache madison "$p" | awk '{print $3}' | tr '\n' ' ')" >&2; \
        exit 1; }; \
    done && \
    apt-get install -y --no-install-recommends "libibverbs1=${VERBS_VERSION}" "ibverbs-providers=${VERBS_VERSION}" && \
    test "$(dpkg-query -W -f '${Version}' libibverbs1) $(dpkg-query -W -f '${Version}' ibverbs-providers)" = "${VERBS_VERSION} ${VERBS_VERSION}" && \
    rm -rf /var/lib/apt/lists/*
COPY bin/ /opt/inkling2/bin/
COPY lib/ /opt/inkling2/lib/
COPY doc/ /opt/inkling2/doc/
COPY VERSION SHA256SUMS /opt/inkling2/
RUN cd /opt/inkling2 && sha256sum -c --quiet SHA256SUMS && chmod 0755 bin/* && \
    test -x bin/tensorfold-native && test -x bin/tf-inkling2-decode && test -x bin/tf-inkling2-fold && test -f lib/libtfgrammar.so
ARG RELEASE=unknown
ARG ASSETS=unknown
LABEL org.opencontainers.image.title="inkling2-sparks" \
      org.opencontainers.image.description="Inkling-2-Small at TP4 on four GB10 hosts (TensorFold native engine), no weights" \
      org.opencontainers.image.licenses="Apache-2.0" \
      inkling2.release=${RELEASE} \
      inkling2.assets=${ASSETS} \
      inkling2.base=${BASE_IMAGE} \
      inkling2.verbs=${VERBS_VERSION}
# The libraries in lib/ come first, so every rank loads the same NCCL and cuBLAS builds as the host install.
# NVIDIA_DISABLE_REQUIRE: the 13.3 runtime image asks for a CUDA 13.3 driver; the binaries run on any 580+ driver
# through CUDA's minor-version compatibility (the host install does the same).
ENV PATH=/opt/inkling2/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/inkling2/lib \
    NVIDIA_DISABLE_REQUIRE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HOME=/tmp
WORKDIR /tmp
ENTRYPOINT ["/opt/inkling2/bin/tensorfold-native"]
CMD ["--help"]
