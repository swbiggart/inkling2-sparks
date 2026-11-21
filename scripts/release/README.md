# Release scripts (Inkling-2 TP4)

- `build-assets.sh DIST --engine-src DIR [--version V] [--out DIR] [--ref REF]`: the release tarball. `DIST/bin/` holds
  the aarch64 sm_121 binaries and `tp_mailbox.fatbin` (`scripts/build-engine.sh --out DIST` writes them), `DIST/lib/`
  NCCL and cuBLAS (bundled unmodified), `libtfgrammar.so` and the image decoders `libturbojpeg.so.0` and
  `libwebpdecoder.so.3` (all required; their licence texts are copied from `LICENSES/` into `lib/`), `DIST/plan/` the
  ranges (the plan's, plus the DSpark drafter's lines on every rank) and digest manifest, optional `DIST/data/`. The
  recipe, the weight and vision-tower copy tools and the benchmark tools come from the committed tree at REF;
  `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` and `LICENSES/` from the engine source DIR (upstream at `ENGINE_COMMIT`
  with `patches/`; its tree must equal `patches/APPLIED_TREE`), since they cover the code compiled into the binaries.
  `SHA256SUMS` inside, a `.sha256` beside it.
- `build-codecs.sh OUT [--src DIR]`: the image decoders, `OUT/libturbojpeg.so.0` (libjpeg-turbo
  3.1.4.1) and `OUT/libwebpdecoder.so.3` (libwebp 1.6.0), for aarch64 Linux (glibc 2.28
  or later), built unmodified from the projects' release tarballs, checked against pinned sha256s (`--src DIR` takes
  the two tarballs from DIR). Needs zig 0.17.0, cmake and ninja.
- `stage-dist.sh RELEASE DIST`: copies `lib/`, `plan/` and `data/` of a published release tarball into DIST (they do
  not depend on the engine build), checked against its `SHA256SUMS`.
- `scrub-check.sh [--names FILE] [--authors] range A..B | tree [REF]`: looks for host names (from a private list),
  private IPv4 addresses, home and build-host paths, and email addresses. Prints each hit; exits 1 on any.

Rebuilding a release's tarball from source, from that release's commit (its tag), with V its version (v0.1, v0.2, ...):

```
V=v0.3
scripts/build-engine.sh                                   # build/engine-src, build/dist/bin
scripts/release/stage-dist.sh tensorfold-inkling2-tp4-${V}-aarch64-sm121.tar.gz build/dist
scripts/release/build-assets.sh build/dist --engine-src build/engine-src --version ${V}
```

Before publishing: `scrub-check.sh --authors range BASE..HEAD` and `scrub-check.sh tree HEAD` clean.
