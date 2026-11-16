# Release scripts (Inkling-2 TP4)

- `build-assets.sh DIST --engine-src DIR [--version V] [--out DIR] [--ref REF]`: the release tarball. `DIST/bin/` holds
  the aarch64 sm_121 binaries and `tp_mailbox.fatbin` (`scripts/build-engine.sh --out DIST` writes them), `DIST/lib/`
  NCCL and cuBLAS (bundled unmodified), `DIST/plan/` the ranges (the plan's, plus the DSpark drafter's lines on every
  rank) and digest manifest, optional `DIST/data/`. The recipe, the copy tool and the benchmark tools come from the
  committed tree at REF; `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` and `LICENSES/` from the engine source DIR
  (upstream at `ENGINE_COMMIT` with `patches/`; its tree must equal `patches/APPLIED_TREE`), since they cover the code
  compiled into the binaries. `SHA256SUMS` inside, a `.sha256` beside it.
- `stage-dist.sh RELEASE DIST`: copies `lib/`, `plan/` and `data/` of a published release tarball into DIST (they do
  not depend on the engine build), checked against its `SHA256SUMS`.
- `scrub-check.sh [--names FILE] [--authors] range A..B | tree [REF]`: looks for host names (from a private list),
  private IPv4 addresses, home and build-host paths, and email addresses. Prints each hit; exits 1 on any.

Rebuilding the v0.1 tarball from source:

```
scripts/build-engine.sh                                   # build/engine-src, build/dist/bin
scripts/release/stage-dist.sh tensorfold-inkling2-tp4-v0.1-aarch64-sm121.tar.gz build/dist
scripts/release/build-assets.sh build/dist --engine-src build/engine-src --version v0.1
```

Before publishing: `scrub-check.sh --authors range BASE..HEAD` and `scrub-check.sh tree HEAD` clean.
