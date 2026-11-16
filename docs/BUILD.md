# Building the engine from source

The release binaries are TensorFold's native engine with the Inkling-2 family: upstream
[TensorFold](https://github.com/ashhart/TensorFold) at the commit in [`ENGINE_COMMIT`](../ENGINE_COMMIT)
(`223da7659b2f1ec7a6fe666cb064d839382d6fce`) with the patches in [`patches/`](../patches/) applied in order.

## The patches

| Patch | What it adds or changes |
| --- | --- |
| `0001` family | the Inkling-2 family (`zig/src/families/inkling2/`: config, plan and weights, the tensor-parallel CUDA runner with the KV page pool, the decode tool), its server text layer, structured output and chat-template additions, `/health`'s `kv`, and the native engine's family hooks |
| `0002` kernels | the CUDA kernels (`zig/kernels/cuda/inkling2_*.cu`: attention, MoE, convolution, head, residual, fused, checkpoint mode) and their launch wrappers |
| `0003` batching | the lanes core for several streams a decode step and interleaved prefill |
| `0004` DSpark | the drafter kernel and runner, the native fold tool `tf-inkling2-fold`, the draft depth rule |
| `0005` prompt cache | exact prefix snapshots, the `shared_first` rules, the store budget (upstream core edits) |
| `0006` tooling | the checkpoint-mode oracle and checks (`tools/zig/inkling2_*.py`, PyTorch), template and tokenizer parity |
| `0007` build glue | `zig build` steps for the family, its kernels and tools |
| `0008` notices | the Inkling-2 credits in `NOTICE` and `THIRD_PARTY_NOTICES.md` (in the fork's wording: the recipe and release scripts are in this repository), and the license texts they cite |

Applied with `git am --committer-date-is-author-date` and the committer `scripts/build-engine.sh` sets, the series gives
commit `dee0906d3d18db50231d63c3112b5bf19942c210`, tree `bef143dea5d7866559f69e617fd408bbc9104d5d`
([`patches/APPLIED_TREE`](../patches/APPLIED_TREE)). That is the engine branch
[`inkling2-sparks-v0.1`](https://github.com/swbiggart/TensorFold-upstream/tree/inkling2-sparks-v0.1) of our TensorFold
fork, at commit
[`dee0906d3d18db50231d63c3112b5bf19942c210`](https://github.com/swbiggart/TensorFold-upstream/tree/dee0906d3d18db50231d63c3112b5bf19942c210),
and the release branch's tree restricted to the engine's paths, except that `NOTICE` and `THIRD_PARTY_NOTICES.md` say
where the recipe lives. The branch never moves; a later version gets its own branch.

Other ids you may meet: the tarball's `VERSION` names the `source commit` of this repository it was packed from (the
container image is tagged with it). The tarball's binaries are the gated build, compiled from the same engine source
as the fork commit above (only `NOTICE` and `THIRD_PARTY_NOTICES.md`, which are not compiled, differ).

## The toolchain

[`scripts/build-engine.Dockerfile`](../scripts/build-engine.Dockerfile): NVIDIA's `nvidia/cuda:13.3.1-devel-ubuntu24.04`
pinned by digest (`sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87`, the multi-arch index; its
linux/amd64 image `sha256:03c372fd9c65fe7739279f8c65473b315dc61efaaffab03e1e65bc7be7aee61e` has the base layers of the
image the release was built in), with ca-certificates, curl, git, xz-utils, cmake, nasm and python3 from Ubuntu, and
Zig 0.17.0 from ziglang.org checked by sha256 (x86_64 `1cbe9df9...e2026`, aarch64 `9e8d1166...34ff8`). nvcc is
CUDA 13.3.1's; g++ 13 is its host compiler. The image holds no sources. Structured output's `libtfgrammar.so` is built
with [xgrammar](https://github.com/mlc-ai/xgrammar) v0.2.8 at commit `97787376faee5ed8466cfad57c99855e4ce2f6aa`
(and its `3rdparty/dlpack` submodule), which `scripts/build-engine.sh` clones (or takes with `--xgrammar DIR`).

## The build

`scripts/build-engine.sh` clones upstream at `ENGINE_COMMIT` (or takes `--src DIR`), applies the patches, checks the
tree, builds the image for the build machine's architecture and runs, with the checkout mounted at `/src/tf` and Zig's
global cache at `/tmp/.cache/zig` (the release build's paths), as the calling user:

```
zig build native tf-inkling2-decode tf-tp-test tf-inkling2-fold tf-inkling2-plan -Doptimize=ReleaseFast \
  -Dtarget=aarch64-linux-gnu.2.28 -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dtp-nvcc=/usr/local/cuda/bin/nvcc \
  --prefix zig-out-arm
zig build grammar -Dxgrammar=/xg -Doptimize=ReleaseFast -Dtarget=aarch64-linux-gnu.2.28 --prefix zig-out-arm
nvcc -fatbin -O3 -std=c++17 -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_121,code=sm_121 \
  -gencode=arch=compute_90,code=compute_90 -o zig-out-arm/tp_mailbox.fatbin zig/src/fabric/tp/mailbox.cu
```

It copies `tensorfold-native`, `tf-inkling2-decode`, `tf-inkling2-plan`, `tf-inkling2-fold`, `tf-tp-test` and
`tp_mailbox.fatbin` to `build/dist/bin/`, `libtfgrammar.so` to `build/dist/lib/`, and writes `build/dist/BUILD-INFO` (commits, image, toolchain, sha256s).

## Reproduction check (v0.1)

On an x86-64 Linux build machine with Docker, from nothing: a fresh clone of
`https://github.com/ashhart/TensorFold` at `223da7659`, `git am` of the eight patches (commit `dee0906d3`, tree
`bef143dea5d7`, as above), a fresh clone of xgrammar at `97787376` (tag v0.2.8, with `3rdparty/dlpack`), the toolchain
image built from the Dockerfile, an empty Zig cache; 463 s of build. Compared with the v0.1 release's `bin/` and
`lib/libtfgrammar.so` (the binaries every v0.1 gate ran):

| File | Size | sha256 of the release's file | Rebuilt |
| --- | ---: | --- | --- |
| `tf-inkling2-fold` | 4,558,136 | `881fc152169da442f69c78cb3a6e3928394a8ba0ab763bdd18d6c1a815cf2fc3` | equal |
| `tf-inkling2-plan` | 6,192,616 | `f4a7e1c3bd949b74f52892520c29ef3b475be5a9ce6dd6d78af113fc53e5abf0` | equal |
| `tf-tp-test` | 8,271,192 | `f8153271182d41307ff186b6d57375d4b0f24dc0a94acfefe10a816436ab74fb` | equal |
| `tp_mailbox.fatbin` | 344,784 | `5e1f6360b6b8330956101897e9ec0edd898ba7e9b7188b9afc4c28936fefee29` | equal |
| `lib/libtfgrammar.so` | 2,993,720 | `33e430dbd081f40597201787641d1f1186e05442c2f7f23d010f359a28edb45c` | equal |
| `tf-inkling2-decode` | 21,624,296 | `617702552b882cfb8e37f87653ebcb7a6dd49745607e516539876370ee864491` | `12352ca37ff2eff45c743728790131ce776310c4163ab2b60dd4e5a2596aa4e0`: 34 bytes differ, all in `.strtab` |
| `tensorfold-native` | 113,782,240 | `4c42f47e9c9a6dc094ba6dd4c9d219cbf801d48e86771eefbcfdd68d093f77b2` | `21531f469ec64ff468d16fdf613f3b70ea38d90294e88249e3cba5912da5d422`: 6,452 bytes differ, 90 in `.strtab` and 6,362 inside two embedded upstream cubins |

The two that differ have the same size, and every difference comes from a symbol name:

- **Zig's anonymous-symbol numbers** (`__anon_N`) in the ELF symbol table (`.strtab`), which is not loaded. They come
  from the order of Zig's semantic analysis and change from build to build.
- **nvcc's internal-linkage names** in the device code of two upstream kernels, `zig/kernels/cuda/qmm.cu` and
  `qmm_group.cu` (embedded in `tensorfold-native`'s `.rodata`; 2 of its 122 cubins): the name of CCCL's
  `cuda::std::iter_swap` customization object carries a module id nvcc makes for each compilation
  (`_ZN49_INTERNAL_4ca677d1_12_qmm_group_cu_b71638e9_10849...` against `_ZN48_INTERNAL_4ca677d1_12_qmm_group_cu_62be725f_2287...`).
  This time the id is one character shorter, so the offsets into those cubins' string tables move too (their symbol,
  info and debug-frame sections). `nvdisasm` of both cubins gives the same instructions; the only differing lines are
  the ones naming that object.

So the code and data are the same, and `tf-inkling2-decode`'s loaded image is identical byte for byte. Because of
these names, two builds are not byte-identical in general; the ranks compare their executables' hashes at start-up, so
run one build on all four hosts.

## Making the release tarball

NVIDIA's NCCL and cuBLAS in `lib/`, `plan/` and `data/` (computed from the checkpoint) do not depend on the build;
`scripts/release/stage-dist.sh` takes them from the published tarball (keeping the `libtfgrammar.so` the build made). `scripts/release/build-assets.sh` then packs the
binaries, the recipe and tools at a commit of this repository, and the engine source's `LICENSE`, `NOTICE`,
`THIRD_PARTY_NOTICES.md` and `LICENSES/` (TensorFold's, with the Inkling-2 section: they cover the code compiled into
the binaries and the bundled libraries):

```
scripts/release/stage-dist.sh tensorfold-inkling2-tp4-v0.1-aarch64-sm121.tar.gz build/dist
scripts/release/build-assets.sh build/dist --engine-src build/engine-src --version v0.1
```

Checked for v0.1: the release asset `tensorfold-inkling2-tp4-v0.1-aarch64-sm121.tar.gz` (sha256
`27d4eff27404b7b3bc5b7eb0d3aa3fb2c2890879499851593cf51dc8a9bcfedb`) is this script's output at commit `836c1a0` of this repository with the gated build's
`bin/` and `lib/`, the release's `plan/` and `data/`, and the engine source above. Against the tarball every release
gate ran on (the same binaries, libraries, plan and data), its 129 members are the same except `VERSION`,
`SHA256SUMS`, the Markdown files, `NOTICE` (the fork's wording) and four recipe files changed after that run
(`bench.sh`, `setup.sh`, `setup/lib.sh`, `setup/test-lib.sh`); the fresh-install gates (host, container, checkpoint
copy) ran again on the asset itself. A tarball made from rebuilt binaries would differ in `bin/tensorfold-native` and
`bin/tf-inkling2-decode` as described above.

