# Third-party notices

This repository holds the Inkling-2-Small TP4 recipe for [TensorFold](https://github.com/ashhart/TensorFold), the tools
it runs, and `patches/`, the engine changes on upstream TensorFold. This file credits the outside code those include
and the sources they follow, with their licenses; the license texts are in [`LICENSES/`](LICENSES/).

The release tarball ships the engine's own `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` and `LICENSES/` (TensorFold's,
with this section added by the notices patch), which also cover the upstream code compiled into its binaries.

## TensorFold

[TensorFold](https://github.com/ashhart/TensorFold), Copyright 2026 TensorFold contributors, Apache License 2.0
([the license text](LICENSE); its notice is reproduced in [`NOTICE`](NOTICE)); code written before TensorFold 0.6.0
keeps its MIT notice in [`LICENSES/MIT.txt`](LICENSES/MIT.txt).

- `patches/*.patch` are changes to TensorFold at the commit in `ENGINE_COMMIT`; their context lines are TensorFold's.
- `tools/bench_openai.py`, `bench_cache.py`, `bench_concurrent.py`, `bench_args.py`, `bench_auth.py`,
  `bench_evidence.py`, `bench_metrics.py`, `bench_stream.py`, `bench_thinking.py`, `bench_timing.py` and
  `bench_prompts.json` are TensorFold's files at that commit, unmodified (`tools/bench_dspark.py` is ours and uses them).
- `LICENSE` is TensorFold's copy of the Apache License 2.0.

## Inkling-2-Small

The Inkling-2 family (`zig/src/families/inkling2/`, `zig/kernels/cuda/inkling2_*.cu`, `tools/zig/inkling2_*.py`), its
text layer (`zig/src/server/inkling2_reply.zig`, `chat_inkling2.zig`, `inkling2_text_test.zig`), the chat-template
additions made for it in `zig/src/core/template/`, the recipe in `recipe/inkling2-tp4/` and the release scripts in
`scripts/` are written for TensorFold: Copyright 2026 Scott Biggart, Apache License 2.0 (see `NOTICE`). The engine
files among them are added to TensorFold at commit `223da7659` by `patches/`; paths under `zig/`, `build.zig` and
`tools/zig/` (other than `inkling2_rank_copy.py`, `inkling2_rank_copy_test.py` and `inkling2_dspark_ranges.py`, which
are in this repository) are paths in that patched engine source (`scripts/build-engine.sh` makes it).

The outside code these files include is listed first, with its license and copyright. Then come the sources whose
numerics or behaviour they reimplement, the upstream TensorFold code they reuse, the projects credited as references
only, the third-party parts of the release binaries, and the model data.

### Code included from outside projects

**Transformers** ([huggingface/transformers](https://github.com/huggingface/transformers),
`src/transformers/models/inkling2/` and `src/transformers/conversion_mapping.py` at commit `109a7d19`), Apache
License 2.0 (see [the license text](LICENSES/Apache-2.0.txt)). `models/inkling2/` is Copyright 2026 the HuggingFace
Team; `conversion_mapping.py` is Copyright (C) 2025 the HuggingFace Inc. team. Transformers ships no NOTICE file.
These parts are adapted from it, modified for TensorFold, and carry that copyright and license where they appear:

- `tools/zig/inkling2_ref.py` writes out the `inkling2_text` weight renames and the gate/up de-interleave (the
  `Interleave` converter) of `conversion_mapping.py` (`LAYER_RENAMES`, `deinterleave`) and restates the decoder loop
  of `Inkling2TextModel.forward` (`masks`, `run_layer`, `finish`). Its `LazyExperts.forward` is
  `Inkling2Experts.forward`, changed to fetch each expert's weights from the checkpoint on first use.
- `tools/zig/inkling2_op_ref.py`: `tau_of` and `dyn_conv` split the attention's log scaling and
  `Inkling2DynamicShortConvolution.forward` into ops.
- `tools/zig/inkling2_moe_check.py`: `exact_sums` is `Inkling2Experts.forward` with fp64 GEMMs.
- `zig/src/core/template/template.zig` (`aliasReasoning`) is `_alias_reasoning_fields` from Transformers'
  `utils/chat_template_utils.py` (Copyright 2024 The HuggingFace Team, Apache License 2.0), rewritten in Zig. As
  `apply_chat_template` does, it copies a message's first non-null `reasoning_content`, `thinking` or `reasoning` into
  `reasoning_content`, and into `thinking` when the message has none. It applies to every chat template, not only
  Inkling-2's.

The other oracle tools (`inkling2_ref_check.py`, `inkling2_attn_check.py`, `inkling2_f64.py`, and
`inkling2_dspark_ref.py`) import Transformers' Inkling-2 modules, masks, cache and tokenizer at run
time. Where they recompute part of the math themselves (the attention in `inkling2_f64.py`, `Inkling2NormedEmbedding`
in `inkling2_dspark_ref.py`), they copy no Transformers code. Transformers itself is not included.

**TokenSpeed** ([lightseekorg/tokenspeed](https://github.com/lightseekorg/tokenspeed)), Copyright (c) 2026 LightSeek
Foundation, MIT License (see [the license text](LICENSES/TokenSpeed-MIT.txt)). TokenSpeed ships no NOTICE file.

- `tools/zig/inkling2_ref.py` ports TokenSpeed's Torch reference for the block-scaled experts
  (`tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/block_scaled.py` at commit `f27ae32d`, the same file as at
  `04ab2e08`: `_fp4_round`, `reference_block_quantize`, `_decode_values`, `reference_capped_swiglu` and the slot fold
  of `torch_block_scaled_moe_routes`). It keeps TokenSpeed's copyright and permission notice above `_ts_fp4_round`.
- The DSpark drafter (`tools/zig/inkling2_dspark_ref.py`, `zig/kernels/cuda/inkling2_dspark.cu`,
  `zig/src/families/inkling2/cuda_dspark.zig`) ports the drafter math of TokenSpeed's native
  Inkling-2 draft (`python/tokenspeed/runtime/models/inkling2_mtp.py`, `execution/drafter/inkling2.py` and
  `execution/drafter/dflash.py`, including `Inkling2MTPHead.refresh_fold` and the bf16 `RotaryEmbedding` cache),
  under the same copyright and license.

### Numerics and behaviour followed

These files reimplement another project's math or behaviour. They copy none of its code, and the project is not
included.

- **Transformers** (the copyright and license above). The model math, its rounding points and the tensor layouts
  follow `modeling_inkling2.py` and `configuration_inkling2.py`, and each file names the module it follows:
  `zig/kernels/cuda/inkling2_attn.cu`, `inkling2_conv.cu`, `inkling2_head.cu`, `inkling2_moe.cu` and
  `inkling2_res.cu` (attention and its L2 clips, the short convolutions, the head and softcap, the router and
  experts, the block residuals), the router order in `inkling2_ck.cu`, and `zig/src/families/inkling2/config.zig`
  (the configuration defaults), `roles.zig`, `plan.zig`, `cuda_attn.zig`, `cuda_conv.zig`, `cuda_forward.zig`,
  `cuda_head.zig`, `cuda_moe.zig`, `cuda_ops.zig`, `cuda_res.zig` and `cuda_weights.zig`. The gate/up row interleave
  of the split experts (`plan.zig`) reads the rows as the `Interleave` converter in `conversion_mapping.py` does.
- **TokenSpeed** (the copyright and license above). The checkpoint-mode kernels compute the numerics of its Torch
  reference, one rounding per Torch op: `zig/kernels/cuda/inkling2_ck.cu`, `zig/src/families/inkling2/cuda_moe.zig`
  (checkpoint mode) and `quant.zig`. The slot order of the fold (`tokenspeed_slot_order` in
  `tools/zig/inkling2_ref.py`, after `torch_block_scaled_topk`) and the fp32 unembed output of checkpoint mode
  (`cuda_head.zig`) follow TokenSpeed. The gate/up interleave in `plan.zig` also matches TokenSpeed's
  `gate_up[:, 0::2]` read. The quantization manifest parser in `quant.zig` follows the checks of TokenSpeed's
  `Inkling2QuantConfig` (`python/tokenspeed/runtime/layers/quantization/inkling2.py`): the group keys it accepts and
  the FP32, BF16, NVFP4 and MXFP8 group settings. The drafter and
  vision tensor names in `roles.zig` follow TokenSpeed's `inkling2_mtp.py` and `inkling2_vision.py`.
  `tools/zig/inkling2_ref_check.py --tokenspeed TS_DIR` runs TokenSpeed's own `block_scaled.py` from a checkout the
  user supplies, unmodified (its unused imports stubbed); it is not copied or shipped.
- **Four Over Six** ([mit-han-lab/fouroversix](https://github.com/mit-han-lab/fouroversix); J. Cook, J. Guo, G. Xiao,
  Y. Lin, S. Han, "Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling",
  [arXiv:2512.02010](https://arxiv.org/abs/2512.02010)). The checkpoint's `scale_selection_rule: fouroversix_mae`
  names this method. For each block of 16, the checkpoint-mode NVFP4 activation quantizer (`quant.zig`,
  `inkling2_ck.cu`, `tools/zig/inkling2_ref.py`) picks the scale with the lower reconstruction MAE, between a
  candidate that maps the block maximum to 6 and one that maps it to 4. We use the method as TokenSpeed implements
  it; no Four Over Six code is copied or linked.
- **PyTorch** ([pytorch/pytorch](https://github.com/pytorch/pytorch)), BSD-3-Clause, Copyright (c) 2016- Facebook,
  Inc. and the other holders listed in [the license text](LICENSES/PyTorch-BSD-3.txt). The oracle and its checks
  (`tools/zig/inkling2_*.py`) run on PyTorch, imported at run time. The CUDA kernels (`inkling2_ck.cu`,
  `inkling2_conv.cu`, `inkling2_moe.cu`, with `cuda_moe.zig` and `cuda_ops.zig`) reproduce the rounding PyTorch's
  CUDA ops were measured to apply: one rounding per op, division by a Python scalar as a multiply by its fp32
  reciprocal, and the summation order of `sum(-1)`. That behaviour was matched by testing against PyTorch's outputs.
- **Inkling-2-Small's reply grammar.** The text layer (`zig/src/server/inkling2_reply.zig`, `chat_inkling2.zig` and
  its test `inkling2_text_test.zig`) reads the reply grammar of the checkpoint: the `response_template` in
  `tokenizer_config.json`, the typed-part marker tokens and `chat_template.jinja`, published by Thinking Machines in
  [thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) under the Apache
  License 2.0. Its handling of text outside a part follows Transformers' response-template parser
  (`utils/chat_parsing/response_parser.py`, Copyright 2026 The HuggingFace Team, Apache License 2.0). No code or
  template text is copied from either; the test builds its own small template, and the release files are read only
  at run time.
- **Jinja2** ([pallets/jinja](https://github.com/pallets/jinja)), Copyright 2007 Pallets, BSD 3-Clause License. The
  chat-template `float` filter added for Inkling-2 (`zig/src/core/template/filters.zig`) follows the behaviour of
  Jinja2's `do_float` filter: `float(value)`, or the given default (`0.0` if none) where Python raises `TypeError` or
  `ValueError`.
- **Python's `float()` of a string** (`zig/src/core/template/pyfloat.zig`, used by that filter) follows CPython
  3.12's `float()` on a `str` ([python/cpython](https://github.com/python/cpython), Python Software Foundation
  License Version 2): Unicode whitespace and decimal digits read as ASCII, then Python's float-literal grammar, with
  underscores between digits and `inf`/`infinity`/`nan`. Its table of decimal-digit runs (the zero of each run of
  ten `Nd` characters) is data from the [Unicode Character Database](https://www.unicode.org/ucd/) 15.0.0 as Python
  3.12's `unicodedata` reports it; Copyright © 1991-2026 Unicode, Inc., Unicode License V3
  ([`LICENSES/Unicode-3.0.txt`](LICENSES/Unicode-3.0.txt), from <https://www.unicode.org/license.txt>).

### Upstream TensorFold code reused

The family reuses upstream [TensorFold](https://github.com/ashhart/TensorFold) code under the same license (Copyright
2026 TensorFold contributors, Apache License 2.0; see `LICENSE` and `NOTICE`; code from before 0.6.0, including the
shipped `tools/bench_openai.py` and `tools/bench_concurrent.py`, keeps its MIT notice in `LICENSES/MIT.txt`):

- `mma_fp4` in `zig/kernels/cuda/inkling2_ck.cu` is `zig/kernels/cuda/mma4.cuh`'s, and `mma_mx8` is its MXFP8
  counterpart. `inkling2_head.cu` keeps `dense_rows.cu`'s summation order. The lane backend in
  `zig/src/families/inkling2/cuda_lanes.zig` follows `minimax_m3/cuda_lanes.zig` and `glm/cuda_lanes.zig`.
- The family builds on the N-rank collectives and the multirank RoCE wire with its four-rank relay in
  `zig/src/fabric/tp/`, the GLM family's CUDA rank, shard and boundary code, the MiniMax-M3 family's CUDA skeleton and
  DSpark drafter, and `tf-tp-test`. `zig/src/families/inkling2/cuda_wire.zig` borrows
  `glm/cuda_roce_multi.zig`.
- The recipe's NCCL `NCCL_CUMEM_ENABLE=1` and `NCCL_RUNTIME_CONNECT=1` settings follow
  `docs/recipes/glm-5.3-flash-cuda.md`.
- The recipe's RoCE gathers (`TF_INK2_ROCE_MULTI=1`, on by default, through `cuda_wire.zig` with the shipped
  `bin/tp_mailbox.fatbin`; and the opt-in `TF_ROCE=1`) run upstream's RoCE transports (`zig/src/cuda/roce.zig`, `zig/src/fabric/tp/`) unchanged.
  These adapt [b12x](https://github.com/local-inference-lab/b12x)'s RoCEnante one-shot all-gather (Copyright 2026
  Luke Alonso and the b12x contributors, Apache License 2.0), which upstream TensorFold credits in the CUDA section of
  its `THIRD_PARTY_NOTICES.md` (shipped in the release tarball).

### References credited (nothing included)

These shaped the recipe or the design. None of their code, text or files is included, so their licenses place no
conditions on TensorFold; they are credited by name.

- **MiaAI-Lab's four-Spark DeepSeek recipe**
  ([MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold),
  commit `c744e6c7`), Copyright 2026 MiaAI-Lab, Apache License 2.0. The recipe's NCCL settings over RoCE
  (`NCCL_CROSS_NIC`, subnet-aware routing with a subnet for each PCIe half, no P2P or SHM transport, four channels,
  `NCCL_ALGO=Ring`, `NCCL_IB_MERGE_NICS=0` and, on the switchless ring, a per-rank NCCL graph file) follow its
  `scripts/nodes.sh` (`config.example.sh`, `rank.sh`, `fabric/host/`, `fabric/ring-experimental/`,
  `tools/zig/inkling2_rank.sh`). The ring's measurements used per-rank NCCL graph files generated for the same four
  hosts by its `graph_xml`; those files are not included. The checks in `smoke.py` (chat, thinking, a tool-call round
  trip, streamed equals whole, a long prompt) follow its `tools/validate.py`. Two decode ideas follow its engine
  patches: 0019 (graph-capture agreement) for one CUDA graph a decode step on every rank, and 0037 (NVFP4 checkpoint)
  for running the experts on the FP4 tensor cores. That recipe credits its own sources, among them MiaAI-Lab's
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  recipe and webzone's PCIe-twin rails.
- **MiaAI-Lab's switchless-ring guide**
  ([DeepSeek-v4.1-Flash-DGX-Sparks `docs/switchless-ring.md`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/64b02dea8681fb1d8e3a659ea432a8e9e9ba3995/docs/switchless-ring.md),
  Copyright (C) 2026 zurih, AGPL-3.0-or-later). The ring addressing in `recipe/inkling2-tp4/fabric/ring-experimental/`
  follows it: one /24 per cable, `.10` on the originating rank's port, `.11` on the next rank's, and MTU 9000.
  TensorFold extends it to the second PCIe half.
- **NVIDIA Developer Forums** ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/)). The rule to reboot a host after any fabric cable change (`recipe/inkling2-tp4/README.md`,
  `fabric/README.md`, `fabric/switch/README.md`, `fabric/host/README.md`) and the PCIe slot power check in `rank.sh`
  and `status.sh` follow a report by forum user daikw in
  [ConnectX-7 Inter-Spark Link Capped at ~13 Gbps](https://forums.developer.nvidia.com/t/connectx-7-inter-spark-link-capped-at-13-gbps-expected-200-gbps-pcie-power-throttling-27w/363461/4):
  after recabling, a link stayed near 13 Gb/s until the host was rebooted with its cables in place. We saw the same
  on our hosts.
- **z-lab DFlash** ([z-lab/dflash](https://github.com/z-lab/dflash), MIT License, Copyright © 2026 Z Lab). The DSpark
  drafter's block design (a block of 8 rows, the anchor's embedding followed by mask embeddings, drafted in one
  non-causal pass over the target's hidden-state taps) is DFlash's. It reached TensorFold through TokenSpeed, whose
  `Inkling2DSpark` drafter extends TokenSpeed's own `DFlash` class. The drafter contains no z-lab code.
- **vLLM's Inkling-2-Small recipe** ([vllm-project/vllm](https://github.com/vllm-project/vllm),
  `recipes/models/thinkingmachines/Inkling-2-Small.yaml`, read at commit `43c1db8c`; Copyright contributors to the
  vLLM project, Apache License 2.0; <https://recipes.vllm.ai/thinkingmachines/Inkling-2-Small>) was a design
  reference for the placement (attention split across the ranks by heads, the experts spread over the group:
  `plan.zig`, `cuda_tp.zig`) and for DSpark at the checkpoint's full width of 7 drafts (`cuda_dspark.zig`).
- **Community Inkling-Small recipes for DGX Spark**:
  [0xSero/Inkling-Small-2-Sparks](https://github.com/0xSero/Inkling-Small-2-Sparks),
  [Reederey87/inkling-small-nvfp4-2x-dgx-spark](https://github.com/Reederey87/inkling-small-nvfp4-2x-dgx-spark) and
  [PixelML/Inkling-Small-NVFP4-Dual-DGX-Spark](https://github.com/PixelML/Inkling-Small-NVFP4-Dual-DGX-Spark). Their
  published speeds and findings (fp32 reductions in decode, a CUDA graph for each batch size, the draft block width)
  were planning reference points.
- **0xSero's inkling2-exl3** ([0xSero/inkling2-exl3](https://github.com/0xSero/inkling2-exl3), MIT License). Ideas
  only: `/health`'s KV capacity and admission report (`server/status_routes.zig` `kv`, in the engine patches), and the
  measuring process behind `recipe/inkling2-tp4/bench.sh` and `tools/inkling2_bench.py` (a one-command speed sweep with
  JSON out and a check command, a fixed realistic prompt set measured on the server's clock and at temperature 0.7 too,
  a speed ledger against a bandwidth floor, concurrent needle sessions). No code was copied; the prompts
  (`tools/inkling2_prompts.json`) and the needle text are our own.
- **BertholomusAI's four-Spark TensorFold recipes**
  ([deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10),
  [mistral-large-4-tensorfold-tp4-4xgb10](https://github.com/bertholomus/mistral-large-4-tensorfold-tp4-4xgb10),
  [glm-5.3-tensorfold-tp4-4xgb10](https://github.com/bertholomus/glm-5.3-tensorfold-tp4-4xgb10); Apache-2.0) were consulted when planning
  the four-host layout of `recipe/inkling2-tp4/`, and [spark-attic](https://github.com/bertholomus/spark-attic)
  (MIT OR Apache-2.0) for GB10 memory headroom.
- **NVIDIA PTX ISA 8.7/8.8 documentation** ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/)) **and LLVM's NVPTX backend** ([NVPTXUsage](https://llvm.org/docs/NVPTXUsage.html)). `mma_mx8` in `inkling2_ck.cu` uses the
  `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32 ... ue8m0` instruction form and its
  scale-operand layout as NVIDIA's PTX ISA documentation gives them ("Block Scaling for mma.sync"); LLVM's list of
  valid NVPTX block-scale combinations confirmed which forms exist.
- **perftest** ([linux-rdma/perftest](https://github.com/linux-rdma/perftest), GPL-2.0 or the OpenIB.org BSD
  license). `recipe/inkling2-tp4/rank.sh` (`bw-server`, `bw-client`) and `status.sh --bw` run the host's installed
  `ib_write_bw` and read its output. The raw-RDMA figures in `recipe/inkling2-tp4/fabric/*/README.md` were measured
  with perftest 24.01. Install it from your distribution (the `perftest` package).
- **MikroTik RouterOS** ([mikrotik.com](https://mikrotik.com)). `recipe/inkling2-tp4/fabric/switch/crs812-fabric.rsc` is a RouterOS configuration script
  written for TensorFold, for a MikroTik CRS812-8DS-2DQ-2DDQ on RouterOS 7.20, checked against the switch's own
  configuration export. It contains no MikroTik code. MikroTik and RouterOS are trademarks of SIA Mikrotīkls;
  TensorFold is not affiliated with or endorsed by MikroTik.

### Release binaries: toolchain and NVIDIA components

- **Zig.** The executables in `bin/` of the release archives (`tensorfold-native`, `tf-inkling2-decode`,
  `tf-inkling2-plan`, `tf-inkling2-fold`, `tf-tp-test`) are built with [Zig](https://ziglang.org) 0.17.0 and include compiled code from
  Zig's standard library and its compiler_rt. MIT License (Expat), Copyright (c) Zig contributors; see
  [the license text](LICENSES/Zig-MIT.txt).
- **NVIDIA CUDA Toolkit.** Those executables and `bin/tp_mailbox.fatbin` contain GPU code compiled by NVIDIA's CUDA
  13.3 compiler (`nvcc`) from TensorFold's own sources. That code includes inline functions from the CUDA headers
  (`cuda_bf16.h`, `cuda_fp16.h`, `cuda_fp8.h` and others), Copyright NVIDIA Corporation, distributed in object form as
  the [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html) permits; they remain under its terms,
  not under the Apache License. The release bundles no NVIDIA compiler or CUDA runtime; the CUDA driver
  (`libcuda.so.1`) comes from the host. Two NVIDIA libraries are bundled in `lib/`, below.
- **NVIDIA NCCL** (`lib/libnccl.so.2`, version 2.30.7 for CUDA 13, aarch64) is bundled unmodified, the same bytes as
  NVIDIA's `libnccl2_2.30.7-1+cuda13.3_arm64.deb` (sha256 in `lib/SHA256SUMS` of the install). NCCL is Copyright (c)
  2015-2026 NVIDIA CORPORATION & AFFILIATES, mostly under the Apache License 2.0 with parts (the bundled DOCA GPUNetIO
  code) under the BSD 3-Clause license; its `LICENSE.txt` from [NVIDIA/nccl](https://github.com/NVIDIA/nccl) at tag
  `v2.30.7-1` is [LICENSES/NCCL-LICENSE.txt](LICENSES/NCCL-LICENSE.txt) and is also shipped as
  `lib/LICENSE-NCCL.txt`. NCCL is not covered by this repository's Apache License notice for its own work.
- **NVIDIA cuBLAS / cuBLASLt** (`lib/libcublas.so.13` and `lib/libcublasLt.so.13`, version 13.1.1.3, aarch64) are
  bundled unmodified (the same bytes as NVIDIA's `nvidia-cublas` 13.1.1.3 wheel for aarch64). They are NVIDIA
  proprietary software under the [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/), which lists them
  among the redistributable components (Attachment A); they are not covered by the Apache License. The EULA text as
  NVIDIA ships it in that wheel is [LICENSES/NVIDIA-CUDA-EULA.txt](LICENSES/NVIDIA-CUDA-EULA.txt), also shipped as
  `lib/LICENSE-NVIDIA-CUDA-EULA.txt`; the libraries ship with that text and without modification, as it requires.
  Do not repackage them without the EULA text. The container image takes the same files from `lib/`; it is built on each host and never pushed to
  a registry.
- **libtfgrammar** (`lib/libtfgrammar.so`, aarch64; structured output) is TensorFold's grammar library
  (`zig/src/grammar/tf_grammar.cc`) built with the C++ sources of [xgrammar](https://github.com/mlc-ai/xgrammar) v0.2.8
  (commit `97787376faee5ed8466cfad57c99855e4ce2f6aa`, Copyright (c) 2024 by XGrammar Contributors, Apache License 2.0)
  and the headers of [dlpack](https://github.com/dmlc/dlpack) (Apache License 2.0) and
  [picojson](https://github.com/kazuho/picojson) (BSD 2-Clause License, Copyright 2009-2010 Cybozu Labs, Inc. and
  2011-2014 Kazuho Oku). Their texts are [LICENSES/xgrammar-Apache-2.0.txt](LICENSES/xgrammar-Apache-2.0.txt),
  [LICENSES/xgrammar-NOTICE.txt](LICENSES/xgrammar-NOTICE.txt), [LICENSES/dlpack-Apache-2.0.txt](LICENSES/dlpack-Apache-2.0.txt)
  and [LICENSES/picojson-BSD-2-Clause.txt](LICENSES/picojson-BSD-2-Clause.txt), also shipped in `lib/` beside the library.
  `scripts/build-engine.sh` clones xgrammar at that commit to build it.

### Model data

`zig/src/families/inkling2/fixtures/config.json` and `hf_quant_config.json` are copied unchanged from
[thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) (Thinking Machines) at
revision `aac974122acd33ddb0d3e1f54433a9c5dd2249ce`, licensed under the Apache License 2.0 (see
[the license text](LICENSES/Apache-2.0.txt)); `inventory.tsv` lists that revision's safetensors headers. The release
ships no NOTICE file and no copyright line, so there is no further notice to carry. Its model card also links Thinking
Machines' Acceptable Use Policy for the model.

The release tarball holds data derived from that checkpoint at the same revision, under the same license. These files
are computed by TensorFold's tools and are not files from the original repository:

- `plan/ranges-w4-r*.tsv`: byte ranges and tensor names read from the checkpoint's safetensors headers;
  `plan/inkling2-digests-w4.tsv`: sha256 digests of the checkpoint's bytes (`tf-inkling2-plan`).
- `data/chat_short.ids.json`, `data/tool_call.ids.json` and the prompt ids in `data/long_4000.safetensors`:
  TensorFold's own prompts, rendered with the checkpoint's chat template and tokenizer.
- `data/decode.json` and the other tensors in `data/long_4000.safetensors` (hidden states, routes, logits): outputs of
  the model at that revision, computed by `tools/zig/inkling2_ref.py`.

The model weights are not included in TensorFold or its release assets. The DSpark fold file
(`inkling2-dspark-fold.safetensors`, computed from the unembed and the drafter's `hidden_out`) is derived weights;
each user builds it locally from their own copy of the weights with the release's `tf-inkling2-fold` (the recipe's
`setup.sh` and `weights.sh` do this on each host) or with `inkling2_dspark_ref.py fold`, and release assets must not
include it. "Inkling" and
"Thinking Machines" are named only to identify the source; TensorFold is not affiliated with or endorsed by Thinking
Machines.
