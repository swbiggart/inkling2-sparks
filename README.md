# Inkling-2-Small on 4× DGX Spark: exact TP4

v0.2

by Scott Biggart · [@woolsey](https://x.com/woolsey)

[![Follow on X](https://img.shields.io/badge/Follow_on_X-@woolsey-000000?logo=x&logoColor=white)](https://x.com/woolsey)
![Version v0.2](https://img.shields.io/badge/version-v0.2-blue)
![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)

Serve `thinkingmachines/Inkling-2-Small`, exact to the checkpoint's own weights, split four ways (tensor parallel, TP4)
across four DGX Spark-class GB10 hosts on one 200G switch. `tensorfold-native` serves an OpenAI- and
Anthropic-compatible HTTP API on rank 0, which samples and drives the other three ranks. DSpark speculative decoding is
on by default.

Inkling-2-Small is [Thinking Machines'](https://thinkingmachines.ai) model (Apache-2.0, on
[Hugging Face](https://huggingface.co/thinkingmachines/Inkling-2-Small)); this recipe only serves it. This project is
not affiliated with or endorsed by Thinking Machines.

## What it is

- **Weights:** the checkpoint's bytes at revision `aac974122acd33ddb0d3e1f54433a9c5dd2249ce`, unchanged; each host holds
  its rank's quarter (about 92 GiB).
- **Arithmetic:** experts quantize activations as the checkpoint specifies (NVFP4 for layers 0-39, MXFP8 for 40-47).
  Attention computes in fp32 and rounds to bf16 once, at its output.
- **Split:** attention by heads, each expert four ways, the output head by vocabulary, the input embedding by vocabulary
  over ranks 1-3 (rank 0 holds none).
- **DSpark:** the checkpoint's drafter (MLP projections in MXFP8) proposes tokens; the model checks every one. Past
  32,768 positions a stream drafts only while drafting beats plain decode at its depth.
- **Exact:** the model's weights and quantization unchanged; replies independent of timing, drafting, caching,
  concurrency, prompt chunking, KV paging and CUDA graphs ([Correctness](#correctness)).

Changes by version: [CHANGELOG.md](CHANGELOG.md).

## Requirements

- **Hosts:** four GB10 hosts (DGX Spark or OEM) on NVIDIA's Ubuntu 24.04-based OS, each with about 116 GiB of memory
  available, a ConnectX-7 200G port, driver 580+ (tested: 580.178.04), `libibverbs1`, `ibverbs-providers`, `python3`
  3.9+ and 100 GiB free under `ROOT` (108 GiB with `--container`, which also needs Docker Compose v2 and the NVIDIA
  Container Toolkit). `setup.sh`'s preflight checks these and names what is missing.
- **Network:** a 200GbE switch, one port a host (a QSFP56-DD 400G-to-2x200G breakout per two hosts on a CRS812), or,
  experimentally, a switchless ring ([fabric setup](recipe/inkling2-tp4/docs/INSTALL-MANUAL.md#step-2-fabric)); a LAN
  for ssh and the HTTP endpoint.
- **Control machine:** bash, curl and key-based ssh to the four hosts (it can be one of them); `sudo -n` only for `--fabric-install` and for
  Docker outside the `docker` group.
- **Optional:** for `CHECKPOINT`, a Linux host on the switch with the full checkpoint (366 GiB) and `python3`; Python
  3.11+ for `tools/bench_*.py`.

## Quick start

With the fabric configured:

```
git clone --branch v0.2 --depth 1 https://github.com/swbiggart/inkling2-sparks && cd inkling2-sparks/recipe/inkling2-tp4
cp hosts.conf.example my-sparks.conf      # HOSTS (ssh names in rank order) and ROOT (the install folder on the hosts)
./setup.sh my-sparks.conf \
  --release https://github.com/swbiggart/inkling2-sparks/releases/download/v0.2/tensorfold-inkling2-tp4-v0.2-aarch64-sm121.tar.gz \
  --sha256 4aa5a3f0a5639435320b354b7d16893c824c85987ff2bf28f9b2bc9b801470be
```

The sha256 pins the v0.2 tarball (998 MB). `setup.sh` installs, fetches weights, starts and runs the smoke and reference
checks; it stops at the first failure and skips finished steps. `./setup.sh --help` lists the options
([details](recipe/inkling2-tp4/docs/INSTALL-MANUAL.md#what-setupsh-does)).

- **Weights:** each host downloads its quarter from Hugging Face (the download resumes). With a full checkpoint on the
  switch, set `CHECKPOINT=host:/path` in the hosts file (or pass `--checkpoint host:/path`) to hash-check it and copy
  it over the fabric ([weights](recipe/inkling2-tp4/weights/README.md#from-a-full-checkpoint)); `--weights-dir DIR`
  verifies rank folders in place.
- **Container:** add `--container` to every command (not mixed with a host install). The image holds no weights and
  stays off registries ([details](recipe/inkling2-tp4/docs/INSTALL-MANUAL.md#container)).
- **Later:** `./setup.sh my-sparks.conf stop | start | status | smoke`; `stop check start` runs the W=4 decode check.
  Manual steps: [INSTALL-MANUAL.md](recipe/inkling2-tp4/docs/INSTALL-MANUAL.md).

HTTP listens on `HTTP_HOST:HTTP_PORT` on rank 0 (loopback by default); the model id is `model`:

```
curl -s http://127.0.0.1:8890/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "model", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

## Benchmarks

Four GB10 hosts on one 200G switch at the release defaults (DSpark on unless a row says plain). Change is v0.2 against
v0.1, relative (for a share too, not in points). Commands and methods:
[docs/BENCHMARKS.md](recipe/inkling2-tp4/docs/BENCHMARKS.md).

### One stream

Tokens per second.

| Workload | v0.1 | v0.2 | Change |
| --- | ---: | ---: | ---: |
| Plain | 31.0 | 33.9 | +9.4% |
| DSpark: tool call | 70.2 | 72.8 | +3.7% |
| DSpark: code | 60.8 | 79.7 | +31.1% |
| DSpark: thinking | 56.3 | 68.0 | +20.8% |
| DSpark: raw code completion | 52.7 | 56.8 | +7.8% |
| DSpark: prose | 40.0 | 53.9 | +34.7% |

### Total throughput

Total tokens per second while all streams are running, v0.1 and v0.2 on the same prompts; 256-token replies unless a
row says 128. `MAX_STREAMS=8` rows use `CONTEXT=32768`.

| Setting | Streams | v0.1 | v0.2 | Change |
| --- | ---: | ---: | ---: | ---: |
| Defaults (DSpark) | 1 | 83.5 | 94.1 | +12.7% |
| Defaults (DSpark) | 2 | 96.4 | 114.0 | +18.3% |
| Defaults (DSpark) | 4 | 91.4 | 118.5 | +29.6% |
| Defaults (DSpark), 128-token replies | 1 | 78.2 | 93.2 | +19.2% |
| Defaults (DSpark), 128-token replies | 2 | 91.4 | 115.1 | +25.9% |
| Defaults (DSpark), 128-token replies | 4 | 81.9 | 118.7 | +44.9% |
| `MAX_STREAMS=8`, plain | 1 | 30.4 | 33.5 | +10.2% |
| `MAX_STREAMS=8`, plain | 2 | 45.8 | 50.3 | +9.8% |
| `MAX_STREAMS=8`, plain | 4 | 62.4 | 68.2 | +9.3% |
| `MAX_STREAMS=8`, plain | 8 | 78.6 | 86.7 | +10.3% |
| `MAX_STREAMS=8`, DSpark | 1 | 82.9 | 95.2 | +14.8% |
| `MAX_STREAMS=8`, DSpark | 2 | 97.4 | 113.8 | +16.8% |
| `MAX_STREAMS=8`, DSpark | 4 | 92.0 | 116.9 | +27.1% |
| `MAX_STREAMS=8`, DSpark | 8 | 78.2 | 87.9 | +12.4% |

### Time to first token

Seconds. Cached is the same prompt again. The 512k rows use `CONTEXT=524288`.

| Prompt | v0.1 | v0.2 | Change |
| --- | ---: | ---: | ---: |
| A short prompt | 0.11 | 0.10 | -9.1% |
| 2k, cold | 1.32 | 1.18 | -10.6% |
| 8k, cold | 5.84 | 4.78 | -18.2% |
| 32k, cold | 30.6 | 20.7 | -32.4% |
| 128k (126,000 tokens), cold | 221 | 102 | -53.8% |
| 512k (519,983 tokens), cold | 2,754 | 826 | -70.0% |
| 2k, cached | 0.18 | 0.15 | -16.7% |
| 8k, cached | 0.18 | 0.15 | -16.7% |
| 32k, cached | 0.19 | 0.16 | -15.8% |
| 128k (126,000 tokens), cached | 0.189 | 0.158 | -16.4% |
| 512k (519,983 tokens), cached | 12.1 | 0.27 | -97.8% |
| 101,859 tokens, cold | 158.7 | 80.4 | -49.3% |
| 101,859 tokens, cached | 0.1 | 0.1 | 0.0% |
| 24,000 tokens, nothing else running | 31.7 | 22.0 | -30.6% |
| 24,000 tokens, beside one decoding stream | 82.2 | 58.1 | -29.3% |

### Long context

Decode tokens per second after a long prompt, one stream, 256-token reply. Book rows (a book as the system message) run
the default unless marked plain; 1M is a synthetic state.

| Context | v0.1 | v0.2 | Change |
| --- | ---: | ---: | ---: |
| 8k (8,163 tokens), plain | 30.3 | 33.3 | +9.9% |
| 32k (33,404 tokens), plain | 29.2 | 32.1 | +9.9% |
| 64k (67,804 tokens), plain | 27.6 | 30.4 | +10.1% |
| 128k (129,173 tokens), plain | 25.2 | 28.0 | +11.1% |
| 512k (519,970 tokens, the book), plain | 15.96 | 18.22 | +14.2% |
| 1M, plain | 10.8 | 12.5 | +15.7% |
| 8k (8,163 tokens), default | 50.8 | 66.6 | +31.1% |
| 32k (33,404 tokens), default | 67.0 | 69.9 | +4.3% |
| 64k (67,804 tokens), default | 61.8 | 73.8 | +19.4% |
| 128k (129,173 tokens), default | 51.5 | 67.8 | +31.7% |
| Book, about 126,250 tokens, prose | 24.8 | 27.3 | +10.1% |
| Book, about 126,250 tokens, code | 33.1 | 44.8 | +35.3% |
| Book, about 520,000 tokens, prose | 16.8 | 25.2 | +50.0% |
| Book, about 520,000 tokens, code | 21.7 | 26.4 | +21.7% |

### Install and load

Seconds. Each install ends with the smoke checks and the W=4 decode check passing.

| | v0.1 | v0.2 | Change |
| --- | ---: | ---: | ---: |
| Fresh install on the hosts (`setup.sh` from an empty folder, weights verified in place) | 341 | 342 | +0.3% |
| The same with `--container` (also the image built and copied to the other three hosts) | 579 | 588 | +1.6% |
| Fresh install with the weights copied from a full checkpoint (`CHECKPOINT=host:/path`) | 508 | 520 | +2.4% |
| The copy itself, a rank (about 92 GiB) | 75-91 | 83-85 | +10.7% to -6.6% |
| Start to ready (weights in the page cache) | 23 | 23 | 0.0% |

### `bench.sh`

What [Measuring your install](#measuring-your-install) runs. The needle uses `CONTEXT=32768` and `MAX_STREAMS=8`.

| | v0.1 | v0.2 | Change |
| --- | ---: | ---: | ---: |
| Prompt set, temperature 0, all 16 rows (tok/s) | 54.1 | 60.1 | +11.1% |
| Prompt set, temperature 0.7, all 16 rows (tok/s) | 47.2 | 46.6 | -1.3% |
| Needle, 8 sessions at once (s) | 207 | 148 | -28.5% |
| Ledger: one stream plain, share of the memory-bandwidth floor (%) | 68.5 | 74.3 | +8.5% |

## Correctness

Hashes are xxh64 of the decode logits. Detail: [docs/CORRECTNESS.md](recipe/inkling2-tp4/docs/CORRECTNESS.md).

- **Decode equals the whole-prompt forward** bit for bit at W=4, all 48 layers: 80/80 (`037a50ab15855ea6`) and 64/64
  after 130,944 tokens (`be7904f5a734c3ed`), in any KV page order, with and without CUDA graphs.
- **1M positions:** the decode check is bit-equal at 1,048,000 tokens on a synthetic state (`d148c4204f687da7`).
- **Attention** gives a row the same bits in every kernel form. On real layers every output lies inside the bound a
  correct fp32 computation must meet (checked against float64).
- **Same reply** with drafting on or off, alone or beside 1 or 3 other streams, cached or cold.
- **Ranks agree:** ranks with different settings, builds or weight digests refuse to start.
- **Structured output** equals the reference under the same masks for a `json_object` reply and a required tool call.

Not claimed: W=4 equal to W=1 in the last bits (attention partials are summed in a fixed rank order), or replies equal
to PyTorch's or v0.1's at near-ties.

### Agreement with the fp32 reference

Share of 2,447 forced positions where each build's token is the fp32 reference's top token:

| Reference | v0.1 | v0.2 | p (exact McNemar) |
| --- | ---: | ---: | ---: |
| fp32, token by token, experts decoded to bf16 | 98.2% | 97.4% | 0.0175 |
| fp32, one pass with the checkpoint's experts (as the servers compute) | 97.7% | 97.6% | 0.888 |

Every difference is where the reference's top two tokens are less than 1 logit apart (token by token).
Mean logit distance over the reference's top 64 (token by token): v0.1 0.432, v0.2 0.425 (Wilcoxon p 0.105).

## Settings you may change

In `release-defaults.sh`; each must be the same on all four ranks. No documented setting changes the output's bits. Details:
[docs/SETTINGS.md](recipe/inkling2-tp4/docs/SETTINGS.md).

| Setting | Default | What it does |
| --- | --- | --- |
| `CONTEXT` (`--context`) | 131072 | the largest request (prompt plus reply), up to 1,048,576 |
| `MAX_STREAMS` (`TF_INK2_MAX_STREAMS`) | 4 | most streams at once (1-8) |
| `DSPARK` (`TF_INK2_DSPARK`) | 1 | DSpark drafting |
| `PROMPT_CACHE_GIB` (`--prompt-cache-gib`) | 1.5 | prompt-cache arena a rank; 0 turns it off |
| `MEMORY_RESERVE_GIB` | 8 | memory each host keeps free |
| `DECODE_SHARE` (`--decode-share`) | 0.5 | a long prompt's time share beside decoding streams |
| `ROCE_MULTI` (`TF_INK2_ROCE_MULTI`), `MAILBOX_MAX_KB` | 1, 640 | small decode exchanges over RoCE writes; needs the switch |
| `XROWS` (`TF_INK2_XROWS`) | 1 | by-rows layer sums in prompt chunks; `WARMUP=0` skips the start-up warm-up |

To change one: `echo CONTEXT=524288 >> my-sparks.conf`, then `./setup.sh my-sparks.conf stop config start`.

## Context, streams and the KV page pool

The server runs as many streams as fit at a whole `CONTEXT` each, at most `MAX_STREAMS`, on one shared pool of
256-position KV pages sized from the memory free at start. Spare pages hold kept prompts.

| `CONTEXT` | Streams (default `MAX_STREAMS=4`) | Pool positions | 128k prompts kept with every stream idle |
| ---: | ---: | ---: | ---: |
| 131,072 (default) | 4 | 944,128 | 7 |
| 262,144 | 3 | 1,008,128 | 7 |
| 524,288 | 1-2, depending on free memory at start | 1,230,592 | 9 |
| 1,048,576 (plain) | 1 | 1,406,464 | 10 |

Rank 0's log reports the choice: `inkling2: context 131072 -> 4 streams on a KV page pool (...): N pages of 256
positions ...`. A request longer than `CONTEXT` gets HTTP 400 naming the limit; one whose pages are not free waits, and
kept prompts give their pages back first. Nothing streams before the first token, so long prompts need a long client
read timeout.

## API

| Endpoint | Supports |
| --- | --- |
| `/v1/chat/completions`, `/v1/completions`, `/v1/models` (OpenAI) | streaming, tools, reasoning, structured output |
| `/v1/responses` (OpenAI) | streaming events, function tools, reasoning items |
| `/v1/messages`, `/v1/messages/count_tokens` (Anthropic) | streaming events, `tool_use` blocks, thinking blocks |
| `/metrics` (also `/v1/metrics`), `/v1/load` | Prometheus text (`tensorfold:requests_running`, `tensorfold:time_to_first_token_seconds`, prompt-cache and draft counters); JSON load |
| `/health` | state, memory and `kv`: the page pool, streams and `admission` (`open`, `streams_full`, `waiting_for_pages`) ([fields](recipe/inkling2-tp4/docs/SETTINGS.md#health)) |

**Structured output** (`/v1/chat/completions`): `response_format` `json_object` or `json_schema`, and `tool_choice`
`"required"` or a named function (one call, `parallel_tool_calls: false`), constrain the answer after thinking (xgrammar
0.2.8 masks). `"auto"` stays unconstrained, `strict` tools included. `guided_regex`, `guided_choice` and
`guided_grammar`, and their `structured_outputs` forms, get HTTP 400
([details](recipe/inkling2-tp4/docs/SETTINGS.md#structured-output)).

## Measuring your install

`bench.sh` takes the server config `setup.sh` writes beside the hosts file (`my-sparks.serve.conf`) and writes JSON to
`./bench-results/`:

```
./bench.sh my-sparks.serve.conf check      # GPU-free: config, binaries, libraries and rank folders on every host
./bench.sh my-sparks.serve.conf sweep      # decode at 1/2/4 streams, one-stream workloads, cold prompts
./bench.sh my-sparks.serve.conf ledger bench-results/sweep-*.json  # each row against a memory-bandwidth floor
```

`prompts` runs 16 chat, code and agentic prompts; `needle` runs concurrent passphrase-recall sessions. Greedy rows
record token hashes to compare DSpark with plain ([details](recipe/inkling2-tp4/docs/BENCHMARKS.md#measuring-your-install)).

## Security

- HTTP listens on loopback (`HTTP_HOST=127.0.0.1`) by default, without TLS. On any other address, start with an API
  key, or anyone who can reach it can use the server; without one, `start.sh` and rank 0's log only warn.
- **API keys:** start with `EXTRA_ARGS="--api-key-file /path/on/rank0"` (`--api-key KEY` and `TENSORFOLD_API_KEY` are
  visible in `ps`, in `ROOT/recipe/config.sh` on every host and in the container's `.env`). Clients send
  `Authorization: Bearer KEY` or `x-api-key: KEY` on every route but `/health`, which then omits `kv`
  ([details](recipe/inkling2-tp4/docs/SETTINGS.md#api-keys)).
- Rank-to-rank traffic (NCCL's bootstrap on `NCCL_PORT` and its own listeners, RoCE on `TP_PORT` and `TP_PORT + 7`) is
  unauthenticated; keep the fabric subnets unrouted.
- `tools/zig/inkling2_rank_copy.py serve` (not run by the recipe) binds 127.0.0.1 unless `--bind` says otherwise and
  accepts any client unless `--allow` lists them; set both to use it across hosts.
- Vulnerabilities: [SECURITY.md](SECURITY.md).

## Limitations

- Text only; checkpoint mode only.
- At 2 to 4 streams on text whose drafts seldom land (the drafter's first guess kept about 3 times in 10), the
  cross-stream draft-row cut can be 20-26% slower than with it off; `TF_INK2_SHARED_CUT=0` turns it off. One
  stream and 5 or more streams are unaffected.
- NCCL over RoCE without GPU Direct RDMA. Tested on one switch and one host order.
- Small memory headroom: stream count and pool size depend on the memory free at start.

## This repository

A recipe for [TensorFold](https://github.com/ashhart/TensorFold): the engine is upstream TensorFold at
[`ENGINE_COMMIT`](ENGINE_COMMIT) (`223da7659`) plus the `git am` series in [`patches/`](patches/).
[`recipe/inkling2-tp4/`](recipe/inkling2-tp4/) holds `release-defaults.sh`, `config.example.sh`, `hosts.conf.example`,
`setup.sh` and `setup/`, the run scripts (`start.sh`, `stop.sh`, `status.sh`, `rank.sh`, `check.sh`, `bench.sh`,
`weights.sh`, `smoke.py`, `prompt_ids.py`), `container/`, `fabric/`, `weights/` and the docs:
[INSTALL-MANUAL](recipe/inkling2-tp4/docs/INSTALL-MANUAL.md), [CORRECTNESS](recipe/inkling2-tp4/docs/CORRECTNESS.md), [BENCHMARKS](recipe/inkling2-tp4/docs/BENCHMARKS.md),
[SETTINGS](recipe/inkling2-tp4/docs/SETTINGS.md), [weights](recipe/inkling2-tp4/weights/README.md).
[`tools/`](tools/) holds the weight copy, drafter ranges and benchmark tools; [`scripts/`](scripts/) the engine build
and release packaging.

## Build from source

The binaries are `ENGINE_COMMIT` plus `patches/`, the tree of branch
[`inkling2-sparks-v0.2`](https://github.com/swbiggart/TensorFold-upstream/tree/inkling2-sparks-v0.2) of our TensorFold
fork, commit [`c55b38d79`](https://github.com/swbiggart/TensorFold-upstream/tree/c55b38d7931b63f26d3cca5a80bfacca3088e053),
built with Zig 0.17.0, CUDA 13.3.1 and xgrammar v0.2.8 for aarch64 sm_121 in a pinned toolchain image
([`scripts/build-engine.Dockerfile`](scripts/build-engine.Dockerfile)). On x86-64 or aarch64 Linux with git and Docker
(no GPU):

```
scripts/build-engine.sh        # clone, git am patches/, build: build/dist/bin/, build/dist/lib/ and build/dist/BUILD-INFO
```

Copy `build/dist/bin/` and `build/dist/lib/libtfgrammar.so` to `ROOT/bin/` and `ROOT/lib/` on every host, or make a
tarball for `setup.sh --release` with the release's NCCL, cuBLAS, `plan/` and `data/`:

```
scripts/release/stage-dist.sh tensorfold-inkling2-tp4-v0.2-aarch64-sm121.tar.gz build/dist
scripts/release/build-assets.sh build/dist --engine-src build/engine-src --version my-build
```

Details: [docs/BUILD.md](docs/BUILD.md).

## Support

Open an issue in this repository (not upstream TensorFold or Thinking Machines) with the output of
`./setup.sh my-sparks.conf status`. Help is best effort.

## Removing it

```
./setup.sh my-sparks.conf stop                      # add --container for the container path
# on each host, if you ran --fabric-install:  sudo ROOT/recipe/fabric/host/uninstall.sh --flush
# on each host:  rm -rf ROOT   (binaries, recipe and, unless you used --weights-dir, the weights)
# on each host, container path:  docker image ls inkling2-sparks, then docker image rm that image
rm -rf ~/.cache/inkling2-setup                      # on the control machine: the downloaded tarball
```

## Credits and licenses

Our work here (the recipe, the tools and `patches/`) is Copyright 2026 Scott Biggart, Apache License 2.0
([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). Outside code keeps its own license
([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), [`LICENSES/`](LICENSES/)). The tarball carries the engine's
`LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` and `LICENSES/`, covering everything compiled into its binaries. The model weights are not part of this release.

- **Thinking Machines:** [Thinking Machines](https://thinkingmachines.ai) created Inkling-2-Small: the model, its
  weights and their quantization, its chat template and tokenizer, and the drafter (`mtp.safetensors`) that DSpark runs,
  released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) on
  [Hugging Face](https://huggingface.co/thinkingmachines/Inkling-2-Small) with an
  [Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy). We use its config fixtures, reply
  grammar and the data in `plan/` and `data/` derived from the checkpoint.
- **Included, adapted or ported:** [Transformers](https://github.com/huggingface/transformers) (Apache-2.0, HuggingFace
  Team; its [Inkling-2 model code](https://github.com/huggingface/transformers/tree/main/src/transformers/models/inkling2)
  is Hugging Face's integration); [TokenSpeed](https://github.com/lightseekorg/tokenspeed) (MIT, LightSeek Foundation);
  kortexa-ai's [TensorFold fork](https://github.com/kortexa-ai/TensorFold) (Apache-2.0, by francip): the fp32
  attention arithmetic of its `tf_ik_attention` kernel at commit `e2614fe`.
- **Followed:** [Four Over Six](https://github.com/mit-han-lab/fouroversix) (MIT, HAN
  Lab); [PyTorch](https://github.com/pytorch/pytorch) (BSD-3-Clause); [Jinja2](https://github.com/pallets/jinja)
  (BSD-3-Clause, Pallets) and [CPython](https://github.com/python/cpython) (PSF-2.0), with the
  [Unicode Character Database](https://www.unicode.org/ucd/) (Unicode-3.0); upstream
  [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0), including the tensor-parallel fabric, the GLM and
  MiniMax-M3 CUDA families, the FP4 mma and the GLM recipe's NCCL settings; [b12x](https://github.com/local-inference-lab/b12x)
  RoCEnante (Apache-2.0), behind the RoCE gathers.
- **Ideas:** 0xSero's [inkling2-exl3](https://github.com/0xSero/inkling2-exl3) (MIT): `/health`'s KV capacity and
  admission report, the process behind `bench.sh` (one-command sweep, fixed prompt set, bandwidth ledger, needle
  sessions), and FP8 weight-only projections for the DSpark drafter (theirs e4m3 with a scale per row; ours MXFP8, e4m3 with a
  power-of-two scale per 32 values). No code was copied; the prompts and the needle text are our own.
- **References:**
  - MiaAI-Lab: [four-Spark DeepSeek recipe](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold)
    (Apache-2.0); [switchless-ring guide](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/64b02dea8681fb1d8e3a659ea432a8e9e9ba3995/docs/switchless-ring.md)
    (AGPL-3.0-or-later).
  - NVIDIA Developer Forums:
    [ConnectX-7 Inter-Spark Link Capped at ~13 Gbps](https://forums.developer.nvidia.com/t/connectx-7-inter-spark-link-capped-at-13-gbps-expected-200-gbps-pcie-power-throttling-27w/363461/4).
  - [z-lab DFlash](https://github.com/z-lab/dflash) (MIT), by way of TokenSpeed.
  - [vLLM's Inkling-2-Small recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Small) (Apache-2.0).
  - Community Inkling-Small Spark recipes: [0xSero](https://github.com/0xSero/Inkling-Small-2-Sparks),
    [PixelML](https://github.com/PixelML/Inkling-Small-NVFP4-Dual-DGX-Spark),
    [Reederey87](https://github.com/Reederey87/inkling-small-nvfp4-2x-dgx-spark).
  - BertholomusAI's four-Spark TensorFold recipes: [DeepSeek](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10),
    [Mistral Large](https://github.com/bertholomus/mistral-large-4-tensorfold-tp4-4xgb10),
    [GLM](https://github.com/bertholomus/glm-5.3-tensorfold-tp4-4xgb10); [spark-attic](https://github.com/bertholomus/spark-attic)
    (GB10 memory headroom).
  - NVIDIA's [PTX ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/);
    [LLVM's NVPTX backend](https://llvm.org/docs/NVPTXUsage.html).
  - [perftest](https://github.com/linux-rdma/perftest) (`ib_write_bw`, GPL-2.0 or BSD; install it on each host).
  - [MikroTik RouterOS](https://mikrotik.com) (MikroTik and RouterOS are trademarks of SIA Mikrotīkls; not affiliated or
    endorsed).
- **Release binaries and libraries:** the binaries include [Zig](https://ziglang.org)'s standard library (MIT, Zig
  contributors) and GPU code built with [NVIDIA's CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), whose
  compiled header code stays under the [CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/). `lib/` bundles
  NVIDIA's [NCCL](https://github.com/NVIDIA/nccl) 2.30.7 (Apache-2.0 and BSD-3-Clause; `LICENSES/NCCL-LICENSE.txt`) and
  cuBLAS / cuBLASLt 13.1.1.3 (NVIDIA proprietary, redistributed unmodified under the CUDA Toolkit EULA;
  `LICENSES/NVIDIA-CUDA-EULA.txt`); they are not covered by this work's Apache-2.0 notice. `lib/libtfgrammar.so` is
  TensorFold's grammar library built with [xgrammar](https://github.com/mlc-ai/xgrammar) v0.2.8 (Apache-2.0, XGrammar
  Contributors) and the dlpack and picojson headers it bundles (Apache-2.0, BSD-2-Clause); their licence texts and
  xgrammar's notice are in `LICENSES/` and `lib/`.
