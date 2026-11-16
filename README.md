# Inkling-2-Small on 4× DGX Spark: exact TP4

v0.1

by Scott Biggart · [@woolsey](https://x.com/woolsey)

[![Follow on X](https://img.shields.io/badge/Follow_on_X-@woolsey-000000?logo=x&logoColor=white)](https://x.com/woolsey)
![Version v0.1](https://img.shields.io/badge/version-v0.1-blue)
![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)

Serve `thinkingmachines/Inkling-2-Small`, exact to the checkpoint's own weights, split four ways (tensor parallel, TP4)
across four DGX Spark-class GB10 hosts on one 200G switch. `tensorfold-native` runs an OpenAI- and Anthropic-compatible
HTTP API on rank 0, which samples and drives the other three ranks. DSpark speculative decoding is on by default and
never changes a reply. One stream decodes plain at about 31.0 tok/s; DSpark takes code to about 60.8 tok/s and tool
calls to about 70.2 tok/s; four streams together reach about 100.7 tok/s; a repeated prompt starts in about 0.18 s. The
streams share one KV page pool sized from the hosts' free memory, so one setting, `CONTEXT`, says how long a request may
be (up to the model's 1,048,576 tokens), and the server runs as many streams as fit (at most four by default). One
command (`setup.sh`, on the hosts or with a pinned `--container`) installs, starts and checks everything.

Inkling-2-Small is [Thinking Machines'](https://thinkingmachines.ai) model, released under the Apache License 2.0 on
[Hugging Face](https://huggingface.co/thinkingmachines/Inkling-2-Small); this recipe only serves it, and its weights are
theirs. This project is not affiliated with or endorsed by Thinking Machines.

Exact means the checkpoint's own weights and quantization, unchanged, and replies that do not depend on timing,
drafting, caching or concurrency; what is checked, and what is not claimed, is under [Correctness](#correctness).

## What it is

- **Weights:** the checkpoint's own bytes at revision `aac974122acd33ddb0d3e1f54433a9c5dd2249ce`, unchanged. Each host
  holds only its rank's quarter (about 92 GiB), downloaded from Hugging Face or copied from a full checkpoint you
  already have.
- **Arithmetic:** checkpoint mode. The experts quantize their activations as the checkpoint specifies (NVFP4 for
  layers 0-39, MXFP8 for 40-47) and run on the block-scaled tensor cores, following TokenSpeed's reference; everything
  else follows Transformers' Inkling-2 model.
- **Placement:** attention split by heads, every expert split four ways (a quarter of each expert's rows on each rank,
  balanced at any batch size), the output head split by vocabulary, and the input embedding split by vocabulary over
  ranks 1-3, so rank 0, which also runs the HTTP side, holds none of it.
- **Exactness:** each rank's partial sums are added in a fixed rank order, in fp32, so a reply does not depend on
  timing, drafting, caching or the other streams ([Correctness](#correctness)).
- **Defaults** (`release-defaults.sh`): requests up to 131,072 tokens, up to four streams, DSpark on, the RoCE mesh
  wire, a 1.5 GiB prompt-cache arena a rank and 8 GiB of each host's memory held back ([Settings](#settings-you-may-change)).

## Requirements

- **Hosts:** four GB10 hosts (DGX Spark or an OEM GB10 system), each with about 116 GiB of memory available and a
  ConnectX-7 200G port, on NVIDIA's Ubuntu 24.04-based OS for these systems (tested with driver 580.178.04).
- **Network:** a 200GbE switch with one port per host (a QSFP56-DD 400G-to-2x200G breakout per two hosts on the CRS812
  used here), or, experimentally, a switchless ring; a LAN for ssh and the HTTP endpoint. Fabric setup:
  [docs/INSTALL-MANUAL.md](recipe/inkling2-tp4/docs/INSTALL-MANUAL.md#step-2-fabric), `fabric/`.
- **Software on each host:** NVIDIA driver 580 or later (`libcuda.so.1`), rdma-core's `libibverbs1` and
  `ibverbs-providers`, and `python3` 3.9 or later. For `--container`: Docker with Compose v2 and the NVIDIA Container
  Toolkit. `setup.sh`'s preflight checks each of these and names what is missing.
- **Disk:** 100 GiB free under `ROOT` on each host (108 GiB with `--container`); the weights are about 92 GiB a host.
  With `CHECKPOINT`, a Linux host on the switch holding the full checkpoint (366 GiB) with `python3`.
- **Control machine:** bash, curl and key-based ssh to all four hosts (it can be one of them). `sudo -n` is used only
  for `--fabric-install`, and for Docker when the ssh user is not in the `docker` group.
- **Benchmarks:** Python 3.9 or later on rank 0's host for `bench.sh`; 3.11 or later for `tools/bench_*.py` (standard
  library only).

## Quick start

From any machine with key-based ssh to the four hosts, with the fabric cabled and configured:

```
git clone --branch v0.1 --depth 1 https://github.com/swbiggart/inkling2-sparks && cd inkling2-sparks/recipe/inkling2-tp4
cp hosts.conf.example my-sparks.conf      # HOSTS (ssh names in rank order) and ROOT (the install folder on the hosts)
./setup.sh my-sparks.conf \
  --release https://github.com/swbiggart/inkling2-sparks/releases/download/v0.1/tensorfold-inkling2-tp4-v0.1-aarch64-sm121.tar.gz \
  --sha256 27d4eff27404b7b3bc5b7eb0d3aa3fb2c2890879499851593cf51dc8a9bcfedb
```

The sha256 is the v0.1 tarball's (991 MB); it is written here, in git, so the download is checked against the
release rather than against a file next to it. The v0.1 asset is never replaced; a fix ships as a new version.

`setup.sh` installs and starts everything and stops at the first step that fails: preflight, assets, libs, weights,
DSpark folds, manifest and config, then start, status, smoke and the oracle check. Every step skips what is already
done, so a second run checks and starts ([docs/INSTALL-MANUAL.md](recipe/inkling2-tp4/docs/INSTALL-MANUAL.md#what-setupsh-does)).

**Weights.** By default each host downloads its quarter (about 92 GiB) from Hugging Face at the pinned revision (it
resumes). If you already have the full checkpoint on one of the four hosts or another Linux host on the switch, set
`CHECKPOINT=host:/path` in the hosts file: `setup.sh` checks every file against the pinned revision's Hugging Face
hashes (any difference is refused), then each host copies its bytes from it over the fabric in about
75-91 s, byte for byte the download's ([weights/README.md](recipe/inkling2-tp4/weights/README.md#from-a-full-checkpoint)).

Afterwards: `./setup.sh my-sparks.conf stop | start | status | smoke`, or `stop check start` for the W=4 decode check
(it needs the server stopped). `./setup.sh --help` lists the options (`--weights-dir DIR` verifies rank folders the
hosts already hold, in place).

**Container.** The same install with each rank in a container (Docker Compose v2 and the NVIDIA Container Toolkit on
the hosts): add `--container` to every command above, for example
`./setup.sh my-sparks.conf --release URL --sha256 HEX --container` with the two values above. The image has no
weights, its base image is pinned by digest, and it is built on rank 0's host and copied to the others, never to a
registry ([docs/INSTALL-MANUAL.md](recipe/inkling2-tp4/docs/INSTALL-MANUAL.md#container)). Run the host install or the
container, not both. By hand: the numbered steps in [docs/INSTALL-MANUAL.md](recipe/inkling2-tp4/docs/INSTALL-MANUAL.md).

Once it is up, HTTP listens on `HTTP_HOST:HTTP_PORT` on rank 0 (loopback by default) and the model id is `model`:

```
curl -s http://127.0.0.1:8890/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "model", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

## Benchmarks

Four GB10 hosts on one 200G switch, the release defaults (requests up to 131,072 tokens, four streams, the prompt
cache on, DSpark on unless a column says plain). Medians. Methods, command lines and every other row:
[docs/BENCHMARKS.md](recipe/inkling2-tp4/docs/BENCHMARKS.md). Speedup is DSpark over plain.

### One stream

Tokens per second. DSpark's rate depends on how predictable the text is.

| Workload | Plain | DSpark | Speedup |
| --- | ---: | ---: | ---: |
| Tool call | 31.0 | 70.2 | 2.3x |
| Code | 31.0 | 60.8 | 2.0x |
| Thinking | 31.0 | 56.3 | 1.8x |
| Raw code completion | 31.0 | 52.7 | 1.7x |
| Prose | 31.0 | 40.0 | 1.3x |

### Total throughput

Tokens per second across all streams, 256-token replies, both columns with `CONTEXT=32768` and `MAX_STREAMS=8`.
At the defaults, DSpark gives 85.2 / 95.6 / 100.7 at 1 / 2 / 4 streams.

| Streams | Plain | DSpark |
| ---: | ---: | ---: |
| 1 | 30.7 | 86.0 |
| 2 | 45.9 | 93.8 |
| 4 | 62.6 | 102.6 |

### Time to first token

Seconds. Cached is the same prompt again: its state is kept on the pool's pages.

| Prompt | Cold | Cached |
| ---: | ---: | ---: |
| 2k | 1.32 | 0.18 |
| 8k | 5.84 | 0.18 |
| 32k | 30.6 | 0.19 |
| 128k (126,000 tokens) | 221 | not measured |
| 512k (519,983 tokens) | 2,754 | 12.1 |

A 101,859-token prompt takes 158.7 s cold and 0.1 s cached. The 512k row runs with `CONTEXT=524288`. A 24,000-token
prompt beside one decoding stream: 82.2 s (31.7 s with nothing else running).

### Long context

Decode after a long prompt, one stream, a 256-token reply, tokens per second. The default drafts by depth
(below); 1M decodes plain (the decode tool on a synthetic state). The 512k row is a book as the system message (the
same book at about 126,000 tokens: 24.8).

| Context | Plain | Default (DSpark by depth) |
| ---: | ---: | ---: |
| 8k (8,163 tokens) | 30.3 | 50.8 |
| 32k (33,404 tokens) | 29.2 | 67.0 |
| 128k (129,173 tokens) | 25.2 | 51.5 |
| 512k (519,970 tokens) | not measured | 16.8 |
| 1M | 10.8 | plain |

### Install and load

| | Seconds |
| --- | ---: |
| Fresh install on the hosts (`setup.sh` from an empty folder, weights verified in place) | 341 |
| Container path, on top of that | 301 |
| Fresh install with the weights copied from a full checkpoint (`CHECKPOINT=host:/path`) | 508 |
| Start to ready (weights in the page cache) | 23 |

Every install path passed the smoke test, the oracle check and the decode check.

## Correctness

The release's gates check, on this setup (hashes are xxh64 of the decode logits; every gate, with its hash, method and
build, is in [docs/CORRECTNESS.md](recipe/inkling2-tp4/docs/CORRECTNESS.md)):

- **Decode equals the whole-prompt forward.** Each decode step's logits equal the same row of the whole-prompt
  forward at the same world size (W=4), bit for bit, across all 48 layers (80/80, `f35e48068f72e8a9`; 64/64 after a
  130,944-token prompt, `1556b802e179e958`), with the KV cache on the page pool in any page order.
- **Greedy equals the oracle on the recorded prompts.** On the two recorded prompts (59 tokens), greedy replies equal
  the checkpoint-mode oracle's (the checkpoint run through Transformers' Inkling-2 modules in PyTorch, with the experts
  quantized as the checkpoint specifies) token for token.
- **The ranks agree** (ranks with different settings, builds or weight digests refuse to start). **Drafting never
  changes output** (drafted replies equal plain ones token for token, at every depth the depth rule prices).
  **Concurrent equals alone** at 1, 2 and 4 streams and beside a deep stream, and **a cached prompt equals a cold
  one**, including several streams resuming one kept prompt and kept prompts given back under memory pressure.
- **1M positions:** the decode check is bit-equal at 1,048,000 tokens on a synthetic state (every layer's caches filled
  with fixed values; 64/64, `39deddf85440d71e`).
- **Structured output:** constrained greedy replies equal the checkpoint-mode oracle run under the same masks for a
  JSON object and a required tool call, alone and in groups of 2 and 4, with DSpark on and off (other cases:
  [docs/CORRECTNESS.md](recipe/inkling2-tp4/docs/CORRECTNESS.md)).

The experts (the split MoE) and the embedding (bit copies) are bit-equal to a one-rank run. The attention partials of
the four ranks are summed in a fixed rank order and can differ from a one-rank run in the last bits, so the whole model
at W=4 is not claimed to equal W=1. Nor are replies claimed to equal the oracle's in general: the kernels round where
the oracle does but sum in their own fixed order, so a reply can depart from the oracle at a near-tie. Exact here
means the checkpoint's weights and quantization unchanged and replies that do not depend on timing, drafting, caching
or concurrency, not equality to PyTorch's arithmetic.

## Settings you may change

All are in `release-defaults.sh` and must be the same on all four ranks; set one in your config only to differ from
the release. Details: [docs/SETTINGS.md](recipe/inkling2-tp4/docs/SETTINGS.md).

| Setting | Default | What it does |
| --- | --- | --- |
| `CONTEXT` (`--context`) | 131072 | the largest request (prompt plus reply), up to 1,048,576; [below](#context-streams-and-the-kv-page-pool) |
| `MAX_STREAMS` (`TF_INK2_MAX_STREAMS`) | 4 | the most streams served at once (1-8) |
| `DSPARK` (`TF_INK2_DSPARK`) | 1 | DSpark drafting with the checkpoint's own drafter, priced by each stream's depth; never changes output |
| `PROMPT_CACHE_GIB` (`--prompt-cache-gib`) | 1.5 | the prompt cache's arena a rank (kept prompts' small fixed parts; their KV stays on the pool's pages); 0 turns the cache off |
| `MEMORY_RESERVE_GIB` | 8 | memory each host keeps free; the page pool takes the rest after the load |
| `DECODE_SHARE` (`--decode-share`) | 0.5 | a long prompt beside decoding streams runs in 512-row chunks taking this share of the time |
| `ROCE_MULTI` (`TF_INK2_ROCE_MULTI`), `MAILBOX_MAX_KB` | 1, 640 | small decode exchanges over RoCE writes instead of NCCL; same bits; needs the switch |
| `XROWS` (`TF_INK2_XROWS`) | 1 | by-rows layer sums in prompt chunks: same bits, less exchange time; `start.sh` sends a warm-up prompt (`WARMUP=0` skips it) |

No setting changes the output's bits.

## Context, streams and the KV page pool

`CONTEXT` (`--context`) is the setting to choose: the largest request the server takes, prompt plus reply, up to the
model's 1,048,576 tokens. At start-up each rank counts its memory after the 8 GiB reserve, the prompt cache's arena
and the graph cache; the rank with the least room decides. The server runs as many streams as fit at a whole
`CONTEXT` each, at most `MAX_STREAMS` (4 by default; up to 8, the most the exactness gates ran together), and every
stream's keys and values live on one shared pool of pages (256 positions a page) that takes the rest of that room. A
request takes the pages it needs as it grows and gives them back when it ends, so short requests leave room for long
ones, and the pages beyond the streams hold kept prompts. Rank 0's log says it, for example
`inkling2: context 131072 -> 4 streams on a KV page pool (...): N pages of 256 positions ...`.

| `CONTEXT` | Streams (default `MAX_STREAMS=4`) | Pool positions | 128k prompts kept with every stream idle |
| ---: | ---: | ---: | ---: |
| 131,072 (default) | 4 | 950,784 | 7 |
| 262,144 | 3 | 965,888 | 7 |
| 524,288 | 1 | 1,197,824 | 9 |
| 1,048,576 (plain) | 1 | 1,492,480 | 11 |

The pool's size follows the memory free when the server starts, so a count at the margin can differ by one between
starts, and a host running other work serves less. A request longer than `CONTEXT` is refused with HTTP 400 naming
the limit; a request whose pages are not free yet waits, and kept prompts give their pages back first. At
`CONTEXT=1048576` the drafter's caches do not fit beside the stream, so the recipe decodes plainly there. To change
it: `echo CONTEXT=524288 >> my-sparks.conf` (or `MAX_STREAMS=8`), then `./stop.sh my-sparks.conf && ./start.sh my-sparks.conf`.

**DSpark by depth.** Each draft reads the whole context, so drafting costs more as a stream grows. Up to 32,768
positions a stream drafts as usual; past that the server prices each round at the stream's own depth, times a plain
round and drafts only while drafts beat it, growing a deep stream's chain only as its drafts land. Short streams beside
a deep one draft as before. It is scheduling only: the tokens are the same bits with drafting on, off or by depth.

**Long prompts.** A 519,983-token document answers in 2,754 s and recalls the fact
placed in it, so clients need a long read timeout (the server streams nothing before the first token). The 1M
setting is checked bit for bit on a synthetic 1M-token state ([Correctness](#correctness)); decode there runs at
10.8 tok/s. Details: [docs/BENCHMARKS.md](recipe/inkling2-tp4/docs/BENCHMARKS.md#context-streams-and-the-pool).

## API

Rank 0 serves HTTP on `HTTP_HOST:HTTP_PORT`. Checked on this release (four GB10s):

| Endpoint | Status |
| --- | --- |
| OpenAI Chat Completions `/v1/chat/completions`, Completions `/v1/completions`, `/v1/models` | work: streaming, tools, reasoning, structured output ([below](#structured-output)) |
| OpenAI Responses `/v1/responses` | works: streaming events, function tools (`function_call` items), reasoning items |
| Anthropic Messages `/v1/messages`, `/v1/messages/count_tokens` | work: streaming events, `tool_use` blocks, thinking blocks |
| Metrics `/metrics` (also `/v1/metrics`) | Prometheus text format (`tensorfold:requests_running`, `tensorfold:time_to_first_token_seconds`, prompt-cache and draft counters, memory, ...); `/v1/load` (JSON load) |
| Health `/health` | JSON: state, memory, and `kv`: the page pool and admission ([docs/SETTINGS.md](recipe/inkling2-tp4/docs/SETTINGS.md#health)) |

**Health and capacity.** `/health`'s `kv` reports the page pool in pages and positions (total, free, available,
kept by the prompt cache), the largest request, the prompt cache's states and bytes, the streams active against their
cap, the requests waiting, and `admission`: `open`, `streams_full` (new requests queue for a stream) or
`waiting_for_pages`. Field by field: [docs/SETTINGS.md](recipe/inkling2-tp4/docs/SETTINGS.md#health).

**API keys.** Started with a key (`EXTRA_ARGS="--api-key-file /path/on/rank0"`, `--api-key KEY` or
`TENSORFOLD_API_KEY`), the server wants it on every route but `/health` (which then stays minimal, without `kv`), as
`Authorization: Bearer KEY` or `x-api-key: KEY`. There is no TLS: keep `HTTP_HOST=127.0.0.1` unless the endpoint must
be reachable from the network; with `HTTP_HOST` not loopback and no key, `start.sh` and rank 0's log warn, and the
server still starts. Details: [docs/SETTINGS.md](recipe/inkling2-tp4/docs/SETTINGS.md#api-keys).

### Structured output

On `/v1/chat/completions`, `response_format` `{"type": "json_object"}` or `{"type": "json_schema", ...}` constrains
the answer to JSON (matching the schema), and `tool_choice` `"required"` or a named function (`{"type": "function",
"function": {"name": ...}}`) constrains the reply to calls of the offered tools or of that one (a single call with
`parallel_tool_calls: false`). Thinking stays free; the answer after it is constrained. `tool_choice` `"auto"` stays
unconstrained, `strict` tools included. Each constrained token is drawn under the grammar's mask
(`lib/libtfgrammar.so`, built with xgrammar 0.2.8); DSpark and the other streams never change a constrained reply.
`guided_regex`, `guided_choice` and `guided_grammar` (and their `structured_outputs` forms) are refused with HTTP 400.

## Measuring your install

`bench.sh` runs the release's own measurements against your server and writes JSON to `./bench-results/`:

```
./bench.sh my-sparks.conf check        # GPU-free: the config, binaries, libraries, plan and rank folders on every host
./bench.sh my-sparks.conf sweep        # decode at 1/2/4 streams (DSpark and plain), the one-stream workloads, cold prompts
./bench.sh my-sparks.conf ledger bench-results/sweep-*.json    # each row against a memory-bandwidth floor
```

`prompts` runs 16 realistic chat, code and agentic prompts, and `needle` concurrent sessions that each find their own
passphrase.

Greedy rows record their token hashes, so the sweep also shows DSpark equal to plain on your hosts
([docs/BENCHMARKS.md](recipe/inkling2-tp4/docs/BENCHMARKS.md#measuring-your-install)).

## Security

- HTTP listens on loopback (`HTTP_HOST=127.0.0.1`) by default and has no TLS. If you set `HTTP_HOST` to any other
  address, start the server with an API key: without one, anyone who can reach that address can use the server.
  `start.sh` and rank 0's log warn when that is so, and the server still starts.
- Give the key with `--api-key-file`. A key passed as `--api-key KEY` or `TENSORFOLD_API_KEY` is visible to local users:
  in `ps` on rank 0, in `ROOT/recipe/config.sh` on all four hosts and in the container's `.env`.
- The rank-to-rank traffic (NCCL's bootstrap on `NCCL_PORT` and its own listeners, the RoCE wire on `TP_PORT` and
  `TP_PORT + 7`) has no authentication. Keep the fabric subnets unrouted: anything that can reach them can reach the
  ranks.
- `tools/zig/inkling2_rank_copy.py serve` (not run by the recipe) binds `0.0.0.0` and allows any client by default;
  pass `--bind` and `--allow` if you use it.
- To report a vulnerability, see [SECURITY.md](SECURITY.md).

## Limitations

- Text only: no image input. Checkpoint mode only (no bf16 activations for the experts).
- Structured output covers JSON (object or schema) and required or named tool calls: no regex, choice or grammar
  constraints.
- While a long prompt fills beside decoding streams, they keep about half their rate (`DECODE_SHARE`) and that prompt
  takes longer than alone. A prompt of hundreds of thousands of tokens takes most of an hour to fill; the longest
  real prompt run is 519,983 tokens, and 1M is checked on a synthetic state.
- Collectives run on NCCL over RoCE without GPU Direct RDMA. One host order and one switch were measured; the
  switchless ring is experimental. `CHECKPOINT` needs a Linux source host with python3 on the fabric.
- Memory headroom is small: the streams and the pool's size depend on the memory free when the server starts.

## This repository

A recipe for [TensorFold](https://github.com/ashhart/TensorFold), as other TensorFold recipes are: the scripts and docs
here, and the engine as upstream TensorFold at a pinned commit plus a short patch series.

| Path | What it is |
| --- | --- |
| [`recipe/inkling2-tp4/`](recipe/inkling2-tp4/) | the recipe: `setup.sh`, the start/stop/status/check scripts, `container/`, `fabric/`, `setup/`, `weights/`, `docs/` |
| [`tools/`](tools/) | the weight copy tool (`zig/inkling2_rank_copy.py`), the drafter ranges tool (`zig/inkling2_dspark_ranges.py`), `bench.sh`'s tool (`inkling2_bench.py`, its prompt set and model sheet, and its tests in `tests/`), and the benchmark clients (`bench_dspark.py` and upstream's `bench_*.py` it uses, with upstream's `bench_prompts.json`) |
| [`ENGINE_COMMIT`](ENGINE_COMMIT) | the upstream TensorFold commit the engine is built from (`223da7659`) and its repository |
| [`patches/`](patches/) | the engine changes, as `git am` patches on that commit: the Inkling-2 family (with the KV page pool and structured output), its CUDA kernels, batching, DSpark, the prompt cache, tooling, build glue and notices |
| [`scripts/build-engine.sh`](scripts/build-engine.sh) | builds the release binaries from `ENGINE_COMMIT` + `patches/` in a pinned toolchain image ([`scripts/build-engine.Dockerfile`](scripts/build-engine.Dockerfile)) |
| [`scripts/release/`](scripts/release/) | `build-assets.sh` (the release tarball), `stage-dist.sh`, `scrub-check.sh` |

The release tarball is the install path; `setup.sh --release` takes it. To build the engine yourself instead, see
[Build from source](#build-from-source).

## Build from source

The release binaries are built from upstream TensorFold at `ENGINE_COMMIT` with `patches/` applied (the same tree as the
engine branch [`inkling2-sparks-v0.1`](https://github.com/swbiggart/TensorFold-upstream/tree/inkling2-sparks-v0.1) of
our TensorFold fork, commit
[`dee0906d3`](https://github.com/swbiggart/TensorFold-upstream/tree/dee0906d3d18db50231d63c3112b5bf19942c210); the
branch never moves, and a later version gets its own), with Zig 0.17.0 and CUDA 13.3.1, cross-compiled for aarch64 and
sm_121; structured output's `libtfgrammar.so` also needs xgrammar v0.2.8, which the script clones at a pinned commit.
On an x86-64 or aarch64 Linux machine with git and Docker (no GPU needed):

```
scripts/build-engine.sh        # clone, git am patches/, build: build/dist/bin/, build/dist/lib/ and build/dist/BUILD-INFO
```

Checked for v0.1 on an x86-64 build machine from fresh clones: five of the seven files (with `libtfgrammar.so`) are
byte for byte the release's; `tensorfold-native` and `tf-inkling2-decode` differ only in symbol names (the same code
and data; see the build notes).
Copy `build/dist/bin/` to `ROOT/bin/` and `build/dist/lib/libtfgrammar.so` to `ROOT/lib/` on every host, or make a
tarball for `setup.sh --release` with the release's NCCL, cuBLAS, `plan/` and `data/`:

```
scripts/release/stage-dist.sh tensorfold-inkling2-tp4-v0.1-aarch64-sm121.tar.gz build/dist
scripts/release/build-assets.sh build/dist --engine-src build/engine-src --version my-build
```

Details, the toolchain pins and how the reproduction was checked: [docs/BUILD.md](docs/BUILD.md).

## More detail (recipe/inkling2-tp4/docs/)

[docs/INSTALL-MANUAL.md](recipe/inkling2-tp4/docs/INSTALL-MANUAL.md) (what `setup.sh` does, the container, the manual steps),
[docs/CORRECTNESS.md](recipe/inkling2-tp4/docs/CORRECTNESS.md) (every gate with its hash, method and build),
[docs/BENCHMARKS.md](recipe/inkling2-tp4/docs/BENCHMARKS.md) (all results and commands, context and streams, 1M, 512k),
[docs/SETTINGS.md](recipe/inkling2-tp4/docs/SETTINGS.md) (each setting, `/health`, API keys),
[weights/README.md](recipe/inkling2-tp4/weights/README.md) (rank folders, download, checkpoint copy).

In [`recipe/inkling2-tp4/`](recipe/inkling2-tp4/): `release-defaults.sh`, `config.example.sh` and `hosts.conf.example`, `setup.sh` and `setup/`,
`container/`, `start.sh`, `stop.sh`, `status.sh`, `rank.sh`, `check.sh` (the W=4 decode check), `bench.sh`, `weights.sh` and
`weights/`, `smoke.py` and `prompt_ids.py`, `fabric/`. Changes by version: [CHANGELOG.md](CHANGELOG.md).

## Support

Open an issue in this repository (not in upstream TensorFold or with Thinking Machines) and attach the output of
`./setup.sh my-sparks.conf status`. Help is best effort; v0.1 was tested on one switch and one host order.

## Removing it

```
./setup.sh my-sparks.conf stop                      # add --container for the container path
# on each host, if you ran --fabric-install:  sudo ROOT/recipe/fabric/host/uninstall.sh --flush
# on each host:  rm -rf ROOT   (binaries, recipe and, unless you used --weights-dir, the weights)
# on each host, container path:  docker image ls inkling2-sparks, then docker image rm that image
rm -rf ~/.cache/inkling2-setup                      # on the control machine: the downloaded tarball
```

## Credits and licenses

Our work in this repository (the recipe, the tools and the engine patches in `patches/`, which add the Inkling-2 family
to TensorFold) is Copyright 2026 Scott Biggart, under the Apache License 2.0 (see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE)). The engine is upstream [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0) at
`ENGINE_COMMIT` with those patches. Outside code keeps its own license. Every project below is credited in detail in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md); the license texts are in [`LICENSES/`](LICENSES/). The release
tarball carries the engine's own `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` and `LICENSES/`, which cover everything
compiled into its binaries.

- **Thinking Machines:** [Thinking Machines](https://thinkingmachines.ai) created Inkling-2-Small: the model, its
  weights and their quantization as released, its chat template and tokenizer, and the drafter (`mtp.safetensors`) that
  DSpark runs. They released it under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) on
  [Hugging Face](https://huggingface.co/thinkingmachines/Inkling-2-Small), with an
  [Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy). We use its config fixtures, reply
  grammar and the data in `plan/` and `data/` derived from the checkpoint. This project is not affiliated with or
  endorsed by Thinking Machines.
- **Included, adapted or ported:** [Transformers](https://github.com/huggingface/transformers) (Apache-2.0, HuggingFace
  Team; its [Inkling-2 model code](https://github.com/huggingface/transformers/tree/main/src/transformers/models/inkling2)
  is Hugging Face's integration); [TokenSpeed](https://github.com/lightseekorg/tokenspeed) (MIT, LightSeek Foundation).
- **Followed:** [Four Over Six](https://github.com/mit-han-lab/fouroversix) (MIT, HAN
  Lab); [PyTorch](https://github.com/pytorch/pytorch) (BSD-3-Clause); [Jinja2](https://github.com/pallets/jinja)
  (BSD-3-Clause, Pallets) and [CPython](https://github.com/python/cpython) (PSF-2.0), with the
  [Unicode Character Database](https://www.unicode.org/ucd/) (Unicode-3.0); upstream
  [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0), including the tensor-parallel fabric, the GLM and
  MiniMax-M3 CUDA families, the FP4 mma and the GLM recipe's NCCL settings; [b12x](https://github.com/local-inference-lab/b12x)
  RoCEnante (Apache-2.0), behind the RoCE gathers.
- **Ideas:** 0xSero's [inkling2-exl3](https://github.com/0xSero/inkling2-exl3) (MIT): `/health`'s KV capacity and
  admission report, and the process behind `bench.sh` (a one-command sweep with JSON out and a check command, a fixed
  realistic prompt set measured on the server's clock, a ledger against a bandwidth floor, concurrent needle
  sessions). No code was copied; the prompts and the needle text are our own.
- **References:** MiaAI-Lab's
  [four-Spark DeepSeek recipe](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold)
  (Apache-2.0) and its [switchless-ring guide](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/64b02dea8681fb1d8e3a659ea432a8e9e9ba3995/docs/switchless-ring.md)
  (AGPL-3.0-or-later); the NVIDIA Developer Forums thread
  [ConnectX-7 Inter-Spark Link Capped at ~13 Gbps](https://forums.developer.nvidia.com/t/connectx-7-inter-spark-link-capped-at-13-gbps-expected-200-gbps-pcie-power-throttling-27w/363461/4);
  [z-lab DFlash](https://github.com/z-lab/dflash) (MIT), by way of TokenSpeed;
  [vLLM's Inkling-2-Small recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Small) (Apache-2.0); the community
  Inkling-Small Spark recipes by [0xSero](https://github.com/0xSero/Inkling-Small-2-Sparks),
  [PixelML](https://github.com/PixelML/Inkling-Small-NVFP4-Dual-DGX-Spark) and
  [Reederey87](https://github.com/Reederey87/inkling-small-nvfp4-2x-dgx-spark), and BertholomusAI's four-Spark
  TensorFold recipes ([DeepSeek](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10),
  [Mistral Large](https://github.com/bertholomus/mistral-large-4-tensorfold-tp4-4xgb10),
  [GLM](https://github.com/bertholomus/glm-5.3-tensorfold-tp4-4xgb10)) and
  [spark-attic](https://github.com/bertholomus/spark-attic) (GB10 memory headroom); NVIDIA's
  [PTX ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/) and
  [LLVM's NVPTX backend](https://llvm.org/docs/NVPTXUsage.html);
  [perftest](https://github.com/linux-rdma/perftest) (`ib_write_bw`, GPL-2.0 or BSD; install it on each host);
  [MikroTik RouterOS](https://mikrotik.com) (MikroTik and RouterOS are trademarks of SIA Mikrotīkls; not affiliated or endorsed).
- **Release binaries and libraries:** the binaries include [Zig](https://ziglang.org)'s standard library (MIT, Zig
  contributors) and GPU code built with [NVIDIA's CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), whose
  compiled header code stays under the [CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/). `lib/` bundles
  NVIDIA's [NCCL](https://github.com/NVIDIA/nccl) 2.30.7 (Apache-2.0 and BSD-3-Clause; `LICENSES/NCCL-LICENSE.txt`) and cuBLAS / cuBLASLt 13.1.1.3
  (NVIDIA proprietary, redistributed unmodified under the CUDA Toolkit EULA; `LICENSES/NVIDIA-CUDA-EULA.txt`), each
  with its licence text beside it; they are not covered by this work's Apache-2.0 notice. `lib/libtfgrammar.so`
  (structured output) is TensorFold's grammar library built with [xgrammar](https://github.com/mlc-ai/xgrammar) v0.2.8
  (Apache-2.0, XGrammar Contributors) and the dlpack and picojson headers it bundles (Apache-2.0, BSD-2-Clause); their
  licence texts and xgrammar's notice are in `LICENSES/` and `lib/`.

The model weights are not part of this release. They are [Thinking Machines'](https://huggingface.co/thinkingmachines/Inkling-2-Small) release on Hugging Face (Apache-2.0 at
the pinned revision, with an Acceptable Use Policy linked from the model card), and each host downloads them from
there or copies them from your own checkpoint.
