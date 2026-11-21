# Inkling-2-Small on 4× DGX Spark: exact TP4

v0.3

by Scott Biggart · [@woolsey](https://x.com/woolsey)

[![Follow on X](https://img.shields.io/badge/Follow_on_X-@woolsey-000000?logo=x&logoColor=white)](https://x.com/woolsey)
![Version v0.3](https://img.shields.io/badge/version-v0.3-blue)
![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)

Serve `thinkingmachines/Inkling-2-Small`, exact to the checkpoint's own weights, split four ways (tensor parallel, TP4)
across four DGX Spark-class GB10 hosts on one 200G switch. `tensorfold-native` serves an OpenAI- and
Anthropic-compatible HTTP API on rank 0, which samples and drives the other three ranks. DSpark speculative decoding and
image input are on by default.

Inkling-2-Small is [Thinking Machines'](https://thinkingmachines.ai) model (Apache-2.0, on
[Hugging Face](https://huggingface.co/thinkingmachines/Inkling-2-Small)); this recipe only serves it. This project is
not affiliated with or endorsed by Thinking Machines.

## What it is

- **Weights:** the checkpoint's bytes at revision `aac974122acd33ddb0d3e1f54433a9c5dd2249ce`, unchanged; each host holds
  its rank's quarter (about 92 GiB), and rank 0 also the vision tower (0.61 GiB).
- **Arithmetic:** experts quantize activations as the checkpoint specifies (NVFP4 for layers 0-39, MXFP8 for 40-47).
  Attention computes in fp32 and rounds to bf16 once, at its output.
- **Split:** attention by heads, each expert four ways, the output head by vocabulary, the input embedding by vocabulary
  over ranks 1-3 (rank 0 holds none); the vision tower runs whole on rank 0.
- **DSpark:** the checkpoint's drafter (MLP projections in MXFP8) proposes tokens; the model checks every one. Past
  32,768 positions a stream drafts only while drafting beats plain decode at its depth, up to 1,048,576.
- **Images:** PNG, JPEG and WebP at their own size, one image token for each 40 x 40 pixel block, decoded with the
  bundled libjpeg-turbo 3.1.4.1 and libwebp 1.6.0.
- **Exact:** the model's weights and quantization unchanged; replies independent of timing, drafting, caching,
  concurrency, prompt chunking, KV paging and CUDA graphs ([Correctness](#correctness)).

## What's new in v0.3

- **Faster prompt fill** (the same bits): cold time to first token 15.3% lower at 2k,
  14.5% at 32k and 10.8% at 126,000 tokens.
- **DSpark at 1M:** 19.43 tok/s by default on a 1,038,466-token book (v0.2: 12.5 plain, on a synthetic
  1,048,000-token state).
- **Image input**, on by default (`VISION=1`).
- **Kept prompts on NVMe** (opt-in): the 128,581-token prompt answers in 0.394 s after a
  restart, against 95.87 s cold.
- **Memory reserve 6 GiB** (was 8): 8 idle 128k prompts kept at the default context (v0.2 7);
  `CONTEXT=524288` serves 2 streams (v0.2 1-2).
- **Costs:** rank 0 downloads the 0.61 GiB tower and loads it at start (start to ready 24 s, v0.2
  23 s). `VISION=0` skips the tower.

Text replies are v0.2's tokens: every v0.2 check hash passes unchanged. Changes: [CHANGELOG.md](CHANGELOG.md).

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
git clone --branch v0.3 --depth 1 https://github.com/swbiggart/inkling2-sparks && cd inkling2-sparks/recipe/inkling2-tp4
cp hosts.conf.example my-sparks.conf      # HOSTS (ssh names in rank order) and ROOT (the install folder on the hosts)
./setup.sh my-sparks.conf \
  --release https://github.com/swbiggart/inkling2-sparks/releases/download/v0.3/tensorfold-inkling2-tp4-v0.3-aarch64-sm121.tar.gz \
  --sha256 27fe1df93f4a28e67c7fee44e47ee57f3e121e407eac99adeafefd2fd1bd069d
```

The sha256 pins the v0.3 tarball (999 MB). `setup.sh` installs, fetches weights, starts and runs the smoke and reference
checks; it stops at the first failure and skips finished steps. `./setup.sh --help` lists the options
([details](recipe/inkling2-tp4/docs/INSTALL-MANUAL.md#what-setupsh-does)).

- **Weights:** each host downloads its quarter from Hugging Face (the download resumes); rank 0 also fetches the
  vision tower as one new file, checked against its sha256. With a full checkpoint on the switch, set `CHECKPOINT=host:/path` in the hosts file (or pass `--checkpoint host:/path`) to
  hash-check it and copy it over the fabric ([weights](recipe/inkling2-tp4/weights/README.md#from-a-full-checkpoint));
  `--weights-dir DIR` verifies rank folders in place.
- **Container:** add `--container` to every command (not mixed with a host install). The image holds no weights and
  stays off registries ([details](recipe/inkling2-tp4/docs/INSTALL-MANUAL.md#container)).
- **Later:** `./setup.sh my-sparks.conf stop | start | status | smoke`; `stop check start` runs the W=4 decode check.
  Manual steps: [INSTALL-MANUAL.md](recipe/inkling2-tp4/docs/INSTALL-MANUAL.md).

HTTP listens on `HTTP_HOST:HTTP_PORT` on rank 0 (loopback by default); the model id is `model`:

```
curl -s http://127.0.0.1:8890/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "model", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

An image goes in as an `image_url` part with a data URL (`data:image/png;base64,...`) or an Anthropic base64 `image`.

## Benchmarks

Four GB10 hosts on one 200G switch at the release defaults (DSpark on unless a row says plain). Change is v0.3 against
v0.2's published figures, relative. Rows marked carried were not measured again: v0.3 decodes with v0.2's launches and
gives the same tokens; plain at 1 and 8 streams, DSpark at 1, 4 and 8 streams and the 128k DSpark book (prose and code) ran on both builds in one hold, the largest change -2.0%, inside that cell's
spread (the largest spread of any cell 3.1%), every token the same. Rows marked slower were measured on v0.3 and are slower than v0.2: the fresh host install (+0.3%), the container install (+1.0%), start to ready (+4.3%), eight plain streams (`MAX_STREAMS=8`) (-0.7%), eight DSpark streams (`MAX_STREAMS=8`) (-0.8%) and the 126,250-token book, prose (0.0%; -0.9% against v0.2 in the same hold). Turning image input off (`VISION=0`) does not remove the decode rows' difference: measured again with the vision tower not loaded, the same cells (plain at 8 streams, DSpark at 8 streams, 128k DSpark book, prose) read -0.3%, -1.0%, -0.9% against v0.2 (with it loaded: -0.7%, -0.8%, -0.9%), so the tower explains at most part of it.
Commands and methods: [docs/BENCHMARKS.md](recipe/inkling2-tp4/docs/BENCHMARKS.md).

### One stream

Tokens per second.

| Workload | v0.2 | v0.3 | Change |
| --- | ---: | ---: | ---: |
| Plain | 33.9 | 33.9 | carried |
| DSpark: tool call | 72.8 | 72.8 | carried |
| DSpark: code | 79.7 | 79.7 | carried |
| DSpark: thinking | 68.0 | 68.0 | carried |
| DSpark: raw code completion | 56.8 | 56.8 | carried |
| DSpark: prose | 53.9 | 53.9 | carried |

### Total throughput

Total tokens per second while all streams are running, on v0.2's prompts; 256-token replies unless a row says 128.
`MAX_STREAMS=8` rows use `CONTEXT=32768`.

| Setting | Streams | v0.2 | v0.3 | Change |
| --- | ---: | ---: | ---: | ---: |
| Defaults (DSpark) | 1 | 94.1 | 94.1 | carried |
| Defaults (DSpark) | 2 | 114.0 | 114.0 | carried |
| Defaults (DSpark) | 4 | 118.5 | 118.5 | carried |
| Defaults (DSpark), 128-token replies | 1 | 93.2 | 93.2 | carried |
| Defaults (DSpark), 128-token replies | 2 | 115.1 | 115.1 | carried |
| Defaults (DSpark), 128-token replies | 4 | 118.7 | 118.7 | carried |
| `MAX_STREAMS=8`, plain | 1 | 33.5 | 33.5 | carried |
| `MAX_STREAMS=8`, plain | 2 | 50.3 | 50.3 | carried |
| `MAX_STREAMS=8`, plain | 4 | 68.2 | 68.2 | carried |
| `MAX_STREAMS=8`, plain | 8 | 86.7 | 86.1 | -0.7% (slower) |
| `MAX_STREAMS=8`, DSpark | 1 | 95.2 | 95.2 | carried |
| `MAX_STREAMS=8`, DSpark | 2 | 113.8 | 113.8 | carried |
| `MAX_STREAMS=8`, DSpark | 4 | 116.9 | 116.9 | carried |
| `MAX_STREAMS=8`, DSpark | 8 | 87.9 | 87.2 | -0.8% (slower) |

### Time to first token

Seconds. Cached is the same prompt again. The 512k rows use `CONTEXT=524288`.

| Prompt | v0.2 | v0.3 | Change |
| --- | ---: | ---: | ---: |
| A short prompt | 0.10 | 0.10 | carried |
| 2k, cold | 1.18 | 1.00 | -15.3% |
| 8k, cold | 4.78 | 4.01 | -16.1% |
| 32k, cold | 20.7 | 17.7 | -14.5% |
| 128k (126,000 tokens), cold | 102 | 91 | -10.8% |
| 512k (519,983 tokens), cold | 826 | 796 | -3.6% |
| 2k, cached | 0.15 | 0.15 | carried |
| 8k, cached | 0.15 | 0.15 | carried |
| 32k, cached | 0.16 | 0.16 | carried |
| 128k (126,000 tokens), cached | 0.158 | 0.158 | carried |
| 512k (519,983 tokens), cached | 0.27 | 0.27 | carried |
| 101,859 tokens, cold | 80.4 | 71.3 | -11.3% |
| 101,859 tokens, cached | 0.1 | 0.1 | carried |
| 24,000 tokens, nothing else running | 22.0 | 17.2 | -21.8% |
| 24,000 tokens, beside one decoding stream | 58.1 | 58.1 | carried |
| 128k chat (128,581 tokens), after a server restart, from NVMe | - | 0.394 | new |

The NVMe row is the median of 3 restarts; the same prompt cold takes 95.87 s.

### Image input

Seconds to the first token of one cold image request, median of 5; one image and a short question ("Describe this image in two sentences."), 16 tokens, temperature 0, a different image each run.

| Image | Image tokens | Time to first token |
| --- | ---: | ---: |
| 448 x 336 | 108 | 0.293 |
| 1024 x 1024 | 676 | 0.612 |
| 1920 x 1080 | 1,296 | 0.973 |
| 3840 x 2160 | 5,184 | 4.294 |
| 4096 x 4096 | 10,609 | 11.012 |

PNG up to 1920x1080, JPEG quality 95 above (3840x2160, 4096x4096).

### Long context

Decode tokens per second after a long prompt, one stream, 256-token reply. Book rows (a book as the system message) run
the default unless marked plain; 1M default is the default (DSpark by depth) on a 1,038,466-token book, 9 replies of 256 tokens, server median; 1M plain is v0.2's figure, on a synthetic
1,048,000-token state.

| Context | v0.2 | v0.3 | Change |
| --- | ---: | ---: | ---: |
| 8k (8,163 tokens), plain | 33.3 | 33.3 | carried |
| 32k (33,404 tokens), plain | 32.1 | 32.1 | carried |
| 64k (67,804 tokens), plain | 30.4 | 30.4 | carried |
| 128k (129,173 tokens), plain | 28.0 | 28.0 | carried |
| 512k (519,970 tokens, the book), plain | 18.22 | 18.22 | carried |
| 1M, plain | 12.5 | 12.5 | carried |
| 1M, default (v0.2: plain) | 12.5 | 19.4 | +55.2% |
| 8k (8,163 tokens), default | 66.6 | 66.6 | carried |
| 32k (33,404 tokens), default | 69.9 | 69.9 | carried |
| 64k (67,804 tokens), default | 73.8 | 73.8 | carried |
| 128k (129,173 tokens), default | 67.8 | 67.8 | carried |
| Book, about 126,250 tokens, prose | 27.3 | 27.3 | 0.0% (slower in the same hold: -0.9%) |
| Book, about 126,250 tokens, code | 44.8 | 44.8 | carried |
| Book, about 520,000 tokens, prose | 25.2 | 25.2 | carried |
| Book, about 520,000 tokens, code | 26.4 | 26.4 | carried |

### Install and load

Seconds. Each install ends with the smoke checks and the W=4 decode check passing.

| | v0.2 | v0.3 | Change |
| --- | ---: | ---: | ---: |
| Fresh install on the hosts (`setup.sh` from an empty folder, weights verified in place) | 342 | 343 | +0.3% (slower) |
| The same with `--container` (also the image built and copied to the other three hosts) | 588 | 594 | +1.0% (slower) |
| Fresh install with the weights copied from a full checkpoint (`CHECKPOINT=host:/path`) | 520 | 520 | carried |
| The copy itself, a rank (about 92 GiB) | 83-85 | 83-85 | carried |
| Start to ready (weights in the page cache) | 23 | 24 | +4.3% (slower) |

### `bench.sh`

What [Measuring your install](#measuring-your-install) runs. The needle uses `CONTEXT=32768` and `MAX_STREAMS=8`.

| | v0.2 | v0.3 | Change |
| --- | ---: | ---: | ---: |
| Prompt set, temperature 0, all 16 rows (tok/s) | 60.1 | 60.1 | carried |
| Prompt set, temperature 0.7, all 16 rows (tok/s) | 46.6 | 46.6 | carried |
| Needle, 8 sessions at once (s) | 148 | 148 | carried |
| Ledger: one stream plain, share of the memory-bandwidth floor (%) | 74.3 | 74.3 | carried |

## Correctness

Hashes are xxh64 of the decode logits. Detail: [docs/CORRECTNESS.md](recipe/inkling2-tp4/docs/CORRECTNESS.md).

- **Text is v0.2's, bit for bit:** every v0.2 check hash passes on the v0.3 build with prompt overlap, DSpark at 1M,
  the NVMe tier and image input on or off; none was recorded again.
- **Decode equals the whole-prompt forward** bit for bit at W=4, all 48 layers: 80/80 (`037a50ab15855ea6`) and 64/64
  after 130,944 tokens (`be7904f5a734c3ed`), in any KV page order, with and without CUDA graphs.
- **1M positions:** the decode check is bit-equal at 1,048,000 tokens on a synthetic state (`d148c4204f687da7`).
- **Attention** gives a row the same bits in every kernel form. On real layers every output lies inside the bound a
  correct fp32 computation must meet (checked against float64).
- **Same reply** with drafting on or off, alone or beside 1 or 3 other streams, cached, cold or read back from NVMe
  after a restart.
- **Ranks agree:** ranks with different settings, builds or weight digests refuse to start.
- **Structured output** equals the reference under the same masks for a `json_object` reply and a required tool call.

Not claimed: W=4 equal to W=1 in the last bits (attention partials are summed in a fixed rank order), or replies equal
to PyTorch's or v0.1's at near-ties.

### Agreement with the fp32 reference

Share of 2,447 forced positions where each build's token is the fp32 reference's top token (measured on v0.2; v0.3's
text tokens are the same):

| Reference | v0.1 | v0.2 | p (exact McNemar) |
| --- | ---: | ---: | ---: |
| fp32, token by token, experts decoded to bf16 | 98.2% | 97.4% | 0.0175 |
| fp32, one pass with the checkpoint's experts (as the servers compute) | 97.7% | 97.6% | 0.888 |

Every difference is where the reference's top two tokens are less than 1 logit apart (token by token).
Mean logit distance over the reference's top 64 (token by token): v0.1 0.432, v0.2 0.425 (Wilcoxon p 0.105).

### Images

The pixels equal Transformers' image processor's bit for bit, and the tower's output is the same bits on every rank.
On 14 recorded image prompts, scored token by token against the fp32 reference forced along the server's
replies, no position a reply serves (up to its first end-of-reply token) departs where the reference's top two tokens
are 1 logit or more apart: 696 of 723 agree, the largest margin at a departure
0.59. Of all 896 forced positions 862 agree; 6 departures of
1 logit or more lie past the end of a reply, where no reply is served
([details](recipe/inkling2-tp4/docs/CORRECTNESS.md#images)). Replies are the same with DSpark on or off and beside
other streams.

## Settings you may change

In `release-defaults.sh`; each must be the same on all four ranks. No documented setting changes the output's bits. Details:
[docs/SETTINGS.md](recipe/inkling2-tp4/docs/SETTINGS.md).

| Setting | Default | What it does |
| --- | --- | --- |
| `CONTEXT` (`--context`) | 131072 | the largest request (prompt plus reply), up to 1,048,576 |
| `MAX_STREAMS` (`TF_INK2_MAX_STREAMS`) | 4 | most streams at once (1-8) |
| `DSPARK` (`TF_INK2_DSPARK`) | 1 | DSpark drafting, at every context |
| `VISION` (`--vision`) | 1 | image input; 0 serves text only and skips the tower |
| `PROMPT_CACHE_GIB` (`--prompt-cache-gib`) | 1.5 | prompt-cache arena a rank; 0 turns it off |
| `TF_INK2_NVME` (in `OPT_ENV`) | off | a directory on each host's NVMe for kept prompts; `TF_INK2_NVME_GIB` caps it (64), `TF_INK2_NVME_FLOOR_GIB` leaves free (32) |
| `MEMORY_RESERVE_GIB` | 6 | memory each host keeps free |
| `DECODE_SHARE` (`--decode-share`) | 0.5 | a long prompt's time share beside decoding streams |
| `ROCE_MULTI` (`TF_INK2_ROCE_MULTI`), `MAILBOX_MAX_KB` | 1, 640 | small decode exchanges over RoCE writes; needs the switch |
| `XROWS` (`TF_INK2_XROWS`) | 1 | by-rows layer sums in prompt chunks; `WARMUP=0` skips the start-up warm-up |

To change one: `echo CONTEXT=524288 >> my-sparks.conf`, then `./setup.sh my-sparks.conf stop config start`.

## Context, streams and the KV page pool

The server runs as many streams as fit at a whole `CONTEXT` each, at most `MAX_STREAMS`, on one shared pool of
256-position KV pages sized from the memory free at start. Spare pages hold kept prompts. At the release defaults
(reserve 6 GiB, image input on):

| `CONTEXT` | Streams (default `MAX_STREAMS=4`) | Pool positions | 128k prompts kept with every stream idle |
| ---: | ---: | ---: | ---: |
| 131,072 (default) | 4 | 1,148,160 | 8 |
| 262,144 | 4 | 1,147,136 | 8 |
| 524,288 | 2 | 1,158,656 | 8 |
| 1,048,576 | 1 | 1,384,704 | 10 |

Rank 0's log reports the choice: `inkling2: context 131072 -> 4 streams on a KV page pool (...): N pages of 256
positions ...`. A request longer than `CONTEXT` gets HTTP 400 naming the limit; one whose pages are not free waits, and
kept prompts give their pages back first. Nothing streams before the first token, so long prompts need a long client
read timeout.

## API

| Endpoint | Supports |
| --- | --- |
| `/v1/chat/completions`, `/v1/completions`, `/v1/models` (OpenAI) | streaming, tools, reasoning, structured output, `image_url` parts (data URLs) |
| `/v1/responses` (OpenAI) | streaming events, function tools, reasoning items |
| `/v1/messages`, `/v1/messages/count_tokens` (Anthropic) | streaming events, `tool_use` blocks, thinking blocks, `image` blocks (base64) |
| `/metrics` (also `/v1/metrics`), `/v1/load` | Prometheus text (`tensorfold:requests_running`, `tensorfold:time_to_first_token_seconds`, prompt-cache and draft counters); JSON load |
| `/health` | state, memory and `kv`: the page pool, streams and `admission` (`open`, `streams_full`, `waiting_for_pages`) ([fields](recipe/inkling2-tp4/docs/SETTINGS.md#health)) |

**Structured output** (`/v1/chat/completions`): `response_format` `json_object` or `json_schema`, and `tool_choice`
`"required"` or a named function (one call, `parallel_tool_calls: false`), constrain the answer after thinking (xgrammar
0.2.8 masks). `"auto"` stays unconstrained, `strict` tools included. `guided_regex`, `guided_choice` and
`guided_grammar`, and their `structured_outputs` forms, get HTTP 400
([details](recipe/inkling2-tp4/docs/SETTINGS.md#structured-output)).

**Images:** at most 4,096 pixels a side, 10,816 image tokens and 4 images a
request, 10 MiB an image; above a limit, or for an image URL, HTTP 400 names the numbers.

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
- The server fetches no image URL. Prompts kept on NVMe are readable by anyone who can read that directory.
- Rank-to-rank traffic (NCCL's bootstrap on `NCCL_PORT` and its own listeners, RoCE on `TP_PORT` and `TP_PORT + 7`) is
  unauthenticated; keep the fabric subnets unrouted.
- `tools/zig/inkling2_rank_copy.py serve` (not run by the recipe) binds 127.0.0.1 unless `--bind` says otherwise and
  accepts any client unless `--allow` lists them; set both to use it across hosts.
- Vulnerabilities: [SECURITY.md](SECURITY.md).

## Limitations

- Text and images in; text out. No audio or video. Checkpoint mode only.
- At 2 to 4 streams on text whose drafts seldom land (the drafter's first guess kept about 3 times in 10), the
  cross-stream draft-row cut can be 20-26% slower than with it off; `TF_INK2_SHARED_CUT=0` turns it off. One
  stream and 5 or more streams are unaffected.
- The NVMe tier keeps text prompts only; a prompt with images stays in memory. One server a host for each directory.
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
[`tools/`](tools/) holds the weight and vision-tower copy, drafter ranges and benchmark tools; [`scripts/`](scripts/)
the engine and image-decoder builds and release packaging.

## Build from source

The binaries are `ENGINE_COMMIT` plus `patches/`, the tree of branch
[`inkling2-sparks-v0.3`](https://github.com/swbiggart/TensorFold-upstream/tree/inkling2-sparks-v0.3) of our TensorFold
fork, commit [`7431a17b1`](https://github.com/swbiggart/TensorFold-upstream/tree/7431a17b155f969798480732ab718ce76170608f),
built with Zig 0.17.0, CUDA 13.3.1 and xgrammar v0.2.8 for aarch64 sm_121 in a pinned toolchain image
([`scripts/build-engine.Dockerfile`](scripts/build-engine.Dockerfile)). On x86-64 or aarch64 Linux with git and Docker
(no GPU):

```
scripts/build-engine.sh        # clone, git am patches/, build: build/dist/bin/, build/dist/lib/ and build/dist/BUILD-INFO
```

`scripts/release/build-codecs.sh` builds the image decoders from their signed release tarballs. Copy `build/dist/bin/`,
`build/dist/lib/libtfgrammar.so` and the decoders to `ROOT/bin/` and `ROOT/lib/` on every host, or make a tarball for
`setup.sh --release` with the release's NCCL, cuBLAS, `plan/` and `data/`:

```
scripts/release/stage-dist.sh tensorfold-inkling2-tp4-v0.3-aarch64-sm121.tar.gz build/dist
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
# on each host, if you set TF_INK2_NVME (after stop): rm -rf DIR, if DIR holds only the tier's files; in a shared
#   DIR, remove only the tier's own entries: its ink2-<16 hex> folders and .ink2-lock.r<rank> files
# on each host, container path:  docker image ls inkling2-sparks, then docker image rm that image
rm -rf ~/.cache/inkling2-setup                      # on the control machine: the downloaded tarball
```

## Credits and licenses

Our work here (the recipe, the tools and `patches/`) is Copyright 2026 Scott Biggart, Apache License 2.0
([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). Outside code keeps its own license
([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), [`LICENSES/`](LICENSES/)). The tarball carries the engine's
`LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` and `LICENSES/`, covering everything compiled into its binaries. The model weights are not part of this release.

- **Thinking Machines:** [Thinking Machines](https://thinkingmachines.ai) created Inkling-2-Small: the model, its
  weights and their quantization, its vision tower, its chat template, tokenizer and image processor, and the drafter
  (`mtp.safetensors`) that DSpark runs, released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) on
  [Hugging Face](https://huggingface.co/thinkingmachines/Inkling-2-Small) with an
  [Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy). We use its config fixtures, reply
  grammar and the data in `plan/` and `data/` derived from the checkpoint.
- **Included, adapted or ported:** [Transformers](https://github.com/huggingface/transformers) (Apache-2.0, HuggingFace
  Team; its [Inkling-2 model code](https://github.com/huggingface/transformers/tree/main/src/transformers/models/inkling2)
  is Hugging Face's integration, as is the image processor the server's pixels follow); [TokenSpeed](https://github.com/lightseekorg/tokenspeed) (MIT, LightSeek Foundation);
  kortexa-ai's [TensorFold fork](https://github.com/kortexa-ai/TensorFold) (Apache-2.0, by francip): the fp32
  attention arithmetic of its `tf_ik_attention` kernel at commit `e2614fe`.
- **Followed:** [Four Over Six](https://github.com/mit-han-lab/fouroversix) (MIT, HAN
  Lab); [PyTorch](https://github.com/pytorch/pytorch) (BSD-3-Clause); [Pillow](https://github.com/python-pillow/Pillow)
  (MIT-CMU): its decoder settings; [Jinja2](https://github.com/pallets/jinja)
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
  xgrammar's notice are in `LICENSES/` and `lib/`. `lib/libturbojpeg.so.0` is [libjpeg-turbo](https://libjpeg-turbo.org)
  3.1.4.1 (IJG, BSD-3-Clause and zlib licenses; this software is based in part on the work of the
  Independent JPEG Group) and `lib/libwebpdecoder.so.3` is [libwebp](https://chromium.googlesource.com/webm/libwebp)
  1.6.0 (BSD-3-Clause with Google's patent grant), both built unmodified from their signed release tarballs;
  their licence texts are in `LICENSES/` and `lib/`.
