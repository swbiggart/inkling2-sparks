# Inkling-2-Small at TP4: settings in detail

Paths in this file are relative to `recipe/inkling2-tp4/` (or the unpacked release folder, for `tools/`, `data/`, `plan/`), unless a link says otherwise.

The full text behind the short table in the [README](https://github.com/swbiggart/inkling2-sparks/blob/v0.3/README.md#settings-you-may-change).

## Settings you may change

All of these are in `release-defaults.sh`; they are the same on all four ranks (they go into the rank fingerprint,
and ranks that differ refuse to start). Set one in your config only to differ from the release.

- **Context** (`CONTEXT`, `--context`; 131072): the largest request the server takes, prompt plus reply, up to the
  model's 1,048,576 tokens. A longer request is refused with HTTP 400 naming the limit. `DSPARK` applies as set at
  every context: at `CONTEXT=1048576` the drafter's caches (about 1 GiB a rank) fit beside the stream and the 1.5 GiB
  prompt-cache arena, taking pool room kept prompts would have had.
- **Streams** (`MAX_STREAMS`, `TF_INK2_MAX_STREAMS`; 4. `PARALLEL`, `--parallel`; auto): at start-up the server
  counts, on every rank, the memory after its load, the reserve, the prompt cache's arena and the graph cache; the
  rank with the least room decides. With `PARALLEL=auto` it serves as many streams as fit at a whole `CONTEXT` each,
  at most `MAX_STREAMS` (1 to 8; 8 is the most the exactness gates ran together). A number in `PARALLEL` is served as
  asked (the cap does not apply) and refused at start-up when it does not fit. A context with no room for one stream
  is refused at start-up, naming the largest context that fits.
- **The KV page pool** (no setting): every stream's keys and values (the eight global layers', and the drafter's) live
  on one pool of pages of 256 positions, 2.25 MiB a page on a rank with DSpark and 2.0 MiB plain. The pool takes the
  room left beside the streams' fixed parts (the sliding layers' rings and the convolution states), the prompt
  cache's arena and the graph cache, less a 256 MiB margin. A stream takes pages as it grows and returns them when it
  ends; a request is admitted when a stream is free and its pages are available, and otherwise waits. Rank 0's log
  line `inkling2: context C -> N streams on a KV page pool (...)` gives the counts, the pool's pages and positions,
  and the largest request. A stream's pages are one run of consecutive pages whenever the pool has one, and a stream
  on one run decodes with the same kernels as a fixed window; any layout of pages gives the same bits.
- **DSpark** (`DSPARK=1`, `TF_INK2_DSPARK`; on): speculative decoding with the checkpoint's own drafter, which needs
  the drafter's tensors in each rank folder (`weights.sh` adds them) and the folds file
  `inkling2-dspark-fold.safetensors` beside them. The folds (the drafter head's `hidden_out` multiplied into the
  unembed: fp64 sums, bf16 stores) are derived from the weights, so the release does not ship them: each host builds
  its own from its rank folder, in `setup.sh`'s folds step or by hand:

  ```
  ROOT/bin/tf-inkling2-fold MODEL MODEL/inkling2-dspark-fold.safetensors --rank R --world 4
  ```

  It needs no PyTorch and takes about a second on a GB10. A rank's file holds only that rank's vocabulary rows (the
  rest are zero; its header names the rows); without `--rank` the tool reads a whole checkpoint and writes the full
  file, byte for byte the same as `tools/zig/inkling2_dspark_ref.py fold` (a PyTorch tool in the engine source, `patches/`). `DSPARK=0` turns drafting off.
- **DSpark by depth** (no setting): drafting costs more as a stream's context grows, since every draft and verify
  reads the whole context. Up to 32,768 positions a stream's rounds are priced from the short-context cost table, at
  least one draft a round. There a lone greedy stream (no structured output) also cuts a draft short
  before the model checks it when the drafter's own probabilities say the next guess will probably be rejected.
  With 2 to 4 streams a shared round gives its draft rows to the likeliest drafts across the streams;
  `TF_INK2_SHARED_CUT=0` (or `off` or `false` in any case; any other value leaves it on) turns this off, the same
  replies either way. Set it on every rank: at start the ranks compare the mode it resolves to and refuse to start
  when that differs.
  Past 32,768 a round of d drafts is priced at the stream's own depth (a table measured at 3,040,
  32,768, 131,072, 262,144 and 524,000 positions, interpolated, plus what such rounds took beyond their price on this
  server), and a plain round competes: drafts must beat it by 5 %, with a probe after a run of plain rounds. A deep
  stream runs plain until a plain round past 32,768 has been timed, starts at one draft a round and reaches at most
  one draft past the most its rounds have kept. In a round shared with short streams each deep stream pays its own
  depth's cost, so it drafts less or not at all and the short streams draft as before. Drafts change the work, never
  the tokens (every draft is verified greedily against the target), so drafted replies equal plain ones.
- **The prompt cache** (`PROMPT_CACHE_GIB`, `--prompt-cache-gib`; 1.5 GiB a rank, on): an exact prefix cache (a
  resumed prompt gives the same bits as a cold one). A kept prompt holds its whole pages of the pool by reference and
  one slot of the 1.5 GiB arena for the rest (the sliding layers' rings, the convolution states, the drafter's local
  rings and the tail past its last whole page); a prompt with images is kept with its images' digest in the key.
  Streams that resume it adopt its pages (shared pages are never written; a stream writes only past its own position)
  and copy back only the arena slot. When a stream needs pages,
  kept prompts give theirs back, least valuable first, never one a waiting request will resume. It is passed
  explicitly; left out, the server sizes the arena itself; 0 turns the cache off.
- **Memory reserve** (`MEMORY_RESERVE_GIB`, `TENSORFOLD_MEMORY_RESERVE_GIB`; 6): memory each host keeps free. The
  streams and the page pool take the rest after the load.
- **Long prompts beside decoding streams** (`DECODE_SHARE`, `--decode-share`; 0.5): while other streams decode, a long
  prompt is cut into 512-row chunks that take this share of the time; the decoders keep about half their rate
  instead of stopping for the whole prompt, and that prompt's time to first token grows. A prompt with nothing else
  decoding runs at full speed. Output is the same either way (prompt chunking is exact).
- **Image input** (`VISION`, `--vision` on every rank; 1, on): image parts in user and tool messages (OpenAI
  `image_url` with a data URL, Anthropic `image` with base64 data; image URLs are refused) as PNG, JPEG or WebP. There
  is no resize: an image is one token for each 40 x 40 pixel block, padded, and its pixels are the ones Transformers'
  processor gives the model (alpha is laid over the processor's grey checkerboard). A request is refused with HTTP 400
  naming the numbers when an image is larger than 4,096 pixels a side or over its share of the request's
  10,816 image tokens (`--vision-image-tokens`; split evenly over the images), when it holds more than
  4 images (`--vision-max-images`), or when an image is over 10 MiB or its images
  together over 20 MiB. Rank 0 decodes images one at a time and runs the vision tower whole, from
  one new file in its rank folder, `inkling2-vision.safetensors` (0.61 GiB; `weights.sh` and `setup.sh`'s
  weights step add it while `VISION=1`). JPEG and WebP decode in the release's `lib/` (libjpeg-turbo
  3.1.4.1, libwebp 1.6.0). `VISION=0` serves text only with v0.2's launches.
- **The RoCE mesh wire** (`ROCE_MULTI=1`, `TF_INK2_ROCE_MULTI`; on). Small decode exchanges (up to `MAILBOX_MAX_KB`,
  640 KiB a rank) go over RoCE writes instead of NCCL; the bytes and their order are the same, so replies are
  bit-identical, and every rank falls back to NCCL together if the probe fails. It needs `bin/tp_mailbox.fatbin` and
  two free TCP ports on rank 0's fabric address (`TP_PORT` and `TP_PORT + 7`). Needs the switch; for the ring see
  `fabric/ring-experimental/`.
- **By-rows layer sums** (`XROWS`, `TF_INK2_XROWS`; on): the same bits, less exchange time in prompt chunks. Its
  first use costs a first long prompt a moment, so `start.sh` sends one warm-up prompt once the server is ready
  (`warmup.sh`; `WARMUP=0` skips it). Needs a switch (every pair of ranks connected). For prompt chunks of 576 rows or
  more, with by-rows sums on, `release_env` also sets `TF_INK2_XOVERLAP=1` (each half's by-rows sum crosses the wire
  under the other half's compute) and `TF_INK2_XCODES=1` (the MoE exchange carries each rank's quantized values in
  place of bf16 ones): the same bits; `=0` in `OPT_ENV` turns one off.

No setting changes the output's bits.

**The embedding** (no setting): the input embedding table is split by vocabulary rows over ranks 1-3 (rank 0, which
also runs the HTTP side and the sampler, holds none of it). Rank 0 still loads and digest-checks the whole table, hands
each rank its rows at start-up and frees it. In a forward each rank copies the rows it holds, one gather in rank order
brings every row to every rank, and the norm follows: bit copies only, so the embedding equals a one-rank run.

Also available, off: `TF_ROCE=1` (upstream's RoCE gather, instead of the mesh wire; set `ROCE_MULTI=0`), and
`TF_INK2_LATE_MASKS=0`, which gives a stream with structured output a DSpark pass of its own instead of the round's
one shared drafter pass with its masks applied after (the replies are the same either way).

To add an opt-in setting on every rank: with `setup.sh`, put it in the hosts file's `OPT_ENV`, for example
`OPT_ENV=(TF_INK2_GRAPH=0)`; setup writes it into the config's `EXTRA_ENV`. With a hand-written config, add it to
`EXTRA_ENV` directly. Either way it comes after the release's settings, so it overrides one of the same name.

**The prompt cache on NVMe** (opt-in, off by default):
`OPT_ENV=(TF_INK2_NVME=/a/dir)` in the hosts file (`EXTRA_ENV` in a manual config), an absolute directory on each
host's own disk, also writes kept text prompts of 1,024
tokens or more there while no stream runs, one file a rank, so they survive a restart and the device cache's
evictions; a prompt whose longest kept prefix is on
disk is read back into fresh pages, the same bits as a resume from memory. `TF_INK2_NVME_GIB` caps a rank's files
(64 GiB); `TF_INK2_NVME_FLOOR_GIB` is the free space a write must leave on the disk
(32 GiB). Files go under `<dir>/ink2-<identity>/`, the identity being the program, the checkpoint,
the settings, the world and the layout: a new build or setting starts cold, and at start each rank removes only its
own files of other identities. The tier deletes only files it wrote. What it costs: disk space up to the cap (each kept state of a 128k prompt is about 1.15 GiB a host, and one prompt can keep several states), and disk writes in idle turns only (a request arriving waits at most one write chunk). With `--container`, only the
container's mounts are seen and kept: use a directory under `RUN` (`ROOT/serve`, mounted read-write), e.g.
`TF_INK2_NVME=ROOT/serve/nvme` with ROOT spelled out. Set it on every rank (the ranks refuse to start
when it differs) and use one server a host for each directory. Prompts with images are not written.

## Health

`GET /health` on rank 0 returns JSON: `status`, `model`, `warming`, `memory`, the live counters and, on this release,
`kv`, rank 0's view of the page pool as its engine last reported it (at a stream's open and close and while a request
waits; nothing is read per decode step):

| Field | Meaning |
| --- | --- |
| `page_positions` | positions a page holds (256) |
| `pages_total`, `positions_total` | the pool's size |
| `pages_free`, `positions_free` | pages no stream and no kept prompt holds |
| `pages_available`, `positions_available` | free and not promised to an admitted stream |
| `pages_kept_by_prompt_cache`, `positions_kept_by_prompt_cache` | pages kept prompts hold (while a stream runs on a resumed prompt's pages they count as kept and as taken) |
| `largest_request_positions` | the largest request the server takes (`CONTEXT`) |
| `prompt_cache_states`, `prompt_cache_bytes`, `prompt_cache_budget_bytes` | kept prompts and the arena's use and size |
| `streams_active`, `streams`, `max_streams` | streams running, streams served, the cap |
| `requests_waiting` | requests queued |
| `admission` | `open` (a request starts now if its pages are free), `streams_full` (every stream is busy; new requests queue for one) or `waiting_for_pages` (a request waits for pages) |
| `waiting_request_pages`, `requests_waited_for_pages` | pages the waiting request needs; how many requests have waited for pages since start |

Started with API keys, `/health` stays minimal and leaves out `kv`.

## API keys

Started with a key, the server wants it on every route but `/health`, as `Authorization: Bearer KEY` or
`x-api-key: KEY` (else 401). In the config: `EXTRA_ARGS="--api-key-file /path/on/rank0"` (a key a line, mode 0600),
`EXTRA_ARGS="--api-key KEY"` (repeatable) or `EXTRA_ENV+=(TENSORFOLD_API_KEY=KEY)`; `--metrics-open` leaves `/metrics`,
`/v1/metrics` and `/v1/load` open for a scraper. `start.sh` counts a 401 as up; its warm-up (skipped with a warning),
`smoke.py` and `setup.sh`'s smoke send no key. There is no TLS: keep `HTTP_HOST=127.0.0.1` unless the endpoint must be
reachable from the network. With `HTTP_HOST` not loopback and no key configured, `start.sh` (and `setup.sh`'s start,
host or container) prints a warning and rank 0's log has one; the server still starts.

## Structured output

Structured output has no setting: a request asks for it (README, [API](https://github.com/swbiggart/inkling2-sparks/blob/v0.3/README.md#api)).
A greedy constrained pick is the highest-scoring allowed token, ties to the lower id like every greedy pick.
