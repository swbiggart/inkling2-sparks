# Inkling-2-Small TP4: changes

## v0.3

- **Prompt fill:** in prompt chunks of 576 rows or more, each half's by-rows attention sum crosses the wire under the
  other half's compute (`TF_INK2_XOVERLAP`), and the MoE exchange carries each rank's quantized values in place of bf16
  ones (`TF_INK2_XCODES`); both on, the same bits, and a rank that differs refuses to start. Cold time to first token
  -15.3% at 2,048 tokens, -16.1% at 8,192, -14.5% at 32,768 and
  -10.8% at 126,000.
- **DSpark at 1M:** `CONTEXT=1048576` drafts by depth instead of decoding plainly, with the 1.5 GiB prompt-cache arena
  kept and one more price point for the depth rule at 1M: 19.43 tok/s by default on a
  1,038,466-token book (v0.2: 12.5 plain, on a synthetic 1,048,000-token state); the idle pool at 1M keeps 10 128k prompts (as v0.2's plain 1M).
- **Image input** (`VISION=1`, the default; `VISION=0` serves text only with v0.2's launches): PNG, JPEG and WebP parts
  in user and tool messages (data URLs and base64; image URLs refused), at their own size, up to 4,096 pixels
  a side, 10,816 image tokens and 4 images a request, refused above with HTTP 400 naming
  the numbers. The checkpoint's vision tower runs on rank 0 from one new file, `inkling2-vision.safetensors`
  (0.61 GiB, 422 tensors, fetched and checked by `weights.sh` / `copy-rank.sh --vision`); images join the
  prompt cache's key. Checks in [CORRECTNESS.md](docs/CORRECTNESS.md#images).
- **Prompt cache on NVMe** (opt-in, `TF_INK2_NVME=<dir>` in `OPT_ENV`): kept text prompts are written to each host's
  disk in idle time and read back after a restart or eviction (0.394 s against 95.87 s cold for
  a 128,581-token prompt); cap `TF_INK2_NVME_GIB` (64), free-space floor
  `TF_INK2_NVME_FLOOR_GIB` (32); a new build or setting starts cold.
- **Memory reserve 6 GiB** (`MEMORY_RESERVE_GIB`, was 8); the default config's memory gate floor is 6.25 GiB.
- **Exactness:** text replies are v0.2's tokens; every v0.2 check hash passes, none recorded again
  ([CORRECTNESS.md](docs/CORRECTNESS.md)).
- **Fixes:** a setting string a rank cannot parse stops every rank instead of leaving the others waiting;
  `TF_INK2_SHARED_CUT` takes `off` and `false` in any case.

## v0.2

- **Attention in fp32:** scores, softmax and probabilities stay in fp32 on the tensor cores and are rounded to bf16
  once, at the output (kortexa-ai's `tf_ik_attention` arithmetic, with fixed 256-key splits merged in order). Replies
  can differ from v0.1's at near-ties ([CORRECTNESS.md](docs/CORRECTNESS.md#agreement-with-the-fp32-reference)); the
  weights, quantization and the rest of the model are unchanged.
- **Decode in fewer launches:** a sliding layer's projections in one launch and a global layer's in three (at one row; a
  window of 2 to 16 rows adds one launch that copies each projection's columns to its rows), the router on the
  checkpoint's own bf16 weights instead of a copy widened to fp32, and each convolution (keys, values, the attention and
  MLP outputs) writing its state in the launch that computes it. The same bits as before
  ([docs/CORRECTNESS.md](docs/CORRECTNESS.md#changes-that-keep-the-bits)). Several streams' convolutions now run in one
  launch per layer instead of one per stream; in its own measurement no cell at 1 to 8 streams, plain or DSpark, got
  slower, and the README's Total throughput table shows the totals.
- **Leaner drafts:** the drafter's global-layer attention packed by key head from 16 key blocks (4,096 keys) up, and its
  merge reading eight blocks at a time; these two give the same proposals, bit for bit. A lone greedy stream (no
  structured output), up to 32,768 positions, cuts a draft short before the model checks it when the drafter's own
  probabilities say the next guess will probably be rejected (scheduling only). The drafter's MLP projections run in
  MXFP8 (e4m3 with a power-of-two scale for every 32 values), quantized at start-up from its bf16 weights (its attention
  projections and its context key and value projections stay bf16): the proposals change, the replies do not. The
  vocabulary step now uses a wider kernel form when it scores 5 or more rows at once, in both the verify head and the
  drafter: the drafter's pass in the 8-stream check falls from 31.7 to 15.8 ms, with no change at one stream, and the
  README's Total throughput table shows what it does to total output. With 2 to 4 streams, shared rounds give their
  draft rows to the likeliest drafts across streams and draft a confident stream deeper (+5.9% / +5.5% at 2 / 4 streams;
  on text whose drafts seldom land it can be 20-26% slower; `TF_INK2_SHARED_CUT=0` turns it off).
- **Oracle data:** `data/` holds the fp32 reference's replies with the reference's top-two margins at every position,
  and an extended set of seven prompts. `smoke.py oracle` reports where a reply departs from the reference
  (information); `smoke.py yardstick` is the token gate (top-1 at least 438 of 445, top-2 at least 443). v0.2 scores 434
  and 443, top-1 below its bar ([docs/CORRECTNESS.md](docs/CORRECTNESS.md#agreement-with-the-fp32-reference)).
- **`bench.sh` from the tarball:** the release tarball carries `tools/inkling2_bench.py`, `inkling2_prompts.json`,
  `bench_prompts.json` and `inkling2_sheet.json`, so `bench.sh` runs without a clone.
- **`inkling2_rank_copy.py serve`** listens on 127.0.0.1 by default; `--bind ADDR` (including `0.0.0.0`) sets another
  address. `setup.sh`'s `CHECKPOINT` path passes its own `--bind`.
- **Multi-stream figures:** the total output while every stream of a level is still decoding, on prompts whose replies
  run 256+ visible tokens, for v0.1 and v0.2 alike, so the v0.1 column differs from v0.1's published DSpark multi-stream
  figures ([docs/BENCHMARKS.md](docs/BENCHMARKS.md#total-throughput)).

## v0.1

The first release: `thinkingmachines/Inkling-2-Small` at its pinned revision, exact TP4 on four GB10 hosts.

- **Serving:** checkpoint mode on the checkpoint's own bytes; attention split by heads, every expert split four ways,
  the output head by vocabulary and the input embedding by vocabulary over ranks 1-3 (rank 0 holds none); fixed-order
  fp32 sums, so replies do not depend on timing, drafting, caching or the other streams.
- **KV page pool:** every stream's keys and values on one pool of 256-position pages sized from the memory left after
  the 8 GiB reserve. `CONTEXT` is the largest request (up to 1,048,576 tokens); the server runs as many streams as fit
  at it, at most `MAX_STREAMS` (4; up to 8); pages beyond the streams hold kept prompts.
- **DSpark** speculative decoding with the checkpoint's own drafter, on by default, priced by each stream's depth
  (deep streams draft only while drafts pay); drafted replies equal plain ones.
- **Prompt cache:** exact; a kept prompt holds its pages by reference, and resuming streams adopt them.
- **API:** OpenAI Chat Completions, Completions and Responses, Anthropic Messages, metrics, and `/health` with the
  page pool and admission state (`kv`). Structured output on Chat Completions: `response_format` `json_object` /
  `json_schema` and `tool_choice` `"required"` / a named function. Optional API keys; a start-up warning when the
  endpoint is open to the network without one.
- **Install:** `setup.sh` on the hosts or with `--container`; weights from Hugging Face or, with
  `CHECKPOINT=host:/path`, copied over the fabric from a full checkpoint you already have (checked against the pinned
  revision's Hugging Face hashes first).
- **Measuring:** `bench.sh` (install check, speed sweep, prompt set, needle tests, speed ledger).
