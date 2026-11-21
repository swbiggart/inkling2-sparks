# Changelog

The recipe's own changelog is [recipe/inkling2-tp4/CHANGELOG.md](recipe/inkling2-tp4/CHANGELOG.md); this one adds the
engine commit and the release asset.

## v0.3

Faster prompt fill, DSpark at 1M, image input, kept prompts on NVMe and a 6 GiB reserve; the model's weights and
quantization are unchanged, and text replies are v0.2's tokens.

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
  prompt cache's key. Checks in [CORRECTNESS.md](recipe/inkling2-tp4/docs/CORRECTNESS.md#images).
- **Prompt cache on NVMe** (opt-in, `TF_INK2_NVME=<dir>` in `OPT_ENV`): kept text prompts are written to each host's
  disk in idle time and read back after a restart or eviction (0.394 s against 95.87 s cold for
  a 128,581-token prompt); cap `TF_INK2_NVME_GIB` (64), free-space floor
  `TF_INK2_NVME_FLOOR_GIB` (32); a new build or setting starts cold.
- **Memory reserve 6 GiB** (`MEMORY_RESERVE_GIB`, was 8); the default config's memory gate floor is 6.25 GiB.
- **Exactness:** text replies are v0.2's tokens; every v0.2 check hash passes, none recorded again
  ([CORRECTNESS.md](recipe/inkling2-tp4/docs/CORRECTNESS.md)).
- **Fixes:** a setting string a rank cannot parse stops every rank instead of leaving the others waiting;
  `TF_INK2_SHARED_CUT` takes `off` and `false` in any case.
- **Speed:** the README compares every row with v0.2's published figure; rows marked carried are v0.2's measurements
  (v0.3 decodes with v0.2's launches; plain at 1 and 8 streams, DSpark at 1, 4 and 8 streams and the 128k DSpark book (prose and code) checked on both builds, largest change -2.0%). Rows marked slower were measured on v0.3 and are slower than v0.2: the fresh host install (+0.3%), the container install (+1.0%), start to ready (+4.3%), eight plain streams (`MAX_STREAMS=8`) (-0.7%), eight DSpark streams (`MAX_STREAMS=8`) (-0.8%) and the 126,250-token book, prose (0.0%; -0.9% against v0.2 in the same hold). Turning image input off (`VISION=0`) does not remove the decode rows' difference: measured again with the vision tower not loaded, the same cells (plain at 8 streams, DSpark at 8 streams, 128k DSpark book, prose) read -0.3%, -1.0%, -0.9% against v0.2 (with it loaded: -0.7%, -0.8%, -0.9%), so the tower explains at most part of it. Costs:
  rank 0 downloads and loads the tower (start to ready 24 s, v0.2 23 s).
- **Tools:** `scripts/release/build-codecs.sh` builds the two image decoders from their signed release tarballs;
  `build-assets.sh` packs them with their licence texts and `tools/zig/inkling2_vision_copy.py`.
- **Engine:** upstream TensorFold `223da7659` plus `patches/` 0001-0008; fork branch `inkling2-sparks-v0.3`, commit
  `7431a17b1`.
- **Release asset:** `tensorfold-inkling2-tp4-v0.3-aarch64-sm121.tar.gz`, sha256
  `27fe1df93f4a28e67c7fee44e47ee57f3e121e407eac99adeafefd2fd1bd069d`.

## v0.2

Attention in fp32, decode steps in fewer launches and leaner drafts at depth; the model's weights and quantization are
unchanged (the drafter, which only proposes tokens, runs its MLP projections from an FP8 copy of its own weights).

- **Attention:** scores, softmax and probabilities in fp32 on the tensor cores, rounded to bf16 once at the output,
  following kortexa-ai's `tf_ik_attention` (kortexa-ai/TensorFold `e2614fe`, Apache-2.0), with fixed 256-key splits on
  the global and sliding-window layers merged in a fixed order, paged keys, CUDA-graph forms and per-row handling of
  non-finite and subnormal inputs. v0.1 rounded scores and probabilities to bf16 where Transformers' eager attention
  does. Replies can differ from v0.1's at near-ties.
- **Exactness:** unchanged in meaning, and now also checked across prompt chunking, KV paging and CUDA graphs
  ([CORRECTNESS.md](recipe/inkling2-tp4/docs/CORRECTNESS.md)). The token gate (`smoke.py yardstick`, forced along the
  fp32 PyTorch reference's replies at 445 positions): 434 top-1 matches (bar 438: below it) and 443 top-2 (bar 443);
  agreement with the reference in
  [CORRECTNESS.md](recipe/inkling2-tp4/docs/CORRECTNESS.md#agreement-with-the-fp32-reference). New attention checks
  against float64 and between every kernel form.
- **Decode:** a sliding layer's projections in one launch and a global layer's in three (at one row; a window of 2 to 16
  rows adds one launch that copies each projection's columns to its rows), the router on the checkpoint's own bf16
  weights instead of a copy widened to fp32, and each convolution (keys, values, the attention and MLP outputs) writing
  its state in the launch that computes it. The same bits as before, checked against the earlier kernels
  ([CORRECTNESS.md](recipe/inkling2-tp4/docs/CORRECTNESS.md#changes-that-keep-the-bits)). Several streams' convolutions
  now run in one launch per layer instead of one per stream; in its own measurement no cell at 1 to 8 streams, plain or
  DSpark, got slower, and the README's Total throughput table shows the totals.
- **DSpark:** the drafter's global-layer attention packed by key head from 16 key blocks (4,096 keys) up, and its merge
  reading eight blocks at a time; these two give the same proposals, bit for bit. A lone greedy stream (no structured
  output), up to 32,768 positions, cuts a draft short before the model checks it when the drafter's own probabilities
  say the next guess will probably be rejected (scheduling only). The drafter's MLP projections run in MXFP8 (e4m3 with
  a power-of-two scale for every 32 values), quantized at start-up from its bf16 weights (its attention projections and
  its context key and value projections stay bf16): the proposals change, the replies do not. The vocabulary step now
  uses a wider kernel form when it scores 5 or more rows at once, in both the verify head and the drafter: the drafter's
  pass in the 8-stream check falls from 31.7 to 15.8 ms, with no change at one stream, and the README's Total throughput
  table shows what it does to total output. With 2 to 4 streams, shared rounds give their draft rows to the likeliest
  drafts across streams and draft a confident stream deeper (+5.9% / +5.5% at 2 / 4 streams; on text whose drafts seldom
  land it can be 20-26% slower; `TF_INK2_SHARED_CUT=0` turns it off).
- **Multi-stream figures:** the total output while every stream of a level is still decoding, on prompts whose replies
  run 256+ visible tokens, for v0.1 and v0.2 alike, so the v0.1 column differs from v0.1's published DSpark multi-stream
  figures.
- **Speed:** every plain, time-to-first-token, install and start-up row within 5% of v0.1 or faster except the
  checkpoint copy, a rank row (+10.7% to -6.6%), where the two fastest ranks copy more slowly but the copy as a whole
  ends sooner (85 s against 91 s); DSpark rows can move more, because the drafts' acceptance moves: one stream plain
  +9.4%, plain after 129,173 tokens +11.1%, DSpark by depth after 129,173 tokens +31.7%, a 32,768-token prompt's time to
  first token -32.4%. The README and `docs/BENCHMARKS.md` compare every speed row with v0.1.
- **Tools:** `bench.sh` runs from the release tarball; `inkling2_rank_copy.py serve` listens on 127.0.0.1 unless
  `--bind` says otherwise; `smoke.py yardstick`, with the extended prompt set in `data/ext/`; `build-assets.sh` and
  `stage-dist.sh` copy `data/` with its folders.
- **Engine:** upstream TensorFold `223da7659` plus `patches/` 0001-0008; fork branch `inkling2-sparks-v0.2`, commit
  `c55b38d79`.
- **Release asset:** `tensorfold-inkling2-tp4-v0.2-aarch64-sm121.tar.gz`, sha256
  `4aa5a3f0a5639435320b354b7d16893c824c85987ff2bf28f9b2bc9b801470be`.

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
- **Engine:** upstream TensorFold `223da7659` plus `patches/` 0001-0008 (the Inkling-2 family with the KV page pool and
  structured output, its CUDA kernels, batching, DSpark, the prompt cache, tooling, build glue, notices); fork commit
  `dee0906d3`.
- **Release asset:** `tensorfold-inkling2-tp4-v0.1-aarch64-sm121.tar.gz`, sha256 `27d4eff27404b7b3bc5b7eb0d3aa3fb2c2890879499851593cf51dc8a9bcfedb`.
