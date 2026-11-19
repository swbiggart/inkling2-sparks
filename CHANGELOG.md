# Changelog

The recipe's own changelog is [recipe/inkling2-tp4/CHANGELOG.md](recipe/inkling2-tp4/CHANGELOG.md); this one adds the
engine commit and the release asset.

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
