# Inkling-2-Small at TP4: correctness gates

Paths in this file are relative to `recipe/inkling2-tp4/` (or the unpacked release folder, for `tools/`, `data/`, `plan/`), unless a link says otherwise.

Every gate of the release with its hash; the summary is in the [README](https://github.com/swbiggart/inkling2-sparks/blob/v0.3/README.md#correctness).

## What exact means here

The checkpoint's own weights and quantization, unchanged (the drafter, which only proposes tokens, runs its MLP
projections from an FP8 copy of its own weights), and replies that do not depend on timing, drafting, caching (in memory
or on NVMe), concurrency, how a prompt is split into chunks, which KV pages a stream gets, or whether a step runs as a
CUDA graph. Attention computes in fp32 and rounds to bf16 once, at its output. Exact does not mean equal to a PyTorch
run token for token: replies can differ from the fp32 reference's at near-ties
([below](#agreement-with-the-fp32-reference)).

Text replies are v0.2's tokens: every v0.2 check hash below passes on the release build, and none was recorded again.

## What the gates check

- **Decode equals the whole-prompt forward.** Each decode step's logits equal the same row of the whole-prompt forward
  at the same world size (W=4), bit for bit, across all 48 layers, with the KV cache on the page pool in whatever pages
  a stream gets, with and without CUDA graphs.
- **The token gate.** Forced along the replies of the checkpoint run in PyTorch with fp32 attention, at 445 positions,
  the server's next token is the reference's top token at least 438 times and in its top two at least 443 times
  ([below](#agreement-with-the-fp32-reference)).
- **The attention kernels** compute the arithmetic written below: every kernel form gives the same bits for a row, a
  row's bits depend only on its own inputs, and real layers fall inside a float64 range of that arithmetic
  ([below](#attention-checks)).
- **The ranks agree.** Every rank computes the same logits and the same token; ranks with different settings (the
  prompt-fill, image and NVMe switches included), builds or weight digests refuse to start, and so does every rank when
  one cannot read a setting.
- **Drafting never changes output.** Drafted replies equal plain ones token for token, at every depth up to 1,048,576
  and with the depth rule drafting or not, with the drafter's FP8 projections, and with drafts cut short by the
  drafter's own probabilities.
- **The prompt-fill overlap and MoE codes, the new decode kernels and the drafter's key packing give the same bits** as
  the paths they replace ([below](#changes-that-keep-the-bits)). The drafter's other changes alter only which tokens it
  proposes.
- **Nothing else changes output:** other streams (concurrent equals alone), the prompt cache (a resumed prompt equals a
  cold one, also when several streams share one kept prompt, when kept prompts are given back, and when a state is read
  back from NVMe after a restart), the pages a stream gets (any order; poisoned new pages), and structured output's
  masks with DSpark on or off.
- **Images** reach the tower as the pixel values of Transformers' processor, bit for bit; the tower gives the same bits
  on every rank however its work is cut; on the recorded image prompts, replies keep the reference's top token at every
  served position where its top two are 1 logit or more apart ([below](#images)).

What this does and does not say about W=4 against one rank: the experts (the split MoE), the embedding (bit copies of
the table's rows) and the vision tower (whole on rank 0) are bit-equal to a one-rank run. The attention partials of the
four ranks are summed in a fixed rank order, the same on every rank and every run, and can differ from a one-rank run
in the last bits. The release does not claim that the whole model at W=4 equals W=1.

## The attention arithmetic

For each query row and head, on every layer:

1. **Scores.** The query and key vectors are bf16, as the model stores them. Their dot products run on the tensor cores
   with fp32 accumulation, times 1/sqrt(128), plus the layer's relative-position bias (a bf16 table). The scores stay
   fp32.
2. **Softmax.** Keys are taken in 64-key tiles at fixed positions. A running maximum and a running sum are kept in fp32
   and rescaled tile by tile (an online softmax). The probabilities are never rounded.
3. **Probabilities times values.** The tensor cores multiply bf16 operands, so each fp32 probability is written as the
   sum of three bf16 parts (high, middle, low), and all three are multiplied by the values and summed in fp32.
4. **Output.** The sum is divided by the softmax total, multiplied by the sigmoid of the head's output gate and by the
   attention sink's factor, sigmoid(running maximum + log of the total - the sink), in fp32, and rounded to bf16 once.
5. **Long contexts.** On the global layers the keys are cut into splits of 256 keys at fixed positions. Each split keeps
   its own maximum, total and sum, and the splits are merged in position order with the same fp32 steps. The
   sliding-window layers do the same over their window: 256-key splits at fixed positions, merged in order from the
   split that holds the window's first key. The splits depend only on positions, so a row gets the same bits however
   many rows run with it, at any depth, in a CUDA graph or not.
6. **Rare inputs.** Keys outside a row's range load as zero and are never read. Subnormal bf16 inputs and subnormal
   probability parts are flushed to zero, element by element. A row whose own query, keys or values hold an infinity or
   NaN is recomputed on its own by a scalar path; the other rows of its block keep their result.

This is the arithmetic of kortexa-ai's `tf_ik_attention` (kortexa-ai/TensorFold, commit `e2614fe`, Apache-2.0), with
the splits of step 5 and the per-element and per-row rules of step 6 added (credits in `THIRD_PARTY_NOTICES.md`).
v0.1 rounded the scores and probabilities to bf16 at the points Transformers' eager attention does.

## Changes that keep the bits

Each change below changes how prompt chunks, decode steps, drafts or the prompt cache run, not what they compute: each
was checked against the path it replaces, on the same inputs, byte for byte. The drafter changes marked below change
its proposals, never the reply.

- **Prompt chunks: the exchange under the compute, and MoE codes (the same bits).** On prompt chunks of 576 rows or
  more (`TF_INK2_XOVERLAP=1` and `TF_INK2_XCODES=1` in `release-defaults.sh`), a layer's attention runs as two row
  halves, and each half's by-rows attention sum crosses the wire while the other half computes. Each half is a chunk of
  its own, in the same kernel forms, its partials after the first half's. The MoE's input exchange carries each rank's
  quantized columns in place of bf16 values that every rank then quantized: each rank quantizes its own columns as the
  whole-row quantizer would (NVFP4 with the row's maximum, gathered first). Checked on the change's own commits
  before the release build: the experts' outputs equal the bf16 exchange's (0 of 2,048 rows differ, an NVFP4 and an
  MXFP8 layer, 4 and 2 ranks); prompt-cache resumes inside each half and on its edges equal; W=2's `740a5e9600f24fa1` and
  the decode check's hashes at 600, 3,040 and 32,768 tokens (the gates below) unchanged with both on. A rank whose
  switches differ refuses, on every rank.
- **DSpark at 1,048,576 positions (proposals only).** `CONTEXT=1048576` drafts by depth, with one more price point for
  the depth rule at 1,048,000 positions; the drafter only proposes and the model checks every token. Checked before the
  release build: on a synthetic 1,048,000-token state drafted equals plain, 64 of 64 tokens, real and mixed drafts; on
  a book of about 1,037,000 tokens, DSpark against `DSPARK=0`, the same token hash over all 36 replies.
- **Prompt cache on NVMe (a restored state equals a cold one, bit for bit).** With `TF_INK2_NVME` set
  (off by default), kept text prompt states of 1,024 tokens or
  more are also written to each host's disk in idle turns. A stream that resumes one reads every rank's file into fresh
  pages and checks it first (its identity, the prompt's tokens, a hash of the bytes); a file that fails on any rank is
  refused on every rank and the prompt filled cold. The identity (program, checkpoint, settings, world, rank, page
  layout) makes a new build or setting start cold; a rank whose switch or cap differs refuses, on every rank; prompts
  with images are not written. Checked at W=2 (8 layers) on the tier's own commits before the release build: states
  written at 12 cuts and read into poisoned pages give `740a5e9600f24fa1` 12 of 12, also when read by a new process; DSpark's
  rounds and proposals after a read equal a cold run's; a flipped byte or a failed read on one rank is refused on both
  and the cold fill is equal.
- **Decode steps in fewer launches.** At one row a sliding layer's projections run as one launch and a global layer's as
  three; a window of 2 to 16 rows takes one more launch, which copies each projection's columns to its rows (two
  launches instead of six, and four instead of eight). The router reads the checkpoint's own bf16 weights instead of a
  copy widened to fp32 (the widening is exact, and the sums run in the same order); each convolution (keys, values, the
  attention and MLP outputs) writes its state in the launch that computes it, up to 16 rows. Checked against the earlier
  kernels: the logits of every decode step and the KV, convolution-state and history bytes each step leaves behind;
  windows on both sides of where the merged forms stop (15, 16 and 17 rows); prompt chunks; prompt-cache restores, at a
  page boundary and inside a page; pages moved and shuffled; CUDA graphs against eager launches; drafted rounds. Byte
  for byte on one GPU, at the one-, two- and four-rank shapes, for the first eight layers up to about 3,300 positions;
  on the four GB10s the decode checks keep their hashes at 3,040 and 32,768 positions. No byte differed in 516
  comparisons of the stream state between the earlier and the new kernels (about 33 GB), nor in 252 more with the new
  kernels reading the earlier fp32 router copy, at two settings of the merged launches.
- **Leaner drafts.** The drafter's global-layer attention is packed by key head (each key is read once for all the query
  heads that share it) from 16 key blocks up, and its merge loads eight blocks at a time, in the same order. Checked
  against the earlier kernels: at every block count from 1 to 40, including 15, 16 and 17 where the packed form takes
  over, in the paged and the table form, at the one-, two- and four-rank head shapes, the attention and the merge give
  the same bytes (5,400 cases, 0 differ); the drafter proposes the same tokens at all 4,893 positions of a long text; on
  the four GB10s the decode checks keep their hashes and drafted replies equal plain ones at 131,000, 262,000 and
  524,000 positions. No byte differed.
- **Shorter drafts (scheduling only).** A lone greedy stream (no structured output), up to 32,768 positions, cuts a
  draft short before the model checks it when the drafter's own probabilities say the next guess will probably be
  rejected. Only how many drafted tokens the model checks changes; drafted replies equal plain ones on the server with
  the cut active (the gate below).
- **FP8 drafter projections (proposals change, replies do not).** The drafter's MLP projections run in MXFP8 (8-bit e4m3
  values with a power-of-two scale for every 32 values), quantized at start-up from the checkpoint's bf16 drafter; the
  rest of the drafter, including its attention projections and its context key and value projections, stays bf16. The
  drafter proposes different tokens at some positions (at 219 of 3,336 positions on 15 texts, with acceptance 0.32%
  lower overall and at most 0.79% lower for one kind of text); the model checks every proposal, and drafted replies
  equal plain ones (the gates below). The model's own weights are not touched.
- **A wider unembedding for drafts and verification (the same bits).** The wider vocabulary step adds up every logit in
  the same fp32 order, with the same bf16 rounding and tie-breaks, at up to 64 rows in both the verify head and the
  drafter (v0.2 uses at most 32 in the verify head and 56 in the drafter), so the check hashes are unchanged and all 45
  requests gave the same tokens with it on and off.
- **Several streams' convolution state in one launch (the same bits).** Outputs are unchanged: the op checks pass
  199/199, including streams with and without draft history and 8 streams of 8 rows, every stream's tokens equal its
  solo run, and every request's tokens are identical with the change on and off.
- **Expert tiles for forwards of 9 to 16 rows (the same bits).** Outputs are unchanged: the check hashes at 600, 3,040
  and 32,768 tokens are identical with and without the tiles, and the expert outputs are bit-equal to the previous tiles
  at every row count.
- **Shared rounds of several streams (scheduling only).** How a round of several DSpark streams shares its draft rows
  among them, and how many rows a round may hold, decide only which drafts the model checks and when: drafted replies
  equal plain ones and concurrent replies equal a stream alone in the gates below.

## Agreement with the fp32 reference

The reference is the checkpoint run through Transformers' Inkling-2 modules in PyTorch, with attention in the fp32 form
above and TF32 off. Forced along the reference's replies, each build (a build of each release that adds only a logit
dump) picks one greedy token at each position. Share of 2,447 forced positions where it is the reference's top token
(measured on v0.2; v0.3's text tokens are the same):

| Reference | v0.1 | v0.2 | p (exact McNemar) |
| --- | ---: | ---: | ---: |
| fp32, token by token, experts decoded to bf16 | 98.2% | 97.4% | 0.0175 |
| fp32, one pass with the checkpoint's experts (as the servers compute) | 97.7% | 97.6% | 0.888 |

Every difference is where the reference's top two tokens are less than 1 logit apart (token by token). Mean logit
distance over the reference's top 64 (token by token): v0.1 0.432, v0.2 0.425 (Wilcoxon p 0.105).

**The token gate** (`smoke.py yardstick`, with the reference's replies in `data/`: two recorded prompts, 59 positions,
and an extended set of seven, thinking, a tool call, C code, JSON, German, a multi-turn chat and a 2,147-token
catalogue, 386 positions) forces the server along the reference's replies at 445 positions, through `/v1/completions`
with token ids, one greedy token each. Its bars, 438 top-1 and 443 top-2 matches, are the lowest scores of six valid
PyTorch ways of running the checkpoint against the same reference. v0.3 scores 434 and 443, the same as
v0.2, since its text tokens are v0.2's (top-1 below its bar, so `smoke.py yardstick` prints FAIL); v0.1 scored 438 and
445.

## Attention checks

These run the kernels outside the server, on one host (`tf-inkling2-attn`), with the release build's kernels; the gate
table below says where each ran. The attention kernels are v0.2's, unchanged in v0.3; the error figures below were
measured on them for v0.2.

- **Every form gives the same bits.** The kernel has several forms (one prompt chunk, one decode row, the key splits
  with their merge, the split walk for long chunks, packed heads). On random inputs at the rank's head shapes, at 32
  positions and 15 row counts, sliding and global layers, contiguous and paged keys: every form equals every other,
  every row equals its own one-row run at chunk sizes 1 to 2,048 and at positions that are not multiples of the tile,
  and every split size gives the same result as the unsplit form below its key count. Also at 1,048,000 positions.
- **CUDA graphs.** Each form is captured once and replayed at other positions, across tile, page and split boundaries,
  with its keys moved to other pages and its page table permuted; every replay equals an eager launch. Planted faults (a
  stale position, keys left at their old pages) must fail, and do.
- **Rare inputs.** Planted rows with a very peaked softmax, subnormal queries, keys and values, a NaN in one row's
  window, and NaN or subnormal values in slots outside the window: every other row equals its own clean run. A control
  build with a block-wide fallback (one row's rare input changing its neighbours) fails, as it must.
- **Against float64, real layers.** Captured inputs of six layers (sliding and global, early, middle and late) from a
  4,015-token prompt: every output falls inside the float64 range of the arithmetic above, and the error against a
  float64 computation, in bf16 steps at the head's output scale, is smaller than v0.1's attention at the median and the
  99th percentile on all six layers (99th percentile 0.23-0.30 against v0.1's 0.54-1.35; median 0.008-0.027 against
  0.023-0.107).
- **All 48 layers.** Captured inputs of every layer for the recorded prompts and a 4,015-token prompt, plus the first
  4,096 rows of a 32,768-token document: every layer inside the float64 range and every kernel form byte-equal, 192 of
  192 layer runs.

## Images

An image (PNG, JPEG or WebP, in a user or tool message) is decoded on rank 0 at its own size, with no resize: one image
token for each 40 x 40 pixel block, the edge blocks padded; a transparent image is laid over the processor's grey
checkerboard, as Transformers' processor does. The checkpoint's vision tower runs whole on rank 0, and its output rows
take the image's place in the prompt on every rank. Images over the limits (4,096 pixels a side,
10,816 image tokens a request split evenly over its images, 4 images, 10 MiB
an image, 20 MiB a request) and image URLs are refused with HTTP 400 naming the numbers. The
prompt cache keys a prompt with images by a digest of their pixels, so it never resumes a state of other images.

These checks ran on the image path's own commits before the release build, on the four GB10s unless a line says
otherwise; the release build's image rows are in the gate table.

- **Pixels** (host tests, aarch64 Linux and macOS). The pixel values equal those of Transformers' processor (Pillow and
  torchvision, fp32 `pixel_values`) bit for bit, with the bundled libjpeg-turbo 3.1.4.1 and libwebp
  1.6.0: 46 small fixtures (PNG RGB, RGBA, palette, grey, with transparency, 16-bit, 1-bit, interlaced;
  JPEG baseline, progressive, 4:4:4, 4:2:2, 4:2:0, grey; WebP lossy, lossless and with alpha; EXIF orientations), 40
  decoded and 6 refused as the processor's route refuses them, and 15 large ones from 1024x768 to 4096x4096 (one,
  4096x4097, refused).
- **Tower.** Its output has one set of bits: the same whole and layer by layer, in row and query blocks, repeated,
  behind another image and on another GPU (13 fixtures, one sha256 each, one sm_120 GPU), and at W=4 equal to that pin
  on all four ranks. Prompts cut and resumed inside an image (cuts at 1, 600, 601, 700 and 2,048 around an image at
  600) equal the uncut run, 6 of 6; a poisoned control fails on all four ranks, as it must.
- **Replies.** 14 recorded image prompts (1 to 4 images, up to 10,816 image tokens, 64-token
  greedy replies). The reference is the checkpoint run through Transformers' Inkling-2 modules in PyTorch with fp32
  attention and the checkpoint's quantization, forced along the server's replies (the server takes no forced prefix
  with images), so both see the same prefix at every position. The rule: no departure where the reference's top two
  tokens are 1 logit or more apart, at every position a reply serves, up to and including its first end-of-reply token
  (id 200006). It holds on the recorded prompts: 696 of 723 served positions agree
  (96.3%), the largest margin at a departure 0.59 logit. Over all 896
  positions, including those the requests forced past the end of a reply, 862 agree (96.2%);
  6 departures of 1 logit or more lie past the end of a reply and are not served.
- **Same reply.** DSpark on and off (a second server, started fresh): all 14 replies identical. The
  10,816-token request (four 2048x2048 images) beside three text streams equals it alone, and each stream
  equals its alone run.
- **Refusals.** A 4097-pixel side, 5 images, a request over its even token share and an image URL: HTTP 400, each
  naming its numbers; afterwards the server serves, and a repeated image request equals its first reply.

**The tower against float64** (one sm_120 GPU; published as measured, not a bar). Op by op (45 ops on each of 4
fixtures, on the same inputs), the tower is no farther from a float64 computation than torch's default form (within a
ratio of 1 + 1e-4). Run whole, at 2 of 6 fixtures some layers are farther from float64 than the farthest torch form
that runs: 4096x4096 (against the flash form) layers +0.03% to +2.7%; 1920x1080, one layer +0.14%. At 3840x2160 the
tower is 2-4% nearer than every torch form, and at the other three fixtures within torch's forms. Torch's own f32core
form (the same rounding scheme, run by torch) fails the same comparison at the same two fixtures.

Not claimed: image replies equal to PyTorch's at near-ties, or the tower equal to PyTorch's arithmetic.

## Gates on the release build

Hashes are xxh64 of the decode logits. Every gate below ran on the release build (fork commit `7431a17b1`)
unless its row says otherwise, on the four GB10s unless a row says W=2 (one host with two GPUs and 8 or 24 of the 48
layers); `gates.sh require` on it: 63 of 63 release gates closed, every one a pass except
the token gate's top-1 bar (434/445, as v0.2: the same tokens). Every hash is v0.2's; none was recorded again.

| Gate | Result |
| --- | --- |
| Decode and drafter changes against the kernels they replace ([above](#changes-that-keep-the-bits)) | PASS, every byte equal (on the change's own commits before the release build, one sm_120 GPU, layers 0-7, up to about 3,300 positions); drafter PASS, every byte equal (on its own commits before the release build, one sm_120 GPU, 1 to 40 key blocks, one-rank proposals) |
| Prompt-fill overlap and MoE codes against the bf16 exchange ([above](#changes-that-keep-the-bits)) | PASS, 0 of 2,048 rows differ (on the change's own commits before the release build, one sm_120 GPU, an NVFP4 and an MXFP8 layer, 4 and 2 ranks); W=2 with both on `740a5e9600f24fa1` |
| Decode check, W=4, 3,040-token prompt, 80 steps, graphs and eager | bit-equal 80/80, `037a50ab15855ea6` both; 30.74 / 33.57 ms a step |
| Decode check with the prompt-fill overlap and MoE codes on (the release default), 600, 3,040 and 32,768 tokens | `6fa828521eb3db2f`, `495d9f842f8f5ffe`, `fa269df6a30eb569` |
| The same with the pool's pages shuffled, with new pages poisoned (`TF_KV_POISON`), and with every new sequence on another run of pages | bit-equal 80/80, `037a50ab15855ea6` in each, drafted rounds included |
| Drafted equals plain, real and mixed drafts | 80/80 each, `037a50ab15855ea6` |
| Drafted equals plain on the server with the depth cut active: one greedy stream alone at 600, 3,040 and 32,768 tokens, DSpark against `DSPARK=0`, token hashes | PASS, the same token hash at each length |
| 4 and 8 streams stepped together equal their solo runs | PASS at 4 and 8 |
| 130,944-token check, and drafted equals plain at 8k / 32k / 64k / 131k | bit-equal 64/64, `be7904f5a734c3ed`; 16/16 at each depth |
| Synthetic 1,048,000-token state, W=4, with the drafter loaded (deep-draft and deep) | bit-equal 64/64, `d148c4204f687da7` in both; drafted equals plain; the fill read back intact on all four ranks |
| DSpark at `CONTEXT=1048576` (now drafting by depth) against `DSPARK=0`: the default (DSpark by depth) on a 1,038,466-token book, 9 replies of 256 tokens, server median, token hashes | PASS, the same token hash over all 18 replies (9 at the default on the release build, 9 with DSPARK=0 on a pre-release build of the same book) |
| Prompt-cache resumes in the decode tool (resume, inside a page, drafted, 4 streams) on kept pages | PASS; proposals `7f474a06c0397152`; the pool idle at the end with every page free or kept |
| Attention forms, W=4 head shape on a GB10 (sm_121), and W=2 / W=1 shapes on sm_120, every split size | every form equal, 19,225 forwards, 0 rows differ; rare-input counter 0 |
| Attention rare inputs (planted) | 7/7 PASS; control 5/5 failed as required |
| Attention against float64, six real layers (W=2, sm_120) | 6/6 layers inside the float64 range ([error figures above](#attention-checks)) |
| Attention in CUDA graphs: replays across boundaries, moved keys, moved page tables | 117,337 replays, 0 rows differ; 345 of 345 planted faults failed as required |
| Attention, all 48 layers, real captures (recorded prompts, 4,015 tokens, the first 4,096 rows of a 32,768-token document) | every layer inside the float64 range and every kernel form byte-equal, 192 of 192 layer runs; rare-input path taken by 0 rows (v0.2's run, on an earlier build's x86 tool with the same kernels, carried by byte-equal digests) |
| Token gate (yardstick): server forced along the reference's replies, 445 positions | top-1 434/445 (bar 438): below the bar, as v0.2 (the same tokens); in the reference's top two 443/445 (bar 443): PASS |
| Free-running greedy replies against the reference (information): `chat_short`, `tool_call`, the extended set | EQUAL; departs at token 10 (a near-tie, the reference's top two 0.11 apart); extended set 1 EQUAL, 6 DEPARTS, 0 DIFFER of 7 |
| Concurrent equals alone, 1 / 2 / 4 streams, greedy and seeded T=1 | PASS, 28/28 equal |
| Prompt cache: resumed equals cold (54 requests against a cache-off reference), and a ~100k-token needle cold then repeated | PASS, 54/54 equal; the same reply both times |
| Three streams resuming one kept prompt together; six documents in three orders under a capped pool (kept prompts given back) | PASS |
| Admission: requests together equal alone while some wait for pages; a request over the pool refused by name | PASS, 9/9 together equal alone |
| `CONTEXT=524288` (the default recipe otherwise): a 524,288-token request admitted with its pages, one token more refused (HTTP 400); free-running replies against the reference (information) | PASS; `chat_short` EQUAL, `tool_call` departs at token 10, extended set 1 EQUAL, 6 DEPARTS, 0 DIFFER |
| Stream counts at `CONTEXT` 32k to 1M, DSpark and plain | PASS, the counts in the README |
| `/health` `kv`: idle, one stream, five requests on four streams (`streams_full`, one waiting), idle again | PASS |
| Structured output: constrained greedy replies against the reference under the same masks, a `json_object` reply and a required tool call | PASS |
| Default config (image input on): 4 streams of 131,072 admitted; MemAvailable during a 1, 2 and 4-stream run stays above the reserve (6.25 GiB floor, every rank) | PASS, lowest 7.56 GiB GiB on rank 0, 7.63-8.22 GiB GiB on ranks 1-3 (over the longer benchmark session: 7.56 GiB GiB / 7.63-8.22 GiB GiB) |
| Speed rows against the README | PASS, 9 of 9 of 9 of 9 rows within tolerance |
| Smoke: chat, reasoning effort high, a tool-call round trip, a 10k-token prompt, stream equals whole | PASS |
| Start-up refusals: a follower-only `TF_ROCE` (mesh on, off), one rank without the mesh wire, one rank missing the kernel image, `TF_INK2_MAX_STREAMS=9`, one rank with a different attention setting, an unknown attention setting | 7 of 7 refused, every rank stopped, the log naming the rank and the setting |
| Fresh installs from the release tarball: host, container, and a one-layer `CHECKPOINT` drop-in (each rank's layer copied from a checkpoint host and checked against the download) | PASS, PASS, PASS; the decode check `037a50ab15855ea6` on the host and container installs |
| Image pixels against Transformers' processor: 46 small fixtures and 15 large ones up to 4096x4096 (on the image path's own commits before the release build) | PASS, 46/46 and 15/15 equal on every field (size, grid, RGB and pixel values' sha256, refusal) |
| Vision tower at W=4: its output against the one-GPU pin, prompts cut and resumed inside an image, a poisoned control (on the image path's own commits) | equal to the pin on all four ranks; 6/6 equal; the control fails on all four ranks, as required |
| Image replies against the fp32 reference, forced along the server's replies, 14 prompts (on the image path's own commits) | served positions 696/723 agree, no departure at a margin of 1 logit or more (largest 0.59): PASS; all positions 862/896 (information) |
| Image replies with DSpark on and off; the 10,816-token request beside three text streams; four requests over the limits (on the image path's own commits) | PASS, 14/14 identical; beside equals alone; 4 of 4 HTTP 400 naming the numbers |
| Image input on the release build | the image checks re-run on the release build: 23 of 26 checks passed, 0 failed, 3 not measured (the tower's own time at the three smaller sizes, which the logs cannot separate); top-1 replies equal to the recorded image run's on 28/28 |
| Prompt cache on NVMe, W=2, 8 layers: states written at 12 cuts and read into poisoned pages, by the same and a new process; DSpark after a read; faults planted on one rank; one rank's switch or cap different (on the tier's own commits before the release build) | `740a5e9600f24fa1` 12/12 and 12/12; rounds and proposals equal a cold run's; every fault refused on both ranks, the cold fill equal; both ranks refuse |
| Prompt cache on NVMe at W=4: replies after server restarts, read from disk, against the tier off | 10 of 10 checks PASS: the 128,581-token chat prompt written to NVMe behind the reply on all four ranks, read back after each of 3 restarts with replies equal to the tier off, the tier's files under its 64 GiB cap and removed after (the 1-hour soak is not part of this run) |
| W=2, 8 layers: decode check, eager, prompt chunks, drafted (real, deferred, mixed), 4 streams, pages shuffled / poisoned / moved | `740a5e9600f24fa1` throughout |
| W=2: prompt cache on kept pages, every cut (inside a page, on a page boundary), with poisoned pages; negative control | `1e172489fdf500de`, proposals `36e9f7d3ccc916de`; the control fails as it must |
| W=2, 24 layers; 8 layers after a 1,000-token prompt (check, eager, scratch) | `21d17d8865873c73`; `b49345d551d715f7` |
| W=2: deep at 524,000 positions; synthetic 131k / 524k / 1,048,000 with and without the drafter | 524,000 `50315c56d82083e1`; synthetic 1,048,000 `ae0c5108db94b469` |
| Digest integrity: every byte loaded matches the manifest (with image input on, rank 0's 422 tower tensors too) | every server start reports "contents checked" |

## Running the checks yourself (step 6)

**Decode check** (server stopped): each decode step must equal the whole-prompt forward at the same W bit for bit. With
`setup.sh`: `./setup.sh my-sparks.conf stop check start` (add `--container` for the container path). By hand after `setup.sh` (ROOT
as in `my-sparks.conf`; on the manual path, `my-sparks.conf` and `../../data/long_4000.safetensors`, as in
[INSTALL-MANUAL.md](INSTALL-MANUAL.md)):

```
./stop.sh my-sparks.serve.conf
./check.sh my-sparks.serve.conf ROOT/data/long_4000.safetensors 3040 80    # prints PASS: ... bit-equal 80/80 ...
```

**The token gate**, **smoke** and the benchmarks run where the endpoint is reachable (rank 0's host when HTTP is on
loopback), from the unpacked release folder:

```
python3 recipe/inkling2-tp4/smoke.py http://127.0.0.1:8890 yardstick data/decode.json \
  chat_short=data/chat_short.ids.json tool_call=data/tool_call.ids.json --ext data/ext
                                      # v0.3 prints YARDSTICK: top-1 434/445 (< 438), in the oracle's top-2 443/445 (>= 443): FAIL
                                      # (the same as v0.2; top-1 below its bar: Agreement with the fp32 reference)
python3 recipe/inkling2-tp4/smoke.py http://127.0.0.1:8890 oracle data/decode.json \
  chat_short=data/chat_short.ids.json tool_call=data/tool_call.ids.json --ext data/ext
                                      # free-running replies: EQUAL, or DEPARTS at j (information)
python3 recipe/inkling2-tp4/smoke.py http://127.0.0.1:8890 smoke --long-tokens 8000
```

From `recipe/inkling2-tp4/` on the machine that ran `setup.sh`:

```
./bench.sh my-sparks.serve.conf sweep        # DSpark equals plain on every greedy row (exact_vs_other_mode)
```

The yardstick sends 445 one-token requests, one after another. The reference files come from
`tools/zig/inkling2_ref.py --attn fp32` (in the engine source: upstream TensorFold at `ENGINE_COMMIT` with `patches/`
applied, as `scripts/build-engine.sh` makes it): `data/decode.json` and `data/ext/` in its default mode (the stored
expert weights decoded to bf16), `data/so-cases/` in its checkpoint mode (activations quantized as the checkpoint's
quantization config requires, TokenSpeed's reference). Each token carries the reference's top-two margin at its
position.

**Structured output:** Two cases are gated against the reference under the same masks (a `json_object` reply and a
required tool call: 9 of 9 equal each), and four more are recorded: a named tool call equals the reference 9 of 9; a
JSON schema, a long string and a schema after thinking depart from it (first differences at tokens 33, 22 and 99). Every
constrained, plain and sampled reply in them is the same with DSpark's shared pass, with `TF_INK2_LATE_MASKS=0` and with
`DSPARK=0`, and in a group the same as alone; they are not claimed equal to the reference.
