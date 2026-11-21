# Inkling-2-Small at TP4: benchmarks

Paths in this file are relative to `recipe/inkling2-tp4/` (or the unpacked release folder, for `tools/`, `data/`, `plan/`), unless a link says otherwise.

Every result of v0.3 against v0.2's published figures, with its method and command (the headline tables are in the
[README](https://github.com/swbiggart/inkling2-sparks/blob/v0.3/README.md#benchmarks)).

## Results

Four GB10 hosts on one 200G switch, NCCL 2.30.7 over RoCE, the release defaults (`CONTEXT=131072`, four streams, the 1.5
GiB prompt-cache arena, the 6 GiB memory reserve, image input on, DSpark on unless a row says plain). Medians. v0.3 was
measured on 20 November 2026 with v0.2's method: each speed row ran 3 times on one server start and the row is the
median, except that the 2,048- to 32,768-token time-to-first-token rows, the 126,000-token cached time-to-first-token
row, the book rows and the 519,970-token plain decode row are medians of 9, and the 101,859-, 24,000- and 519,983-token
prompts, the 8-session needle, the prompt set (each of its 16 rows) and the installs ran once; the 1M, image and NVMe
rows say how they ran; the eight-stream rows and the 126,250-token prose book row measured on v0.3 are means of
the two v0.3 runs of the guard hold. Rows printed carried are v0.2's measurements: v0.3 decodes with v0.2's launches and gives the
same tokens; plain at 1 and 8 streams, DSpark at 1, 4 and 8 streams and the 128k DSpark book (prose and code) ran on both builds in one hold, the largest change -2.0%, inside that cell's
spread (the largest spread of any cell 3.1%), every token the same. Rows marked slower were measured on v0.3 and are slower than v0.2: the fresh host install (+0.3%), the container install (+1.0%), start to ready (+4.3%), eight plain streams (`MAX_STREAMS=8`) (-0.7%), eight DSpark streams (`MAX_STREAMS=8`) (-0.8%) and the 126,250-token book, prose (0.0%; -0.9% against v0.2 in the same hold). Turning image input off (`VISION=0`) does not remove the decode rows' difference: measured again with the vision tower not loaded, the same cells (plain at 8 streams, DSpark at 8 streams, 128k DSpark book, prose) read -0.3%, -1.0%, -0.9% against v0.2 (with it loaded: -0.7%, -0.8%, -0.9%), so the tower explains at most part of it.

Every table compares v0.3 with v0.2's published figures. Change is v0.3 against v0.2, from the figures as printed: for
tokens per second and for shares of the floor higher is better, for seconds lower is better; the change of a share is
relative, not in points.

### One stream

The server's own decode rate, one stream, temperature 0: `tools/bench_dspark.py --reps 5` (the tool call and thinking
run to their natural end, the others 256 tokens). Plain is a `DSPARK=0` server decoding 256 tokens of three prompts
(raw code, prose, code; `tools/bench_openai.py --tokens 256 --reps 3`); the README's plain row is the prose prompt, at
one decimal.

| Workload | Tokens a round, v0.2 | Tokens a round, v0.3 | v0.2 tok/s | v0.3 tok/s | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain, raw code |  |  | 33.74 | 33.74 | carried |
| Plain, prose |  |  | 33.85 | 33.85 | carried |
| Plain, code |  |  | 33.77 | 33.77 | carried |
| DSpark: tool call, to its end | 4.55 | carried | 72.8 | 72.8 | carried |
| DSpark: code (chat) | 4.94 | carried | 79.7 | 79.7 | carried |
| DSpark: thinking, effort high, to its end | 4.07 | carried | 68.0 | 68.0 | carried |
| DSpark: code (raw completion) | 3.25 | carried | 56.8 | 56.8 | carried |
| DSpark: prose (chat, thinking off) | 2.97 | carried | 53.9 | 53.9 | carried |

DSpark's rate depends on how predictable the text is, so the table is a range, not a promise. A short reply such as
the tool call spends a larger share of its time starting up.

### Total throughput

Tokens per second across all streams, `tools/bench_concurrent.py --levels 1,2,4,8 --temperatures 0 --tokens 256 --reps 3
--prompts tools/bench_prompts.json`: request i sends item i of the prompt list (tools/bench_prompts.json, prompts whose
replies run at least 256 visible tokens), greedy, 256 tokens. A cell is the total output while every stream of a level
is still decoding: the tokens all streams produce in the window from the latest first chunk to the earliest last chunk
(the chunk arrival times in the run's `client_intervals`), over the window's length; the median of each run's reps, then
of the three runs. The run's json keeps the arrivals, not each chunk's token count, so each chunk after a request's
first counts that request's (tokens - 1) / (chunks - 1): its first chunk is one token, and plain decode carries about
one token a chunk (254 to 256 chunks for 256 tokens). One-stream cells keep the time from first to last token (the two
agree within 0.05 tok/s there). With DSpark at the defaults the total is 94.1 tok/s on one stream,
114.0 on two and 118.5 on four: about flat beyond two, as DSpark already batches the drafted
rows into each step; plain decode scales with the number of streams. The defaults run up to four streams
(`CONTEXT=131072`), at 256-token replies and again at 128 (`--levels 1,2,4 --tokens 128`); the `MAX_STREAMS=8` rows use
`CONTEXT=32768` and `MAX_STREAMS=8`, so that eight streams fit, and plain there is a `DSPARK=0` server with the same
settings.

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

DSpark's gain falls with more streams: at eight it gives 87.2 tok/s against 86.1 plain.
`bench.sh sweep` at the defaults gave each stream the same token hash with DSpark and plain at 1, 2 and 4 streams.

### Time to first token

Seconds. Cold: `tools/bench_cache.py --require-cold`, three prompts of exactly this length (the 126,000-token row: one
prompt). Cached: a document of about this length (2,027, 8,019 and 31,923 tokens; 125,377 for 128k) asked once, then the
same request again. The 101,859-token row is a document with one fact in it, cold, then the same request again (the
prompt-cache gate); the 519,983-token rows are [below](#one-request-at-512k-tokens). The short prompt is
`tools/bench_openai.py`'s, on the release's server.

| Prompt | v0.2 | v0.3 | Change |
| --- | ---: | ---: | ---: |
| A short prompt | 0.10 | 0.10 | carried |
| 2,048 tokens, cold | 1.18 | 1.00 | -15.3% |
| 8,192 tokens, cold | 4.78 | 4.01 | -16.1% |
| 32,768 tokens, cold | 20.7 | 17.7 | -14.5% |
| 126,000 tokens, cold | 102 | 91 | -10.8% |
| A 519,983-token document, cold (`CONTEXT=524288`) | 826 | 796 | -3.6% |
| 2,048 tokens, cached | 0.15 | 0.15 | carried |
| 8,192 tokens, cached | 0.15 | 0.15 | carried |
| 32,768 tokens, cached | 0.16 | 0.16 | carried |
| 126,000 tokens, cached | 0.158 | 0.158 | carried |
| A 519,983-token document, cached | 0.27 | 0.27 | carried |
| 101,859 tokens, cold | 80.4 | 71.3 | -11.3% |
| 101,859 tokens, cached | 0.1 | 0.1 | carried |
| 24,000 tokens, nothing else running (fresh server) | 22.0 | 17.2 | -21.8% |
| 24,000 tokens, beside one decoding stream (`--decode-share 0.5`) | 58.1 | 58.1 | carried |

Cold prompts fill faster than on v0.2: in prompt chunks of 576 rows or more, each half's attention sum crosses the
fabric under the other half's compute, and the MoE exchange carries each rank's quantized values in place of bf16 ones
(the same bits). A repeated prompt resumes from its kept state: the kept pages are adopted, not copied, and only the
state's small fixed part comes back from the arena. Another question on the same document resumes the document too:
0.20 s up to 32k. A document that extends an earlier request's document inside the same message is not resumed: the
cache keeps states at message boundaries and at the end of each prompt.

### Prompt cache on NVMe

With `TF_INK2_NVME` set (in `OPT_ENV`; off by default), kept text prompts of
1,024 tokens or more are also written to each host's disk and read back after a restart.
Measured with the 128k chat prompt
(128,581 tokens: a long document as the system message): asked once with the tier on, the server
restarted, then the same request again; the median of 3 restarts. Cold is the same prompt with
nothing kept.

| Measure | v0.3 |
| --- | ---: |
| The 128k chat prompt, cold, s | 95.87 |
| The same after a server restart, read from NVMe, s | 0.394 |
| Read from NVMe for that restore, GiB a rank | 1.15 |

The replies after each restart are the same tokens as with the tier off.

### Images

Seconds to the first token of one cold image request (a new image each run), median of 5;
one image and a short question ("Describe this image in two sentences."), 16 tokens, temperature 0, a different image each run. An image is not resized: one image token for each 40 x 40 pixel block, padded.

| Image | Image tokens | Time to first token |
| --- | ---: | ---: |
| 448 x 336 | 108 | 0.293 |
| 1024 x 1024 | 676 | 0.612 |
| 1920 x 1080 | 1,296 | 0.973 |
| 3840 x 2160 | 5,184 | 4.294 |
| 4096 x 4096 | 10,609 | 11.012 |

PNG up to 1920x1080, JPEG quality 95 above (3840x2160, 4096x4096).
The vision tower's own time, from rank 0's log, median of 5: 0.988 s at 3840 x 2160 and
4.400 s at 4096 x 4096 (at the three smaller sizes it is too short to separate in the log).

### Decode at depth

One stream, the server's decode rate of a 256-token reply after a long prompt, tokens per second. Plain is a
`DSPARK=0` server; default is the release (DSpark by depth). The first rows are one prompt set at four lengths. The
book rows are a book as the system message, cut to length (prose: the reply continues the story; code: a program asked
after it), the default only, 3 rounds of 3 replies a kind (the median of all nine), each request resumed from the
book's kept state. The 512k plain row decodes after the same book, cut to 519,970 tokens. The 1M rows
(`CONTEXT=1048576`): the default (DSpark by depth) on a 1,038,466-token book, 9 replies of 256 tokens, server median; v0.2's 1M figure is a synthetic 1,048,000-token state in the decode tool,
plain (v0.2's default at 1M).

| Prompt tokens | v0.2 | v0.3 | Change |
| --- | ---: | ---: | ---: |
| 8,163, plain | 33.3 | 33.3 | carried |
| 33,404, plain | 32.1 | 32.1 | carried |
| 67,804, plain | 30.4 | 30.4 | carried |
| 129,173, plain | 28.0 | 28.0 | carried |
| 519,970 (`CONTEXT=524288`), plain | 18.22 | 18.22 | carried |
| 1M, plain | 12.5 | 12.5 | carried |
| 1M, default (v0.2: plain) | 12.5 | 19.4 | +55.2% |
| 8,163, default | 66.6 | 66.6 | carried |
| 33,404, default | 69.9 | 69.9 | carried |
| 67,804, default | 73.8 | 73.8 | carried |
| 129,173, default | 67.8 | 67.8 | carried |
| Book, about 126,250, prose | 27.3 | 27.3 | 0.0% (slower in the same hold: -0.9%) |
| Book, about 126,250, code | 44.8 | 44.8 | carried |
| Book, about 520,000 (`CONTEXT=524288`), prose | 25.2 | 25.2 | carried |
| Book, about 520,000 (`CONTEXT=524288`), code | 26.4 | 26.4 | carried |

Each kind's 9 replies at each length had one token hash. At 1M the replies with DSpark
were the same tokens as plain ones (`DSPARK=0` on a pre-release build, the same book).

### Install and load

Seconds.

| Measure | v0.2 | v0.3 | Change |
| --- | ---: | ---: | ---: |
| Host install, from an empty folder to a served smoke (`setup.sh`, wall time, weights verified in place with `--weights-dir`) | 342 | 343 | +0.3% (slower) |
| Container install, the same from an empty folder with `--container` (also the image built and copied to the other three hosts) | 588 | 594 | +1.0% (slower) |
| Host install with `CHECKPOINT=host:/path` (the full checkpoint on a fifth host on the switch, every file hashed, each rank's bytes copied over the fabric, verified) | 520 | 520 | carried |
| The copy itself, a rank, from the checkpoint (about 92 GiB) | 83-85 | 83-85 | carried |
| Start to ready, default config (weights in the page cache; rank 0 also loads the vision tower) | 23 | 24 | +4.3% (slower) |

| Check | v0.2 | v0.3 |
| --- | --- | --- |
| Smoke and token checks at the end of `setup.sh`, every path | smoke PASS; the reference check reports `chat_short` EQUAL (information) | smoke PASS; the reference check reports `chat_short` EQUAL (information) |
| W=4 decode check on each install (`setup.sh CONFIG stop check`) | PASS, 80/80 bit-equal, `037a50ab15855ea6`, host and container | PASS, 80/80 bit-equal, `037a50ab15855ea6`, host and container |

The host install is the weights check (every byte re-read and hashed on all four hosts) plus preflight, assets and
libs, folds (each rank's DSpark folds file built from nothing, byte for byte equal to the one already on the host),
manifest and config, and the start. With `VISION=1` (the default) rank 0's weights include the vision tower,
0.61 GiB more. With `CHECKPOINT`, the source's hashes are cached on its host, so a second install from it skips
that step. Not exercised: the base-image pull of the container build (the hosts had the layers cached) and
`--fabric-install`.

### Memory while serving

| Measure | Result |
| --- | ---: |
| MemAvailable at ready, default config, rank 0 / ranks 1-3 | 8.03 GiB / 8.17-8.23 GiB |
| Lowest MemAvailable over a benchmark run, rank 0 / ranks 1-3 | 7.56 GiB / 7.63-8.22 GiB |
| Lowest MemAvailable while the default config serves 1, 2 and 4 streams (the release check needs 6.25 GiB), rank 0 / ranks 1-3 | 7.56 GiB / 7.63-8.22 GiB |

The page pool takes the room after the 6 GiB reserve, with the vision tower loaded on rank 0.

## Context, streams and the pool

Start-up counts on the four GB10s, the release's recipe (reserve 6 GiB, image input on) with only `CONTEXT`, `DSPARK`
and `MAX_STREAMS` set (raw lines: rank 0's log, `inkling2: context C -> N streams on a KV page pool (...)`). Pages are
256 positions, 2.25 MiB a rank with DSpark and 2.0 MiB plain. "Kept" is how many 128k prompts the pool holds as kept
prompts with every stream idle.

| `CONTEXT` | DSpark | Streams (`MAX_STREAMS=4`) | Pool positions | 128k prompts kept | Room after load on the rank with the least, GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32,768 | on | 4 | 1,150,208 | requests up to 32k | 14.21 |
| 131,072 (default) | on | 4 | 1,148,160 | 8 | 14.19 |
| 131,072 | off | 4 | 1,415,168 | 10 | 15.03 |
| 262,144 | on | 4 | 1,147,136 | 8 | 14.18 |
| 524,288 | on | 2 | 1,158,656 | 8 | 13.99 |
| 524,288 | off | 2 | 1,442,816 | 11 | 15.00 |
| 1,048,576 | on (the default) | 1 | 1,384,704 | 10 | 14.04 |

With `MAX_STREAMS=8`: 8 / 8 streams at 131,072 (DSpark / plain) and 8
at 32,768. A count at the margin depends on the memory free when the server starts; the pool's size follows the same
reading, so it moves by some hundreds of pages between starts of the same configuration.

**1M, one stream with DSpark** (`CONTEXT=1048576`):
ready in 25 s (v0.2, plain: 24 s).
Decode at 1M tokens of context: 19.4 tok/s by default and 12.5 tok/s
plain (v0.2: 12.5 tok/s, plain). The 1.5 GiB prompt-cache arena is kept; the drafter's caches (about 1 GiB a rank)
take pool room kept prompts had: 10 128k prompts kept, against 10 on v0.2 (plain). `DSPARK=0` serves 1M
plain.

How 1M is checked bit for bit: a real 1M-token prompt takes about 45 minutes to fill (2,670.3 s for the
1,038,466-token book above; the global layers' attention grows with the square of the length), so `tf-inkling2-decode --exp deep --fill synth` writes every layer's keys and values for positions 0
to 1,047,998, the sliding layers' rings, the convolution states and the drafter's caches with fixed pseudo-random
values on every rank (no forward), runs 64 decode steps from there and compares their logits with one 64-row prompt
forward over the same positions, bit for bit; with the drafter it also decodes the same tokens in drafted rounds. Then
every rank reads its global layers' filled positions back and the ranks' verdicts are gathered: one changed row on any
rank fails the check. This covers what changes with the length (attention indexing, the key splits and grids past 2^20
positions, the op scratch, the log scaling of the queries), not the model's output on a real 1M-token text: that runs
on the 1,038,466-token book above, and a 519,983-token prompt gives the same reply cold and resumed (below).

ROOT is the one in `my-sparks.conf` (the release's `data/` on the hosts); on the manual path, run it with `my-sparks.conf`
and `../../data/long_4000.safetensors` from `recipe/inkling2-tp4/` of the unpacked release:

```
./check.sh my-sparks.serve.conf ROOT/data/long_4000.safetensors 1048000 64 -- --exp deep-draft --fill synth --context 1048592
```

## One request at 512k tokens

`CONTEXT=524288` in the config serves requests up to 524,288 tokens (prompt plus reply) with the release's other
settings; DSpark drafts by depth. The same four ranks and weights serve it; restart with the other config to switch.

```
echo CONTEXT=524288 >> my-sparks.conf
./setup.sh my-sparks.conf stop config start    # add --container for the container path
```

| Measure | v0.2 | v0.3 | Change |
| --- | ---: | ---: | ---: |
| A 519,983-token document (book text) with one fact inserted at 37 %, a cold prefill: time to first token, s | 826 | 796 | -3.6% |
| The same prompt again (resumed from its kept state), s | 0.27 | 0.27 | carried |
| Decode after the 519,970-token book, default (DSpark by depth), tok/s | 25.2 | 25.2 | carried |
| Decode after the 519,970-token book, plain (`DSPARK=0`), tok/s | 18.22 | 18.22 | carried |

The time to first token is measured at the client; the greedy reply states the fact: yes, in its reasoning, quoting the
inserted sentence with the code (the 128-token reply ends there); the resumed request gives the same reply (token hash
eebc3258f678).

What is checked at this length: the decode check at 524,000 positions (W=2: one prefill, then 64 decode steps whose
logits equal one prompt forward over the same positions bit for bit); drafted replies equal plain ones at about 520,000
tokens on the four GB10s; the free-running replies on the release's recorded prompts with `CONTEXT=524288` against the
reference (`chat_short` EQUAL, `tool_call` departs at token 10, extended set 1 EQUAL, 6 DEPARTS, 0 DIFFER, information);
a request of exactly 524,288 tokens is admitted with its pages and one token more is refused by name. A prompt this long
takes 796 s to fill, and the server streams nothing before the first token, so clients need a long
read timeout.

## Measuring your install

`bench.sh CONFIG COMMAND` (CONFIG is the server config: `my-sparks.serve.conf` after `setup.sh`, `my-sparks.conf` on
the manual path) runs `tools/inkling2_bench.py` (Python 3.9 or later, standard library only) on rank 0's
host against the running server and copies its JSON back to `./bench-results/`. It runs from a clone of this
repository or from the unpacked release tarball:

```
./bench.sh my-sparks.serve.conf check [--sums] # GPU-free; the server may run or not
./bench.sh my-sparks.serve.conf sweep        # decode, workloads and cold prompts at the defaults (~7 minutes)
./bench.sh my-sparks.serve.conf prompts      # 16 rows x temperature 0 and 0.7, 2,048 tokens each (~21 minutes)
./bench.sh my-sparks.serve.conf needle       # 8 sessions at once: MAX_STREAMS=8 and CONTEXT=32768
./bench.sh my-sparks.serve.conf needle --sessions 1 --tokens 120000   # one deep needle at the defaults
./bench.sh my-sparks.serve.conf ledger bench-results/sweep-*.json [--prompts bench-results/prompts-*.json]
```

- **check:** the config's values; on every host the binaries, the kernel image, the libraries and that the binaries
  resolve against them, the plan files, the rank folder (config, shards, digests, the DSpark folds when `DSPARK=1`)
  and Python on rank 0's host; `--sums` also checks every file of the release's `SHA256SUMS`. Exit 1 if anything
  failed.
- **sweep:** decode at 1, 2, 4 and 8 streams within `MAX_STREAMS` (DSpark and plain), the one-stream workloads, cold
  prompts of 2k and 8k tokens of generated text (prompt tokens over the server's own prefill time, and the client's
  time to first token); medians of 3. Greedy rows must give the same token hashes across repeats and between DSpark
  and plain, and the JSON says so (`repeatable`, `exact_vs_other_mode`).
- **prompts:** a fixed set of 16 authored rows (`tools/inkling2_prompts.json`: 5 chat, 5 code, 6 agentic with tools
  and tool-call history), each at temperature 0 and 0.7 (seeded), 2,048 tokens a row so rows compare across builds
  (`--natural` stops at the model's end instead).
- **needle:** each session gets its own generated log text with one passphrase line at its depth and the question at
  the end; exact = its own passphrase in the reply and no other session's.
- **ledger:** for each sweep row, the bytes a rank reads a step (from the checkpoint's sizes by role,
  `tools/inkling2_sheet.json`) over a streaming bandwidth (250 GB/s by default), plus the gathers' latency: a floor
  for the step's time, and the row's share of it. Prefill and depth rows are not in it.

On this release, against v0.2's published figures (the prompt set in tokens per second, at the defaults):

| Measure | v0.2 | v0.3 | Change |
| --- | ---: | ---: | ---: |
| Needle, 8 sessions at once, ~28k tokens each, depths across the text (`CONTEXT=32768`, `MAX_STREAMS=8`): exact | 8 of 8 | 8 of 8 | carried |
| The same: wall time, s | 148 | 148 | carried |
| Prompt set, temperature 0, chat rows | 48.8 | 48.8 | carried |
| Prompt set, temperature 0, code rows | 75.3 | 75.3 | carried |
| Prompt set, temperature 0, agentic rows | 69.6 | 69.6 | carried |
| Prompt set, temperature 0, all 16 rows | 60.1 | 60.1 | carried |
| Prompt set, temperature 0.7, chat rows | 38.5 | 38.5 | carried |
| Prompt set, temperature 0.7, code rows | 50.4 | 50.4 | carried |
| Prompt set, temperature 0.7, agentic rows | 57.6 | 57.6 | carried |
| Prompt set, temperature 0.7, all 16 rows | 46.6 | 46.6 | carried |
| Ledger: one stream plain, share of the floor, % | 74.3 | 74.3 | carried |
| Sweep at the defaults: errors, greedy repeatable, DSpark equal to plain | 0, yes, yes | 0, yes, yes | carried |

## Other benchmark commands (step 7)

The tools the methods above name (Python 3.11 or later, standard library only), against the server:

```
python3 tools/bench_openai.py http://127.0.0.1:8890 model --tokens 256 --reps 3 --temperatures 0
python3 tools/bench_dspark.py http://127.0.0.1:8890/v1 model --reps 5
python3 recipe/inkling2-tp4/prompt_ids.py MODEL 8192 prefill-8192.json        # needs `pip install tokenizers`
python3 tools/bench_cache.py http://127.0.0.1:8890 model --tokens 32 --repeats 1 --streams 1 \
  --prompt-file prefill-8192.json --expected-prompt-tokens 8192 --require-cold
python3 tools/bench_concurrent.py http://127.0.0.1:8890 model --levels 1,2,4 --temperatures 0 --tokens 256
```

Concurrent equals alone, with `PARALLEL=2` in the config:

```
python3 tools/bench_concurrent.py http://127.0.0.1:8890 model --levels 2 --alone --temperatures 0 --tokens 128
```
