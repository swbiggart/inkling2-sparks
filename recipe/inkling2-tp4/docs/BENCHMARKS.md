# Inkling-2-Small at TP4: benchmarks

Paths in this file are relative to `recipe/inkling2-tp4/` (or the unpacked release folder, for `tools/`, `data/`, `plan/`), unless a link says otherwise.

Every result of v0.2 against v0.1, with its method and command (the headline tables are in the
[README](../../../README.md#benchmarks)).

## Results

Four GB10 hosts on one 200G switch, NCCL 2.30.7 over RoCE, the release defaults (`CONTEXT=131072`, four streams, the 1.5
GiB prompt-cache arena, DSpark on unless a row says plain). Medians. v0.2 was measured on 19 November 2026: each speed
row ran 3 times on one server start and the row is the median, except that the 2,048- to 32,768-token
time-to-first-token rows, the 126,000-token cached time-to-first-token row, the book rows and the 519,970-token plain
decode row are medians of 9, and the 101,859-, 24,000- and 519,983-token prompts, the 1M decode, the 8-session needle,
the prompt set (each of its 16 rows) and the installs ran once.

Every table compares v0.2 with v0.1. Change is v0.2 against v0.1, from the figures as printed: for tokens per second
and for shares of the floor higher is better, for seconds lower is better; the change of a share is relative, not in
points.

### One stream

The server's own decode rate, one stream, temperature 0: `tools/bench_dspark.py --reps 5` (the tool call and thinking
run to their natural end, the others 256 tokens). Plain is a `DSPARK=0` server decoding 256 tokens of three prompts
(raw code, prose, code; `tools/bench_openai.py --tokens 256 --reps 3`); the README's plain row is the prose prompt, at
one decimal.

| Workload | v0.1 tok/s | v0.2 tok/s | Change | Tokens a round, v0.1 | Tokens a round, v0.2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain, raw code | 30.73 | 33.74 | +9.8% | | |
| Plain, prose | 30.98 | 33.85 | +9.3% | | |
| Plain, code | 30.95 | 33.77 | +9.1% | | |
| DSpark: tool call, to its end | 70.2 | 72.8 | +3.7% | 5.23 | 4.55 |
| DSpark: code (chat) | 60.8 | 79.7 | +31.1% | 4.32 | 4.94 |
| DSpark: thinking, effort high, to its end | 56.3 | 68.0 | +20.8% | 3.94 | 4.07 |
| DSpark: code (raw completion) | 52.7 | 56.8 | +7.8% | 3.76 | 3.25 |
| DSpark: prose (chat, thinking off) | 40.0 | 53.9 | +34.7% | 2.39 | 2.97 |

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
agree within 0.05 tok/s there). v0.1 is measured the same way on the same prompts, so its column differs from the
figures published with v0.1 (v0.1's shorter prompts, and the client's own `aggregate_tps`, timed to the last visible
text). With DSpark at the defaults the total is 94.1 tok/s on one stream, 114.0 on two and 118.5 on four: about flat
beyond two, as DSpark already batches the drafted rows into each step; plain decode scales with the number of streams.
The defaults run up to four streams (`CONTEXT=131072`), at 256-token replies and again at 128 (`--levels 1,2,4 --tokens
128`); the `MAX_STREAMS=8` rows use `CONTEXT=32768` and `MAX_STREAMS=8`, so that eight streams fit, and plain there is a
`DSPARK=0` server with the same settings.

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

DSpark's gain falls with more streams: at eight it is about 1% over plain (87.9 against 86.7 tok/s). `bench.sh sweep` at
the defaults gave each stream the same token hash with DSpark and plain at 1, 2 and 4 streams.

### Time to first token

Seconds. Cold: `tools/bench_cache.py --require-cold`, three prompts of exactly this length (the 126,000-token row: one
prompt). Cached: a document of about this length (2,027, 8,019 and 31,923 tokens; 125,377 for 128k) asked once, then the
same request again. The 101,859-token row is a document with one fact in it, cold, then the same request again (the
prompt-cache gate); the 519,983-token rows are [below](#one-request-at-512k-tokens). The short prompt is
`tools/bench_openai.py`'s, on the release's server.

| Prompt | v0.1 | v0.2 | Change |
| --- | ---: | ---: | ---: |
| A short prompt | 0.11 | 0.10 | -9.1% |
| 2,048 tokens, cold | 1.32 | 1.18 | -10.6% |
| 8,192 tokens, cold | 5.84 | 4.78 | -18.2% |
| 32,768 tokens, cold | 30.6 | 20.7 | -32.4% |
| 126,000 tokens, cold | 221 | 102 | -53.8% |
| A 519,983-token document, cold (`CONTEXT=524288`) | 2,754 | 826 | -70.0% |
| 2,048 tokens, cached | 0.18 | 0.15 | -16.7% |
| 8,192 tokens, cached | 0.18 | 0.15 | -16.7% |
| 32,768 tokens, cached | 0.19 | 0.16 | -15.8% |
| 126,000 tokens, cached | 0.189 | 0.158 | -16.4% |
| A 519,983-token document, cached | 12.1 | 0.27 | -97.8% |
| 101,859 tokens, cold | 158.7 | 80.4 | -49.3% |
| 101,859 tokens, cached | 0.1 | 0.1 | 0.0% |
| 24,000 tokens, nothing else running (fresh server) | 31.7 | 22.0 | -30.6% |
| 24,000 tokens, beside one decoding stream (`--decode-share 0.5`) | 82.2 | 58.1 | -29.3% |

A repeated prompt resumes from its kept state: the kept pages are adopted, not copied, and only the state's small fixed
part comes back from the arena. Another question on the same document resumes the document too: 0.20 s up to 32k. A
document that extends an earlier request's document inside the same message is not resumed: the cache keeps states at
message boundaries and at the end of each prompt.

### Decode at depth

One stream, the server's decode rate of a 256-token reply after a long prompt, tokens per second. Plain is a
`DSPARK=0` server; default is the release (DSpark by depth). The first rows are one prompt set at four lengths. The
book rows are a book as the system message, cut to length (prose: the reply continues the story; code: a program asked
after it), the default only, 3 rounds of 3 replies a kind (the median of all nine), each request resumed from the
book's kept state. The 512k plain row decodes after the same book, cut to 519,970 tokens.

| Prompt tokens | v0.1 | v0.2 | Change |
| --- | ---: | ---: | ---: |
| 8,163, plain | 30.3 | 33.3 | +9.9% |
| 33,404, plain | 29.2 | 32.1 | +9.9% |
| 67,804, plain | 27.6 | 30.4 | +10.1% |
| 129,173, plain | 25.2 | 28.0 | +11.1% |
| 519,970 (`CONTEXT=524288`), plain | 15.96 | 18.22 | +14.2% |
| 1,048,000 (synthetic state, decode tool), plain (`CONTEXT=1048576`) | 10.8 | 12.5 | +15.7% |
| 8,163, default | 50.8 | 66.6 | +31.1% |
| 33,404, default | 67.0 | 69.9 | +4.3% |
| 67,804, default | 61.8 | 73.8 | +19.4% |
| 129,173, default | 51.5 | 67.8 | +31.7% |
| Book, about 126,250, prose | 24.8 | 27.3 | +10.1% |
| Book, about 126,250, code | 33.1 | 44.8 | +35.3% |
| Book, about 520,000 (`CONTEXT=524288`), prose | 16.8 | 25.2 | +50.0% |
| Book, about 520,000 (`CONTEXT=524288`), code | 21.7 | 26.4 | +21.7% |

Each kind's 9 replies at each length had one token hash.

### Install and load

Seconds.

| Measure | v0.1 | v0.2 | Change |
| --- | ---: | ---: | ---: |
| Host install, from an empty folder to a served smoke (`setup.sh`, wall time, weights verified in place with `--weights-dir`) | 341 | 342 | +0.3% |
| Container install, the same from an empty folder with `--container` (also the image built and copied to the other three hosts) | 579 | 588 | +1.6% |
| Host install with `CHECKPOINT=host:/path` (the full checkpoint on a fifth host on the switch, every file hashed, each rank's bytes copied over the fabric, verified) | 508 | 520 | +2.4% |
| The copy itself, a rank, from the checkpoint (about 92 GiB) | 75-91 | 83-85 | +10.7% to -6.6% |
| Start to ready, default config (weights in the page cache) | 23 | 23 | 0.0% |

| Check | v0.2 |
| --- | --- |
| Smoke and token checks at the end of `setup.sh`, every path | smoke PASS; the reference check reports `chat_short` EQUAL (information) |
| W=4 decode check on each install (`setup.sh CONFIG stop check`) | PASS, 80/80 bit-equal, `037a50ab15855ea6`, host and container |

The host install is the weights check (every byte re-read and hashed on all four hosts) plus preflight, assets and
libs, folds (each rank's DSpark folds file built from nothing, byte for byte equal to the one already on the host),
manifest and config, and the start. With `CHECKPOINT`, the source's hashes are cached on its host, so a second install
from it skips that step. Not exercised: the base-image pull of the container build (the hosts had the layers cached)
and `--fabric-install`.

### Memory while serving

| Measure | Result |
| --- | ---: |
| MemAvailable at ready, default config, rank 0 / ranks 1-3 | 10.85 GiB / 9.60-10.59 GiB |
| Lowest MemAvailable over a benchmark run, rank 0 / ranks 1-3 | 9.69 GiB / 8.99-9.71 GiB |

The page pool takes the room after the 8 GiB reserve. Over the default config's benchmark run, MemAvailable stays above
the reserve on every rank (lowest 8.99 GiB), and the ranks' lowest values are within three quarters of a GiB of each
other.

## Context, streams and the pool

Start-up counts on the four GB10s, the release's recipe with only `CONTEXT`, `DSPARK` and `MAX_STREAMS` set (raw
lines: rank 0's log, `inkling2: context C -> N streams on a KV page pool (...)`). Pages are 256 positions, 2.25 MiB a
rank with DSpark and 2.0 MiB plain. "Kept" is how many 128k prompts the pool holds as kept prompts with every stream
idle.

| `CONTEXT` | DSpark | Streams (`MAX_STREAMS=4`) | Pool positions | 128k prompts kept | Room after load on the rank with the least, GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32,768 | on | 4 | 996,352 | requests up to 32k | 12.89 |
| 131,072 (default) | on | 4 | 944,128 | 7 | 12.44 |
| 131,072 | off | 4 | 1,240,832 | 9 | 13.70 |
| 262,144 | on | 3 | 1,008,128 | 7 | 12.84 |
| 524,288 | on | 1-2, depending on free memory at start | 1,230,592 | 9 | 12.71 |
| 524,288 | off | 2 | 1,161,216 | 8 | 12.85 |
| 1,048,576 | off (the recipe) | 1 | 1,406,464 | 10 | 12.86 |

With `MAX_STREAMS=8`: 7 / 8 streams at 131,072 (DSpark / plain) and 8 at 32,768. A count at the margin depends on the
memory free when the server starts; the pool's size follows the same reading, so it moves by some hundreds of pages
between starts of the same configuration.

**1M, one plain stream** (`CONTEXT=1048576`): ready in 24 s (v0.1: 24 s). Decode at 1,048,000 tokens of context: 79.91
ms a step, 12.5 tok/s (v0.1: 92.4 ms, 10.8 tok/s).

How 1M is checked: a real 1M-token prompt would take hours to fill (the global layers' attention grows with the square
of the length), so `tf-inkling2-decode --exp deep --fill synth` writes every layer's keys and values for positions 0
to 1,047,998, the sliding layers' rings, the convolution states and the drafter's caches with fixed pseudo-random
values on every rank (no forward), runs 64 decode steps from there and compares their logits with one 64-row prompt
forward over the same positions, bit for bit; with the drafter it also decodes the same tokens in drafted rounds. Then
every rank reads its global layers' filled positions back and the ranks' verdicts are gathered: one changed row on any
rank fails the check. This covers what changes with the length (attention indexing, the key splits and grids past 2^20
positions, the op scratch, the log scaling of the queries), not the model's output on a real 1M-token text; that is
checked end to end at about 520,000 tokens (below).

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

| Measure | v0.1 | v0.2 | Change |
| --- | ---: | ---: | ---: |
| A 519,983-token document (book text) with one fact inserted at 37 %, a cold prefill: time to first token, s | 2,754 | 826 | -70.0% |
| The same prompt again (resumed from its kept state), s | 12.1 | 0.27 | -97.8% |
| Decode after the 519,970-token book, default (DSpark by depth), tok/s | 16.8 | 25.2 | +50.0% |
| Decode after the 519,970-token book, plain (`DSPARK=0`), tok/s | 15.96 | 18.22 | +14.2% |

On v0.2 the time to first token is measured at the client (v0.1's 2,754 s is its request's 2,761.5 s less the 127 tokens
after the first at 16.96 tok/s); the greedy reply states the fact: yes, in its reasoning, quoting the inserted sentence
with the code (the 128-token reply ends there); the resumed request gives the same reply (token hash eebc3258f678).

What is checked at this length: the decode check at 524,000 positions (W=2: one prefill, then 64 decode steps whose
logits equal one prompt forward over the same positions bit for bit); drafted replies equal plain ones at about 520,000
tokens on the four GB10s; the free-running replies on the release's recorded prompts with `CONTEXT=524288` against the
reference (`chat_short` EQUAL, `tool_call` departs at token 10, extended set 1 EQUAL, 6 DEPARTS, 0 DIFFER, information);
a request of exactly 524,288 tokens is admitted with its pages and one token more is refused by name. A prompt this long
takes about 14 minutes to fill, and the server streams nothing before the first token, so clients need a long read
timeout.

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

On this release, against v0.1 (the prompt set in tokens per second, at the defaults):

| Measure | v0.1 | v0.2 | Change |
| --- | ---: | ---: | ---: |
| Needle, 8 sessions at once, ~28k tokens each, depths across the text (`CONTEXT=32768`, `MAX_STREAMS=8`): exact | 8 of 8 | 8 of 8 | same |
| The same: wall time, s | 207 | 148 | -28.5% |
| Prompt set, temperature 0, chat rows | 41.0 | 48.8 | +19.0% |
| Prompt set, temperature 0, code rows | 70.2 | 75.3 | +7.3% |
| Prompt set, temperature 0, agentic rows | 68.8 | 69.6 | +1.2% |
| Prompt set, temperature 0, all 16 rows | 54.1 | 60.1 | +11.1% |
| Prompt set, temperature 0.7, chat rows | 35.8 | 38.5 | +7.5% |
| Prompt set, temperature 0.7, code rows | 58.4 | 50.4 | -13.7% |
| Prompt set, temperature 0.7, agentic rows | 50.4 | 57.6 | +14.3% |
| Prompt set, temperature 0.7, all 16 rows | 47.2 | 46.6 | -1.3% |
| Ledger: one stream plain, share of the floor, % | 68.5 | 74.3 | +8.5% |
| Sweep at the defaults: errors, greedy repeatable, DSpark equal to plain | 0, yes, yes | 0, yes, yes | same |

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
