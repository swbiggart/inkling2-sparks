# Inkling-2-Small at TP4: benchmarks

Paths in this file are relative to `recipe/inkling2-tp4/` (or the unpacked release folder, for `tools/`, `data/`, `plan/`), unless a link says otherwise.

All results, methods and command lines for release v0.1 (the headline tables are in the [README](../../../README.md#benchmarks)).

## Results

Four GB10 hosts on one 200G switch, NCCL 2.30.7 over RoCE, the release defaults (`CONTEXT=131072`, four streams, the
1.5 GiB prompt-cache arena, DSpark on unless a row says plain), measured on 15-16 November 2026. Medians. Each
speed row ran three times, each on its own server start, and the first run is dropped (the median is over the other
two runs' repeats).

### One stream

The server's own decode rate, one stream, temperature 0: `tools/bench_dspark.py --reps 5` (the tool call and thinking
run to their natural end, the others 256 tokens). Plain is a `DSPARK=0` server decoding 256 tokens of three prompts
(raw code, prose, code; `tools/bench_openai.py --tokens 256 --reps 3`): 30.98 tok/s on the prose prompt (31.0 in
the README), 30.73 on raw code and 30.95 on code.

| Workload | DSpark tok/s | Tokens a round |
| --- | ---: | ---: |
| Tool call, to its end | 70.2 | 5.23 |
| Code (chat) | 60.8 | 4.32 |
| Thinking, effort high, to its end | 56.3 | 3.94 |
| Code (raw completion) | 52.7 | 3.76 |
| Prose (chat, thinking off) | 40.0 | 2.39 |

DSpark's rate depends on how predictable the text is, so the table is a range, not a promise. A short reply such as
the tool call spends a larger share of its time starting up.

### Total throughput

Tokens per second across all streams, `tools/bench_concurrent.py --levels 1,2,4,8 --tokens 256 --reps 3`: request i
sends item i of the prompt list, greedy, 256 tokens; aggregate = the streams' tokens after their first over the span
from the first stream's first token to the last stream's last. Every row is at one setting, `CONTEXT=32768` and
`MAX_STREAMS=8`, so that eight streams fit; plain is a `DSPARK=0` server with the same settings. At the defaults
(`CONTEXT=131072`, four streams) DSpark gives 85.2 / 95.6 / 100.7 at 1 / 2 / 4 streams, and 77.5 / 96.5 / 75.3 with
128-token replies.

| Streams | Plain | DSpark |
| ---: | ---: | ---: |
| 1 | 30.7 | 86.0 |
| 2 | 45.9 | 93.8 |
| 4 | 62.6 | 102.6 |
| 8 | 79.0 | 81.0 |

DSpark's aggregate peaks at two to four streams and falls back toward plain at eight, where every step is already full
of work. `bench.sh sweep` at the defaults gave each stream the same token hash with DSpark and plain at 1, 2 and 4
streams.

### Time to first token

| Prompt | Cold | Cached | Method |
| ---: | ---: | ---: | --- |
| 2,048 tokens | 1.32 | 0.18 | cold: `tools/bench_cache.py --require-cold`, three prompts of exactly this length; cached: a document of about this length (2,027 tokens) asked once, then the same request again, 9 times |
| 8,192 tokens | 5.84 | 0.18 | same (cached: 8,019 tokens) |
| 32,768 tokens | 30.6 | 0.19 | same (cached: 31,923 tokens) |
| 126,000 tokens | 221 | not measured | cold as above; a 101,859-token prompt: 158.7 s cold, 0.1 s cached |
| A 519,983-token document | 2,754 | 12.1 | `CONTEXT=524288`, [below](#one-request-at-512k-tokens) |
| A short prompt | 0.11 | | `tools/bench_openai.py`, the release's server |

A repeated prompt resumes from its kept state: the kept pages are adopted, not copied, and only the state's small fixed
part comes back from the arena (0.18-0.19 s up to 32k tokens, 0.1 s at 101,859, 12.1 s at 519,983). Another question
on the same document resumes the document too: 0.21-0.23 s up to 32k. A 24,000-token prompt beside one decoding
stream (`--decode-share 0.5`): 82.2 s (31.7 s with nothing else running). A document that
extends an earlier request's document inside the same message is not resumed: the cache keeps states at message
boundaries and at the end of each prompt.

### Decode at depth

One stream, the server's decode rate of a 256-token reply after a long prompt, tokens per second. Plain is a
`DSPARK=0` server; default is the release (DSpark by depth). The first rows are one prompt set at four lengths (three
runs, the first dropped):

| Prompt tokens | Plain | Default (DSpark by depth) |
| ---: | ---: | ---: |
| 8,163 | 30.3 | 50.8 |
| 33,404 | 29.2 | 67.0 |
| 67,804 | 27.6 | 61.8 |
| 129,173 | 25.2 | 51.5 |
| 1,048,000 (synthetic state, decode tool) | 10.8 | plain (`CONTEXT=1048576`) |

A book as the system message, cut to length (prose: the reply continues the story; code: a program asked after it),
the default only, 3 rounds of 3 replies a kind (the first round dropped), each request resumed from the book's kept
state:

| Prompt tokens | Prose | Code |
| ---: | ---: | ---: |
| about 126,250 | 24.8 | 33.1 |
| about 520,000 (`CONTEXT=524288`) | 16.8 | 21.7 |

Each kind's 9 replies at each length had one token hash.

### Install and load

| Measure | Result |
| --- | ---: |
| Host install, from an empty folder to a served smoke (`setup.sh`, wall time, weights verified in place with `--weights-dir`) | 341 s |
| Container install on top (image build and copy to the other three hosts, start, smoke) | 301 s |
| Host install with `CHECKPOINT=host:/path` (the full checkpoint on a fifth host on the switch, every file hashed, each rank's bytes copied over the fabric, verified) | 508 s |
| The copy itself, a rank, from the checkpoint (about 92 GiB) | 75-91 s |
| Start to ready, default config (weights in the page cache) | 23 s |
| Smoke and oracle checks at the end of `setup.sh`, every path | PASS (host, container, and the full drop-in install) |
| W=4 decode check on each install (`setup.sh CONFIG stop check`) | PASS, 80/80 bit-equal, `f35e48068f72e8a9` |
| Weights download from Hugging Face, a host | not measured on this release |

The host install is the weights check (every byte re-read and hashed on all four hosts) plus preflight, assets and
libs, folds (each rank's DSpark folds file built from nothing, byte for byte equal to the one already on the host),
manifest and config, and the start. With `CHECKPOINT`, the source's hashes are cached on its host, so a second install
from it skips that step. Not exercised: the base-image pull of the container build (the hosts had the layers cached)
and `--fabric-install`.

### Memory while serving

| Measure | Result |
| --- | ---: |
| MemAvailable at ready, default config, rank 0 / ranks 1-3 | 9.96 / 9.86-9.98 |
| Lowest MemAvailable over a benchmark run, rank 0 / ranks 1-3 | 8.85 / 8.60-8.75 |

The page pool takes the room after the 8 GiB reserve, so MemAvailable while serving sits a little above the reserve
on every rank; the ranks end within a few tenths of a GiB of each other since the embedding moved off rank 0.

## Context, streams and the pool

Start-up counts on the four GB10s, the release's recipe with only `CONTEXT`, `DSPARK` and `MAX_STREAMS` set (raw
lines: rank 0's log, `inkling2: context C -> N streams on a KV page pool (...)`). Pages are 256 positions, 2.25 MiB a
rank with DSpark and 2.0 MiB plain. "Kept" is how many 128k prompts the pool holds as kept prompts with every stream
idle.

| `CONTEXT` | DSpark | Streams (`MAX_STREAMS=4`) | Pool positions | 128k prompts kept | Room after load on the rank with the least, GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32,768 | on | 4 | 952,320 | requests up to 32k | 12.51 |
| 131,072 (default) | on | 4 | 950,784 | 7 | 12.50 |
| 131,072 | off | 4 | 1,231,616 | 9 | 13.63 |
| 262,144 | on | 3 | 965,888 | 7 | 12.48 |
| 524,288 | on | 1 | 1,197,824 | 9 | 12.43 |
| 524,288 | off | 2 | 1,246,720 | 9 | 13.51 |
| 1,048,576 | off (the recipe) | 1 | 1,492,480 | 11 | 13.51 |

With `MAX_STREAMS=8`: 7 / 8 streams at 131,072
(DSpark / plain) and 8 at 32,768. A count at the margin
depends on the memory free when the server starts; the pool's size follows the same reading, so it moves by some
hundreds of pages between starts of the same configuration.

**1M, one plain stream** (`CONTEXT=1048576`): ready in 24 s. Decode at
1,048,000 tokens of context:
92.4 ms a step (10.8 tok/s).

How 1M is checked: a real 1M-token prompt would take hours to fill (the global layers' attention grows with the square
of the length), so `tf-inkling2-decode --exp deep --fill synth` writes every layer's keys and values for positions 0
to 1,047,998, the sliding layers' rings, the convolution states and the drafter's caches with fixed pseudo-random
values on every rank (no forward), runs 64 decode steps from there and compares their logits with one 64-row prompt
forward over the same positions, bit for bit; with the drafter it also decodes the same tokens in drafted rounds. Then
every rank reads its global layers' filled positions back and the ranks' verdicts are gathered: one changed row on any
rank fails the check. This covers what changes with the length (attention indexing and grids past 2^20 positions, the
split counts, the op scratch, the log scaling of the queries), not the model's output on a real 1M-token text; that
is checked end to end at about 520,000 tokens (below).

```
./check.sh my-sparks.conf data/long_4000.safetensors 1048000 64 -- --exp deep-draft --fill synth --context 1048592
```

## One request at 512k tokens

`CONTEXT=524288` in the config serves requests up to 524,288 tokens (prompt plus reply) with the release's other
settings; DSpark drafts by depth. The same four ranks and weights serve it; restart with the other config to switch.

```
echo CONTEXT=524288 >> my-sparks.conf
./stop.sh my-sparks.conf && ./start.sh my-sparks.conf
```

| Measure | Result |
| --- | ---: |
| A 519,983-token document (book text) with one fact inserted at 37 %, a cold prefill: time to first token (the request's 2,761.5 s less its 127 later tokens at its 16.96 tok/s) | 2,754 s |
| Its greedy reply states the fact | yes, in its reasoning (sent as the system message, the book makes the model reason first; the fact is in the reply's first 128 tokens of reasoning) |
| Decode after it, default (DSpark by depth) | 16.8 tok/s |
| The same prompt again (resumed from its kept state) | 12.1 s, the same reply |

What is checked at this length: the decode check at 524,000 positions (W=2: one prefill, then 64 decode steps whose
logits equal one prompt forward over the same positions bit for bit); drafted replies equal plain ones at about
520,000 tokens on the four GB10s; the server's greedy replies equal the oracle's; a request of exactly 524,288 tokens
is admitted with its pages and one token more is refused by name. A prompt this long takes most of an hour to fill,
and the server streams nothing before the first token, so clients need a read timeout of an hour or more.

## Measuring your install

`bench.sh CONFIG COMMAND` runs `tools/inkling2_bench.py` (Python 3.9 or later, standard library only) on rank 0's
host against the running server and copies its JSON back to `./bench-results/`:

```
./bench.sh my-sparks.conf check [--sums]     # GPU-free; the server may run or not
./bench.sh my-sparks.conf sweep              # ~6 minutes at the defaults
./bench.sh my-sparks.conf prompts            # 16 rows x temperature 0 and 0.7, 2,048 tokens each (~25 minutes)
./bench.sh my-sparks.conf needle             # 8 sessions at once: MAX_STREAMS=8 and CONTEXT=32768
./bench.sh my-sparks.conf needle --sessions 1 --tokens 120000   # one deep needle at the defaults
./bench.sh my-sparks.conf ledger bench-results/sweep-*.json [--prompts bench-results/prompts-*.json]
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

On this release:

| Measure | Result |
| --- | ---: |
| Needle, 8 sessions at once, ~28k tokens each, depths across the text (`CONTEXT=32768`, `MAX_STREAMS=8`) | 8 of 8 exact, 207 s |
| Prompt set, temperature 0, chat / code / agentic rows | 41.0 / 70.2 / 68.8 (all 16 rows: 54.1) |
| Prompt set, temperature 0.7, chat / code / agentic rows | 35.8 / 58.4 / 50.4 (all 16 rows: 47.2) |
| Ledger: one stream plain, share of the floor | 68.5 % |
| Ledger: 4 streams DSpark aggregate, share of the floor | 75.1 % |
| Sweep at the defaults: errors, greedy repeatable, DSpark equal to plain | 0, yes, yes |

## Other benchmark commands (step 7)

The release's earlier measurements came from these tools (Python 3.10 or later, standard library only), which still
work against the server:

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
