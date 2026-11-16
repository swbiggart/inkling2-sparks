# Inkling-2-Small at TP4: correctness gates

Paths in this file are relative to `recipe/inkling2-tp4/` (or the unpacked release folder, for `tools/`, `data/`, `plan/`), unless a link says otherwise.

Every gate of the release, with its hash, method and build. The summary is in the [README](../../../README.md#correctness).

## What the gates check

- **Decode equals the whole-prompt forward.** Each decode step's logits equal the same row of the whole-prompt
  forward at the same world size (W=4), bit for bit, across all 48 layers, with the KV cache on the page pool in
  whatever pages a stream gets.
- **Greedy equals the oracle on the recorded prompts.** On the two recorded prompts (59 tokens), greedy replies equal
  the checkpoint-mode oracle's (the checkpoint run through Transformers' Inkling-2 modules in PyTorch, with the experts
  quantized as the checkpoint specifies) token for token.
- **The ranks agree.** Every rank computes the same logits and the same token; ranks with different settings, builds
  or weight digests refuse to start.
- **Drafting never changes output.** Drafted replies equal plain ones token for token, at every depth and with the
  depth rule drafting or not.
- **Nothing else changes output:** other streams (concurrent equals alone), the prompt cache (a resumed prompt equals
  a cold one, also when several streams share one kept prompt and when kept prompts are given back), the pages a
  stream gets (any order; poisoned new pages), and structured output's masks with DSpark on or off.

What this does and does not say about W=4 against one rank: the experts (the split MoE) and the embedding (bit
copies of the table's rows) are bit-equal to a one-rank run. The attention partials of the four ranks are summed in a
fixed rank order, the same on every rank and every run, and can differ from a one-rank run in the last bits. The
release does not claim that the whole model at W=4 equals W=1.

What it does and does not say against the oracle: the kernels round where the oracle does but sum in their own fixed
order, so a reply can depart from the oracle's at a near-tie (the structured-output cases at the end record
such departures). The oracle gate shows equality on its recorded prompts, not on every prompt.

## Gates on the release build

Hashes are xxh64 of the decode logits. Every gate below ran on the release build (the four
GB10s unless a row says W=2, which is one host with two GPUs and 8 or 24 of the 48 layers); `gates.sh require` on it:
54 of 54 release gates passed.

| Gate | Result |
| --- | --- |
| Decode check, W=4, 3,040-token prompt, 80 steps, graphs and eager | bit-equal 80/80, `f35e48068f72e8a9` both; 33.67 / 36.90 ms a step |
| The same with the pool's pages shuffled, with new pages poisoned (`TF_KV_POISON`), and with every new sequence on another run of pages | bit-equal 80/80, `f35e48068f72e8a9` in each, drafted rounds included |
| Drafted equals plain, real and mixed drafts | 80/80 each, `f35e48068f72e8a9` |
| 4 and 8 streams stepped together equal their solo runs | PASS at 4 and 8 |
| 130,944-token check, and drafted equals plain at 8k / 32k / 64k / 131k | bit-equal 64/64, `1556b802e179e958`; 16/16 at each depth |
| Synthetic 1,048,000-token state, W=4, with the drafter (deep-draft) and without (deep) | bit-equal 64/64, `39deddf85440d71e` in both; drafted equals plain; the fill read back intact on all four ranks |
| Prompt-cache resumes in the decode tool (resume, inside a page, drafted, 4 streams) on kept pages | PASS; proposals `ca56c24ce01aafcb`; the pool idle at the end with every page free or kept |
| Server greedy against the oracle: `chat_short`, `tool_call` | 11 of 11, 48 of 48 |
| Concurrent equals alone, 1 / 2 / 4 streams, greedy and seeded T=1 | PASS, 28/28 equal |
| Prompt cache: resumed equals cold (54 requests against a cache-off reference), and a ~100k-token needle cold then repeated | PASS, 54/54 equal, 25 resumed; the same reply both times |
| Three streams resuming one kept prompt together; six documents in three orders under a capped pool (kept prompts given back) | PASS, 3/3 resumed together equal alone, every reply equal in every order |
| Admission: requests together equal alone while some wait for pages; a request over the pool refused by name | PASS, 9/9 together equal alone, 2 requests waited |
| `CONTEXT=524288` (the default recipe otherwise): oracle; a 524,288-token request admitted with its pages, one token more refused (HTTP 400); a short request beside it when the pool has room | PASS |
| Stream counts at `CONTEXT` 32k to 1M, DSpark and plain | PASS, the counts in the table above |
| `/health` `kv`: idle, one stream, five requests on four streams (`streams_full`, one waiting), idle again | PASS |
| Structured output: constrained greedy replies against the oracle under the same masks, JSON object and required tool call | PASS |
| Default config: 4 streams of 131,072 admitted; MemAvailable over a benchmark run stays above the reserve (8.25 GiB floor, every rank) | PASS, 10.52 / 9.44-10.37 |
| Speed rows against the README | PASS, 9 of 9 rows within tolerance |
| Smoke: chat, reasoning effort high, a tool-call round trip, a 10k-token prompt, stream equals whole | PASS |
| Start-up refusals: a follower-only `TF_ROCE` (mesh on, off), one rank without the mesh wire, one rank missing the kernel image, `TF_INK2_MAX_STREAMS=9` | 5 of 5 refused, every rank stopped, the log naming the rank and the setting |
| Fresh installs from the release tarball: host, container, and with `CHECKPOINT` (a copy of one layer from a full checkpoint, every run equal to the download's; a wrong file refused, nothing copied) | PASS, PASS, PASS (104 runs equal to the download's on 4 ranks); the decode check `f35e48068f72e8a9` on each |
| W=2, 8 layers: decode check, eager, prompt chunks, drafted (real, deferred, mixed), 4 streams, pages shuffled / poisoned / moved | `34cdc28b0ce0ba99` throughout |
| W=2: prompt cache on kept pages, every cut (inside a page, on a page boundary), with poisoned pages; negative control | `f690e2f35594d965`, proposals `0d5b07d25b925c22`; the control fails as it must |
| W=2, 24 layers; 8 layers after a 1,000-token prompt (check, eager, scratch) | `710e6f88ec3a2cce`; `ab0a7ec5f66ade58` |
| W=2: deep at 524,000 positions; synthetic 131k / 524k / 1,048,000 with and without the drafter | 524,000 `a633e50751e18d35`; synthetic 1,048,000 `16eeccc6b5a59371` |
| Digest integrity: every byte loaded matches the manifest | every server start reports "contents checked" |

Exactness across modes on the four GB10s, from the depth rule's measurements (one token hash for every request body
across the rule, always drafting, plain within the DSpark server and a `DSPARK=0` server, at about 3k, 32k, 127k and
520k tokens of prompt, prose and code; concurrent equals alone with a 127k request beside three short ones):
237 replies of 27 request bodies over three runs, each body one token hash. In the release's own measurements the ~520k
document gave the same token hash cold and resumed, and each depth-rule kind's 9 replies one hash.

## Running the checks yourself (step 6)

**Decode check** (server stopped): each decode step must equal the whole-prompt forward at the same W bit for bit. With
`setup.sh`: `./setup.sh my-sparks.conf stop check start` (add `--container` for the container path). By hand:

```
./stop.sh my-sparks.conf
./check.sh my-sparks.conf ../../data/long_4000.safetensors 3040 80    # prints PASS: ... bit-equal 80/80 ...
```

**Greedy against the oracle**, **smoke** and the benchmarks run where the endpoint is reachable (rank 0's host when
HTTP is on loopback), from the unpacked release folder:

```
python3 recipe/inkling2-tp4/smoke.py http://127.0.0.1:8890 oracle data/decode.json \
  chat_short=data/chat_short.ids.json tool_call=data/tool_call.ids.json       # token for token
python3 recipe/inkling2-tp4/smoke.py http://127.0.0.1:8890 smoke --long-tokens 8000
./bench.sh my-sparks.conf sweep        # DSpark equals plain on every greedy row (exact_vs_other_mode)
```

The oracle files come from `tools/zig/inkling2_ref.py` (in the engine source: upstream TensorFold at `ENGINE_COMMIT` with `patches/` applied, as `scripts/build-engine.sh` makes it): the checkpoint run through Transformers' own Inkling-2
modules in PyTorch, with the experts quantized as the checkpoint specifies (TokenSpeed's reference).

**Structured output:** four recorded cases besides the gate's (a schema, a long string, thinking then a schema, a named
tool) depart from the oracle where its own unconstrained reply does, at a near-tie in free text. Every constrained,
plain and sampled reply in them is the same with DSpark's shared pass, with `TF_INK2_LATE_MASKS=0` and with
`DSPARK=0`, and in a group the same as alone; they are not claimed equal to the oracle.
