# Changelog

The recipe's own changelog is [recipe/inkling2-tp4/CHANGELOG.md](recipe/inkling2-tp4/CHANGELOG.md); this one adds the
engine commit and the release asset.

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
