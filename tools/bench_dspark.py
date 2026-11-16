# bench_dspark.py BASE MODEL [--tokens 256] [--reps 3] [--label L] [--only a,b] [--natural-cap 1024]
# The README's per-workload DSpark rows (recipe/inkling2-tp4/README.md). Five workloads through bench_openai.stream
# (this folder's tools: one warm-up, then REPS streamed requests at temperature 0, seeds 1234..): bench_openai's
# fibonacci-raw and gpu-chat-no-think prompts, a code prompt (code-lru-chat below), a tool call (the recipe smoke
# test's get_weather request with its tools) and thinking with reasoning_effort high (the smoke test's bat-and-ball
# question). Example: python3 tools/bench_dspark.py http://127.0.0.1:8890/v1 model-split --reps 5
#
# Method:
# - The tool call and thinking run to their NATURAL END (ignore_eos off, max_tokens = --natural-cap): the reply the
#   user gets, not the model writing on after its own end. The prose/code/completion workloads keep bench_openai's
#   fixed --tokens with ignore_eos (their replies run past 256 tokens anyway, so nothing after an EOS is counted
#   unless the reply ended early: "finish" and "tokens" show it).
# - Every workload reports two rates a request, medians over the reps:
#     server_tps : the server's own decode rate (the final chunk's tensorfold.tokens_per_second = (tokens - 1) /
#                  seconds from prefill done to the last token)
#     client_tps : the client's wall clock, (tokens - 1) / first-to-last output event (content, reasoning or tool-call
#                  deltas; bench_openai's text-piece span leaves tool deltas out and ends at the last text piece)
#   pieces_tps (bench_openai's own decode_tps) is kept only for comparison with earlier tables; do not quote it.
# - Also the server's draft counts ("speculative": rounds, drafted, accepted, summed over the measured requests), the
#   tokens a round, each request's token count, finish reason and token sha. One JSON line a workload.
import argparse, json, os, statistics, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_openai, bench_stream, bench_thinking

WEATHER = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}]
EXTRA = [
    {"name": "tool-call-weather", "kind": "chat", "prompt": "What's the weather in Paris right now? Use celsius.",
     "extra": {"tools": WEATHER}, "natural": True},
    {"name": "thinking-high", "kind": "chat",
     "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball "
               "cost? Think it through.", "extra": {"reasoning_effort": "high"}, "natural": True},
]
CODE = {"name": "code-lru-chat", "kind": "chat",
        "prompt": "Write a Python class LRUCache with get and put in O(1) time, with type hints, docstrings and a few "
                  "unit tests using unittest."}

current = {}
_Request = urllib.request.Request
def request(url, data=None, headers={}, **kw):
    if data is not None and current.get("body"):
        body = json.loads(data); body.update(current["body"]); data = json.dumps(body).encode()
    return _Request(url, data=data, headers=headers, **kw)
bench_openai.urllib.request.Request = request

_receipt = bench_stream.Collector.receipt
def receipt(self):
    rt = self.runtime or {}
    return {**_receipt(self), "speculative": self.speculative, "runtime_sha": rt.get("token_sha"),
            "server_tps": rt.get("tokens_per_second"), "server_s": rt.get("seconds")}
bench_stream.Collector.receipt = receipt

def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None

p = argparse.ArgumentParser()
p.add_argument("base"); p.add_argument("model")
p.add_argument("--tokens", type=int, default=256); p.add_argument("--reps", type=int, default=3)
p.add_argument("--natural-cap", type=int, default=1024)
p.add_argument("--label", default=""); p.add_argument("--only", default="")
a = p.parse_args()
items = bench_thinking.configure(bench_openai.PROMPTS + [CODE], bench_thinking.discover(a.base, a.model, None))
items += [{**x, "thinking_applicable": False} for x in EXTRA]   # thinking left at the server default (on)
for item in items:
    if a.only and item["name"] not in a.only.split(","):
        continue
    natural = bool(item.get("natural"))
    current.clear(); current.update(item)
    current["body"] = {**item.get("extra", {}), **({"ignore_eos": False} if natural else {})}
    n_tok = a.natural_cap if natural else a.tokens
    seeds = [1234 + i for i in range(a.reps)]
    bench_openai.stream(a.base, a.model, item, n_tok, 0.0, seeds[0])  # warm-up
    runs = [bench_openai.stream(a.base, a.model, item, n_tok, 0.0, s) for s in seeds]
    spec = {k: sum((r.get("speculative") or {}).get(k, 0) or 0 for r in runs) for k in ("rounds", "drafted", "accepted")}
    toks = sum(r.get("tokens") or 0 for r in runs)
    client = [(r["tokens"] - 1) / (r["last_output_s"] - r["first_output_s"]) for r in runs
              if r.get("tokens") and r.get("last_output_s") and r.get("first_output_s") is not None and r["last_output_s"] > r["first_output_s"]]
    print(json.dumps({"label": a.label, "prompt": item["name"], "mode": "natural end" if natural else f"{a.tokens} ignore_eos",
                      "server_tps_median": med([r.get("server_tps") for r in runs]),
                      "server_tps_all": [round(r["server_tps"], 2) if r.get("server_tps") else None for r in runs],
                      "client_tps_median": med(client), "client_tps_all": [round(x, 2) for x in client],
                      "pieces_tps_median": med([r["decode_tps"] for r in runs]),
                      "ttft_s_median": med([r["ttft_s"] for r in runs]),
                      "tokens": [r.get("tokens") for r in runs], "finish": [r.get("finish_reasons") for r in runs],
                      "tool_calls": [r.get("tool_calls") for r in runs], "speculative": spec,
                      "tokens_per_round": round(toks / spec["rounds"], 3) if spec["rounds"] else None,
                      "token_sha": [r.get("runtime_sha") for r in runs], "first_speculative": runs[0].get("speculative")}), flush=True)
