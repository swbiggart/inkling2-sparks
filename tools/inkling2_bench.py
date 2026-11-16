#!/usr/bin/env python3
"""inkling2_bench.py: speed sweep, realistic prompt set, needle tests and speed ledger for Inkling-2-Small at TP4.

Standard library only (Python 3.9 or later), so it runs on a bare host next to the server. The process ideas come
from 0xSero's inkling2-exl3 (MIT): a one-command sweep with JSON out, a fixed realistic prompt set measured on the
server's own clock, a ledger against a bandwidth floor, and needle tests with concurrent sessions. No code is copied.
recipe/inkling2-tp4/bench.sh runs it on rank 0's host with the config's port and MAX_STREAMS.

  sweep   BASE [MODEL] [--levels 1,2,4,8] [--max-streams 4] [--modes dspark,plain] [--reps 3] [--tokens 256]
          [--workloads all|none|a,b] [--prefill 2048,8192] [--out FILE]
      Decode at each concurrency level within --max-streams (levels above it are listed as skipped): request i sends
      item i of bench_prompts.json (greedy, ignore_eos, --tokens), all at once; aggregate tok/s = the streams' tokens
      after their first / (last token of any stream - first token of any stream), the README's "Total throughput"
      method. Per stream also the server's own decode rate (tensorfold.tokens_per_second: prefill excluded) and
      DSpark's counts (tokens a round = acceptance length, accepted / drafted). Then one stream on the README's five
      workloads (bench_dspark.py's prompts; tool call and thinking to their natural end), then cold prompts of about
      the --prefill sizes (a unique first line keeps the prompt cache out): time to first token and prompt tokens /
      the server's prefill seconds. "plain" sends "draft": false on the same server. Every figure is the median of
      --reps; greedy replies must be equal across reps and across modes (same bits), "exact" says whether they were.
      A reply counts only when it completed ([DONE] seen, no error) and carried its token sha; a stream that ends
      early is an error, and a missing sha is never equal to another.
  prompts BASE [MODEL] [--set FILE] [--temperatures 0,0.7] [--seed 1234] [--max-tokens 2048] [--natural]
          [--concurrency 1] [--modes dspark] [--only a,b] [--out FILE]
      The fixed prompt set (inkling2_prompts.json: 16 authored chat, code and agentic/tool rows). Each row generates
      --max-tokens tokens (ignore_eos; --natural stops at the model's end instead) and reports the server's decode
      rate with prefill excluded, DSpark's tokens a round and the token sha; a sampled row is seeded (seed + row).
  needle  BASE [MODEL] [--sessions 8] [--tokens 28000] [--depths 0.1,...] [--concurrent] [--out FILE]
      Long-context retrieval: each session gets its own synthetic haystack (generated text, no copyrighted
      material) with one passphrase at its depth, all sessions sent at once (--concurrent, default when sessions > 1).
      Exact retrieval: the reply must state the session's own passphrase and no other session's. Exit 1 otherwise.
  ledger  SWEEP.json [--sheet FILE] [--bw-gbs 250] [--gathers 145] [--gather-us 12.5] [--prompts PROMPTS.json]
      Per row: the bytes a rank reads a step (weights, the experts R rows route to, K/V), the floor = bytes /
      streaming bandwidth + gathers x latency, the floor's tok/s and the row's % of it. Prints a table and JSON.
  sheet   CHECKPOINT_DIR [--out FILE]
      The ledger's model sheet: tensor bytes by role from the checkpoint's safetensors headers and config.json.

BASE is the server, e.g. http://127.0.0.1:8890 (a trailing /v1 is accepted). MODEL defaults to the first id of
/v1/models. --api-key-env NAME sends the key in that environment variable.
"""
import argparse
import hashlib
import json
import os
import random
import statistics
import struct
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
API_KEY = None

# ------------------------------------------------------------------------------------------------ HTTP and streams


def base_url(base):
    base = base.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def request(base, path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    if API_KEY:
        req.add_header("Authorization", "Bearer " + API_KEY)
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(base, path):
    try:
        with request(base, path, timeout=20) as r:
            return json.loads(r.read().decode() or "null")
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"error": type(e).__name__ + ": " + str(e)[:200]}


def model_id(base, model):
    if model and model != "auto":
        return model
    m = get_json(base, "/v1/models")
    try:
        return m["data"][0]["id"]
    except (KeyError, IndexError, TypeError):
        sys.exit(f"no model id from {base}/v1/models: {m}")


def stream(base, body, timeout=3600, gate=None):
    """POST a streamed request; returns the reply's figures (client clock: time.perf_counter seconds)."""
    path = "/v1/chat/completions" if "messages" in body else "/v1/completions"
    body = dict(body, stream=True, stream_options={"include_usage": True})
    out = {"error": None, "first": None, "last": None, "usage": {}, "runtime": {}, "speculative": {}, "finish": None,
           "content": [], "reasoning": [], "tools": {}}
    try:
        if gate is not None:
            gate.wait(timeout=60)
        out["sent"] = time.perf_counter()
        with request(base, path, body, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                x = line[5:].strip()
                if x == "[DONE]":
                    out["done"] = True
                    break
                ev = json.loads(x)
                if "error" in ev:
                    out["error"] = str(ev["error"])[:300]
                    break
                out["usage"] = ev.get("usage") or out["usage"]
                out["runtime"] = ev.get("tensorfold") or out["runtime"]
                out["speculative"] = ev.get("speculative") or out["speculative"]
                for ch in ev.get("choices") or []:
                    out["finish"] = ch.get("finish_reason") or out["finish"]
                    d = ch.get("delta") or {}
                    text = ch.get("text") or d.get("content") or ""
                    reas = d.get("reasoning_content") or d.get("reasoning") or ""
                    calls = d.get("tool_calls") or []
                    if text:
                        out["content"].append(text)
                    if reas:
                        out["reasoning"].append(reas)
                    for c in calls:
                        t = out["tools"].setdefault(c.get("index", 0), {"name": "", "arguments": ""})
                        f = c.get("function") or {}
                        t["name"] += f.get("name") or ""
                        t["arguments"] += f.get("arguments") or ""
                    if text or reas or calls:
                        now = time.perf_counter()
                        out["first"] = out["first"] if out["first"] is not None else now
                        out["last"] = now
        if not out.get("done") and not out["error"]:
            out["error"] = "incomplete response: the stream ended before [DONE]"
    except Exception as e:  # noqa: BLE001 - a failed request is reported in its row
        if gate is not None:
            gate.abort()
        out["error"] = out["error"] or f"{type(e).__name__}: {str(e)[:200]}"
    return summarize(out)


def summarize(o):
    u, rt, sp = o["usage"] or {}, o["runtime"] or {}, o["speculative"] or {}
    n = u.get("completion_tokens")
    ok = o.get("done") and not o["error"]
    if not ok and not o["error"]:
        o["error"] = "incomplete response"
    span = (o["last"] - o["first"]) if o["first"] is not None and o["last"] is not None else None
    details = u.get("prompt_tokens_details") or {}
    pre = rt.get("prefill_seconds")
    p = u.get("prompt_tokens")
    return {
        "ok": bool(ok), "error": o["error"], "tokens": n, "prompt_tokens": p, "cached_tokens": details.get("cached_tokens"),
        "finish": o["finish"], "server_tps": rt.get("tokens_per_second"), "server_decode_s": rt.get("seconds"),
        "prefill_s": pre, "prefill_tps": (p / pre) if p and pre else None,
        "ttft_s": (o["first"] - o["sent"]) if o["first"] is not None and "sent" in o else None,
        "client_tps": (n - 1) / span if ok and n and n > 1 and span else None,
        "rounds": sp.get("rounds"), "drafted": sp.get("drafted"), "accepted": sp.get("accepted"),
        "token_sha": rt.get("token_sha256") or rt.get("token_sha"), "sampling": rt.get("sampling"),
        "text": "".join(o["content"]), "reasoning_chars": sum(len(x) for x in o["reasoning"]),
        "tool_calls": [o["tools"][k] for k in sorted(o["tools"])] or None,
        "first": o["first"], "last": o["last"],
    }


def together(base, bodies, timeout=3600):
    """Send the bodies at once (a barrier lines up the starts); results in order."""
    gate = threading.Barrier(len(bodies)) if len(bodies) > 1 else None
    out = [None] * len(bodies)

    def run(i):
        out[i] = stream(base, bodies[i], timeout, gate)

    th = [threading.Thread(target=run, args=(i,)) for i in range(len(bodies))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return out


def same_digests(*groups):
    """True only when every digest is present and all are one value: a missing digest is never equal to anything."""
    xs = [x for g in groups for x in g]
    return bool(xs) and all(xs) and len(set(xs)) == 1


def med(xs, nd=2):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), nd) if xs else None


def rnd(x, nd=2):
    return None if x is None else round(x, nd)


def spec_sum(runs):
    s = {k: sum((r.get(k) or 0) for r in runs) for k in ("rounds", "drafted", "accepted")}
    toks = sum(r.get("tokens") or 0 for r in runs)
    s["tokens_per_round"] = round(toks / s["rounds"], 3) if s["rounds"] else None
    s["accept_rate"] = round(s["accepted"] / s["drafted"], 3) if s["drafted"] else None
    s["drafts_per_round"] = round(s["drafted"] / s["rounds"], 3) if s["rounds"] else None
    return s


def meta(base, model, args, tool):
    return {"tool": f"inkling2_bench.py {tool}", "format": 1, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base": base, "model": model, "python": sys.version.split()[0],
            "args": {k: v for k, v in vars(args).items() if k not in ("func", "api_key_env")},
            "server": {"models": get_json(base, "/v1/models"), "health": get_json(base, "/health")}}


def write(doc, path):
    s = json.dumps(doc, indent=1)
    if path and path != "-":
        with open(path, "w", encoding="utf-8") as f:
            f.write(s + "\n")
        print(f"wrote {path}", file=sys.stderr)
    else:
        print(s)


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ prompts


def chat_body(model, prompt, tokens, temperature=0.0, seed=1234, thinking=False, extra=None):
    b = {"model": model, "max_tokens": tokens, "temperature": temperature, "seed": seed,
         "messages": [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt}
    if not thinking:
        b["chat_template_kwargs"] = {"enable_thinking": False}
    b.update(extra or {})
    return b


def item_body(model, item, tokens, mode, temperature=0.0, seed=1234, ignore_eos=True):
    """bench_prompts.json / bench_dspark.py items: chat (thinking off unless the item keeps it) or completion."""
    if item["kind"] == "chat":
        b = chat_body(model, item["prompt"], tokens, temperature, seed, item.get("thinking", False), item.get("extra"))
    else:
        b = {"model": model, "prompt": item["prompt"], "max_tokens": tokens, "temperature": temperature, "seed": seed}
    b["ignore_eos"] = ignore_eos
    if mode == "plain":
        b["draft"] = False
    return b


def load_json(name, path=None):
    p = path or os.path.join(HERE, name)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


WEATHER = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}]
# The README's one-stream rows (tools/bench_dspark.py's five workloads, same prompts and settings)
WORKLOADS = [
    {"name": "fibonacci-raw", "row": "Raw code completion", "kind": "completion",
     "prompt": "Write a short Python function that computes the Fibonacci sequence and explain it."},
    {"name": "gpu-chat-no-think", "row": "Prose", "kind": "chat",
     "prompt": "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example."},
    {"name": "code-lru-chat", "row": "Code", "kind": "chat",
     "prompt": "Write a Python class LRUCache with get and put in O(1) time, with type hints, docstrings and a few "
               "unit tests using unittest."},
    {"name": "tool-call-weather", "row": "Tool call", "kind": "chat", "thinking": True, "natural": True,
     "prompt": "What's the weather in Paris right now? Use celsius.", "extra": {"tools": WEATHER}},
    {"name": "thinking-high", "row": "Thinking", "kind": "chat", "thinking": True, "natural": True,
     "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball "
               "cost? Think it through.", "extra": {"reasoning_effort": "high"}},
]

# ------------------------------------------------------------------------------------------------ synthetic text

PLACES = ["north depot", "river yard", "east annex", "harbor office", "hill station", "south mill", "old granary",
          "west quay", "canal works", "upper store", "lower dock", "market hall", "glass house", "stone barn"]
GOODS = ["copper fittings", "pine boards", "wool bales", "glass jars", "iron hinges", "rope coils", "tin cans",
         "clay pipes", "linen sheets", "oak barrels", "brass valves", "paper reams", "salt sacks", "seed packets"]
PEOPLE = ["the clerk", "the foreman", "a courier", "the night guard", "the surveyor", "an apprentice", "the buyer",
          "the inspector", "a carter", "the bookkeeper", "the harbormaster", "a mason"]
VERBS = ["shipped", "counted", "repaired", "weighed", "stacked", "returned", "labelled", "inspected", "moved", "sorted"]
WEATHER_WORDS = ["clear", "windy", "overcast", "rainy", "foggy", "mild", "cold", "bright", "humid", "still"]
NOUNS = ["AMBER", "FALCON", "CEDAR", "ORBIT", "MAPLE", "QUARTZ", "HARBOR", "LANTERN", "COBALT", "MEADOW", "TIMBER",
         "VELVET", "SUMMIT", "PEBBLE", "RIVET", "SAFFRON", "GLACIER", "ANVIL", "HAZEL", "PRISM"]


def filler(rng, words):
    """Deterministic ledger-style sentences of about WORDS words (generated; no outside text)."""
    out, n, day = [], 0, rng.randint(1, 300)
    while n < words:
        day += rng.randint(0, 2)
        s = (f"Day {day}: {rng.choice(PEOPLE)} at the {rng.choice(PLACES)} {rng.choice(VERBS)} "
             f"{rng.randint(2, 480)} {rng.choice(GOODS)} for the {rng.choice(PLACES)}; the weather was "
             f"{rng.choice(WEATHER_WORDS)} and the tally closed at {rng.randint(100, 9999)}.")
        out.append(s)
        n += len(s.split())
    return out


def calibrate(base, model):
    """Prompt tokens a filler word, measured once with a 1-token request (cold: its own first line)."""
    rng = random.Random(7)
    words = 1500
    text = f"Calibration {os.getpid()}-{time.time_ns()}.\n" + " ".join(filler(rng, words))
    r = stream(base, {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0.0})
    if not r["ok"] or not r["prompt_tokens"]:
        sys.exit(f"calibration request failed: {r['error']}")
    return r["prompt_tokens"] / len(text.split())


# ------------------------------------------------------------------------------------------------ sweep


def sweep(a):
    base, model = a.base, a.model
    doc = meta(base, model, a, "sweep")
    modes = [m for m in a.modes.split(",") if m]
    levels = [int(x) for x in a.levels.split(",") if x]
    run_levels = [n for n in levels if n <= a.max_streams]
    doc["skipped"] = [f"c{n}: over MAX_STREAMS={a.max_streams}" for n in levels if n > a.max_streams]
    items = load_json("bench_prompts.json", a.decode_prompts)
    seeds = [a.seed + i for i in range(a.reps)]
    doc["decode"], doc["workloads"], doc["prefill"] = [], [], []
    shas = {}
    for n in run_levels:
        for mode in modes:
            bodies = [item_body(model, items[i % len(items)], a.tokens, mode, 0.0, a.seed) for i in range(n)]
            log(f"decode c{n} {mode}: warm-up + {a.reps} reps")
            together(base, bodies)
            reps = []
            for _ in seeds:
                runs = together(base, bodies)
                good = [r for r in runs if r["ok"] and r["first"] is not None and r["tokens"]]
                agg = None
                if good:  # a reply with no output events (no content, reasoning or tool deltas) cannot be timed
                    t0, t1 = min(r["first"] for r in good), max(r["last"] for r in good)
                    agg = sum(r["tokens"] - 1 for r in good) / (t1 - t0) if t1 > t0 else None
                reps.append({"aggregate_tps": agg, "timed": len(good), "runs": runs})
            allruns = [r for rep in reps for r in rep["runs"]]
            per_stream = [[rep["runs"][i]["token_sha"] for rep in reps] for i in range(n)]
            shas[(n, mode)] = [s[0] for s in per_stream]
            row = {"row": f"c{n} {mode}", "streams": n, "mode": mode, "tokens": a.tokens,
                   "aggregate_tps_median": med([r["aggregate_tps"] for r in reps]),
                   "aggregate_tps_all": [rnd(r["aggregate_tps"]) for r in reps],
                   "timed_streams": min(r["timed"] for r in reps),
                   "server_tps_median": med([r["server_tps"] for r in allruns]),
                   "client_tps_median": med([r["client_tps"] for r in allruns]),
                   "ttft_s_median": med([r["ttft_s"] for r in allruns], 3),
                   "prompt_tokens": [r["prompt_tokens"] for r in reps[0]["runs"]],
                   "errors": [r["error"] for r in allruns if not r["ok"]],
                   "repeatable": all(len(set(s)) == 1 and s[0] for s in per_stream),
                   "token_sha": [s[0] for s in per_stream], "speculative": spec_sum(allruns)}
            doc["decode"].append(row)
            log(f"  {row['row']}: aggregate {row['aggregate_tps_median']} tok/s {row['aggregate_tps_all']}, "
                f"server {row['server_tps_median']} a stream, tokens a round {row['speculative']['tokens_per_round']}")
    for n in run_levels:
        if (n, "dspark") in shas and (n, "plain") in shas:
            for r in doc["decode"]:
                if r["streams"] == n:
                    r["exact_vs_other_mode"] = same_digests(shas[(n, "dspark")], shas[(n, "plain")])
    wl = [] if a.workloads == "none" else [w for w in WORKLOADS if a.workloads == "all" or w["name"] in a.workloads.split(",")]
    wsha = {}
    for w in wl:
        for mode in modes:
            toks = a.natural_cap if w.get("natural") else a.tokens
            body = item_body(model, w, toks, mode, 0.0, a.seed, ignore_eos=not w.get("natural"))
            log(f"workload {w['name']} {mode}: warm-up + {a.reps} reps")
            stream(base, body)
            runs = [stream(base, dict(body, seed=s)) for s in seeds]
            wsha[(w["name"], mode)] = [r["token_sha"] for r in runs]
            row = {"row": f"one stream {mode} {w['name']}", "readme_row": w["row"], "streams": 1, "mode": mode,
                   "workload": w["name"], "end": "natural" if w.get("natural") else f"{toks} ignore_eos",
                   "server_tps_median": med([r["server_tps"] for r in runs]),
                   "server_tps_all": [rnd(r["server_tps"]) for r in runs],
                   "client_tps_median": med([r["client_tps"] for r in runs]),
                   "ttft_s_median": med([r["ttft_s"] for r in runs], 3),
                   "tokens": [r["tokens"] for r in runs], "prompt_tokens": runs[0]["prompt_tokens"],
                   "finish": [r["finish"] for r in runs], "errors": [r["error"] for r in runs if not r["ok"]],
                   "repeatable": same_digests(wsha[(w["name"], mode)]), "token_sha": runs[0]["token_sha"],
                   "speculative": spec_sum(runs)}
            doc["workloads"].append(row)
            log(f"  {row['row']}: server {row['server_tps_median']} tok/s {row['server_tps_all']}, "
                f"tokens a round {row['speculative']['tokens_per_round']}")
    for r in doc["workloads"]:
        o = wsha.get((r["workload"], "plain" if r["mode"] == "dspark" else "dspark"))
        if o is not None:
            r["exact_vs_other_mode"] = same_digests([o[0]], [r["token_sha"]])
    sizes = [int(x) for x in a.prefill.split(",") if x]
    if sizes:
        per_word = calibrate(base, model)
        doc["prefill_tokens_per_word"] = round(per_word, 4)
        for size in sizes:
            words = int(size / per_word)
            runs = []
            log(f"prefill ~{size} tokens cold: {a.reps} reps")
            for i in range(a.reps):
                rng = random.Random(size * 100 + i)
                text = f"Run {time.time_ns()}-{i}.\n" + " ".join(filler(rng, words)) + "\nSummarize the tally."
                runs.append(stream(base, {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0.0}))
            row = {"row": f"prefill {size} cold", "target_tokens": size,
                   "prompt_tokens": [r["prompt_tokens"] for r in runs],
                   "cached_tokens": [r["cached_tokens"] for r in runs],
                   "ttft_s_median": med([r["ttft_s"] for r in runs], 3), "ttft_s_all": [rnd(r["ttft_s"], 3) for r in runs],
                   "prefill_s_median": med([r["prefill_s"] for r in runs], 3),
                   "prefill_tps_median": med([r["prefill_tps"] for r in runs], 1),
                   "client_prefill_tps_median": med([(r["prompt_tokens"] / r["ttft_s"]) if r["prompt_tokens"] and r["ttft_s"] else None for r in runs], 1),
                   "errors": [r["error"] for r in runs if not r["ok"]]}
            doc["prefill"].append(row)
            log(f"  {row['row']}: {row['prompt_tokens']} tokens, TTFT {row['ttft_s_median']} s, "
                f"{row['prefill_tps_median']} tok/s (server prefill)")
    errs = [e for sec in ("decode", "workloads", "prefill") for r in doc[sec] for e in r.get("errors", [])]
    rep = all(r.get("repeatable", True) for sec in ("decode", "workloads") for r in doc[sec])
    cmp = [r["exact_vs_other_mode"] for sec in ("decode", "workloads") for r in doc[sec] if "exact_vs_other_mode" in r]
    exact = all(cmp) if cmp else None  # None: one mode only, nothing compared (not "equal")
    doc["verdict"] = {"errors": len(errs), "greedy_repeatable": rep, "dspark_equals_plain": exact}
    write(doc, a.out)
    table(doc)
    return 0 if not errs and rep and exact is not False else 1


def table(doc):
    print("row\ttok/s\tserver tok/s a stream\ttokens a round\texact", file=sys.stderr)
    for r in doc["decode"]:
        print(f"{r['row']}\t{r['aggregate_tps_median']}\t{r['server_tps_median']}\t{r['speculative']['tokens_per_round']}\t"
              f"{r.get('exact_vs_other_mode', '-')}", file=sys.stderr)
    for r in doc["workloads"]:
        print(f"{r['row']}\t{r['server_tps_median']}\t{r['server_tps_median']}\t{r['speculative']['tokens_per_round']}\t"
              f"{r.get('exact_vs_other_mode', '-')}", file=sys.stderr)
    for r in doc["prefill"]:
        print(f"{r['row']}\t{r['prefill_tps_median']} (TTFT {r['ttft_s_median']} s)\t-\t-\t-", file=sys.stderr)
    print(f"verdict: {doc['verdict']}", file=sys.stderr)


# ------------------------------------------------------------------------------------------------ prompt set


def prompt_rows(a):
    base, model = a.base, a.model
    rows = load_json("inkling2_prompts.json", a.set)["prompts"]
    if a.only:
        rows = [r for r in rows if r["name"] in a.only.split(",")]
    doc = meta(base, model, a, "prompts")
    doc["rows"] = []
    temps = [float(t) for t in a.temperatures.split(",") if t]
    modes = [m for m in a.modes.split(",") if m]
    jobs = []
    for t in temps:
        for mode in modes:
            for i, r in enumerate(rows):
                b = {"model": model, "max_tokens": a.max_tokens, "temperature": t, "seed": a.seed + i,
                     "messages": r["messages"], "ignore_eos": not a.natural}
                if r.get("tools"):
                    b["tools"] = r["tools"]
                if not r.get("thinking", False):
                    b["chat_template_kwargs"] = {"enable_thinking": False}
                if mode == "plain":
                    b["draft"] = False
                jobs.append((r, t, mode, b))
    log(f"prompt set: {len(rows)} rows x {len(temps)} temperatures x {len(modes)} modes, {a.concurrency} at once")
    stream(base, dict(jobs[0][3], max_tokens=16))  # warm-up
    for k in range(0, len(jobs), a.concurrency):
        batch = jobs[k:k + a.concurrency]
        res = together(base, [j[3] for j in batch])
        for (r, t, mode, _), x in zip(batch, res):
            row = {"row": r["name"], "category": r["category"], "temperature": t, "mode": mode, "seed": a.seed + rows.index(r),
                   "prompt_tokens": x["prompt_tokens"], "tokens": x["tokens"], "finish": x["finish"],
                   "server_tps": rnd(x["server_tps"]), "client_tps": rnd(x["client_tps"]), "ttft_s": rnd(x["ttft_s"], 3),
                   "tokens_per_round": rnd(x["tokens"] / x["rounds"], 3) if x["rounds"] else None,
                   "accept_rate": rnd(x["accepted"] / x["drafted"], 3) if x["drafted"] else None,
                   "rounds": x["rounds"], "drafted": x["drafted"], "accepted": x["accepted"],
                   "tool_calls": len(x["tool_calls"] or []), "token_sha": x["token_sha"], "ok": x["ok"], "error": x["error"]}
            doc["rows"].append(row)
            log(f"  {row['row']} t={t} {mode}: {row['tokens']} tokens, server {row['server_tps']} tok/s, "
                f"tokens a round {row['tokens_per_round']}{' ERROR ' + str(x['error']) if x['error'] else ''}")
    summ = []
    for t in temps:
        for mode in modes:
            rs = [r for r in doc["rows"] if r["temperature"] == t and r["mode"] == mode]
            by = {}
            for r in rs:
                by.setdefault(r["category"], []).append(r["server_tps"])
            summ.append({"temperature": t, "mode": mode, "rows": len(rs), "server_tps_median": med([r["server_tps"] for r in rs]),
                         "by_category": {c: med(v) for c, v in sorted(by.items())},
                         "tokens_per_round": rnd(sum(r["tokens"] or 0 for r in rs) / max(1, sum(r["rounds"] or 0 for r in rs)), 3),
                         "short_rows": [r["row"] for r in rs if (r["tokens"] or 0) < (2048 if a.natural else a.max_tokens)],
                         "errors": [r["row"] for r in rs if not r["ok"]]})
            print(f"t={t} {mode}: median {summ[-1]['server_tps_median']} tok/s (server, prefill excluded), "
                  f"{summ[-1]['by_category']}, tokens a round {summ[-1]['tokens_per_round']}, short rows: "
                  f"{summ[-1]['short_rows'] or 'none'}", file=sys.stderr)
    doc["summary"] = summ
    if 0.0 in temps and len(modes) > 1:
        g = {}
        for r in doc["rows"]:
            if r["temperature"] == 0.0:
                g.setdefault(r["row"], set()).add(r["token_sha"])
        doc["greedy_dspark_equals_plain"] = all(same_digests(list(s)) for s in g.values())
    write(doc, a.out)
    return 0 if not any(s["errors"] for s in summ) else 1


# ------------------------------------------------------------------------------------------------ needles


def needle(a):
    base, model = a.base, a.model
    doc = meta(base, model, a, "needle")
    n = a.sessions
    depths = [float(x) for x in a.depths.split(",")] if a.depths else [round((i + 0.5) / n, 3) for i in range(n)]
    per_word = calibrate(base, model)
    words = int((a.tokens - 120) / per_word)
    rng0 = random.Random(a.seed)
    codes = []
    while len(codes) < n:
        c = f"{rng0.choice(NOUNS)}-{rng0.choice(NOUNS)}-{rng0.randint(1000, 9999)}"
        if c not in codes:
            codes.append(c)
    lockers = [f"{rng0.choice(['A', 'B', 'C', 'D', 'E', 'F'])}{rng0.randint(10, 99)}" for _ in range(n)]
    bodies = []
    for i in range(n):
        rng = random.Random(a.seed * 1000 + i)
        sents = filler(rng, words)
        at = min(len(sents), int(len(sents) * depths[i % len(depths)]))
        sents.insert(at, f"Note for the record: the passphrase for locker {lockers[i]} is {codes[i]}. Keep it safe.")
        doc_text = " ".join(sents)
        q = (f"Session {i} ({time.time_ns()}). Below is a long storehouse log. Read it, then answer the question after it."
             f"\n\n{doc_text}\n\nQuestion: what is the passphrase for locker {lockers[i]}? Reply with the passphrase only.")
        bodies.append(chat_body(model, q, a.max_tokens, 0.0, a.seed, thinking=False))
    log(f"needle: {n} sessions of ~{a.tokens} tokens (calibrated {per_word:.3f} tokens a word), depths {depths}, "
        f"{'at once' if a.concurrent or n > 1 else 'one at a time'}")
    t0 = time.perf_counter()
    res = together(base, bodies, timeout=4 * 3600) if (a.concurrent or n > 1) else [stream(base, b, 4 * 3600) for b in bodies]
    wall = time.perf_counter() - t0
    rows, ok = [], 0
    for i, r in enumerate(res):
        reply = r["text"].strip()
        own = codes[i] in reply
        others = [c for j, c in enumerate(codes) if j != i and c in reply]
        exact = own and not others and r["ok"]
        ok += exact
        rows.append({"session": i, "depth": depths[i % len(depths)], "code": codes[i], "locker": lockers[i],
                     "prompt_tokens": r["prompt_tokens"], "cached_tokens": r["cached_tokens"], "ttft_s": rnd(r["ttft_s"], 2),
                     "prefill_s": rnd(r["prefill_s"], 2), "reply": reply[:200], "own_code": own, "other_codes": others,
                     "exact": exact, "error": r["error"]})
        print(f"{'PASS' if exact else 'FAIL'} session {i}: depth {rows[-1]['depth']}, {r['prompt_tokens']} prompt tokens, "
              f"TTFT {rows[-1]['ttft_s']} s, reply {json.dumps(reply[:80])}{' error ' + str(r['error']) if r['error'] else ''}",
              file=sys.stderr)
    doc["sessions"] = rows
    doc["verdict"] = {"exact": ok, "sessions": n, "wall_s": round(wall, 1)}
    print(f"needle: {ok}/{n} sessions retrieved their own passphrase exactly ({wall:.1f} s)", file=sys.stderr)
    write(doc, a.out)
    return 0 if ok == n else 1


# ------------------------------------------------------------------------------------------------ model sheet


def st_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return {k: v["data_offsets"][1] - v["data_offsets"][0] for k, v in h.items()}


def sheet(a):
    d = a.checkpoint
    cfg = json.load(open(os.path.join(d, "config.json")))
    tc = cfg.get("text_config", cfg)
    sizes = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors") and f.startswith(("model", "mtp")):
            for k, v in st_header(os.path.join(d, f)).items():
                sizes[("mtp." if f.startswith("mtp") else "") + k] = v
    role = {"experts": 0, "router": 0, "unembed": 0, "embed": 0, "dense": 0, "drafter": 0, "vision": 0}
    expert_layers = set()
    for k, v in sizes.items():
        if k.startswith("mtp."):
            role["drafter"] += v
        elif ".visual." in k or k.startswith("model.visual"):
            role["vision"] += v
        elif ".mlp.experts." in k:
            role["experts"] += v
            expert_layers.add(k.split(".layers.")[1].split(".")[0])
        elif ".mlp.gate." in k:
            role["router"] += v
        elif k.endswith("unembed.weight"):
            role["unembed"] += v
        elif k.endswith("embed.weight"):
            role["embed"] += v
        else:
            role["dense"] += v
    ne = tc["n_routed_experts"]
    layers = tc["num_hidden_layers"]
    local = len(tc.get("local_layer_ids") or [])
    kv_layer_token = 2 * tc["num_key_value_heads"] * tc["head_dim"] * 2   # K and V, bf16, all heads
    out = {"source": "checkpoint safetensors headers + config.json (inkling2_bench.py sheet)",
           "checkpoint_bytes": role, "layers": layers, "global_layers": layers - local, "local_layers": local,
           "sliding_window": tc.get("sliding_window_size"), "n_routed_experts": ne, "experts_per_token": tc["num_experts_per_tok"],
           "expert_layers": len(expert_layers), "expert_bytes": role["experts"] / (ne * max(1, len(expert_layers))),
           "hidden_size": tc["hidden_size"], "vocab_size": tc["vocab_size"], "kv_bytes_per_token_layer": kv_layer_token,
           # how the release reads them at TP4 (phase notes: dense bf16 split by rows, experts split 1/4 of each,
           # unembed split by vocab, router logits in fp32 on every rank, the drafter split over the ranks)
           "per_rank": {"dense": "split", "experts": "split", "unembed": "split", "router": "replicated_fp32",
                        "drafter": "split"},
           # distinct experts R rows route to (a layer, measured on the oracle; phase5 dspark notes), R = 1..8
           "distinct_experts": [8, 13.4, 18.0, 22.2, 26.1, 29.7, 33.0, 36.2]}
    write(out, a.out)
    return 0


# ------------------------------------------------------------------------------------------------ ledger


def distinct(sh, rows):
    t = sh["distinct_experts"]
    if rows <= len(t):
        lo = int(rows)
        if rows == lo:
            return t[lo - 1]
        return t[lo - 1] + (t[lo] - t[lo - 1]) * (rows - lo)
    n = sh["n_routed_experts"]
    return n * (1 - (1 - t[-1] / n) ** (rows / len(t)))


def step_bytes(sh, world, rows, contexts, drafter=False):
    cb = sh["checkpoint_bytes"]
    w = float(world)
    router = cb["router"] * (2 if sh["per_rank"].get("router") == "replicated_fp32" else 1 / w)
    fixed = cb["dense"] / w + cb["unembed"] / w + router
    experts = distinct(sh, rows) * sh["expert_layers"] * sh["expert_bytes"] / w
    kvl = sh["kv_bytes_per_token_layer"] / w
    kv = sum(sh["global_layers"] * c * kvl + sh["local_layers"] * min(c, sh["sliding_window"]) * kvl for c in contexts)
    d = cb["drafter"] / w if drafter else 0
    return {"weights": fixed, "experts": experts, "kv": kv, "drafter": d, "total": fixed + experts + kv + d}


def ledger(a):
    sw = json.load(open(a.sweep))
    sh = json.load(open(a.sheet or os.path.join(HERE, "inkling2_sheet.json")))
    bw = a.bw_gbs * 1e9
    lat = a.gathers * a.gather_us * 1e-6
    rows = []

    def add(name, mode, streams, tps, ctx, spec, tokens):
        if not tps:
            return
        if mode == "dspark" and spec and spec.get("rounds"):
            L = spec["tokens_per_round"]
            wdt = 1 + (spec.get("drafts_per_round") or 0)
            b = step_bytes(sh, a.world, streams * wdt, [ctx] * streams, drafter=True)
            t = b["total"] / bw + lat + a.draft_gathers * a.gather_us * 1e-6
            floor = streams * L / t
            how = f"round: {streams} x {wdt:.2f} rows, {L:.2f} tokens a round a stream"
        else:
            b = step_bytes(sh, a.world, streams, [ctx] * streams)
            t = b["total"] / bw + lat
            floor = streams / t
            how = f"step: {streams} rows"
        rows.append({"row": name, "measured_tps": tps, "floor_ms": round(t * 1e3, 2), "floor_tps": round(floor, 1),
                     "pct_of_floor": round(100 * tps / floor, 1), "bytes_a_rank_MB": round(b["total"] / 1e6, 1),
                     "experts_MB": round(b["experts"] / 1e6, 1), "kv_MB": round(b["kv"] / 1e6, 2), "basis": how})

    for r in sw.get("decode", []):
        ctx = int(statistics.mean([p for p in r.get("prompt_tokens") or [] if p] or [100]) + r["tokens"] / 2)
        add(r["row"] + " (aggregate)", r["mode"], r["streams"], r["aggregate_tps_median"], ctx, r["speculative"], r["tokens"])
    for r in sw.get("workloads", []):
        toks = statistics.median([t for t in r["tokens"] if t] or [1])
        ctx = int((r.get("prompt_tokens") or 100) + toks / 2)
        add(r["row"] + f" [{r['readme_row']}]", r["mode"], 1, r["server_tps_median"], ctx, r["speculative"], toks)
    if a.prompts:
        pr = json.load(open(a.prompts))
        for s in pr.get("summary", []):
            rs = [x for x in pr["rows"] if x["temperature"] == s["temperature"] and x["mode"] == s["mode"]]
            ctx = int(statistics.median([(x["prompt_tokens"] or 0) + (x["tokens"] or 0) / 2 for x in rs]))
            spec = {"rounds": sum(x["rounds"] or 0 for x in rs), "tokens_per_round": s["tokens_per_round"],
                    "drafts_per_round": sum(x["drafted"] or 0 for x in rs) / max(1, sum(x["rounds"] or 0 for x in rs))}
            add(f"prompt set t={s['temperature']} {s['mode']} (median row)", s["mode"], 1, s["server_tps_median"], ctx, spec, None)
    out = {"tool": "inkling2_bench.py ledger", "sweep": a.sweep, "sweep_utc": sw.get("utc"),
           "assumptions": {"world": a.world, "stream_bandwidth_GBps": a.bw_gbs, "gathers_a_step": a.gathers,
                           "gather_us": a.gather_us, "draft_gathers": a.draft_gathers, "sheet": a.sheet or "inkling2_sheet.json"},
           "rows": rows}
    print(f"bandwidth floor: bytes a rank a step / {a.bw_gbs:g} GB/s + {a.gathers} gathers x {a.gather_us:g} us", file=sys.stderr)
    print(f"{'row':58s} {'tok/s':>7s} {'floor':>7s} {'%':>6s} {'MB/rank':>8s}  basis", file=sys.stderr)
    for r in rows:
        print(f"{r['row'][:58]:58s} {r['measured_tps']:7.1f} {r['floor_tps']:7.1f} {r['pct_of_floor']:6.1f} "
              f"{r['bytes_a_rank_MB']:8.0f}  {r['basis']}", file=sys.stderr)
    write(out, a.out)
    return 0


# ------------------------------------------------------------------------------------------------ main


def main():
    global API_KEY
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def server(sp):
        sp.add_argument("base")
        sp.add_argument("model", nargs="?", default="auto")
        sp.add_argument("--api-key-env", metavar="NAME")
        sp.add_argument("--seed", type=int, default=1234)
        sp.add_argument("--out", default="-")

    s = sub.add_parser("sweep")
    server(s)
    s.add_argument("--levels", default="1,2,4,8")
    s.add_argument("--max-streams", type=int, default=4)
    s.add_argument("--modes", default="dspark,plain")
    s.add_argument("--reps", type=int, default=3)
    s.add_argument("--tokens", type=int, default=256)
    s.add_argument("--natural-cap", type=int, default=1024)
    s.add_argument("--workloads", default="all")
    s.add_argument("--prefill", default="2048,8192")
    s.add_argument("--decode-prompts", metavar="FILE", help="default: bench_prompts.json next to this file")
    s.set_defaults(func=sweep)
    s = sub.add_parser("prompts")
    server(s)
    s.add_argument("--set", metavar="FILE", help="default: inkling2_prompts.json next to this file")
    s.add_argument("--temperatures", default="0,0.7")
    s.add_argument("--modes", default="dspark")
    s.add_argument("--max-tokens", type=int, default=2048)
    s.add_argument("--natural", action="store_true", help="stop at the model's end (default: ignore_eos, --max-tokens a row)")
    s.add_argument("--concurrency", type=int, default=1)
    s.add_argument("--only", default="")
    s.set_defaults(func=prompt_rows)
    s = sub.add_parser("needle")
    server(s)
    s.add_argument("--sessions", type=int, default=8)
    s.add_argument("--tokens", type=int, default=28000, help="prompt tokens a session (about)")
    s.add_argument("--depths", default="", help="needle depth fractions (default: spread over the sessions)")
    s.add_argument("--max-tokens", type=int, default=24)
    s.add_argument("--concurrent", action="store_true")
    s.set_defaults(func=needle)
    s = sub.add_parser("ledger")
    s.add_argument("sweep")
    s.add_argument("--sheet")
    s.add_argument("--prompts", help="a prompts JSON to add its median rows")
    s.add_argument("--world", type=int, default=4)
    s.add_argument("--bw-gbs", type=float, default=250.0, help="streaming bandwidth a GPU, GB/s")
    s.add_argument("--gathers", type=int, default=145, help="all-gathers a target step")
    s.add_argument("--draft-gathers", type=int, default=11, help="all-gathers a draft pass")
    s.add_argument("--gather-us", type=float, default=12.5, help="latency of one small all-gather, us")
    s.add_argument("--out", default="-")
    s.set_defaults(func=ledger)
    s = sub.add_parser("sheet")
    s.add_argument("checkpoint")
    s.add_argument("--out", default="-")
    s.set_defaults(func=sheet)
    a = p.parse_args()
    if getattr(a, "api_key_env", None):
        API_KEY = os.environ.get(a.api_key_env) or sys.exit(f"--api-key-env {a.api_key_env}: not set")
    if hasattr(a, "base"):
        a.base = base_url(a.base)
        a.model = model_id(a.base, a.model)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
