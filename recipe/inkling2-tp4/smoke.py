#!/usr/bin/env python3
"""Smoke and oracle checks for the Inkling-2-Small TP4 server. Standard library only; run where the endpoint is
reachable (rank 0's host when it serves on loopback).

  smoke.py URL smoke [--long-tokens 8000]   short chat, thinking, a get_weather round trip, stream equals whole,
                                           a long prompt (TTFT and prefill tok/s from the stream)
  smoke.py URL oracle DECODE_JSON IDS... [--ext DIR]
                                           greedy reply of each prompt (token ids, NAME=ids.json) against the
                                           oracle's greedy ids in DECODE_JSON (tools/zig/inkling2_ref.py's decode.json)
  smoke.py URL yardstick DECODE_JSON IDS... [--ext DIR] [--json OUT]
                                           the token gate: the server forced along the oracle's replies, one greedy
                                           token at every position, scored against the oracle's top-1 and top-2
  smoke.py URL prefill N [N...]            cold prompts of about N tokens: TTFT and prompt tokens per second

oracle prints, per prompt, EQUAL or, at the first position j where the server's free-running reply leaves the
oracle's, DIFFER at j (a failure) when DECODE_JSON has no margins (an oracle file made before v0.2), or DEPARTS at j
(information, not a failure) when it has them, with the oracle's gap there and upstream TensorFold's one-bf16-step
label (tools/zig/qwen27_margins.py). --ext DIR (a folder with ORDER, decode.json and NAME.ids.json) adds every prompt
of DIR in ORDER and an EXT summary line.

yardstick is the gate when DECODE_JSON has margins (v0.2's fp32 oracle). At every position j of each oracle reply the
server gets a token-id prompt (the prompt, then the oracle's first j reply tokens) on /v1/completions with
max_tokens 1, temperature 0, ignore_eos and return_token_ids, the release's own API, and its one token is compared with
the oracle's top-1 and top-2 there (DECODE_JSON's margins[j]). It passes when, over YARDSTICK_POSITIONS positions,
at least YARDSTICK_TOP1 tokens equal the oracle's and at least YARDSTICK_TOP2 are in its top-2 (YARDSTICK line).
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.request

MODEL = "model"
EOS = (200006,)  # the oracle replies' end token; a server may stop on it without returning it

# The token gate's bars. The server, forced along the fp32 oracle's greedy replies (chat_short, tool_call and the 7
# prompts of the extended set: 445 positions), must give the oracle's top-1 at >= YARDSTICK_TOP1 positions and a token
# in the oracle's top-2 at >= YARDSTICK_TOP2. They are the worst scores of the reference implementation's own valid
# variants on the same 445 positions, forced the same way: Transformers' eager attention with bf16 and with fp32
# reduction, and the fp32 form the oracle uses, each on the decode path (prefill, then one-row steps) and teacher-forced
# (one forward over the whole sequence). Those six scored 438-445 top-1 matches (each variant's own argmax) and
# 443-445 in the top-2 (with an exact tie at a variant's top counted against it; 444-445 by argmax);
# tools/zig/inkling2_tolerance.py --yardstick reprints them.
YARDSTICK_POSITIONS = 445
YARDSTICK_TOP1 = 438
YARDSTICK_TOP2 = 443


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream(url, body, timeout=1800):
    """Streams a chat or completion; returns (text, reasoning, tool call deltas, finish, usage, t_first, t_end, t0)."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    text, reasoning, tools, finish, usage, t_first = [], [], [], None, None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for c in ev.get("choices", []):
                d = c.get("delta") or {}
                piece = d.get("content") or c.get("text") or ""
                rpiece = d.get("reasoning_content") or d.get("reasoning") or ""
                if (piece or rpiece or d.get("tool_calls")) and t_first is None:
                    t_first = time.perf_counter()
                text.append(piece)
                reasoning.append(rpiece)
                tools.extend(d.get("tool_calls") or [])
                finish = c.get("finish_reason") or finish
    return "".join(text), "".join(reasoning), tools, finish, usage, t_first, time.perf_counter(), t0


def show(title, msg, usage, secs):
    print(f"--- {title} ({secs:.1f} s, usage {json.dumps(usage)})")
    if msg.get("reasoning_content") or msg.get("reasoning"):
        print(f"  reasoning: {(msg.get('reasoning_content') or msg.get('reasoning'))!r}")
    print(f"  content:   {msg.get('content')!r}")
    for t in msg.get("tool_calls") or []:
        print(f"  tool call: {t['function']['name']} {t['function']['arguments']}")


def chat(base, messages, **kw):
    body = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": 400}
    body.update(kw)
    t = time.perf_counter()
    r = post(base + "/v1/chat/completions", body)
    return r, time.perf_counter() - t


WEATHER = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}]


def long_document(n_tokens):
    # about 13 tokens a line
    shelves = "ABCDEFGH"
    lines = [f"Item {i} is stored on shelf {shelves[i % 8]}{i % 7}; it is a {['red', 'blue', 'green', 'amber'][i % 4]} box."
             for i in range(max(1, n_tokens // 14))]
    return "\n".join(lines)


def smoke(a):
    base = a.url
    ok = True
    # 1. short chat
    r, s = chat(base, [{"role": "user", "content": "What is the capital of France?"}])
    show("chat: capital of France", r["choices"][0]["message"], r["usage"], s)
    ok &= "Paris" in (r["choices"][0]["message"].get("content") or "")
    whole = r["choices"][0]["message"].get("content")
    # 2. streaming equals whole
    text, reasoning, _, finish, usage, tf, te, t0 = stream(base + "/v1/chat/completions", {
        "model": MODEL, "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "temperature": 0, "max_tokens": 400})
    print(f"--- stream: capital of France ({te - t0:.1f} s, TTFT {tf - t0:.2f} s, finish {finish}, usage {json.dumps(usage)})")
    print(f"  content:   {text!r}\n  stream == whole: {text == whole}")
    ok &= text == whole
    # 3. thinking
    r, s = chat(base, [{"role": "user", "content": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Think it through."}],
                max_tokens=1500, reasoning_effort="high")
    show("thinking (reasoning_effort high)", r["choices"][0]["message"], r["usage"], s)
    m = r["choices"][0]["message"]
    ok &= bool(m.get("reasoning_content") or m.get("reasoning")) and "0.05" in (m.get("content") or "") + ""
    # 4. tool call round trip
    msgs = [{"role": "user", "content": "What's the weather in Paris right now? Use celsius."}]
    r, s = chat(base, msgs, tools=WEATHER, max_tokens=600)
    m = r["choices"][0]["message"]
    show(f"tool call (finish {r['choices'][0]['finish_reason']})", m, r["usage"], s)
    calls = m.get("tool_calls") or []
    ok &= bool(calls) and calls[0]["function"]["name"] == "get_weather"
    if calls:
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        msgs.append({"role": "tool", "tool_call_id": calls[0].get("id", "call_0"), "name": "get_weather",
                     "content": json.dumps({"city": "Paris", "temperature": 14, "unit": "celsius", "condition": "light rain"})})
        r, s = chat(base, msgs, tools=WEATHER, max_tokens=600)
        show(f"tool result -> answer (finish {r['choices'][0]['finish_reason']})", r["choices"][0]["message"], r["usage"], s)
        ok &= "14" in (r["choices"][0]["message"].get("content") or "")
    # 5. streamed tool call
    text, reasoning, tools, finish, usage, tf, te, t0 = stream(base + "/v1/chat/completions", {
        "model": MODEL, "messages": msgs[:1], "tools": WEATHER, "temperature": 0, "max_tokens": 600})
    names = "".join(t.get("function", {}).get("name") or "" for t in tools)
    args = "".join(t.get("function", {}).get("arguments") or "" for t in tools)
    print(f"--- stream: tool call ({te - t0:.1f} s, finish {finish}): reasoning {reasoning!r}; call {names} {args}")
    ok &= names == "get_weather"
    # 6. long prompt
    doc = long_document(a.long_tokens)
    q = "Which shelf holds item 417, and what colour is it? Answer in one sentence."
    text, reasoning, _, finish, usage, tf, te, t0 = stream(base + "/v1/chat/completions", {
        "model": MODEL, "messages": [{"role": "user", "content": doc + "\n\n" + q}],
        "temperature": 0, "max_tokens": 300, "reasoning_effort": "none"})
    pt = (usage or {}).get("prompt_tokens", 0)
    ct = (usage or {}).get("completion_tokens", 0)
    print(f"--- long prompt: {pt} prompt tokens, TTFT {tf - t0:.2f} s ({pt / (tf - t0):.0f} prompt tok/s), "
          f"{ct} reply tokens in {te - t0:.1f} s, finish {finish}")
    print(f"  reasoning: {reasoning!r}\n  content:   {text!r}  (expected: shelf {'ABCDEFGH'[417 % 8]}{417 % 7}, {['red', 'blue', 'green', 'amber'][417 % 4]})")
    ok &= f"{'ABCDEFGH'[417 % 8]}{417 % 7}" in text + reasoning
    print("SMOKE", "PASS" if ok else "CHECK OUTPUT")
    return 0 if ok else 1


def bf16_spacing(value):
    """One bf16 step at |value| (upstream TensorFold's tools/zig/qwen27_margins.py rule)."""
    m = abs(value)
    if m < 2.0 ** -126:
        return 2.0 ** -133
    return 2.0 ** (math.frexp(m)[1] - 1 - 7)


def near_tie(margin, oracle_token, server_token):
    """Upstream TensorFold's one-bf16-step tie label at a departure, oracle side (information only, never the gate):
    the oracle's row there (decode.json's margins[j]) has the oracle's token first and the server's second, at most one
    bf16 step of the top-1 logit apart (softcapped logits). Returns (near, gap, step, why)."""
    if not margin or margin.get("top1") != oracle_token:
        return False, None, None, "no oracle margin for this row"
    if margin.get("top2") != server_token or margin.get("gap") is None:
        return False, None, None, f"the oracle's runner-up is {margin.get('top2')}, not the server's token"
    gap = float(margin["gap"])
    step = bf16_spacing(float(margin["top1_logit"]))
    return gap <= step, gap, step, ""


def label(margin, oracle_token, server_token):
    """A departure's information line: the oracle's gap there and upstream's one-step label."""
    near, gap, step, why = near_tie(margin, oracle_token, server_token)
    if why:
        return why
    return f"oracle gap {gap:.4f}; upstream one bf16 step {step:.4g}: {'near tie' if near else 'over'}"


def compare(ref, margins, got):
    """The server's free-running ids against the oracle's: (status, j, tokens equal before j, note). status EQUAL;
    DEPARTS (information) when the oracle file has margins; DIFFER (a failure) when it has none (v0.1's strict rule)."""
    n = min(len(got), len(ref))
    first = next((i for i in range(n) if got[i] != ref[i]), None)
    if first is None:
        eos_end = len(got) == len(ref) - 1 and ref[-1] in EOS
        if len(got) >= len(ref) or eos_end:
            return "EQUAL", None, len(ref), " (oracle ends with EOS, which the server does not return)" if eos_end else ""
        first, server = n, "end of reply"
    else:
        server = got[first]
    if margins is None:
        return "DIFFER", first, first, (f" (server {server}, oracle {ref[first]}; no margins in the oracle file: "
                                        "strict rule)")
    row = margins[first] if first < len(margins) else None
    info = label(row, ref[first], server) if server != "end of reply" else "the server stopped"
    return "DEPARTS", first, first, f" (server {server}, oracle {ref[first]}: {info}; information)"


def yardstick_row(margin, oracle_token, token):
    """One forced position: (token is the oracle's top-1, token is in the oracle's top-2). margin: decode.json's
    margins[j]; an exact tie at the oracle's top is listed in either order, so both of its two tokens count."""
    if not margin:
        raise SystemExit("the yardstick needs the oracle's margins (an oracle file made with --attn fp32)")
    top2 = {oracle_token, margin.get("top1"), margin.get("top2")} - {None}
    return token == oracle_token, token in top2


def yardstick_score(prompts):
    """prompts: [(name, oracle ids, margins, forced tokens)], one forced token per oracle position. Returns
    {"positions", "top1", "top2", "misses": [(name, j, token, oracle token, margins[j])]}."""
    out = {"positions": 0, "top1": 0, "top2": 0, "misses": []}
    for name, ref, margins, forced in prompts:
        if margins is None or len(margins) != len(ref):
            raise SystemExit(f"{name}: the yardstick needs a margins row for every oracle token")
        if len(forced) != len(ref):
            raise SystemExit(f"{name}: {len(forced)} forced tokens for {len(ref)} oracle positions")
        for j, (o, t) in enumerate(zip(ref, forced)):
            m1, m2 = yardstick_row(margins[j], o, t)
            out["positions"] += 1
            out["top1"] += m1
            out["top2"] += m2
            if not m1:
                out["misses"].append((name, j, t, o, margins[j]))
    return out


def yardstick_verdict(score):
    """(passed, the YARDSTICK line). The bars hold only on the 445 positions they were measured on."""
    n, t1, t2 = score["positions"], score["top1"], score["top2"]
    ok = n == YARDSTICK_POSITIONS and t1 >= YARDSTICK_TOP1 and t2 >= YARDSTICK_TOP2
    why = "" if n == YARDSTICK_POSITIONS else f"; the bars are for {YARDSTICK_POSITIONS} positions"
    return ok, (f"YARDSTICK: top-1 {t1}/{n} ({'>=' if t1 >= YARDSTICK_TOP1 else '<'} {YARDSTICK_TOP1}), in the "
                f"oracle's top-2 {t2}/{n} ({'>=' if t2 >= YARDSTICK_TOP2 else '<'} {YARDSTICK_TOP2}){why}: "
                f"{'PASS' if ok else 'FAIL'}")


def yardstick_lines(score, out=print):
    """Every position whose forced token is not the oracle's top-1, with upstream's one-step label (information)."""
    for name, j, t, o, m in score["misses"]:
        where = "in the oracle's top-2" if t in {o, m.get("top1"), m.get("top2")} else "outside the oracle's top-2"
        out(f"  {name} j={j}: server {t}, oracle {o} ({where}; {label(m, o, t)})")


def oracle_one(a, name, ids, want):
    ref = want[name]["ids"]
    r = post(a.url + "/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": len(ref), "temperature": 0,
                                         "return_token_ids": True})
    got = r.get("tensorfold", {}).get("token_ids") or []
    status, j, n, note = compare(ref, want[name].get("margins"), got)
    at = "" if status == "EQUAL" else f" at {j}"
    print(f"{name}: prompt {len(ids)} tokens; oracle {len(ref)} tokens, server {len(got)} "
          f"(finish {r['choices'][0].get('finish_reason')}): {status}{at}{note}")
    print(f"  server: {r['choices'][0].get('text')!r}")
    print(f"  oracle: {want[name]['text']!r}")
    return status, n, len(ref)


def load_prompts(a):
    """[(name, prompt ids, oracle entry)]: NAME=ids.json against DECODE_JSON, then --ext DIR's prompts in ORDER."""
    def load(path):
        with open(path) as f:
            return json.load(f)

    want = load(a.decode)
    out = []
    for spec in a.ids:
        name, path = spec.split("=", 1)
        out.append((name, load(path), want[name]))
    if a.ext:
        ext = load(os.path.join(a.ext, "decode.json"))
        with open(os.path.join(a.ext, "ORDER")) as f:
            order = [x.strip() for x in f if x.strip() and not x.startswith("#")]
        out += [(n, load(os.path.join(a.ext, n + ".ids.json")), ext[n]) for n in order]
    names = [n for n, _, _ in out]
    if len(set(names)) != len(names):
        raise SystemExit(f"a prompt is given twice: {names}")
    return out


def oracle(a):
    fails, ext = 0, []
    for i, (name, ids, entry) in enumerate(load_prompts(a)):
        r = oracle_one(a, name, ids, {name: entry})
        fails += r[0] == "DIFFER"
        if i >= len(a.ids):
            ext.append(r)
    if a.ext:
        c = {k: sum(v[0] == k for v in ext) for k in ("EQUAL", "DEPARTS", "DIFFER")}
        n, total = sum(v[1] for v in ext), sum(v[2] for v in ext)
        print(f"EXT summary: {len(ext)} prompts, {c['EQUAL']} EQUAL, {c['DEPARTS']} DEPARTS, {c['DIFFER']} DIFFER; "
              f"{n} of {total} oracle tokens equal before the first departure ({100.0 * n / max(total, 1):.1f}%)")
    return 1 if fails else 0


def force_one(url, ids):
    """The server's greedy token after the token ids `ids`, through the release's own API: a /v1/completions token-id
    prompt with max_tokens 1 (ignore_eos so an end token comes back as a token, return_token_ids for its id)."""
    r = post(url + "/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0,
                                       "ignore_eos": True, "return_token_ids": True})
    got = (r.get("tensorfold") or {}).get("token_ids")
    if not isinstance(got, list) or len(got) != 1:
        raise SystemExit(f"a forced request ({len(ids)} prompt tokens) returned token_ids {got!r}, not one token")
    return got[0]


def yardstick(a):
    rows, forced, t0 = [], {}, time.perf_counter()
    for name, ids, entry in load_prompts(a):
        ref, margins = entry["ids"], entry.get("margins")
        if margins is None:
            raise SystemExit(f"{name}: the oracle file has no margins (the yardstick needs v0.2's fp32 oracle)")
        forced[name] = [force_one(a.url, list(ids) + list(ref[:j])) for j in range(len(ref))]
        one = yardstick_score([(name, ref, margins, forced[name])])
        print(f"{name}: prompt {len(ids)} tokens, {len(ref)} positions: top-1 {one['top1']}, in the oracle's top-2 "
              f"{one['top2']}")
        yardstick_lines(one)
        rows.append((name, ref, margins, forced[name]))
    score = yardstick_score(rows)
    ok, line = yardstick_verdict(score)
    print(f"{score['positions']} forced requests in {time.perf_counter() - t0:.1f} s")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"forced": forced, "positions": score["positions"], "top1": score["top1"],
                       "top2": score["top2"], "passed": ok}, f)
    print(line)
    return 0 if ok else 1


def prefill(a):
    for n in a.sizes:
        doc = long_document(n)
        body = {"model": MODEL, "prompt": doc + f"\nNonce {time.time_ns()}.\nItem 3 is stored on shelf", "max_tokens": 2,
                "temperature": 0}
        text, _, _, _, usage, tf, te, t0 = stream(a.url + "/v1/completions", body)
        pt = (usage or {}).get("prompt_tokens", 0)
        print(f"prefill ~{n}: {pt} prompt tokens, cold TTFT {tf - t0:.2f} s, {pt / (tf - t0):.0f} prompt tok/s; reply {text!r}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--long-tokens", type=int, default=8000)
    o = sub.add_parser("oracle")
    o.add_argument("decode")
    o.add_argument("ids", nargs="+")
    o.add_argument("--ext", default="", help="the extended set's folder (ORDER, decode.json, NAME.ids.json)")
    y = sub.add_parser("yardstick")
    y.add_argument("decode")
    y.add_argument("ids", nargs="+")
    y.add_argument("--ext", default="", help="the extended set's folder (ORDER, decode.json, NAME.ids.json)")
    y.add_argument("--json", default="", help="write the forced tokens and the score here")
    f = sub.add_parser("prefill")
    f.add_argument("sizes", type=int, nargs="+")
    a = p.parse_args()
    a.url = a.url.rstrip("/")
    return {"smoke": smoke, "oracle": oracle, "yardstick": yardstick, "prefill": prefill}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
