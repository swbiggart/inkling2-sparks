#!/usr/bin/env python3
"""Smoke and oracle checks for the Inkling-2-Small TP4 server. Standard library only; run where the endpoint is
reachable (rank 0's host when it serves on loopback).

  smoke.py URL smoke [--long-tokens 8000]   short chat, thinking, a get_weather round trip, stream equals whole,
                                           a long prompt (TTFT and prefill tok/s from the stream)
  smoke.py URL oracle DECODE_JSON IDS...   greedy reply of each prompt (token ids, NAME=ids.json) against the
                                           oracle's greedy ids in DECODE_JSON (tools/zig/inkling2_ref.py's decode.json)
  smoke.py URL prefill N [N...]            cold prompts of about N tokens: TTFT and prompt tokens per second
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request

MODEL = "model"


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


def oracle(a):
    want = json.load(open(a.decode))
    fails = 0
    for spec in a.ids:
        name, path = spec.split("=", 1)
        ids = json.load(open(path))
        ref = want[name]["ids"]
        r = post(a.url + "/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": len(ref), "temperature": 0,
                                             "return_token_ids": True})
        got = r.get("tensorfold", {}).get("token_ids") or []
        n = min(len(got), len(ref))
        first = next((i for i in range(n) if got[i] != ref[i]), None)
        # the server may stop at EOS without returning it; the oracle's reply ends with it
        eos_end = first is None and len(got) == len(ref) - 1 and ref[-1] in (200006,)
        same = first is None and (len(got) >= len(ref) or eos_end)
        print(f"{name}: prompt {len(ids)} tokens; oracle {len(ref)} tokens, server {len(got)} "
              f"(finish {r['choices'][0].get('finish_reason')}): {'EQUAL' if same else f'differs at {first}'}"
              f"{' (oracle ends with EOS, which the server does not return)' if eos_end else ''}")
        print(f"  server: {r['choices'][0].get('text')!r}")
        print(f"  oracle: {want[name]['text']!r}")
        fails += not same
    return 1 if fails else 0


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
    f = sub.add_parser("prefill")
    f.add_argument("sizes", type=int, nargs="+")
    a = p.parse_args()
    a.url = a.url.rstrip("/")
    return {"smoke": smoke, "oracle": oracle, "prefill": prefill}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
