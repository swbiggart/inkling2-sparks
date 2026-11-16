#!/usr/bin/env python3
"""Negative fixtures for tools/inkling2_bench.py (no network): a stream that ends before [DONE] is an error and fails
every verdict; a completed reply without a token sha never counts as equal. Run: python3 tools/tests/test_inkling2_bench.py
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("inkling2_bench", os.path.join(HERE, "..", "inkling2_bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def sse(*events, done=True):
    lines = [b"data: " + json.dumps(e).encode() + b"\n" for e in events]
    return lines + ([b"data: [DONE]\n"] if done else [])


def content(text):
    return {"choices": [{"delta": {"content": text}, "finish_reason": None}]}


def final(sha="abc123", tokens=4):
    return {"choices": [{"delta": {}, "finish_reason": "length"}], "usage": {"prompt_tokens": 10, "completion_tokens": tokens},
            "tensorfold": {"tokens_per_second": 30.0, "token_sha256": sha} if sha else {"tokens_per_second": 30.0}}


class Fake:
    """bench.request stand-in: reply(body) -> the response's lines; GET paths answer small JSON."""

    def __init__(self, reply):
        self.reply = reply

    def __call__(self, base, path, body=None, timeout=60):
        lines = self.reply(body) if body is not None else [json.dumps({"data": [{"id": "m"}]}).encode()]
        r = io.BytesIO(b"".join(lines))
        return contextlib.closing(r) if body is None else contextlib.nullcontext(iter(lines))


class Quiet(unittest.TestCase):
    def setUp(self):
        self.real = bench.request
        self.err = sys.stderr
        sys.stderr = io.StringIO()

    def tearDown(self):
        bench.request = self.real
        sys.stderr = self.err

    def run_stream(self, lines):
        bench.request = Fake(lambda body: lines)
        return bench.stream("http://x", {"messages": [{"role": "user", "content": "hi"}]})


class StreamEnd(Quiet):
    def test_eof_after_content_is_an_error(self):
        r = self.run_stream(sse(content("hello"), done=False))
        self.assertFalse(r["ok"])
        self.assertIn("incomplete", r["error"])

    def test_eof_after_usage_before_done_is_an_error(self):
        r = self.run_stream(sse(content("hello"), final(), done=False))
        self.assertFalse(r["ok"])
        self.assertIn("ended before [DONE]", r["error"])
        self.assertEqual(r["token_sha"], "abc123")

    def test_complete_reply_is_ok(self):
        r = self.run_stream(sse(content("hello"), final()))
        self.assertTrue(r["ok"])
        self.assertIsNone(r["error"])


class Needle(Quiet):
    def args(self, sessions):
        return argparse.Namespace(base="http://x", model="m", seed=1234, out="-", sessions=sessions, tokens=400,
                                  depths="", max_tokens=24, concurrent=True)

    def reply_own_code(self, done):
        def reply(body):
            text = body.get("prompt") or body["messages"][0]["content"]
            if "Question:" not in text:  # calibration
                return sse(content("x"), final(), done=True)
            code = re.search(r"passphrase for locker \S+ is (\S+)\.", text).group(1)
            return sse(content(code), final(), done=done)
        return reply

    def test_needle_prefix_without_done_fails(self):
        bench.request = Fake(self.reply_own_code(done=False))
        bench.calibrate = lambda base, model: 1.3
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bench.needle(self.args(2)), 1)

    def test_needle_complete_passes(self):
        bench.request = Fake(self.reply_own_code(done=True))
        bench.calibrate = lambda base, model: 1.3
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bench.needle(self.args(2)), 0)


class Sweep(Quiet):
    def args(self):
        return argparse.Namespace(base="http://x", model="m", seed=1234, out="-", levels="8", max_streams=4,
                                  modes="dspark,plain", reps=2, tokens=8, natural_cap=16, workloads="gpu-chat-no-think",
                                  prefill="", decode_prompts=None)

    def sweep(self, lines):
        bench.request = Fake(lambda body: lines)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = bench.sweep(self.args())
        return rc, json.loads(out.getvalue())

    def test_workloads_without_token_sha_are_not_equal(self):
        rc, doc = self.sweep(sse(content("a b c"), final(sha=None)))
        self.assertEqual(rc, 1)
        self.assertFalse(doc["verdict"]["greedy_repeatable"])
        self.assertFalse(doc["verdict"]["dspark_equals_plain"])

    def test_workloads_ending_before_done_are_errors(self):
        rc, doc = self.sweep(sse(content("a b c"), final(), done=False))
        self.assertEqual(rc, 1)
        self.assertGreater(doc["verdict"]["errors"], 0)

    def test_workloads_complete_with_sha_pass(self):
        rc, doc = self.sweep(sse(content("a b c"), final()))
        self.assertEqual(rc, 0)
        self.assertTrue(doc["verdict"]["dspark_equals_plain"])

    def test_same_digests(self):
        self.assertFalse(bench.same_digests([None], [None]))
        self.assertFalse(bench.same_digests([]))
        self.assertFalse(bench.same_digests(["a", "b"]))
        self.assertTrue(bench.same_digests(["a"], ["a"]))


if __name__ == "__main__":
    unittest.main()
