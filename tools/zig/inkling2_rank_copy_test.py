"""inkling2_rank_copy.py offline: a fake checkpoint served by a local HTTP server (with injected faults) and by the
tool's own TCP server; fetch, verify and check against it, with short writes, failed syncs and connections, wrong
Content-Range, short header bodies, truncated or recreated shards and missing runs."""
import contextlib
import errno
import hashlib
import http.server
import io
import json
import os
from pathlib import Path
import random
import struct
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling2_rank_copy as rc  # noqa: E402

PAD = 1 << 20  # a tensor the rank does not load, so its runs sit past the first blocks (holes are block-sized)


def shard(tensors):
    """A safetensors file's bytes and each tensor's absolute [lo, hi), from {name: payload}."""
    meta, at = {}, 0
    for name, data in tensors.items():
        meta[name] = {"dtype": "U8", "shape": [len(data)], "data_offsets": [at, at + len(data)]}
        at += len(data)
    head = json.dumps(meta).encode()
    head += b" " * (-len(head) % 8)
    base = 8 + len(head)
    spans = {k: (base + v["data_offsets"][0], base + v["data_offsets"][1]) for k, v in meta.items()}
    return struct.pack("<Q", len(head)) + head + b"".join(tensors.values()), spans


class Faults:
    shift_range = False  # Content-Range one byte off for a run (a range not starting at 0)
    short_header = False  # a header body cut short, with no Content-Length (the connection closes)
    fail_after = None  # every request fails once this many have been answered


class Handler(http.server.BaseHTTPRequestHandler):
    root = None
    faults = Faults()
    served = [0]

    def log_message(self, *a):
        pass

    def do_GET(self):
        f = self.faults
        if f.fail_after is not None and self.served[0] >= f.fail_after:
            self.send_error(503)
            return
        self.served[0] += 1
        p = os.path.join(self.root, self.path.lstrip("/"))
        if not os.path.isfile(p):
            self.send_error(404)
            return
        data = open(p, "rb").read()
        rng = self.headers.get("Range")
        if rng is None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        lo, hi = (int(x) for x in rng.split("=")[1].split("-"))
        body = data[lo:hi + 1]
        clo, chi = lo, hi
        if f.shift_range and lo > 0:
            clo, chi = lo + 1, hi + 1
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {clo}-{chi}/{len(data)}")
        if f.short_header and lo == 0 and hi > 7:
            self.end_headers()
            self.wfile.write(body[:len(body) // 2])
            return
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class RankCopyTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.tmp = Path(folder.name)
        self.src, self.dest = self.tmp / "src", self.tmp / "dest"
        self.src.mkdir()
        rnd = random.Random(7)

        def blob(n):
            return bytes(rnd.getrandbits(8) for _ in range(n))

        s1, sp1 = shard({"model.llm.pad": blob(PAD), "model.llm.a": blob(5000), "model.llm.b": blob(16 * 64)})
        s2, sp2 = shard({"model.llm.pad2": blob(PAD), "model.llm.c": blob(7000), "model.visual.v": blob(3000)})
        mtp, spm = shard({"pad": blob(PAD), "layers.0.x": blob(2000)})
        self.shards = {"model-00001-of-00002.safetensors": s1, "model-00002-of-00002.safetensors": s2, "mtp.safetensors": mtp}
        for name, data in self.shards.items():
            (self.src / name).write_bytes(data)
        index = {"weight_map": {**{k: "model-00001-of-00002.safetensors" for k in sp1}, **{k: "model-00002-of-00002.safetensors" for k in sp2}}}
        for name in rc.SMALL_FILES:
            (self.src / name).parent.mkdir(parents=True, exist_ok=True)
            (self.src / name).write_bytes(json.dumps(index).encode() if name == "model.safetensors.index.json" else f"{name} {blob(4).hex()}".encode())
        lines = ["# file\toffset\tlen\tstride\tcount\tplacement\ttensor"]
        lo, hi = sp1["model.llm.a"]
        lines.append(f"model-00001-of-00002.safetensors\t{lo}\t{hi - lo}\t{hi - lo}\t1\treplicated\tmodel.llm.a")
        lo, _ = sp1["model.llm.b"]
        lines.append(f"model-00001-of-00002.safetensors\t{lo + 8}\t4\t16\t64\tq_cols\tmodel.llm.b")  # strided: 4 of every 16
        lo, hi = sp2["model.llm.c"]
        lines.append(f"model-00002-of-00002.safetensors\t{lo}\t{hi - lo}\t{hi - lo}\t1\texperts\tmodel.llm.c")
        lo, hi = sp2["model.visual.v"]
        lines.append(f"model-00002-of-00002.safetensors\t{lo}\t{hi - lo}\t{hi - lo}\t1\tfirst\tmodel.visual.v")
        lo, hi = spm["layers.0.x"]
        lines.append(f"mtp.safetensors\t{lo}\t{hi - lo}\t{hi - lo}\t1\tlast\tlayers.0.x")
        self.ranges = self.tmp / "ranges.tsv"
        self.ranges.write_text("\n".join(lines) + "\n")
        Handler.root = str(self.src)
        Handler.faults = Faults()
        Handler.served = [0]
        self.http = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.http.daemon_threads = True
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"
        patcher = mock.patch.object(rc, "HTTP_PAUSE", 0.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tool(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = rc.main([str(a) for a in args])
        return code, out.getvalue()

    def fetch(self, *extra):
        return self.tool("fetch", "--http", self.url, "--ranges", self.ranges, "--dest", self.dest, "--jobs", 2, *extra)

    def verify(self, *extra):
        return self.tool("verify", "--ranges", self.ranges, "--dest", self.dest, *extra)

    def check(self, *extra):
        return self.tool("check", "--http", self.url, "--ranges", self.ranges, "--dest", self.dest, "--jobs", 2, *extra)

    def state(self):
        return json.loads((self.dest / rc.STATE).read_text())

    def fetch_all(self):
        """The text runs (default prefix), then the vision and drafter runs."""
        code, out = self.fetch()
        self.assertEqual(code, 0, out)
        code, out = self.fetch("--prefix", "model.visual.", "--prefix", "layers.")
        self.assertEqual(code, 0, out)

    def test_fetch_verify_check_clean(self):
        self.fetch_all()
        self.assertEqual(len(self.state()["runs"]), 5)
        code, out = self.verify()
        self.assertEqual(code, 0, out)
        self.assertIn("coverage: 5 of 5 runs", out)
        self.assertIn("headers: 3 of 3 against the headers recorded at fetch", out)
        code, out = self.check()
        self.assertEqual(code, 0, out)
        self.assertIn("check: 5 of 5 held runs", out)
        for name, data in self.shards.items():  # every held run's bytes are the source's, at the source's offsets
            got = (self.dest / name).read_bytes()
            self.assertEqual(len(got), len(data))
        for r in rc.load_runs(self.ranges, None):
            got = (self.dest / r["file"]).read_bytes()
            want = self.shards[r["file"]]
            for k in range(r["count"]):
                at = r["offset"] + k * r["stride"]
                self.assertEqual(got[at:at + r["len"]], want[at:at + r["len"]])

    def test_coverage_names_missing_runs(self):
        code, out = self.fetch()  # text only: the vision and drafter runs are not fetched
        self.assertEqual(code, 0, out)
        code, out = self.verify()
        self.assertEqual(code, 1, out)
        self.assertIn("missing model.visual.*: 1 runs", out)
        self.assertIn("missing mtp.safetensors: 1 runs", out)
        code, out = self.verify("--expect-missing", "model.visual.", "--expect-missing", "mtp.safetensors")
        self.assertEqual(code, 0, out)
        self.assertIn("(expected: --expect-missing)", out)
        code, out = self.check("--expect-missing", "model.visual.")
        self.assertEqual(code, 1, out)  # the drafter's run is still missing
        self.assertIn("coverage INCOMPLETE", out)
        # a run dropped from the state is missing too
        self.fetch_all()
        st = self.state()
        key = next(k for k in st["runs"] if k.startswith("model-00002"))
        del st["runs"][key]
        (self.dest / rc.STATE).write_text(json.dumps(st))
        code, out = self.verify()
        self.assertEqual(code, 1, out)
        self.assertIn("missing model.", out)

    def test_short_write_is_not_recorded(self):
        real = os.pwrite
        calls = [0]

        def disk_full(fd, data, off):
            if off >= PAD:  # a run's bytes, not a header
                calls[0] += 1
                return real(fd, bytes(data)[:len(data) // 2], off) if calls[0] == 1 else 0
            return real(fd, data, off)

        with mock.patch.object(rc, "_pwrite", disk_full):
            code, out = self.fetch()
        self.assertEqual(code, 1, out)
        self.assertIn("short write", out)
        self.assertLess(len(self.state()["runs"]), 3)
        code, out = self.fetch()  # space again: the rest is copied
        self.assertEqual(code, 0, out)
        code, out = self.verify("--expect-missing", "model.visual.", "--expect-missing", "mtp.safetensors")
        self.assertEqual(code, 0, out)

    def test_short_write_continued(self):
        real = os.pwrite

        def dribble(fd, data, off):
            return real(fd, bytes(data)[:max(1, len(data) // 3)], off)

        with mock.patch.object(rc, "_pwrite", dribble):
            code, out = self.fetch()
        self.assertEqual(code, 0, out)
        code, out = self.verify("--expect-missing", "model.visual.", "--expect-missing", "mtp.safetensors")
        self.assertEqual(code, 0, out)

    def test_sync_failure_fails_the_fetch(self):
        real = rc._fsync

        def failing(fd):
            if threading.current_thread() is not threading.main_thread():
                raise OSError(errno.EIO, "fsync failed")
            return real(fd)

        with mock.patch.object(rc, "_fsync", failing):
            code, out = self.fetch()
        self.assertEqual(code, 1, out)
        self.assertEqual(self.state()["runs"], {})

    def test_worker_connection_failure_fails_the_fetch(self):
        with mock.patch.object(rc.HttpSource, "clone", side_effect=ConnectionRefusedError("no route")):
            code, out = self.fetch()
        self.assertEqual(code, 1, out)
        self.assertIn("ConnectionRefusedError", out)
        self.assertIn("3 runs not copied", out)

    def test_wrong_content_range_rejected(self):
        Handler.faults.shift_range = True
        code, out = self.fetch()
        self.assertEqual(code, 1, out)
        self.assertIn("Content-Range", out)
        self.assertEqual(self.state()["runs"], {})

    def test_short_header_rejected(self):
        Handler.faults.short_header = True
        with self.assertRaises(EOFError):
            self.fetch()

    def test_resume_into_truncated_or_recreated_shard(self):
        self.fetch_all()
        s1, s2 = self.dest / "model-00001-of-00002.safetensors", self.dest / "model-00002-of-00002.safetensors"
        os.truncate(s1, 0)  # truncated
        os.remove(s2)  # recreated
        code, out = self.fetch()
        self.assertEqual(code, 0, out)
        self.assertIn("recorded runs dropped", out)
        self.assertIn("3 of 3 runs to copy", out)
        code, out = self.fetch("--prefix", "model.visual.", "--prefix", "layers.")
        self.assertEqual(code, 0, out)
        self.assertIn("1 of 2 runs to copy", out)  # the vision run (its shard was recreated); the drafter's stays
        code, out = self.verify()
        self.assertEqual(code, 0, out)

    def test_resume_with_holes_in_an_old_state(self):
        """A state from the earlier tool (no shard records): a shard recreated at full size has holes where runs were."""
        probe = self.tmp / "probe"
        with open(probe, "wb") as f:
            f.truncate(4 * PAD)
        fd = os.open(probe, os.O_RDONLY)
        try:
            sparse = not rc.in_data(fd, 2 * PAD)
        finally:
            os.close(fd)
        if not sparse:
            self.skipTest("this file system does not report holes")
        self.fetch_all()
        st = self.state()
        del st["files"]
        (self.dest / rc.STATE).write_text(json.dumps(st))
        s1 = self.dest / "model-00001-of-00002.safetensors"
        size = s1.stat().st_size
        os.remove(s1)
        with open(s1, "wb") as f:
            f.truncate(size)
        code, out = self.fetch()
        self.assertEqual(code, 0, out)
        self.assertIn("2 recorded runs dropped", out)
        code, out = self.verify()
        self.assertEqual(code, 0, out)

    def test_verify_stops_at_end_of_file(self):
        self.fetch_all()
        s2 = self.dest / "model-00002-of-00002.safetensors"
        os.truncate(s2, PAD)  # the runs past here are gone
        result = []
        t = threading.Thread(target=lambda: result.append(self.verify()), daemon=True)
        t.start()
        t.join(30)
        self.assertFalse(t.is_alive(), "verify did not finish")
        code, out = result[0]
        self.assertEqual(code, 1, out)
        self.assertIn("SHORT model-00002", out)
        self.assertIn("HEADER model-00002", out)  # its size is not the recorded one

    def test_header_differs(self):
        self.fetch_all()
        p = self.dest / "model-00001-of-00002.safetensors"
        with open(p, "r+b") as f:
            f.seek(20)
            b = f.read(1)
            f.seek(20)
            f.write(bytes([b[0] ^ 1]))
        code, out = self.verify()
        self.assertEqual(code, 1, out)
        self.assertIn("HEADER model-00001-of-00002.safetensors: differs from the header recorded at fetch", out)
        code, out = self.check()
        self.assertEqual(code, 1, out)
        self.assertIn("HEADER model-00001-of-00002.safetensors: differs from the source's", out)
        # without recorded headers (an older state), verify parses the header and finds the plan's runs in it
        st = self.state()
        del st["files"]
        (self.dest / rc.STATE).write_text(json.dumps(st))
        with open(p, "r+b") as f:
            f.seek(20)
            f.write(b)
        code, out = self.verify()
        self.assertEqual(code, 0, out)
        self.assertIn("3 not recorded", out)

    def test_check_finds_a_source_difference_and_samples(self):
        self.fetch_all()
        name = "model-00002-of-00002.safetensors"
        r = next(r for r in rc.load_runs(self.ranges, None) if r["tensor"] == "model.llm.c")
        data = bytearray(self.shards[name])
        data[r["offset"] + 100] ^= 1
        (self.src / name).write_bytes(bytes(data))
        code, out = self.check()
        self.assertEqual(code, 1, out)
        self.assertIn(f"MISMATCH {rc.run_key(r)}", out)
        code, out = self.check("--max-gib", 4000 / 2**30)  # a sample under 4000 bytes leaves model.llm.c out
        self.assertEqual(code, 0, out)
        self.assertIn("a sample of at most", out)
        self.assertNotIn("check: 5 of 5", out)

    def test_tcp_server_source(self):
        srv = rc.Server(("127.0.0.1", 0), rc.Handler)
        srv.root, srv.allow = str(self.src), set()
        srv.files = sorted(rc.SMALL_FILES)
        srv.shards = set(self.shards)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        addr = f"127.0.0.1:{srv.server_address[1]}"
        code, out = self.tool("fetch", "--server", addr, "--ranges", self.ranges, "--dest", self.dest)
        self.assertEqual(code, 0, out)
        code, out = self.tool("check", "--server", addr, "--ranges", self.ranges, "--dest", self.dest, "--expect-missing", "model.visual.", "--expect-missing", "mtp.safetensors")
        self.assertEqual(code, 0, out)
        self.assertIn("check: 3 of 3 held runs", out)


if __name__ == "__main__":
    unittest.main()
