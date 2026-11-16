#!/usr/bin/env python3
"""One rank's Inkling-2 checkpoint bytes, copied over the network into a folder the runner loads unchanged.

The destination holds every shard (and mtp.safetensors) at its full original size as a sparse file: the header, then
only the byte ranges the rank loads, at their original offsets, with holes elsewhere. The ranges come from
`tf-inkling2-plan --dir MODEL --world W --ranges R` (plan.zig, the same cut the runner's loader reads), filtered to
the tensors the runner loads (`model.llm.*` by default). The small files (config, quantization manifest, index,
tokenizer, template) are copied whole.

  serve  --root MODEL --port P --allow IP[,IP...]      on the host that holds the checkpoint (read-only)
  bench  --server HOST:P --gib N [--jobs J]             link speed from the server, nothing written
  fetch  --server HOST:P | --http URL  --ranges RANK.tsv --dest DIR [--jobs J] [--prefix model.llm. ...]
                                                        --http: range requests on URL/<file> (a Hugging Face
                                                        resolve/<revision> URL); strided runs fetch their span
  check  --server HOST:P | --http URL  --ranges RANK.tsv --dest DIR [--max-gib G]
                                                        coverage (below), every shard header and small file against
                                                        the source's, and each held run's digest against the
                                                        source's disk (--server) or its bytes read again (--http;
                                                        --max-gib: a fixed pseudo-random sample)
  verify --ranges RANK.tsv --dest DIR                   coverage, the headers against those recorded at fetch, and
                                                        every held run re-read from disk (page cache dropped
                                                        first) against its sha256 recorded at fetch
  plan   --ranges RANK.tsv [--prefix ...]               the runs and bytes fetch would copy

fetch from a server hashes each run as the server reads it from the source and again as it arrives (from --http,
only as it arrives; `check` compares with the source later); verify hashes it again from the destination's disk. A
run is recorded (DIR/.rank_copy.json) only once its bytes are written in full and synced; a rerun of fetch skips
recorded runs, except in a shard made anew or resized, or where the file has a hole. Coverage (verify and check):
every run of the ranges file (all of it, or under --prefix) must be recorded; --expect-missing PREFIX reports a
group's missing runs without failing.
Standard library only; no paths or host names are built in.
"""
import argparse
import errno
import hashlib
import json
import os
import re
import socket
import socketserver
import struct
import sys
import threading
import time

CHUNK = 16 << 20
STATE = ".rank_copy.json"


def read_line(f):
    line = f.readline()
    if not line:
        raise EOFError("connection closed")
    return json.loads(line)


def send_json(sock, obj):
    sock.sendall((json.dumps(obj) + "\n").encode())


# ---------------------------------------------------------------- server

class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        ip = self.client_address[0]
        if self.server.allow and ip not in self.server.allow:
            return
        root = self.server.root
        while True:
            try:
                req = read_line(self.rfile)
            except (EOFError, ConnectionError, ValueError):
                return
            op = req.get("op")
            try:
                if op == "list":
                    send_json(self.request, {"files": self.server.files})
                elif op == "file":
                    name = req["file"]
                    if name not in self.server.files:
                        raise ValueError(f"not served: {name}")
                    data = open(os.path.join(root, name), "rb").read()
                    send_json(self.request, {"size": len(data)})
                    self.request.sendall(data)
                elif op == "stat":
                    name = req["file"]
                    if name not in self.server.shards:
                        raise ValueError(f"not a served shard: {name}")
                    with open(os.path.join(root, name), "rb") as f:
                        n = struct.unpack("<Q", f.read(8))[0]
                        f.seek(0)
                        head = f.read(8 + n)
                        size = os.fstat(f.fileno()).st_size
                    send_json(self.request, {"size": size, "header": len(head)})
                    self.request.sendall(head)
                elif op == "run":
                    name = req["file"]
                    if name not in self.server.shards:
                        raise ValueError(f"not a served shard: {name}")
                    off, ln, stride, count = (int(req[k]) for k in ("offset", "len", "stride", "count"))
                    send_json(self.request, {"bytes": ln * count})
                    h = hashlib.sha256()
                    with open(os.path.join(root, name), "rb", buffering=0) as f:
                        fd = f.fileno()
                        if count == 1:
                            done = 0
                            while done < ln:
                                b = os.pread(fd, min(CHUNK, ln - done), off + done)
                                if not b:
                                    raise IOError("short read")
                                h.update(b)
                                self.request.sendall(b)
                                done += len(b)
                        else:
                            per = max(1, CHUNK // ln)
                            k = 0
                            while k < count:
                                n = min(per, count - k)
                                buf = bytearray()
                                for j in range(n):
                                    b = os.pread(fd, ln, off + (k + j) * stride)
                                    if len(b) != ln:
                                        raise IOError("short read")
                                    buf += b
                                h.update(buf)
                                self.request.sendall(buf)
                                k += n
                    self.request.sendall(h.digest())
                elif op == "sha":
                    name = req["file"]
                    if name not in self.server.shards:
                        raise ValueError(f"not a served shard: {name}")
                    off, ln, stride, count = (int(req[k]) for k in ("offset", "len", "stride", "count"))
                    h = hashlib.sha256()
                    with open(os.path.join(root, name), "rb", buffering=0) as f:
                        fd = f.fileno()
                        if count == 1:
                            done = 0
                            while done < ln:
                                b = os.pread(fd, min(CHUNK, ln - done), off + done)
                                if not b:
                                    raise IOError("short read")
                                h.update(b)
                                done += len(b)
                        else:
                            for k in range(count):
                                h.update(os.pread(fd, ln, off + k * stride))
                    send_json(self.request, {"sha256": h.hexdigest()})
                elif op == "bench":
                    n = int(req["len"])
                    send_json(self.request, {"bytes": n})
                    z = bytes(CHUNK)
                    done = 0
                    while done < n:
                        k = min(CHUNK, n - done)
                        self.request.sendall(z[:k])
                        done += k
                else:
                    raise ValueError(f"unknown op {op}")
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as e:  # report and keep serving
                try:
                    send_json(self.request, {"error": str(e)})
                except OSError:
                    return


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(a):
    root = os.path.abspath(a.root)
    files, shards = [], []
    for d, _, names in os.walk(root):
        for n in names:
            rel = os.path.relpath(os.path.join(d, n), root)
            if rel.startswith(".") or os.path.islink(os.path.join(d, n)):
                continue
            (shards if rel.endswith(".safetensors") else files).append(rel)
    srv = Server((a.bind, a.port), Handler)
    srv.root = root
    srv.files = sorted(files)
    srv.shards = set(shards)
    srv.allow = set(x for x in (a.allow or "").split(",") if x)
    print(f"serving {len(shards)} shards and {len(files)} files from {root} on {a.bind}:{a.port}; allow {sorted(srv.allow) or 'any'}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------- client

def connect(server):
    host, port = server.rsplit(":", 1)
    s = socket.create_connection((host, int(port)), timeout=600)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    return s, s.makefile("rb", buffering=1 << 20)


def recv_exact(f, n):
    b = f.read(n)
    if b is None or len(b) != n:
        raise EOFError(f"short read: {0 if b is None else len(b)} of {n}")
    return b


def reply(f):
    r = read_line(f)
    if "error" in r:
        raise RuntimeError(r["error"])
    return r


def load_runs(path, prefixes):
    runs = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        file, off, ln, stride, count, placement, tensor = line.rstrip("\n").split("\t")
        if prefixes and not any(tensor.startswith(p) for p in prefixes):
            continue
        runs.append({"file": file, "offset": int(off), "len": int(ln), "stride": int(stride), "count": int(count), "tensor": tensor, "placement": placement})
    runs.sort(key=lambda r: (r["file"], r["offset"]))
    return runs


def run_key(r):
    return f'{r["file"]}:{r["offset"]}:{r["len"]}:{r["stride"]}:{r["count"]}'


def plan(a):
    runs = load_runs(a.ranges, a.prefix)
    total = sum(r["len"] * r["count"] for r in runs)
    by = {}
    for r in runs:
        by[r["placement"]] = by.get(r["placement"], 0) + r["len"] * r["count"]
    print(f"{len(runs)} runs, {total / 2**30:.2f} GiB: " + ", ".join(f"{k} {v / 2**30:.2f} GiB" for k, v in sorted(by.items())))


def bench(a):
    n = int(a.gib * 2**30)
    per = n // a.jobs
    done = [0] * a.jobs

    def one(i):
        s, f = connect(a.server)
        send_json(s, {"op": "bench", "len": per})
        reply(f)
        left = per
        while left:
            b = f.read(min(CHUNK, left))
            if not b:
                raise EOFError
            left -= len(b)
            done[i] += len(b)
        s.close()

    t = time.time()
    th = [threading.Thread(target=one, args=(i,)) for i in range(a.jobs)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    dt = time.time() - t
    print(f"bench: {sum(done) / 2**30:.2f} GiB in {dt:.1f} s = {sum(done) / dt / 1e6:.0f} MB/s ({a.jobs} connections)")


SMALL_FILES = ["config.json", "hf_quant_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "chat_template.jinja", "processor_config.json", "tiktoken/tokenizer.model"]

# The calls that write and sync the destination (tests replace them to inject faults).
_pwrite = os.pwrite
_fsync = getattr(os, "fdatasync", os.fsync)


def pwrite_all(fd, data, off):
    """All of `data` at `off`: a short write is continued, and a write that makes no progress is an error."""
    mv = memoryview(data)
    while len(mv):
        n = _pwrite(fd, mv, off)
        if n <= 0:
            raise OSError(errno.ENOSPC, f"short write at {off}: {len(mv)} bytes left")
        mv = mv[n:]
        off += n


def read_exact(resp, n, what):
    """Exactly n bytes from a response (several reads if need be)."""
    b = bytearray()
    while len(b) < n:
        x = resp.read(n - len(b))
        if not x:
            raise EOFError(f"{what}: short response, {len(b)} of {n} bytes")
        b += x
    return bytes(b)


def check_header(name, size, head):
    """A safetensors header as read: its 8-byte length, then that many bytes, within the file's size."""
    if len(head) < 8:
        raise EOFError(f"{name}: header of {len(head)} bytes")
    n = struct.unpack("<Q", head[:8])[0]
    if len(head) != 8 + n or 8 + n > size:
        raise EOFError(f"{name}: header of {len(head)} bytes, its length field says {8 + n} (file {size} bytes)")
    return head


class TcpSource:
    """The `serve` side: it hashes each run as it reads it from the source's disk."""

    def __init__(self, server):
        self.server = server
        self.s, self.f = connect(server)

    def clone(self):
        return TcpSource(self.server)

    def close(self):
        self.s.close()

    def files(self):
        send_json(self.s, {"op": "list"})
        return reply(self.f)["files"]

    def file(self, name):
        send_json(self.s, {"op": "file", "file": name})
        return recv_exact(self.f, reply(self.f)["size"])

    def stat(self, name):
        send_json(self.s, {"op": "stat", "file": name})
        st = reply(self.f)
        return st["size"], check_header(name, st["size"], recv_exact(self.f, st["header"]))

    def run(self, r, sink):
        """sink(piece_index_or_None, bytes): count == 1 streams pieces in order; else whole rows. Returns the source digest."""
        send_json(self.s, {"op": "run", **{k: r[k] for k in ("file", "offset", "len", "stride", "count")}})
        n = reply(self.f)["bytes"]
        ln, count = r["len"], r["count"]
        if n != ln * count:
            raise IOError(f"{r['tensor']}: the server offers {n} bytes, the run is {ln * count}")
        if count == 1:
            done = 0
            while done < n:
                b = recv_exact(self.f, min(CHUNK, n - done))
                sink(done, b)
                done += len(b)
        else:
            per = max(1, CHUNK // ln)
            k = 0
            while k < count:
                m = min(per, count - k)
                b = recv_exact(self.f, m * ln)
                sink(k, b)
                k += m
        return recv_exact(self.f, 32)

    def sha(self, r):
        """The source's sha256 (hex) of the run, read from its disk by the server."""
        send_json(self.s, {"op": "sha", **{k: r[k] for k in ("file", "offset", "len", "stride", "count")}})
        return reply(self.f)["sha256"]


HTTP_PAUSE = 2.0  # seconds before a retry (more on each later one)
CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")


class HttpSource:
    """HTTP range requests on `<base>/<file>` (a Hugging Face `resolve/<revision>` URL): no source-side digest."""

    def __init__(self, base, tries=6, pause=None):
        self.base = base.rstrip("/")
        self.tries = tries
        self.pause = HTTP_PAUSE if pause is None else pause

    def clone(self):
        return HttpSource(self.base, self.tries, self.pause)

    def close(self):
        pass

    def _get(self, name, lo, hi):
        """The response for bytes [lo, hi) of `name` (the whole file when hi is None), its range checked."""
        import urllib.request
        req = urllib.request.Request(f"{self.base}/{name}", headers={"Range": f"bytes={lo}-{hi - 1}"} if hi is not None else {})
        for attempt in range(self.tries):
            try:
                resp = urllib.request.urlopen(req, timeout=120)
                if hi is not None:
                    if resp.status != 206:
                        raise IOError(f"{name}: status {resp.status} for a range")
                    m = CONTENT_RANGE.fullmatch(resp.headers.get("Content-Range", "").strip())
                    if not m or int(m.group(1)) != lo or int(m.group(2)) != hi - 1:
                        raise IOError(f"{name}: asked for bytes {lo}-{hi - 1}, Content-Range says {resp.headers.get('Content-Range')!r}")
                    length = resp.headers.get("Content-Length")
                    if length is not None and int(length) != hi - lo:
                        raise IOError(f"{name}: asked for {hi - lo} bytes, Content-Length says {length}")
                return resp
            except Exception:
                if attempt == self.tries - 1:
                    raise
                time.sleep(self.pause * (1 + 2 * attempt))

    def files(self):
        return SMALL_FILES

    def file(self, name):
        resp = self._get(name, 0, None)
        length = resp.headers.get("Content-Length")
        data = resp.read()
        if length is not None and len(data) != int(length):
            raise EOFError(f"{name}: {len(data)} of {length} bytes")
        return data

    def stat(self, name):
        resp = self._get(name, 0, 8)
        size = int(CONTENT_RANGE.fullmatch(resp.headers["Content-Range"].strip()).group(3))
        n = struct.unpack("<Q", read_exact(resp, 8, f"{name}'s header length"))[0]
        if 8 + n > size:
            raise EOFError(f"{name}: header length {n} past the file's {size} bytes")
        return size, check_header(name, size, read_exact(self._get(name, 0, 8 + n), 8 + n, f"{name}'s header"))

    def run(self, r, sink):
        ln, stride, count = r["len"], r["stride"], r["count"]
        span = (count - 1) * stride + ln
        for attempt in range(4):
            try:
                resp = self._get(r["file"], r["offset"], r["offset"] + span)
                if count == 1:
                    done = 0
                    while done < span:
                        b = resp.read(min(CHUNK, span - done))
                        if not b:
                            raise EOFError(f"{r['tensor']}: short body")
                        sink(done, b)
                        done += len(b)
                else:  # the whole strided span, keeping each row's first `ln` bytes
                    k = 0
                    per = max(1, CHUNK // stride)
                    while k < count:
                        m = min(per, count - k)
                        want = (m - 1) * stride + ln if k + m == count else m * stride
                        b = read_exact(resp, want, f"{r['tensor']}'s body")
                        sink(k, b"".join(bytes(b[j * stride:j * stride + ln]) for j in range(m)))
                        k += m
                return None
            except (EOFError, ConnectionError, TimeoutError, OSError):
                if attempt == 3:
                    raise
                sink(None, None)  # restart the run
                time.sleep(self.pause * 2.5)

    def sha(self, r):
        """sha256 (hex) of the run's bytes as read over HTTP now (rows in order, as fetch hashes them)."""
        h = [hashlib.sha256()]

        def sink(at, b):
            if b is None:
                h[0] = hashlib.sha256()
            else:
                h[0].update(b)

        self.run(r, sink)
        return h[0].hexdigest()


def source(a):
    return (HttpSource(a.http), "http") if getattr(a, "http", None) else (TcpSource(a.server), "server")


def parse_key(key):
    name, off, ln, stride, count = key.rsplit(":", 4)
    return {"file": name, "offset": int(off), "len": int(ln), "stride": int(stride), "count": int(count), "tensor": key}


def in_data(fd, off):
    """Whether byte `off` of the file holds data (not a hole, not past the end); True when the system cannot say."""
    try:
        return os.lseek(fd, off, os.SEEK_DATA) == off
    except AttributeError:
        return True
    except OSError as e:
        if e.errno == errno.ENXIO:
            return False
        return True


def no_hole(fd, lo, hi):
    """Bytes [lo, hi) of the file are all data: no hole starts inside them (True when the system cannot say)."""
    try:
        return os.lseek(fd, lo, os.SEEK_HOLE) >= hi
    except AttributeError:
        return True
    except OSError as e:
        if e.errno == errno.ENXIO:
            return False
        return True


def held(fd, r):
    """Every row of the run is data from its first byte to its last: no hole anywhere inside it (a recreated,
    truncated or punched shard has holes there)."""
    for k in range(r["count"]):
        lo = r["offset"] + k * r["stride"]
        if not in_data(fd, lo) or not no_hole(fd, lo, lo + r["len"]):
            return False
    return True


def save_state(state, path):
    with open(path + ".tmp", "w") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)


def fetch(a):
    dest = os.path.abspath(a.dest)
    os.makedirs(dest, exist_ok=True)
    runs = load_runs(a.ranges, a.prefix)
    state_path = os.path.join(dest, STATE)
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"runs": {}}
    state.setdefault("sources", {})
    state.setdefault("files", {})
    state.setdefault("small", {})
    lock = threading.Lock()
    src, tag = source(a)
    for name in src.files():
        data = src.file(name)
        p = os.path.join(dest, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if os.path.exists(p) and open(p, "rb").read() != data:
            print(f"FAIL fetch: {name} differs from the copy already there", flush=True)
            return 1
        with open(p, "wb") as o:
            o.write(data)
        state["small"][name] = hashlib.sha256(data).hexdigest()
    index = json.load(open(os.path.join(dest, "model.safetensors.index.json")))
    shards = sorted(set(index["weight_map"].values()))
    if a.mtp:
        shards.append("mtp.safetensors")
    dropped = 0
    for name in shards:
        size, head = src.stat(name)
        p = os.path.join(dest, name)
        existed = os.path.exists(p)
        fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            st = os.fstat(fd)
            if st.st_size == size and os.pread(fd, len(head), 0) not in (head, bytes(len(head))):
                print(f"FAIL fetch: {name}'s header differs from the copy already there", flush=True)
                return 1
            rec = state["files"].get(name)
            # a shard made or resized now, or not the file the state was recorded on, holds none of the recorded runs
            fresh = not existed or st.st_size != size or (rec is not None and (rec.get("ino"), rec.get("dev"), rec.get("size")) != (st.st_ino, st.st_dev, size))
            if st.st_size != size:
                os.ftruncate(fd, size)
            for key in [k for k in state["runs"] if parse_key(k)["file"] == name]:
                if fresh or not held(fd, parse_key(key)):
                    del state["runs"][key]
                    state["sources"].pop(key, None)
                    dropped += 1
            pwrite_all(fd, head, 0)
            _fsync(fd)
            st = os.fstat(fd)
            state["files"][name] = {"size": size, "header_len": len(head), "header_sha256": hashlib.sha256(head).hexdigest(), "ino": st.st_ino, "dev": st.st_dev}
        finally:
            os.close(fd)
    save_state(state, state_path)
    print(f"{len(src.files())} small files, {len(shards)} shard headers at full size (source: {tag})" + (f"; {dropped} recorded runs dropped (their shard was made anew, resized or has holes there)" if dropped else ""), flush=True)
    src.close()
    todo = [r for r in runs if run_key(r) not in state["runs"]]
    total = sum(r["len"] * r["count"] for r in todo)
    print(f"{len(todo)} of {len(runs)} runs to copy, {total / 2**30:.2f} GiB", flush=True)
    queue = list(reversed(todo))
    moved = [0]
    t0 = time.time()
    errors = []

    def worker():
        mine = None
        fds = {}
        try:
            mine = src.clone()
            while True:
                with lock:
                    if not queue or errors:
                        return
                    r = queue.pop()
                name = r["file"]
                if name not in fds:
                    fds[name] = os.open(os.path.join(dest, name), os.O_RDWR)
                fd = fds[name]
                ln, stride = r["len"], r["stride"]
                h = [hashlib.sha256()]

                def sink(at, b):
                    if b is None:
                        h[0] = hashlib.sha256()
                        return
                    h[0].update(b)
                    if r["count"] == 1:
                        pwrite_all(fd, b, r["offset"] + at)
                    else:
                        mv = memoryview(b)
                        for j in range(len(b) // ln):
                            pwrite_all(fd, mv[j * ln:(j + 1) * ln], r["offset"] + (at + j) * stride)

                digest = mine.run(r, sink)
                if digest is not None and digest != h[0].digest():
                    raise RuntimeError(f"{r['tensor']}: received bytes do not hash as the source's")
                _fsync(fd)  # a run is recorded only once its bytes are on disk
                with lock:
                    state["runs"][run_key(r)] = h[0].hexdigest()
                    state["sources"][run_key(r)] = tag
                    moved[0] += r["len"] * r["count"]
        except BaseException as e:
            with lock:
                errors.append(e)
        finally:
            for fd in fds.values():
                try:
                    os.close(fd)
                except OSError as e:
                    with lock:
                        errors.append(e)
            if mine is not None:
                try:
                    mine.close()
                except OSError:
                    pass

    th = [threading.Thread(target=worker) for _ in range(a.jobs)]
    for x in th:
        x.start()
    last = time.time()
    while any(x.is_alive() for x in th):
        time.sleep(0.2)
        if time.time() - last > 30:
            last = time.time()
            with lock:
                save_state(state, state_path)
            dt = time.time() - t0
            print(f"  {moved[0] / 2**30:.2f} of {total / 2**30:.2f} GiB in {dt:.0f} s ({moved[0] / max(dt, 1e-9) / 1e6:.0f} MB/s)", flush=True)
    for x in th:
        x.join()
    save_state(state, state_path)
    dt = time.time() - t0
    left = [r for r in todo if run_key(r) not in state["runs"]]
    if errors:
        print(f"FAIL fetch: {errors[0]!r}; {len(left)} runs not copied", flush=True)
        return 1
    if left:
        print(f"FAIL fetch: {len(left)} runs not copied (first: {left[0]['tensor']})", flush=True)
        return 1
    print(f"fetch: {moved[0] / 2**30:.2f} GiB in {dt:.0f} s ({moved[0] / max(dt, 1e-9) / 1e6:.0f} MB/s); {len(state['runs'])} runs held", flush=True)
    return 0


def group_of(r):
    """A run's group for reports: its file for the drafter, else the tensor name's first two parts."""
    if r["file"] == "mtp.safetensors":
        return "mtp.safetensors"
    return ".".join(r["tensor"].split(".")[:2]) + ".*"


def coverage(state, a):
    """Every run of the ranges file (under --prefix when given) must be in the state. Returns (ok, the covered runs)."""
    runs = load_runs(a.ranges, a.prefix)
    have = [r for r in runs if run_key(r) in state["runs"]]
    missing = [r for r in runs if run_key(r) not in state["runs"]]
    groups = {}
    for r in missing:
        expected = any(r["tensor"].startswith(p) or r["file"] == p for p in (a.expect_missing or []))
        g = groups.setdefault((group_of(r), expected), [0, 0])
        g[0] += 1
        g[1] += r["len"] * r["count"]
    bad = sum(n for (_, expected), (n, _) in groups.items() if not expected)
    planned = {run_key(r) for r in runs}
    extra = 0 if a.prefix else sum(1 for k in state["runs"] if k not in planned)
    print(f"coverage: {len(have)} of {len(runs)} runs of {os.path.basename(a.ranges)}{' under ' + ','.join(a.prefix) if a.prefix else ''} held"
          + (f", {extra} held runs not in it" if extra else ""))
    for (g, expected), (n, b) in sorted(groups.items()):
        print(f"  missing {g}: {n} runs, {b / 2**30:.2f} GiB{' (expected: --expect-missing)' if expected else ''}")
    return bad == 0 and extra == 0, have


def shards_of(dest, state, runs):
    """The shards a rank folder must hold: the index's, mtp.safetensors when there, any the state or the runs name."""
    names = set(state.get("files", {})) | {r["file"] for r in runs}
    try:
        names |= set(json.load(open(os.path.join(dest, "model.safetensors.index.json")))["weight_map"].values())
    except (OSError, ValueError, KeyError):
        pass
    if os.path.exists(os.path.join(dest, "mtp.safetensors")):
        names.add("mtp.safetensors")
    return sorted(names)


def disk_header(path):
    """(size, header bytes) of a shard on disk; the header is the 8-byte length and what it counts."""
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        b = f.read(8)
        if len(b) != 8:
            return size, b
        n = struct.unpack("<Q", b)[0]
        if 8 + n > size:
            return size, b
        return size, b + f.read(n)


def structural(name, size, head, runs):
    """Without a recorded header: it parses, its tensors end within the file, and each run lies inside a tensor of it."""
    try:
        check_header(name, size, head)
        meta = json.loads(head[8:])
    except (EOFError, ValueError) as e:
        return str(e)
    base = len(head)
    spans = []
    for k, v in meta.items():
        if k == "__metadata__":
            continue
        lo, hi = v["data_offsets"]
        if base + hi > size:
            return f"{k} ends past the file"
        spans.append((base + lo, base + hi))
    spans.sort()
    import bisect
    starts = [s for s, _ in spans]
    for r in runs:
        end = r["offset"] + (r["count"] - 1) * r["stride"] + r["len"]
        i = bisect.bisect_right(starts, r["offset"]) - 1
        if i < 0 or end > spans[i][1]:
            return f"{r['tensor']}'s run is not inside one of its tensors"
    return None


def verify(a):
    dest = os.path.abspath(a.dest)
    state = json.load(open(os.path.join(dest, STATE)))
    ok, have = coverage(state, a)
    planned = load_runs(a.ranges, None)
    # headers: against what fetch recorded, else a structural check against the plan's runs
    recorded = state.get("files", {})
    hbad = 0
    unrec = []
    for name in shards_of(dest, state, planned):
        p = os.path.join(dest, name)
        if not os.path.exists(p):
            print(f"HEADER {name}: no such file")
            hbad += 1
            continue
        size, head = disk_header(p)
        rec = recorded.get(name)
        if rec is not None:
            if size != rec["size"] or len(head) != rec["header_len"] or hashlib.sha256(head).hexdigest() != rec["header_sha256"]:
                print(f"HEADER {name}: differs from the header recorded at fetch (size {size}, header {len(head)} bytes)")
                hbad += 1
        else:
            unrec.append(name)
            why = structural(name, size, head, [r for r in planned if r["file"] == name])
            if why:
                print(f"HEADER {name}: {why}")
                hbad += 1
    for name, want in sorted(state.get("small", {}).items()):
        try:
            got = hashlib.sha256(open(os.path.join(dest, name), "rb").read()).hexdigest()
        except OSError:
            got = None
        if got != want:
            print(f"FILE {name}: differs from the copy recorded at fetch")
            hbad += 1
    nshards = len(shards_of(dest, state, planned))
    print(f"headers: {nshards - len(unrec)} of {nshards} against the headers recorded at fetch"
          + (f", {len(unrec)} not recorded (fetched by an earlier tool): parsed and checked to hold the plan's runs" if unrec else "")
          + f": {'OK' if hbad == 0 else f'{hbad} BAD'}")
    # runs: every recorded run re-read from disk, page cache dropped first
    bad = 0
    total = 0
    t0 = time.time()
    fds = {}
    keys = sorted(state["runs"])
    for key in keys:
        want = state["runs"][key]
        r = parse_key(key)
        name, off, ln, stride, count = r["file"], r["offset"], r["len"], r["stride"], r["count"]
        if name not in fds:
            try:
                fds[name] = os.open(os.path.join(dest, name), os.O_RDONLY)
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(fds[name], 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                fds[name] = None
        fd = fds[name]
        short = fd is None
        h = hashlib.sha256()
        if not short and count == 1:
            done = 0
            while done < ln:
                b = os.pread(fd, min(CHUNK, ln - done), off + done)
                if not b:
                    short = True
                    break
                h.update(b)
                done += len(b)
        elif not short:
            for k in range(count):
                b = os.pread(fd, ln, off + k * stride)
                if len(b) != ln:
                    short = True
                    break
                h.update(b)
        total += ln * count
        if short:
            bad += 1
            print(f"SHORT {key}: the file ends (or is missing) before the run does")
        elif h.hexdigest() != want:
            bad += 1
            print(f"MISMATCH {key}")
    for fd in fds.values():
        if fd is not None:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
    dt = time.time() - t0
    print(f"verify: {len(keys)} runs, {total / 2**30:.2f} GiB re-read in {dt:.0f} s: {'OK' if bad == 0 else f'{bad} MISMATCH'}")
    good = ok and hbad == 0 and bad == 0
    print(f"verify: {'OK' if good else 'FAIL'} (coverage {'OK' if ok else 'INCOMPLETE'}, headers {'OK' if hbad == 0 else 'BAD'}, runs {'OK' if bad == 0 else 'BAD'})")
    return 0 if good else 1


def sample(runs, max_gib):
    """Runs in a fixed pseudo-random order (by their key's sha256), taken while they fit in max_gib (None: all)."""
    if max_gib is None:
        return runs
    budget = int(max_gib * 2**30)
    out, used = [], 0
    for r in sorted(runs, key=lambda r: hashlib.sha256(run_key(r).encode()).digest()):
        b = r["len"] * r["count"]
        if used + b <= budget:
            out.append(r)
            used += b
    return sorted(out, key=lambda r: (r["file"], r["offset"]))


def check(a):
    """The rank folder against the source: coverage, every shard header and small file, and the held runs' digests
    (the server's sha256 of each run read from its disk; over HTTP, the run's bytes read again and hashed)."""
    dest = os.path.abspath(a.dest)
    state = json.load(open(os.path.join(dest, STATE)))
    ok, have = coverage(state, a)
    planned = load_runs(a.ranges, None)
    src, tag = source(a)
    hbad = 0
    names = shards_of(dest, state, planned)
    for name in names:
        size, head = src.stat(name)
        p = os.path.join(dest, name)
        if not os.path.exists(p):
            print(f"HEADER {name}: no such file")
            hbad += 1
            continue
        dsize, dhead = disk_header(p)
        if dsize != size or dhead != head:
            print(f"HEADER {name}: differs from the source's (size {dsize} against {size}, header {len(dhead)} against {len(head)} bytes)")
            hbad += 1
    small = [n for n in src.files() if os.path.exists(os.path.join(dest, n))]
    for name in small:
        if open(os.path.join(dest, name), "rb").read() != src.file(name):
            print(f"FILE {name}: differs from the source's")
            hbad += 1
    src.close()
    print(f"headers: {len(names)} shard headers and {len(small)} small files against the source ({tag}): {'OK' if hbad == 0 else f'{hbad} BAD'}")
    todo = sample(have, a.max_gib)
    want_bytes = sum(r["len"] * r["count"] for r in todo)
    have_bytes = sum(r["len"] * r["count"] for r in have)
    bad, done, errors = [], [0, 0], []
    lock = threading.Lock()
    queue = list(reversed(todo))

    def worker():
        mine = None
        try:
            mine = src.clone()
            while True:
                with lock:
                    if not queue or errors:
                        return
                    r = queue.pop()
                got = mine.sha(r)
                with lock:
                    done[0] += 1
                    done[1] += r["len"] * r["count"]
                    if got != state["runs"][run_key(r)]:
                        bad.append(run_key(r))
        except BaseException as e:
            with lock:
                errors.append(e)
        finally:
            if mine is not None:
                try:
                    mine.close()
                except OSError:
                    pass

    t0 = time.time()
    th = [threading.Thread(target=worker) for _ in range(a.jobs)]
    for x in th:
        x.start()
    last = time.time()
    while any(x.is_alive() for x in th):
        time.sleep(0.2)
        if time.time() - last > 60:
            last = time.time()
            dt = time.time() - t0
            print(f"  {done[1] / 2**30:.2f} of {want_bytes / 2**30:.2f} GiB in {dt:.0f} s ({done[1] / max(dt, 1e-9) / 1e6:.0f} MB/s)", flush=True)
    for x in th:
        x.join()
    for k in bad:
        print(f"MISMATCH {k}")
    if errors:
        print(f"ERROR {errors[0]!r}")
    runs_ok = not bad and not errors and done[0] == len(todo)
    dt = time.time() - t0
    print(f"check: {done[0]} of {len(have)} held runs ({done[1] / 2**30:.2f} of {have_bytes / 2**30:.2f} GiB"
          + (f", a sample of at most {a.max_gib:g} GiB" if a.max_gib is not None else "") + f") against the source ({tag}) in {dt:.0f} s"
          + (f" ({done[1] / max(dt, 1e-9) / 1e6:.0f} MB/s)" if dt > 0 else "") + f": {'OK' if runs_ok else f'{len(bad)} MISMATCH'}")
    good = ok and hbad == 0 and runs_ok
    print(f"check: {'OK' if good else 'FAIL'} (coverage {'OK' if ok else 'INCOMPLETE'}, headers {'OK' if hbad == 0 else 'BAD'}, runs {'OK' if runs_ok else 'BAD'})")
    return 0 if good else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--root", required=True)
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--bind", default="0.0.0.0")
    s.add_argument("--allow", default="")
    b = sub.add_parser("bench")
    b.add_argument("--server", required=True)
    b.add_argument("--gib", type=float, default=4)
    b.add_argument("--jobs", type=int, default=4)
    f = sub.add_parser("fetch")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--server", help="HOST:PORT of `serve`")
    g.add_argument("--http", help="base URL of the files, e.g. a Hugging Face resolve/<revision> URL (HTTP ranges)")
    f.add_argument("--ranges", required=True)
    f.add_argument("--dest", required=True)
    f.add_argument("--jobs", type=int, default=4)
    f.add_argument("--prefix", action="append", default=None, help="tensor-name prefixes to copy (default model.llm.)")
    f.add_argument("--no-mtp", dest="mtp", action="store_false", help="leave mtp.safetensors out (its header too)")
    cover = argparse.ArgumentParser(add_help=False)
    cover.add_argument("--ranges", required=True, help="the rank's ranges file: every run in it must be held")
    cover.add_argument("--prefix", action="append", default=None, help="only the ranges file's tensors under these prefixes (default: all)")
    cover.add_argument("--expect-missing", action="append", default=None, help="a tensor prefix or file whose missing runs are reported but not failed")
    v = sub.add_parser("verify", parents=[cover])
    v.add_argument("--dest", required=True)
    c = sub.add_parser("check", parents=[cover])
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument("--server", help="HOST:PORT of `serve`")
    g.add_argument("--http", help="base URL of the files (HTTP ranges); the runs' bytes are read again and hashed")
    c.add_argument("--dest", required=True)
    c.add_argument("--jobs", type=int, default=4)
    c.add_argument("--max-gib", type=float, default=None, help="check a fixed pseudo-random sample of the held runs, at most this many GiB")
    q = sub.add_parser("plan")
    q.add_argument("--ranges", required=True)
    q.add_argument("--prefix", action="append", default=None)
    a = p.parse_args(argv)
    if a.cmd in ("fetch", "plan") and a.prefix is None:
        a.prefix = ["model.llm."]
    return {"serve": serve, "bench": bench, "fetch": fetch, "verify": verify, "check": check, "plan": plan}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
