"""Inkling-2's vision tower as one NEW file in a rank folder: inkling2-vision.safetensors (--vision; owner decision D5).

The checkpoint keeps the tower's 422 tensors (model.visual.*, 655,386,624 bytes of bf16) as one contiguous span of
model-00024-of-00024.safetensors. This tool writes them, and nothing else, as a safetensors file of their own beside
a rank folder's shards: the shard's own header entries with offsets moved to the new file's data, then the span's
bytes. No existing file is opened for writing: one maker a folder (an exclusive lock, .lock) writes
inkling2-vision.safetensors.part, and publishes it only once its sha256 equals the recipe's pin (--sha256), by a link
that never replaces a file of that name, so a folder holds the whole file or none of it.

  make --http BASE --dest DIR --sha256 HEX     the pinned Hugging Face revision's shard 24, by range requests
  make --from CHECKPOINT_DIR --dest DIR        a full local checkpoint; without --sha256 it builds the .part
                                               and prints the sha256 to pin, and publishes nothing
  check --dest DIR --sha256 HEX                the file is there, whole, and has the pinned digest

The engine checks each tensor again when it loads them (the digest manifest's xxh64 for the tower's rank-0 lines).
Standard library only; run with python3 -I.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import struct
import sys
import time
import urllib.request

SHARD = "model-00024-of-00024.safetensors"
OUT = "inkling2-vision.safetensors"
PREFIX = "model.visual."
TENSORS = 422
BYTES = 655_386_624
CHUNK = 64 << 20


class Local:
    def __init__(self, folder: str):
        self.path = os.path.join(folder, SHARD)

    def read(self, lo: int, n: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(lo)
            b = f.read(n)
        if len(b) != n:
            raise IOError(f"{self.path}: {n} bytes at {lo}, got {len(b)}")
        return b


class Http:
    def __init__(self, base: str, tries: int = 6):
        self.url = f"{base.rstrip('/')}/{SHARD}"
        self.tries = tries

    def read(self, lo: int, n: int) -> bytes:
        for t in range(self.tries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{lo + n - 1}", "User-Agent": "tensorfold-inkling2"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise IOError(f"range request answered {r.status}")
                    b = r.read()
                if len(b) != n:
                    raise IOError(f"{n} bytes at {lo}, got {len(b)}")
                return b
            except Exception as e:  # network: retry with backoff, then give up with the reason
                if t + 1 == self.tries:
                    raise IOError(f"{self.url} bytes {lo}-{lo + n - 1}: {e}") from e
                time.sleep(min(60, 2 ** t))
        raise AssertionError("unreachable")


def tower(src) -> tuple[bytes, int, int]:
    """The new file's header bytes, and the span (start, length) of the tower in the shard's data."""
    (n,) = struct.unpack("<Q", src.read(0, 8))
    if n > 100 << 20:
        raise ValueError(f"{SHARD}: header of {n} bytes")
    head = json.loads(src.read(8, n))
    mine = {k: v for k, v in head.items() if k.startswith(PREFIX)}
    if len(mine) != TENSORS:
        raise ValueError(f"{SHARD}: {len(mine)} tensors under {PREFIX}, expected {TENSORS}")
    spans = sorted((v["data_offsets"][0], v["data_offsets"][1], k) for k, v in mine.items())
    lo, hi = spans[0][0], spans[-1][1]
    at = lo
    for a, b, k in spans:  # contiguous, in order, nothing else inside the span
        if a != at or b < a:
            raise ValueError(f"{SHARD}: {k} at {a}, expected {at}: the tower is not one contiguous span")
        at = b
    for k, v in head.items():
        if not k.startswith(PREFIX) and k != "__metadata__" and v["data_offsets"][0] < hi and v["data_offsets"][1] > lo:
            raise ValueError(f"{SHARD}: {k} lies inside the tower's span")
    if hi - lo != BYTES:
        raise ValueError(f"{SHARD}: the tower spans {hi - lo} bytes, expected {BYTES}")
    out = {}
    if "__metadata__" in head:
        out["__metadata__"] = head["__metadata__"]
    for a, b, k in spans:
        v = dict(mine[k])
        v["data_offsets"] = [a - lo, b - lo]
        out[k] = v
    text = json.dumps(out, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)  # data starts 8-byte aligned, as safetensors writers pad
    return struct.pack("<Q", len(text)) + text, 8 + n + lo, hi - lo


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(CHUNK):
            h.update(b)
    return h.hexdigest()


def install(part: str, final: str) -> bool:
    """Publishes `part` as `final` only if no `final` exists: a hard link (refused when the name exists, so a file
    another process put there in the meantime is never replaced), then the staging name removed."""
    try:
        os.link(part, final)
    except FileExistsError:
        return False
    os.unlink(part)
    return True


def make(a) -> int:
    dest = os.path.abspath(a.dest)
    final = os.path.join(dest, OUT)
    if not a.sha256:
        # without the recipe's pin nothing is published: the file is built in staging and its sha256 printed
        print("no --sha256: the file is built as .part and its sha256 printed, but not published", file=sys.stderr)
    # one maker a folder: an exclusive lock on DEST/inkling2-vision.safetensors.lock, held to the end, so the
    # staging file has one writer (a second maker stops at once, its staging untouched)
    lock = os.open(final + ".lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{final}: another make holds {final}.lock; let it finish", file=sys.stderr)
            return 6
        return made(a, final)
    finally:
        os.close(lock)


def made(a, final: str) -> int:
    part = final + ".part"
    # a publish cut between its link and its unlink leaves .part as a second name of the published file: drop that
    # name (never write through it) and stage afresh
    if os.path.exists(part) and os.stat(part).st_nlink > 1:
        os.unlink(part)
    if os.path.exists(final):
        got = sha256_file(final)
        if a.sha256 and got == a.sha256:
            print(f"{final}: already there, sha256 {got}")
            return 0
        print(f"{final} exists with sha256 {got}{'' if not a.sha256 else ', not the pin ' + a.sha256}; move it away first", file=sys.stderr)
        return 3
    src = Http(a.http) if a.http else Local(a.src)
    head, start, length = tower(src)
    # resumable: a .part with the same header keeps its whole chunks
    have = 0
    if os.path.exists(part):
        with open(part, "rb") as f:
            same = f.read(len(head)) == head
        have = max(0, (os.path.getsize(part) - len(head)) // CHUNK * CHUNK) if same else 0
        if not same:
            os.remove(part)
    fd = os.open(part, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        if have == 0:
            os.ftruncate(fd, 0)
            os.pwrite(fd, head, 0)
        else:
            os.ftruncate(fd, len(head) + have)
            print(f"resuming at {have} of {length} bytes")
        t0 = time.time()
        off = have
        while off < length:
            n = min(CHUNK, length - off)
            b = src.read(start + off, n)
            if os.pwrite(fd, b, len(head) + off) != n:
                raise IOError("short write")
            off += n
            print(f"  {off >> 20} / {length >> 20} MiB ({time.time() - t0:.0f} s)", flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
    got = sha256_file(part)
    if not a.sha256:
        print(f"{part}: sha256 {got} (not published: pass it as --sha256 to publish)")
        return 5
    if got != a.sha256:
        print(f"{part}: sha256 {got}, the pin is {a.sha256}: kept for inspection, not published", file=sys.stderr)
        return 4
    if not install(part, final):
        print(f"{final} appeared while this make ran; left as it is, {part} kept", file=sys.stderr)
        return 3
    print(f"{final}: {TENSORS} tensors, {length} bytes of data, sha256 {got}")
    return 0


def check(a) -> int:
    final = os.path.join(os.path.abspath(a.dest), OUT)
    if not os.path.exists(final):
        print(f"{final}: missing", file=sys.stderr)
        return 3
    got = sha256_file(final)
    if got != a.sha256:
        print(f"{final}: sha256 {got}, the pin is {a.sha256}", file=sys.stderr)
        return 4
    with open(final, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        head = json.loads(f.read(n))
    names = [k for k in head if k != "__metadata__"]
    if len(names) != TENSORS or not all(k.startswith(PREFIX) for k in names):
        print(f"{final}: {len(names)} tensors, expected {TENSORS} under {PREFIX}", file=sys.stderr)
        return 4
    print(f"{final}: whole, {TENSORS} tensors, sha256 {got}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    g = m.add_mutually_exclusive_group(required=True)
    g.add_argument("--http", help="base URL of the pinned revision (…/resolve/REVISION)")
    g.add_argument("--from", dest="src", help="a full local checkpoint folder")
    m.add_argument("--dest", required=True)
    m.add_argument("--sha256")
    c = sub.add_parser("check")
    c.add_argument("--dest", required=True)
    c.add_argument("--sha256", required=True)
    a = ap.parse_args(argv)
    return make(a) if a.cmd == "make" else check(a)


if __name__ == "__main__":
    sys.exit(main())
