#!/usr/bin/env python3
"""Seed a new rank folder from another rank folder already on this host (for example an earlier layout's).

seed_rank_folder.py --old DIR --ranges NEW.tsv --dest DIR [--prefix model.llm.]

For every run of NEW.tsv whose rows all lie inside runs recorded (verified) in DIR/.rank_copy.json of --old, copy
the bytes from --old's shard at the same offsets into --dest's shard (created sparse at the source size, header
copied), and record the run in --dest/.rank_copy.json the way inkling2_rank_copy.py fetch does: the sha256 over the
run's rows in order, source tag "local", and the shard's size/header/inode in "files". fetch then skips those runs;
check --server compares each against the source checkpoint's own disk, check --http against the bytes on Hugging
Face, and the server's digest manifest checks every loaded byte at start-up.

Only use an --old folder whose runs were themselves checked (inkling2_rank_copy.py verify and check). Seed into a new,
empty --dest only: the tool refuses a --dest that already has a state file. Standard library only.
"""
import argparse
import bisect
import hashlib
import json
import os
import struct
import sys

STATE = ".rank_copy.json"


def load_runs(path, prefixes):
    runs = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        file, off, ln, stride, count, placement, tensor = line.rstrip("\n").split("\t")
        if prefixes and not any(tensor.startswith(p) for p in prefixes):
            continue
        runs.append({"file": file, "offset": int(off), "len": int(ln), "stride": int(stride), "count": int(count), "tensor": tensor})
    runs.sort(key=lambda r: (r["file"], r["offset"]))
    return runs


def run_key(r):
    return f'{r["file"]}:{r["offset"]}:{r["len"]}:{r["stride"]}:{r["count"]}'


def parse_key(key):
    name, off, ln, stride, count = key.rsplit(":", 4)
    return name, int(off), int(ln), int(stride), int(count)


def intervals(old_state):
    """Per file, merged [start, end) byte intervals the old folder holds (its recorded runs' rows)."""
    per = {}
    for key in old_state["runs"]:
        name, off, ln, stride, count = parse_key(key)
        lst = per.setdefault(name, [])
        for k in range(count):
            lst.append((off + k * stride, off + k * stride + ln))
    out = {}
    for name, lst in per.items():
        lst.sort()
        merged = []
        for s, e in lst:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out[name] = ([m[0] for m in merged], [m[1] for m in merged])
    return out


def inside(iv, name, s, e):
    if name not in iv:
        return False
    starts, ends = iv[name]
    i = bisect.bisect_right(starts, s) - 1
    return i >= 0 and ends[i] >= e


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(0)
        return f.read(8 + n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--ranges", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--prefix", action="append", default=None)
    a = ap.parse_args()
    prefixes = a.prefix if a.prefix is not None else ["model.llm."]
    old = json.load(open(os.path.join(a.old, STATE)))
    iv = intervals(old)
    runs = load_runs(a.ranges, prefixes)
    os.makedirs(a.dest, exist_ok=True)
    sp = os.path.join(a.dest, STATE)
    if os.path.exists(sp):
        print(f"FAIL: {sp} exists; seed only into a new folder", flush=True)
        return 1
    state = {"runs": {}, "sources": {}, "files": {}, "small": {}}
    seed = [r for r in runs if all(inside(iv, r["file"], r["offset"] + k * r["stride"], r["offset"] + k * r["stride"] + r["len"]) for k in range(r["count"]))]
    tot = sum(r["len"] * r["count"] for r in runs)
    sb = sum(r["len"] * r["count"] for r in seed)
    print(f"{len(seed)} of {len(runs)} runs ({sb / 2**30:.2f} of {tot / 2**30:.2f} GiB) held locally; {(tot - sb) / 2**30:.2f} GiB left to fetch", flush=True)
    fds = {}
    done = 0
    for r in seed:
        name = r["file"]
        if name not in fds:
            src = os.path.join(a.old, name)
            size = os.path.getsize(src)
            head = header(src)
            dp = os.path.join(a.dest, name)
            fd = os.open(dp, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
            os.ftruncate(fd, size)
            os.pwrite(fd, head, 0)
            fds[name] = (os.open(src, os.O_RDONLY), fd, size, head)
        sfd, dfd, size, head = fds[name]
        h = hashlib.sha256()
        for k in range(r["count"]):
            off = r["offset"] + k * r["stride"]
            left, at = r["len"], off
            while left:
                b = os.pread(sfd, min(left, 64 << 20), at)
                if not b:
                    raise RuntimeError(f"{name}: short read at {at}")
                h.update(b)
                n = os.pwrite(dfd, b, at)
                assert n == len(b)
                at += len(b)
                left -= len(b)
        state["runs"][run_key(r)] = h.hexdigest()
        state["sources"][run_key(r)] = "local"
        done += r["len"] * r["count"]
    for name, (sfd, dfd, size, head) in fds.items():
        os.fsync(dfd)
        st = os.fstat(dfd)
        state["files"][name] = {"size": size, "header_len": len(head), "header_sha256": hashlib.sha256(head).hexdigest(), "ino": st.st_ino, "dev": st.st_dev}
        os.close(sfd)
        os.close(dfd)
    with open(sp + ".tmp", "w") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(sp + ".tmp", sp)
    print(f"seeded {done / 2**30:.2f} GiB in {len(fds)} shards", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
