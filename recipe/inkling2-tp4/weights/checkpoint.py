#!/usr/bin/env python3
"""checkpoint.py: a full Inkling-2-Small checkpoint already on a host, as the source of every rank's folder.

setup.sh runs it on the checkpoint's host when CHECKPOINT is set (hosts.conf.example): `check`, then `serve` while the
ranks copy their byte ranges from it with inkling2_rank_copy.py fetch --server (weights/copy-rank.sh --server).

  check DIR [--files TSV] [--jobs J] [--cache FILE]
      DIR against the pinned file list (TSV, default checkpoint-files.tsv beside this script: Hugging Face's sizes
      and hashes of the files a rank folder is made from, at the pinned revision). First every file (symlinks are
      followed, so a Hugging Face cache snapshot folder works) and its size, then every hash: the whole file's
      sha256 for the LFS files (shards, mtp.safetensors, tokenizer files), the git blob sha1 for the small text
      files; J files at a time (default 4). A hash is kept in FILE (a JSON cache) under the file's real path, size,
      mtime, ctime, inode and device (any write changes the ctime), so a rerun on an unchanged checkpoint reads
      nothing again. Exit 0: DIR holds the
      pinned revision's files; 1: it does not (every difference printed, nothing else done); 2: usage.
  addrs --peers IP[,IP...]
      this host's IPv4 addresses (ip -4 address) that share a subnet with one of the peers, one a line
  serve DIR --bind ADDR[,ADDR...] --allow IP[,IP...] [--port P] [--files TSV] [--idle-exit S]
      inkling2_rank_copy.py's server (its protocol: fetch --server and check --server) on each ADDR, serving only the
      pinned files of DIR (read only) to the allowed addresses only. Port 0 (the default) takes a free port from the
      ephemeral range. Prints "listening ADDR:PORT" for each, then "ready". Ends on SIGTERM or SIGINT, or once no
      connection has been open for S seconds (default 1800).
Standard library only; inkling2_rank_copy.py must sit beside this script for `serve`.
"""
import argparse
import hashlib
import ipaddress
import json
import os
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REVISION = "aac974122acd33ddb0d3e1f54433a9c5dd2249ce"
CHUNK = 16 << 20
KEY = ("size", "mtime_ns", "ctime_ns", "ino", "dev", "kind")  # a cached hash counts only while all of these are equal


def load_files(path):
    out = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        name, size, kind, digest = line.rstrip("\n").split("\t")
        if kind not in ("sha256", "git-sha1"):
            raise ValueError(f"{path}: unknown hash kind {kind!r} for {name}")
        out.append({"name": name, "size": int(size), "kind": kind, "hash": digest})
    return out


def file_hash(path, kind):
    """The file's sha256 (kind sha256) or git blob sha1 (kind git-sha1), read once; its pages are dropped after."""
    with open(path, "rb", buffering=0) as f:
        fd = f.fileno()
        if kind == "git-sha1":
            h = hashlib.sha1(b"blob %d\0" % os.fstat(fd).st_size)
        else:
            h = hashlib.sha256()
        buf = bytearray(CHUNK)
        mv = memoryview(buf)
        done = 0
        while True:
            n = f.readinto(mv)
            if not n:
                break
            h.update(mv[:n])
            done += n
            if hasattr(os, "posix_fadvise") and done % (1 << 30) < CHUNK:
                os.posix_fadvise(fd, 0, done, os.POSIX_FADV_DONTNEED)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def load_cache(path):
    try:
        with open(path) as f:
            c = json.load(f)
        return c if isinstance(c, dict) else {}
    except (OSError, ValueError):
        return {}


def save_cache(path, cache):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=0, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def check(a):
    root = os.path.abspath(a.dir)
    files = load_files(a.files)
    total = sum(e["size"] for e in files)
    print(f"checkpoint: {root} against {len(files)} files of Inkling-2-Small at revision {REVISION} ({total / 2**30:.2f} GiB)", flush=True)
    if not os.path.isdir(root):
        print(f"checkpoint: REFUSED: {root} is not a folder")
        return 1
    bad = []
    stats = {}
    for e in files:
        p = os.path.join(root, e["name"])
        try:
            st = os.stat(p)
        except OSError as x:
            bad.append(f"{e['name']}: missing ({x.strerror})")
            continue
        if not os.path.isfile(p):
            bad.append(f"{e['name']}: not a regular file")
        elif st.st_size != e["size"]:
            bad.append(f"{e['name']}: {st.st_size} bytes, the pinned revision's file has {e['size']}")
        else:
            stats[e["name"]] = (os.path.realpath(p), st)
    if bad:
        for b in bad:
            print(f"  DIFFERS {b}")
        print(f"checkpoint: REFUSED: {root} is not Inkling-2-Small at revision {REVISION} ({len(bad)} of {len(files)} files "
              "missing or of another size; listed above). Nothing was copied from it.")
        return 1
    cache = load_cache(a.cache) if a.cache else {}
    todo, cached = [], 0
    for e in files:
        real, st = stats[e["name"]]
        c = cache.get(real)
        if c and [c.get(k) for k in KEY] == [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_dev, e["kind"]]:
            e["got"] = c["hash"]
            cached += 1
        else:
            todo.append(e)
    want = sum(e["size"] for e in todo)
    print(f"checkpoint: every file there at its size; hashing {len(todo)} files ({want / 2**30:.2f} GiB, {a.jobs} at a time)"
          + (f", {cached} unchanged since an earlier check (cached)" if cached else ""), flush=True)
    lock = threading.Lock()
    queue = sorted(todo, key=lambda e: -e["size"])
    done = [0]
    errors = []
    t0 = time.time()

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                e = queue.pop(0)
            real, st = stats[e["name"]]
            try:
                got = file_hash(real, e["kind"])
            except OSError as x:
                with lock:
                    errors.append(f"{e['name']}: read error ({x})")
                continue
            with lock:
                e["got"] = got
                done[0] += e["size"]
                cache[real] = dict(zip(KEY, [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_dev, e["kind"]]), hash=got)

    th = [threading.Thread(target=worker) for _ in range(max(1, a.jobs))]
    for x in th:
        x.start()
    last = time.time()
    while any(x.is_alive() for x in th):
        time.sleep(0.5)
        if time.time() - last > 30:
            last = time.time()
            dt = time.time() - t0
            print(f"  hashed {done[0] / 2**30:.1f} of {want / 2**30:.1f} GiB in {dt:.0f} s ({done[0] / max(dt, 1e-9) / 1e6:.0f} MB/s)", flush=True)
    for x in th:
        x.join()
    dt = time.time() - t0
    if a.cache and todo:
        try:
            save_cache(a.cache, cache)
        except OSError as x:
            print(f"  (hash cache not written: {x})")
    for e in files:
        if "got" in e and e["got"] != e["hash"]:
            errors.append(f"{e['name']}: {e['kind']} {e['got']}, the pinned revision's is {e['hash']}")
    if errors:
        for b in errors:
            print(f"  DIFFERS {b}")
        print(f"checkpoint: REFUSED: {root} is not Inkling-2-Small at revision {REVISION}, or a file is damaged ({len(errors)} of "
              f"{len(files)} files differ; listed above). Nothing was copied from it.")
        return 1
    print(f"checkpoint: OK: {len(files)} files, {total / 2**30:.2f} GiB, sizes and Hugging Face hashes equal to revision {REVISION}"
          + " (" + "; ".join(x for x in [f"{want / 2**30:.2f} GiB hashed in {dt:.0f} s, {want / max(dt, 1e-9) / 1e6:.0f} MB/s" if todo else "",
                                         f"{cached} files from the cache" if cached else ""] if x) + ")", flush=True)
    return 0


def addrs(a):
    peers = [ipaddress.ip_address(x) for x in a.peers.split(",") if x]
    out = subprocess.run(["ip", "-4", "-o", "address", "show"], capture_output=True, text=True, check=True).stdout
    seen = []
    for line in out.splitlines():
        f = line.split()
        if "inet" not in f:
            continue
        iface = ipaddress.ip_interface(f[f.index("inet") + 1])
        if iface.ip.is_loopback or str(iface.ip) in seen:
            continue
        if any(p in iface.network for p in peers):
            seen.append(str(iface.ip))
            print(iface.ip)
    return 0 if seen else 1


def serve(a):
    sys.path.insert(0, HERE)
    import inkling2_rank_copy as rc

    root = os.path.abspath(a.dir)
    files = load_files(a.files)
    allow = set(x for x in a.allow.split(",") if x)
    if not allow:
        print("serve: --allow names no address", file=sys.stderr)
        return 2
    state = {"open": 0, "last": time.time()}
    lock = threading.Lock()

    class Handler(rc.Handler):
        def setup(self):
            with lock:
                state["open"] += 1
            super().setup()

        def finish(self):
            try:
                super().finish()
            finally:
                with lock:
                    state["open"] -= 1
                    state["last"] = time.time()

    servers = []
    for addr in [x for x in a.bind.split(",") if x]:
        s = rc.Server((addr, a.port), Handler)
        s.root = root
        s.files = sorted(e["name"] for e in files if not e["name"].endswith(".safetensors"))
        s.shards = set(e["name"] for e in files if e["name"].endswith(".safetensors"))
        s.allow = allow
        servers.append(s)
        print(f"listening {addr}:{s.server_address[1]}", flush=True)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print(f"ready: {len(files)} files of {root} to {sorted(allow)}; ends after {a.idle_exit} s without a connection", flush=True)
    while not stop.wait(1):
        with lock:
            if state["open"] == 0 and time.time() - state["last"] > a.idle_exit:
                print("serve: no connection for {} s, ending".format(a.idle_exit), flush=True)
                break
    for s in servers:
        s.shutdown()
        s.server_close()
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    default_files = os.path.join(HERE, "checkpoint-files.tsv")
    c = sub.add_parser("check")
    c.add_argument("dir")
    c.add_argument("--files", default=default_files)
    c.add_argument("--jobs", type=int, default=4)
    c.add_argument("--cache", default="")
    d = sub.add_parser("addrs")
    d.add_argument("--peers", required=True)
    s = sub.add_parser("serve")
    s.add_argument("dir")
    s.add_argument("--bind", required=True)
    s.add_argument("--allow", required=True)
    s.add_argument("--port", type=int, default=0)
    s.add_argument("--files", default=default_files)
    s.add_argument("--idle-exit", type=float, default=1800)
    a = p.parse_args(argv)
    return {"check": check, "addrs": addrs, "serve": serve}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
