#!/usr/bin/env python3
"""TCP reachability probe for setup.sh's preflight (standard library only).

  netprobe.py ports BOOT_PORT N        BOOT_PORT, BOOT_PORT + 1 and N ports spread over this host's ephemeral range
                                       (where NCCL's bootstrap and proxy listeners bind)
  netprobe.py listen ADDR SECS PORT... listen on ADDR:PORT for each PORT until every port has had a connection or
                                       SECS pass; prints "ready" once bound, then "got PORT" lines, "busy PORT" for a
                                       port something else holds
  netprobe.py connect ADDR PORT...     connect to each ADDR:PORT (3 s timeout, in parallel); prints "ok PORT" or
                                       "blocked PORT (reason)"; exit 1 if any is blocked

A firewall that drops TCP on the bootstrap interface shows up as "blocked ... timed out". NCCL bootstraps on
BOOT_PORT (NCCL_COMM_ID), the server's liveness lines use BOOT_PORT + 1, and NCCL's own listeners take random
ephemeral ports, so a rule that drops part of that range hangs a start at random.
"""
import socket
import sys
import threading
import time


def ports(boot, n):
    try:
        lo, hi = (int(x) for x in open("/proc/sys/net/ipv4/ip_local_port_range").read().split())
    except OSError:
        lo, hi = 32768, 60999
    out = [boot, boot + 1]
    for k in range(n):
        p = lo + (hi - lo) * (2 * k + 1) // (2 * n)
        if p not in out:
            out.append(p)
    print(" ".join(str(p) for p in out))
    return 0


def listen(addr, secs, plist):
    socks = {}
    for p in plist:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((addr, p))
            s.listen(4)
            s.settimeout(0.2)
            socks[p] = s
        except OSError as e:
            print(f"busy {p} ({e.strerror})", flush=True)
    print("ready", flush=True)
    end = time.time() + secs
    while socks and time.time() < end:
        for p, s in list(socks.items()):
            try:
                c, _ = s.accept()
                c.close()
                print(f"got {p}", flush=True)
                s.close()
                del socks[p]
            except socket.timeout:
                pass
    return 0


def connect(addr, plist):
    res = {}

    def one(p):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        try:
            s.connect((addr, p))
            res[p] = "ok"
        except socket.timeout:
            res[p] = "timed out (dropped by a firewall?)"
        except OSError as e:
            res[p] = e.strerror or str(e)
        finally:
            s.close()

    ts = [threading.Thread(target=one, args=(p,)) for p in plist]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    bad = 0
    for p in plist:
        if res[p] == "ok":
            print(f"ok {p}")
        else:
            print(f"blocked {p} ({res[p]})")
            bad += 1
    return 1 if bad else 0


def main(a):
    if a[:1] == ["ports"]:
        return ports(int(a[1]), int(a[2]))
    if a[:1] == ["listen"]:
        return listen(a[1], float(a[2]), [int(x) for x in a[3:]])
    if a[:1] == ["connect"]:
        return connect(a[1], [int(x) for x in a[2:]])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
