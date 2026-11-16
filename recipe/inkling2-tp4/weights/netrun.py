#!/usr/bin/env python3
"""netrun.py SCRIPT [ARGS...]: run SCRIPT (inkling2_rank_copy.py) with a connect that gives up on a dead address fast.

Python's socket.create_connection tries a host's addresses one after another, each with the whole request timeout
(inkling2_rank_copy.py uses 120 s). Hugging Face's downloads go through CDN host names with many addresses, and on
some networks a few of them never answer (a SYN with no reply), so a request can sit two minutes on one address
before it tries the next, and a fetch of a thousand runs crawls. Here each address gets INKLING2_CONNECT_TIMEOUT
seconds (default 5) to connect, IPv4 addresses first; once connected, the socket takes the caller's timeout as
before. Nothing else changes: the same requests, the same bytes, the same checks. Standard library only.
copy-rank.sh runs the copy tool through this (INKLING2_NETRUN=0 runs it directly).
"""
import os
import runpy
import socket
import sys

_DEFAULT = socket._GLOBAL_DEFAULT_TIMEOUT


def create_connection(address, timeout=_DEFAULT, source_address=None, *args, **kwargs):
    host, port = address[0], address[1]
    limit = float(os.environ.get("INKLING2_CONNECT_TIMEOUT", "5"))
    final = socket.getdefaulttimeout() if timeout is _DEFAULT else timeout
    infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    infos.sort(key=lambda a: a[0] != socket.AF_INET)  # IPv4 first, then IPv6, each in the resolver's order
    err = None
    for family, kind, proto, _, addr in infos:
        s = None
        try:
            s = socket.socket(family, kind, proto)
            s.settimeout(limit if final is None else min(limit, final))
            if source_address:
                s.bind(source_address)
            s.connect(addr)
            s.settimeout(final)
            return s
        except OSError as e:
            err = e
            if s is not None:
                s.close()
    if err is not None:
        raise err
    raise OSError(f"getaddrinfo returned no address for {host}")


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    socket.create_connection = create_connection
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
