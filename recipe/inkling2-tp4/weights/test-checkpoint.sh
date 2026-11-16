#!/usr/bin/env bash
# weights/test-checkpoint.sh [RANK_COPY]: fixture tests of weights/checkpoint.py and copy-rank.sh --server on this
# machine (no hosts; the server listens on 127.0.0.1 only). python3 writes a small stand-in checkpoint (two shards,
# mtp.safetensors, the nine small files) and its pinned list. check: the folder passes; a Hugging Face cache layout
# (symlinks into blobs/) passes; the second check reads nothing (cache); a missing file, a size change and one changed
# byte (same size, same mtime: the cache must not hide it) are each refused and named. serve + copy-rank.sh --server
# (fetch, verify, check against the server): every planned run equals the checkpoint's bytes at its offset, holes
# elsewhere, an unlisted file in the checkpoint folder is not served; an address not allowed gets nothing; the server
# ends by itself after --idle-exit. RANK_COPY: inkling2_rank_copy.py (default: the repository's tools/zig copy, or
# ROOT/tools in an install; without it the serve and copy cases print SKIP). Prints PASS/FAIL lines; exits 0 only if
# all pass.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
tool=${1:-}
if [ -z "${tool}" ]; then
  for c in "${here}/../../../tools/zig/inkling2_rank_copy.py" "${here}/../../tools/inkling2_rank_copy.py"; do
    [ -f "${c}" ] && { tool=${c}; break; }
  done
fi
w=$(mktemp -d "${TMPDIR:-/tmp}/inkling2-ckpt-test.XXXXXX") || exit 1
spid=""
trap '[ -n "${spid}" ] && kill "${spid}" 2> /dev/null; rm -rf "${w}"' EXIT
fails=0
ok() { echo "PASS $*"; }
no() { echo "FAIL $*"; fails=$((fails + 1)); }

# the stand-in checkpoint w/ck, its pinned list w/files.tsv, rank 0's ranges w/ranges-w4-r0.tsv (strided runs too)
python3 -I - "${w}" << 'EOF'
import hashlib, json, os, struct, sys
w = sys.argv[1]
ck = os.path.join(w, "ck")
os.makedirs(os.path.join(ck, "tiktoken"))
def shard(name, tensors):
    head, data, off = {}, b"", 0
    for t, n in tensors:
        b = hashlib.sha256(name.encode() + t.encode()).digest() * (n // 32)  # n is a multiple of 32
        head[t] = {"dtype": "U8", "shape": [n], "data_offsets": [off, off + n]}
        data += b
        off += n
    j = json.dumps(head).encode()
    j += b" " * ((8 - len(j) % 8) % 8)
    with open(os.path.join(ck, name), "wb") as f:
        f.write(struct.pack("<Q", len(j)) + j + data)
    return {t: (8 + len(j) + head[t]["data_offsets"][0], n) for t, n in tensors}
s1 = shard("model-00001-of-00002.safetensors", [("model.llm.a", 4096), ("model.llm.b", 8192), ("vision.x", 4096)])
s2 = shard("model-00002-of-00002.safetensors", [("model.llm.c", 16384), ("model.llm.d", 4096)])
m = shard("mtp.safetensors", [("layers.0.w", 4096)])
index = {"weight_map": {"model.llm.a": "model-00001-of-00002.safetensors", "model.llm.b": "model-00001-of-00002.safetensors",
                        "vision.x": "model-00001-of-00002.safetensors", "model.llm.c": "model-00002-of-00002.safetensors",
                        "model.llm.d": "model-00002-of-00002.safetensors"}}
small = {"config.json": b'{"model_type": "stand-in"}\n', "hf_quant_config.json": b"{}\n",
         "model.safetensors.index.json": json.dumps(index).encode(), "tokenizer.json": b"{}\n", "tokenizer_config.json": b"{}\n",
         "special_tokens_map.json": b"{}\n", "chat_template.jinja": b"{{ messages }}\n", "processor_config.json": b"{}\n",
         "tiktoken/tokenizer.model": b"stand-in tokenizer\n"}
for n, b in small.items():
    open(os.path.join(ck, n), "wb").write(b)
open(os.path.join(ck, "extra.bin"), "wb").write(b"not pinned" * 100)
with open(os.path.join(w, "files.tsv"), "w") as f:
    f.write("# stand-in\n# path\tsize\tkind\thash\n")
    for n in sorted(os.listdir(ck)) + ["tiktoken/tokenizer.model"]:
        p = os.path.join(ck, n)
        if n == "extra.bin" or os.path.isdir(p):
            continue
        b = open(p, "rb").read()
        if n.endswith(".safetensors") or n.startswith("tiktoken/"):
            f.write(f"{n}\t{len(b)}\tsha256\t{hashlib.sha256(b).hexdigest()}\n")
        else:
            f.write(f"{n}\t{len(b)}\tgit-sha1\t{hashlib.sha1(b'blob %d' % len(b) + bytes(1) + b).hexdigest()}\n")
# rank 0: a whole tensor, half of one, every other 512-byte row of one (strided), and the drafter's tensor
a, b, c = s1["model.llm.a"], s1["model.llm.b"], s2["model.llm.c"]
rows = [("model-00001-of-00002.safetensors", a[0], a[1], a[1], 1, "replicated", "model.llm.a"),
        ("model-00001-of-00002.safetensors", b[0], b[1] // 2, b[1] // 2, 1, "split", "model.llm.b"),
        ("model-00002-of-00002.safetensors", c[0], 512, 1024, c[1] // 1024, "split", "model.llm.c"),
        ("mtp.safetensors", m["layers.0.w"][0], 2048, 2048, 1, "split", "layers.0.w")]
with open(os.path.join(w, "ranges-w4-r0.tsv"), "w") as f:
    f.write("# file\toffset\tlen\tstride\tcount\tplacement\ttensor\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
EOF
ck=${w}/ck; files=${w}/files.tsv
cp_py() { python3 -I "${here}/checkpoint.py" "$@"; }

# ---------------------------------------------------------------- check
out=$(cp_py check "${ck}" --files "${files}" --cache "${w}/cache.json" 2>&1); rc=$?
[ ${rc} = 0 ] && grep -q '^checkpoint: OK: 12 files' <<< "${out}" && ok "check: the stand-in checkpoint passes" || { no "check: rc ${rc}"; echo "${out}" | tail -3; }
out=$(cp_py check "${ck}" --files "${files}" --cache "${w}/cache.json" 2>&1); rc=$?
[ ${rc} = 0 ] && grep -q 'hashing 0 files.*12 unchanged since an earlier check' <<< "${out}" && ok "check: a rerun reads nothing (12 hashes from the cache)" || { no "check: cache rerun rc ${rc}"; echo "${out}" | tail -3; }
# a Hugging Face cache snapshot: every file a symlink into blobs/
hf=${w}/hf; mkdir -p "${hf}/blobs" "${hf}/snap/tiktoken"
for f in $(awk -F'\t' '!/^#/{print $1}' "${files}"); do
  cp "${ck}/${f}" "${hf}/blobs/$(basename "${f}").blob"
  ln -s "$( [ "${f#tiktoken/}" != "${f}" ] && echo ../)../blobs/$(basename "${f}").blob" "${hf}/snap/${f}"
done
out=$(cp_py check "${hf}/snap" --files "${files}" 2>&1); rc=$?
[ ${rc} = 0 ] && ok "check: a Hugging Face cache snapshot (symlinks) passes" || { no "check: symlinked snapshot rc ${rc}"; echo "${out}" | tail -3; }
# refusals: a missing file, another size, one changed byte (same size, mtime put back), without and with a cache that
# holds the file's earlier hash (the ctime in the key changes on any write)
bad=${w}/bad; cp -Rp "${ck}" "${bad}"
rm "${bad}/tokenizer_config.json"
out=$(cp_py check "${bad}" --files "${files}" 2>&1); rc=$?
[ ${rc} = 1 ] && grep -q 'DIFFERS tokenizer_config.json: missing' <<< "${out}" && grep -q 'REFUSED' <<< "${out}" && ok "check: a missing file refused and named" || { no "check: missing file rc ${rc}"; echo "${out}" | tail -3; }
cp -p "${ck}/tokenizer_config.json" "${bad}/"; printf 'x' >> "${bad}/config.json"
out=$(cp_py check "${bad}" --files "${files}" 2>&1); rc=$?
[ ${rc} = 1 ] && grep -q 'DIFFERS config.json: .* bytes, the pinned' <<< "${out}" && ok "check: another size refused and named" || { no "check: size rc ${rc}"; echo "${out}" | tail -3; }
cp -p "${ck}/config.json" "${bad}/"
cp_py check "${bad}" --files "${files}" --cache "${w}/cache2.json" > /dev/null 2>&1 || no "check: the copy did not pass before the byte change"
python3 -I - "${bad}/model-00002-of-00002.safetensors" << 'EOF'
import os, sys
p = sys.argv[1]
st = os.stat(p)
with open(p, "r+b") as f:
    f.seek(st.st_size - 100)
    b = f.read(1)
    f.seek(st.st_size - 100)
    f.write(bytes([b[0] ^ 1]))
os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
EOF
out=$(cp_py check "${bad}" --files "${files}" 2>&1); rc=$?
[ ${rc} = 1 ] && grep -q 'DIFFERS model-00002-of-00002.safetensors: sha256' <<< "${out}" && ok "check: one changed byte (same size) refused and named" || { no "check: changed byte rc ${rc}"; echo "${out}" | tail -3; }
out=$(cp_py check "${bad}" --files "${files}" --cache "${w}/cache2.json" 2>&1); rc=$?
[ ${rc} = 1 ] && grep -q 'DIFFERS model-00002-of-00002.safetensors: sha256' <<< "${out}" && ok "check: the changed byte is refused with a cache that held the old hash" || { no "check: the cache hid a changed byte (rc ${rc})"; echo "${out}" | tail -3; }

# ---------------------------------------------------------------- serve + copy-rank.sh --server
if [ ! -f "${tool}" ]; then
  echo "SKIP serve and copy-rank.sh --server: no inkling2_rank_copy.py beside the recipe (pass its path)"
  echo "$([ ${fails} = 0 ] && echo "ALL PASS" || echo "${fails} FAILED")"
  exit $((fails > 0))
fi
cr=${w}/cr; mkdir -p "${cr}"
cp "${here}/copy-rank.sh" "${here}/netrun.py" "${tool}" "${w}/ranges-w4-r0.tsv" "${cr}/"
printf '# digests\n' > "${cr}/inkling2-digests-w4.tsv"
mkdir -p "${w}/srv"; cp "${here}/checkpoint.py" "${tool}" "${w}/srv/" # serve imports the copy tool from beside itself
sv() { # sv ALLOW IDLE: start the server; sets spid and addr
  python3 -I "${w}/srv/checkpoint.py" serve "${ck}" --files "${files}" --bind 127.0.0.1 --allow "$1" --idle-exit "$2" > "${w}/serve.log" 2>&1 &
  spid=$!
  for _ in $(seq 1 40); do grep -q '^ready' "${w}/serve.log" 2> /dev/null && break; sleep 0.25; done
  addr=$(awk '/^listening /{print $2}' "${w}/serve.log")
}
sv 127.0.0.1 600
[ -n "${addr}" ] && ok "serve: listening on ${addr}" || { no "serve: did not start"; cat "${w}/serve.log"; }
out=$(cd "${cr}" && bash copy-rank.sh 0 "${w}/dest" --server "${addr}" --drafter --check-gib 0 2>&1); rc=$?
if [ ${rc} = 0 ] && tail -1 <<< "${out}" | grep -q '^== done'; then ok "copy-rank.sh --server: fetch, verify, check passed"
else no "copy-rank.sh --server: rc ${rc}"; echo "${out}" | tail -8; fi
python3 -I - "${ck}" "${w}/dest" "${w}/ranges-w4-r0.tsv" << 'EOF' && ok "copy: every run equals the checkpoint's bytes, zeros elsewhere, small files equal, extra.bin not served" || no "copy: bytes differ"
import os, sys
ck, dest, ranges = sys.argv[1:]
want = {}
for line in open(ranges):
    if line.startswith("#"):
        continue
    f, off, ln, stride, count, _, _ = line.rstrip("\n").split("\t")
    want.setdefault(f, []).append((int(off), int(ln), int(stride), int(count)))
ok = not os.path.exists(os.path.join(dest, "extra.bin"))
for n in os.listdir(ck):
    p = os.path.join(ck, n)
    if n.endswith(".safetensors"):
        src, got = open(p, "rb").read(), open(os.path.join(dest, n), "rb").read()
        exp = bytearray(len(src))
        hl = 8 + int.from_bytes(src[:8], "little")
        exp[:hl] = src[:hl]
        for off, ln, stride, count in want.get(n, []):
            for k in range(count):
                o = off + k * stride
                exp[o:o + ln] = src[o:o + ln]
        ok &= bytes(exp) == got
    elif os.path.isfile(p) and n != "extra.bin":
        ok &= open(p, "rb").read() == open(os.path.join(dest, n), "rb").read()
ok &= open(os.path.join(ck, "tiktoken/tokenizer.model"), "rb").read() == open(os.path.join(dest, "tiktoken/tokenizer.model"), "rb").read()
sys.exit(0 if ok else 1)
EOF
out=$(cd "${cr}" && bash copy-rank.sh 0 "${w}/dest" --server "${addr}" --drafter --check-gib all 2>&1); rc=$?
[ ${rc} = 0 ] && grep -q '0 of 4 runs to copy' <<< "${out}" && grep -q '^check: OK' <<< "${out}" && ok "copy-rank.sh --server rerun: nothing copied again, every run checked against the server's disk" || { no "rerun rc ${rc}"; echo "${out}" | tail -5; }
kill "${spid}" 2> /dev/null; wait "${spid}" 2> /dev/null; spid=""
sv 192.0.2.1 600
out=$(cd "${w}" && python3 -I "${tool}" fetch --server "${addr}" --ranges "${w}/ranges-w4-r0.tsv" --dest "${w}/dest2" 2>&1); rc=$?
[ ${rc} != 0 ] && [ ! -e "${w}/dest2/config.json" ] && ok "serve: an address not in --allow gets nothing (fetch exit ${rc})" || { no "serve: a disallowed address was served"; echo "${out}" | tail -3; }
kill "${spid}" 2> /dev/null; wait "${spid}" 2> /dev/null; spid=""
sv 127.0.0.1 2
sleep 5
if ! kill -0 "${spid}" 2> /dev/null && grep -q 'no connection for 2' "${w}/serve.log"; then ok "serve: ends by itself after --idle-exit"
else no "serve: still running after --idle-exit"; fi
spid=""

echo "$([ ${fails} = 0 ] && echo "ALL PASS" || echo "${fails} FAILED")"
exit $((fails > 0))
