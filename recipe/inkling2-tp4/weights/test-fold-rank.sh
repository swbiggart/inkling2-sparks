#!/usr/bin/env bash
# weights/test-fold-rank.sh FOLD_BIN: fixture tests of fold-rank.sh's digest record on this machine (no weights, no
# GPU; python3 writes stand-in folds files). FOLD_BIN is a built tf-inkling2-fold: its --record publishes the record.
# The folds step itself is stood in by a wrapper that writes a small valid folds file. Cases: symlinks to a sentinel
# planted at the record, at its earlier mktemp-style names and at the tool's temporary-name pattern; four runs at once
# into one folder; four runs at once while a racer keeps replacing every new temporary entry with a symlink to the
# sentinel. The sentinel must be unchanged after every case. Prints PASS/FAIL lines; exits 0 only if all pass.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
real=${1:?usage: test-fold-rank.sh FOLD_BIN (a built tf-inkling2-fold)}
real=$(cd "$(dirname "${real}")" && pwd)/$(basename "${real}")
[ -x "${real}" ] || { echo "no ${real}" >&2; exit 2; }
w=$(mktemp -d "${TMPDIR:-/tmp}/inkling2-fold-test.XXXXXX") || exit 1
trap 'rm -rf "${w}"' EXIT
sha() { if command -v sha256sum > /dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
fails=0
ok() { echo "PASS $*"; }
no() { echo "FAIL $*"; fails=$((fails + 1)); }

# the stand-in tool: --record goes to the real tool; MODEL OUT --rank R --world 4 writes a valid folds file (V = 8)
fake=${w}/fake-fold
cat > "${fake}" << EOF
#!/usr/bin/env bash
[ "\$1" = --record ] && exec "${real}" "\$@"
python3 -I - "\$2" "\$4" << 'PY'
import json, os, struct, sys
out, r = sys.argv[1], int(sys.argv[2])
V, W = 8, 4
h = {"__metadata__": {"rows": f"{r * V // W}:{(r + 1) * V // W}"},
     "fold_b": {"dtype": "BF16", "shape": [V], "data_offsets": [0, 2 * V]},
     "fold_w": {"dtype": "BF16", "shape": [V, 256], "data_offsets": [2 * V, 2 * V * 257]}}
j = json.dumps(h).encode()
t = f"{out}.fake.{os.getpid()}"
with open(t, "wb") as fh:
    fh.write(struct.pack("<Q", len(j)) + j + bytes(2 * V * 257))
os.replace(t, out)
PY
EOF
chmod +x "${fake}"
sentinel=${w}/sentinel
printf 'sentinel: an original weight shard\n' > "${sentinel}"
s0=$(sha < "${sentinel}")
rec=.inkling2-dspark-fold.sha256
f=inkling2-dspark-fold.safetensors
sentinel_ok() { [ "$(sha < "${sentinel}")" = "${s0}" ]; }
record_ok() { # $1: folder; the record a regular file holding the folds file's digest for rank 3
  local d; d=$(sha "$1/${f}" | cut -c1-64)
  [ -f "$1/${rec}" ] && [ ! -L "$1/${rec}" ] && [ "$(cat "$1/${rec}")" = "${d}  rank 3" ]
}
leftovers() { (cd "$1" && ls -A | grep -E '\.tmp$|\.fake\.|^\.inkling2-dspark-fold\.sha256\.' | grep -vE "${2:-^$}"); }

# ---------------------------------------------------------------- planted links
m=${w}/planted
mkdir -p "${m}"
planted=("${rec}" "${rec}.tmp" "${rec}.AbCd1234" "${rec}.XXXXXXXX" ".${rec}.abababababababab.tmp" ".${rec}.0000000000000000.tmp")
for n in "${planted[@]}"; do ln -s "${sentinel}" "${m}/${n}"; done
bash "${here}/fold-rank.sh" 3 "${m}" "${fake}" > "${w}/p.log" 2>&1; rc=$?
[ ${rc} = 0 ] && ok "planted links: fold-rank.sh exits 0" || { no "planted links: exit ${rc}"; cat "${w}/p.log"; }
sentinel_ok && ok "planted links: the sentinel unchanged" || no "planted links: the sentinel CHANGED"
record_ok "${m}" && ok "planted links: the record is a regular file with the digest (the link at its name replaced)" || no "planted links: record wrong"
kept=1
for n in "${planted[@]:1}"; do [ "$(readlink "${m}/${n}")" = "${sentinel}" ] || kept=0; done
[ ${kept} = 1 ] && ok "planted links: the other planted names untouched" || no "planted links: a planted name changed"
bash "${here}/fold-rank.sh" 3 "${m}" "${fake}" | grep -q kept && ok "planted links: a rerun keeps the file" || no "planted links: rerun did not keep"
# a failing --record (no such folder for the record) exits nonzero and changes nothing
"${real}" --record "${w}/absent/${rec}" "x  rank 3" 2> /dev/null && no "--record into a missing folder succeeded" || ok "--record into a missing folder fails"
"${real}" --record "${m}/${rec}" "$(printf 'two\nlines')" 2> /dev/null && no "--record of two lines succeeded" || ok "--record of two lines is refused"
record_ok "${m}" && sentinel_ok && ok "refused records leave the record and the sentinel" || no "a refused record changed something"

# ---------------------------------------------------------------- four runs at once
m=${w}/four
mkdir -p "${m}"
ln -s "${sentinel}" "${m}/${rec}"
pids=()
for i in 1 2 3 4; do bash "${here}/fold-rank.sh" 3 "${m}" "${fake}" --force > "${w}/four-${i}.log" 2>&1 & pids+=($!); done
bad=0
for p in "${pids[@]}"; do wait "${p}" || bad=$((bad + 1)); done
[ ${bad} = 0 ] && ok "four runs at once: all exit 0" || { no "four runs at once: ${bad} failed"; tail -2 "${w}"/four-*.log; }
sentinel_ok && ok "four runs at once: the sentinel unchanged" || no "four runs at once: the sentinel CHANGED"
record_ok "${m}" && ok "four runs at once: one whole record" || no "four runs at once: record wrong"
[ -z "$(leftovers "${m}")" ] && ok "four runs at once: no temporary file left" || no "four runs at once: left $(leftovers "${m}")"

# ---------------------------------------------------------------- four runs at once against a racer
# the racer replaces every temporary entry it sees (the tool's .NAME.<hex>.tmp names, and the earlier wrapper's
# mktemp names) with a link to the sentinel, as a hostile writer in a shared folder could: runs may then fail, but
# nothing may be written through a link
m=${w}/race
mkdir -p "${m}"
touch "${w}/race.on"
(
  while [ -e "${w}/race.on" ]; do
    for t in "${m}"/.*.tmp "${m}/${rec}".*; do
      [ -e "${t}" ] && [ ! -L "${t}" ] && ln -sfn "${sentinel}" "${t}" 2> /dev/null
    done
  done
) &
racer=$!
pids=()
for i in 1 2 3 4; do
  (for _ in 1 2 3 4 5; do bash "${here}/fold-rank.sh" 3 "${m}" "${fake}" --force > /dev/null 2>&1; done) & pids+=($!)
done
for p in "${pids[@]}"; do wait "${p}"; done
rm -f "${w}/race.on"
wait "${racer}"
sentinel_ok && ok "twenty runs against a racer: the sentinel unchanged" || no "twenty runs against a racer: the sentinel CHANGED"
bash "${here}/fold-rank.sh" 3 "${m}" "${fake}" --force > "${w}/race.log" 2>&1 && record_ok "${m}" &&
  ok "after the racer: a run publishes a whole record" || { no "after the racer: run failed"; cat "${w}/race.log"; }

echo "$([ ${fails} = 0 ] && echo ALL PASS || echo "${fails} FAILED")"
exit $((fails > 0))
