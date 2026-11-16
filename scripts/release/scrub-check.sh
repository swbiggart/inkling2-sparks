#!/usr/bin/env bash
# scrub-check.sh [--names FILE] [--authors] range A..B | tree [REF] [-- PATHSPEC...]
# Scan for private details before anything is published, print every hit, and exit 1 if there is any.
#   range A..B   the lines the range adds (git diff A B), and the range's commit messages
#                (--authors: also each commit's author and committer name and email)
#   tree [REF]   every tracked text file at REF (default: the working tree's tracked files), or only PATHSPEC
# Looks for:
#   host names   read from --names FILE or SCRUB_NAMES (whitespace or comma separated), matched case-sensitively as
#                whole names (a name followed by "-" or a letter is a different name). Keep the list out of the
#                repository: the names are what this check keeps private.
#   private IPv4 192.168.x.x, 10.x.x.x, 100.x.x.x (also with a placeholder for the last octet)
#   paths        home folders (macOS Users, /home/<user>) and the build host's srv/llm tree
#   emails       anything shaped like an email address, and host names ending in .lan (not in the verbatim
#                license texts under LICENSES/, whose contact addresses are the licensors')
# Output: KIND<TAB>FILE:LINE<TAB>TEXT (TEXT cut to 200 characters). Exit 0 clean, 1 hits, 2 usage.
set -uo pipefail
# bytes, not characters: a cut multi-byte character must not make sort or grep drop lines
export LC_ALL=C
names_file=${SCRUB_NAMES_FILE:-}
authors=0
while [ $# -gt 0 ]; do
  case $1 in
    --names) names_file=$2; shift 2;;
    --authors) authors=1; shift;;
    *) break;;
  esac
done
mode=${1:-}
[ -n "${mode}" ] || { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
shift

names=${SCRUB_NAMES:-}
if [ -n "${names_file}" ]; then
  [ -r "${names_file}" ] || { echo "cannot read ${names_file}" >&2; exit 2; }
  names="${names} $(grep -v '^#' "${names_file}")"
fi
names=$(printf '%s' "${names}" | tr ', \t\n' '    ' | tr -s ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' ')
[ -n "${names// /}" ] || echo "warning: no host names given (--names FILE or SCRUB_NAMES); checking the other patterns only" >&2

# patterns are written so that this file does not match itself
kinds=() pats=()
add() { kinds+=("$1"); pats+=("$2"); }
if [ -n "${names// /}" ]; then
  alt=""
  for n in ${names}; do alt="${alt:+${alt}|}$(printf '%s' "${n}" | sed 's/[.[\*^$()+?{|]/\\&/g')"; done
  add host "(^|[^A-Za-z0-9_-])(${alt})([^A-Za-z0-9_-]|$)"
fi
add ipv4 '(^|[^0-9.])19[2]\.168\.[0-9]{1,3}'
add ipv4 '(^|[^0-9.])(1[0]|10[0])\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9A-Za-z<{$*]'
add path '/User[s]/'
add path '/hom[e]/[a-z_][a-z0-9_-]*'
add path '/sr[v]/llm'
add email '[A-Za-z0-9._%+-]+[@][A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}'
add email '[A-Za-z0-9-]+\.la[n]([^A-Za-z0-9-]|$)'

fail() { echo "scrub-check: $*" >&2; exit 2; }
tmp=$(mktemp -d)
trap 'rm -rf "${tmp}"' EXIT
# KIND<TAB>FILE:LINE<TAB>TEXT for every line of a FILE<TAB>LINE<TAB>TEXT stream that matches a pattern
scan() {
  local f=$1 i rc
  for i in "${!pats[@]}"; do
    grep -E -- "${pats[$i]}" "${f}" > "${tmp}/m"; rc=$?
    [ ${rc} -le 1 ] || fail "grep failed (${rc}) on pattern ${i}"
    awk -F'\t' -v k="${kinds[$i]}" \
      '{ t = $0; sub(/^[^\t]*\t[^\t]*\t/, "", t); printf "%s\t%s:%s\t%s\n", k, $1, $2, substr(t, 1, 200) }' "${tmp}/m"
  done
}
case ${mode} in
range)
  rng=${1:?usage: scrub-check.sh range A..B}
  a=${rng%%..*}; b=${rng##*..}; [ -n "${b}" ] || b=HEAD
  git rev-parse --verify -q "${a}^{commit}" > /dev/null || fail "no commit ${a}"
  git rev-parse --verify -q "${b}^{commit}" > /dev/null || fail "no commit ${b}"
  git diff -U0 --no-color --no-ext-diff "${a}" "${b}" > "${tmp}/diff" || fail "git diff failed"
  awk '
    /^\+\+\+ / { f = substr($0, 7); next }
    /^@@ / { split($3, p, ","); n = substr(p[1], 2) + 0; next }
    /^\+/ { printf "%s\t%d\t%s\n", f, n, substr($0, 2); n++; next }
  ' "${tmp}/diff" > "${tmp}/lines"
  git log --format='%h%x09%B%x00' "${a}..${b}" > "${tmp}/log" || fail "git log failed"
  awk 'BEGIN { RS = "\0" } {
    h = $0; sub(/\t.*/, "", h); sub(/^\n/, "", h); m = $0; sub(/^[^\t]*\t/, "", m)
    k = split(m, ls, "\n"); for (i = 1; i <= k; i++) if (ls[i] != "") printf "commit %s message\t%d\t%s\n", h, i, ls[i] }' \
    "${tmp}/log" >> "${tmp}/lines"
  if [ ${authors} = 1 ]; then
    git log --format='commit %h author%x090%x09%an <%ae>%ncommit %h committer%x090%x09%cn <%ce>' "${a}..${b}" >> "${tmp}/lines" \
      || fail "git log failed"
  fi
  scan "${tmp}/lines" > "${tmp}/raw";;
tree)
  ref=""
  if [ $# -gt 0 ] && [ "$1" != -- ]; then ref=$1; shift; fi
  [ "${1:-}" = -- ] && shift
  [ -z "${ref}" ] || git rev-parse --verify -q "${ref}^{commit}" > /dev/null || fail "no commit ${ref}"
  : > "${tmp}/raw"
  for i in "${!pats[@]}"; do
    git grep -n -I -E --full-name -e "${pats[$i]}" ${ref:+"${ref}"} -- ${1+"$@"} > "${tmp}/m"; rc=$?
    [ ${rc} -le 1 ] || fail "git grep failed (${rc}) on pattern ${i}"
    awk -v k="${kinds[$i]}" -v r="${ref}" '{
      s = $0; if (r != "") s = substr(s, length(r) + 2)
      f = s; sub(/:.*/, "", f); s = substr(s, length(f) + 2); l = s; sub(/:.*/, "", l); t = substr(s, length(l) + 2)
      printf "%s\t%s:%s\t%s\n", k, f, l, substr(t, 1, 200) }' "${tmp}/m" >> "${tmp}/raw"
  done;;
*) echo "unknown mode ${mode}: range A..B or tree [REF]" >&2; exit 2;;
esac
# contact addresses inside the verbatim third-party license texts (LICENSES/) are not private details
awk -F'\t' '!($1 == "email" && $2 ~ /^LICENSES\//)' "${tmp}/raw" > "${tmp}/raw.kept" || fail "awk failed"
sort -u "${tmp}/raw.kept" > "${tmp}/hits" || fail "sort failed"
hits=$(wc -l < "${tmp}/hits" | tr -d ' ')
cat "${tmp}/hits"
if [ "${hits}" -gt 0 ]; then echo "scrub-check: ${hits} hit(s)" >&2; exit 1; fi
echo "scrub-check: clean" >&2
