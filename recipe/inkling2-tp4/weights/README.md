# Weights: each rank's folder

Each host holds only its own rank's bytes of `thinkingmachines/Inkling-2-Small`, pinned to revision
`aac974122acd33ddb0d3e1f54433a9c5dd2249ce`. A rank folder looks like the checkpoint to the server: every shard at
its full size as a sparse file (the safetensors header, then only this rank's byte ranges at their original offsets,
holes elsewhere), plus the small files (config, quantization manifest, index, tokenizer, chat template) whole. That
is about 92 GiB on disk a rank instead of the checkpoint's 366 GiB. The weights are the checkpoint's own bytes: no
conversion, no re-quantization.

## What you need

- **The plan files** (in the release tarball's `plan/`, or made by you, below):
  - `ranges-w4-r0.tsv` .. `ranges-w4-r3.tsv`: each rank's byte ranges (file, offset, length, stride, count,
    tensor).
  - `inkling2-digests-w4.tsv`: the content-digest manifest. Each rank hashes every byte it loads and checks it
    against its lines (`TF_INK2_DIGESTS`), so a wrong or corrupt byte stops the server at start-up.
- `tools/zig/inkling2_rank_copy.py` (Python 3.9 or later, standard library only) on the machine that runs
  `weights.sh`; it is copied to the hosts.
- About 92 GiB free on each host (plus headroom), and HTTPS to huggingface.co (or a full checkpoint on the fabric, below).

## Copy

```
./weights.sh my-sparks.conf start plan/                       # all four hosts at once, in the background
./weights.sh my-sparks.conf status                            # running or ended, last log lines
```

On each host, `weights/copy-rank.sh` then:

1. **fetch:** HTTP range requests for the rank's runs from
   `https://huggingface.co/thinkingmachines/Inkling-2-Small/resolve/aac974122acd33ddb0d3e1f54433a9c5dd2249ce`, 8 at
   a time. Each run is hashed as it arrives, and recorded only once its bytes are written and synced. It resumes:
   run `start` again after an interruption and it skips the recorded runs.
2. Copies the digest manifest into `MODEL/inkling2-digests-w4.tsv`.
3. **verify:** coverage (every run of the ranges file is held), the shard headers against those recorded at fetch,
   and every run re-read from disk with the page cache dropped, against its hash from fetch.
4. **check --http:** every shard header and small file against Hugging Face, plus a 3 GiB pseudo-random sample of
   runs read again from Hugging Face and hashed (`--check-gib all` checks every run, about 92 GiB of downloads).

The log ends in `== done` when every step passed. A step that fails stops the script with its exit code (1 seed,
2 fetch, 3 verify, 4 check).

Only `model.llm.*` (the text model) is copied. `--drafter` adds the DSpark drafter's tensors, which only
`TF_INK2_DSPARK=1` needs, and `--folds BIN` (added by `weights.sh` while `DSPARK=1`) then builds the rank's DSpark
folds file `inkling2-dspark-fold.safetensors` in the folder with `BIN` (`ROOT/bin/tf-inkling2-fold`) from the
folder's own unembed rows and `hidden_out` (`weights/fold-rank.sh`: about a second; kept on a rerun when its
recorded digest matches). The vision tower is not used and not copied.

## From a full checkpoint

If you already have the full checkpoint (366 GiB, the pinned revision) on a Linux host with python3 that the four
hosts reach over the fabric, `setup.sh` builds the rank folders from it instead of Hugging Face: set
`CHECKPOINT=HOST:/path` in the hosts file (or pass `--checkpoint HOST:/path`; `/path` alone means the machine
running `setup.sh`). On the checkpoint's host, in a new temporary folder:

1. `weights/checkpoint.py check PATH`: every file of `weights/checkpoint-files.tsv` (the 34 files a rank folder is
   made from, with their sizes and hashes from Hugging Face's file tree at the pinned revision) is present (symlinks
   followed, so a Hugging Face cache snapshot works), has its size, and has its Hugging Face hash (the whole file's
   sha256 for the LFS files, the git blob sha1 for the small text files). Any difference: refused, every differing
   file named, nothing copied. Hashes are cached on that host (`~/.cache/inkling2-setup/checkpoint-hashes.json`) by
   the file's real path, size, mtime, ctime, inode and device, so a rerun on an unchanged checkpoint reads nothing.
2. `weights/checkpoint.py serve PATH`: the copy tool's own server, serving only the pinned files, bound only to the
   host's addresses in the ranks' fabric subnets (or `CHECKPOINT_ADDR`), answering only the four ranks' fabric
   addresses, on a free ephemeral port unless `CHECKPOINT_PORT`. It ends when `setup.sh` stops it after the copies,
   or by itself after 30 minutes without a connection. The copies live only as long as that `setup.sh` run: if it
   stops early (an error, a lost connection), it stops and waits for the copies it started before the server ends;
   run `setup.sh` again and each copy starts again from the runs it already holds.

Each rank then runs `copy-rank.sh R MODEL --server ADDR:PORT`: **fetch --server** (each run hashed by the server as
it reads the source and again on arrival, recorded only once synced; resumable), the digest manifest, **verify**
(every run re-read from disk) and **check --server** (a 3 GiB sample against the source's disk). The folder is byte
for byte the one the download makes, and its stamp is the same. By hand, run `checkpoint.py check` and
`checkpoint.py serve` on the source yourself and pass `--server ADDR:PORT` to `weights.sh ... start`.

## Optional: seed from a folder already on the host

If a host already has a checked rank folder (an earlier layout's, for example), copy what it holds locally first
and fetch only the rest:

```
./weights.sh my-sparks.conf start plan/ --seed-from /path/on/hosts/old-rank-folder
```

`weights/seed_rank_folder.py` copies every run of the new ranges file that lies wholly inside runs recorded in the
old folder's `.rank_copy.json`, at the same offsets, and records it with the hash of the bytes copied. It refuses a
destination that already has a copy state, so it only seeds a new folder. The seeded bytes are only as good as the
old folder: seed only from one that passed verify and check. The digest manifest still checks every byte when the
server loads.

## Making the plan files yourself

`tf-inkling2-plan` is a host tool: the release's `bin/` copy runs on aarch64; on another control machine build it
with `zig build tf-inkling2-plan -Doptimize=ReleaseFast` in the engine source (upstream TensorFold at `ENGINE_COMMIT` with
`patches/` applied; `scripts/build-engine.sh` makes that checkout). The ranges come from the same placement code the server
loads with, and need no checkpoint (the release's headers are built in):

```
for r in 0 1 2 3; do tf-inkling2-plan --world 4 --experts split --ranges $r > plan/plan-w4-r$r.tsv; done
python3 tools/zig/inkling2_dspark_ranges.py /path/to/Inkling-2-Small --world 4 --plan plan/plan-w4 --out plan/ranges-w4
```

The second line adds the DSpark drafter's lines to each rank's file (the plan places the whole drafter on rank 3
only; the server splits it over all four): each rank's slices of the drafter, and `markov_head.hidden_out` whole,
which the server does not load but `tf-inkling2-fold` reads to build the rank's folds file. It reads only the
safetensors headers of the folder it is given (a full checkpoint, or a rank folder that has `mtp.safetensors`).

The digest manifest hashes the source bytes, so it needs the whole checkpoint once (366 GiB, about a minute on a
fast disk):

```
tf-inkling2-plan --dir /path/to/Inkling-2-Small --world 4 --experts split --digests plan/inkling2-digests-w4.tsv
```

Use the plan files that match the binaries: a build with a different placement loads different ranges, and its
digest check fails against a manifest made for another placement.
