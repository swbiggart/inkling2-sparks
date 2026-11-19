# Inkling-2-Small at TP4: install details and the manual steps

Paths in this file are relative to `recipe/inkling2-tp4/` (or the unpacked release folder, for `tools/`, `data/`, `plan/`), unless a link says otherwise.

What `setup.sh` does step by step, its options, the container install, and the seven manual steps for doing it by hand. The short path is the Quick start in the [README](../../../README.md).

## What setup.sh does

`setup.sh` installs and starts everything, and stops at the first step that fails:

1. **preflight** on every host: a GB10 (sm_121), driver 580 or later (CUDA 13), the libraries the binaries load,
   python3, free disk and memory, both fabric ports (speed, MTU, carrier, address, HCA state), the NIC re-probe
   detector, and firewall rules on the bootstrap interface. Then a TCP probe from host to host of rank 0's bootstrap
   and liveness ports and of ephemeral ports on every host's fabric address, where NCCL's listeners bind.
2. **assets:** the release tarball, checked against its `.sha256` and its `SHA256SUMS`, unpacked into `ROOT` on every
   host and checked again there.
3. **libs:** `ROOT/lib`: NCCL 2.30.7 and cuBLAS 13.1.1.3, from the release's own `lib/` (bundled, with their licence
   texts), or with `--libs DIR`, or else from NVIDIA's wheels on PyPI at pinned sha256s.
4. **weights:** each host fetches its rank's bytes (and the DSpark drafter's) from Hugging Face at the pinned
   revision, then verifies and checks them (step 4 below). This is the long step, about 92 GiB a host. It resumes: run `setup.sh` again after an
   interruption. With `CHECKPOINT` (below) the hosts copy the same bytes from a full checkpoint over the fabric
   instead.
5. **folds** (DSpark only): each host builds its rank's DSpark folds file from its own rank folder with the release's
   `tf-inkling2-fold`, in about a second (see DSpark in [SETTINGS.md](SETTINGS.md)).
6. **manifest** and **config:** the digest manifest on every host, and the server config (`my-sparks.serve.conf`,
   beside `my-sparks.conf`). NCCL bootstraps over the fabric port at rank 0's fabric address, not the LAN. The
   release settings come from `release-defaults.sh`: checkpoint mode, split experts in `model-split/`, the digest
   check, DSpark, the RoCE mesh wire, `--context 131072`, `--parallel auto` capped by `MAX_STREAMS=4` (four streams),
   an 8 GiB memory reserve and an explicit `--prompt-cache-gib`. The hosts file only needs to name what differs.
7. **start**, **status** and **smoke:** rank 0, then ranks 1-3. `setup.sh` waits until all four have loaded, prints
   each rank's state, then runs the smoke checks on rank 0's host. When the release ships the reference files, it
   also runs greedy replies against the reference's and reports where they depart (information; the token gate is
   `smoke.py yardstick`, step 6 below).

Every step skips what is already done, so a second run checks and starts. Single steps:

```
./setup.sh my-sparks.conf stop | start | status | smoke
./setup.sh my-sparks.conf stop check start      # the W=4 decode check (step 6 below) needs the server stopped
```

Options (`./setup.sh --help` lists them all):

- `--weights-dir DIR`: the hosts already hold rank folders for this release at `DIR`. `setup.sh` verifies them in
  place (coverage, headers, every byte re-read from disk) and never writes their weights; with DSpark on, the folds
  step adds `inkling2-dspark-fold.safetensors` there if it is missing.
- `--checkpoint SRC` (or `CHECKPOINT=SRC` in the hosts file): build the rank folders from a full Inkling-2-Small
  checkpoint you already have, instead of Hugging Face. `SRC` is `HOST:/path` (an ssh name: one of the four, or
  another Linux host with python3 on the switch) or `/path` on the machine running `setup.sh`; a Hugging Face cache
  snapshot folder works (symlinks are followed). `setup.sh` first checks it against the pinned revision
  (`weights/checkpoint-files.tsv`: every file, its size and its Hugging Face hash; any difference is refused, every
  differing file named, and nothing is copied; the hashes are cached on that host, so a second run reads nothing
  again), then serves it read only on its fabric address(es), to the four ranks' fabric addresses only and only the
  pinned files, while each rank copies, verifies and sample-checks its bytes. The server ends with the step.
  `CHECKPOINT_ADDR` names the source's address when it is not in the ranks' fabric subnets; `CHECKPOINT_PORT` fixes
  the port (default: a free ephemeral port, as NCCL's listeners use). The folders are byte for byte the download's,
  and `--weights-dir` and `--checkpoint` exclude each other.
- `--fabric-install`: run `fabric/host/install.sh RANK` on every host first. It needs root through `sudo -n`, and
  `FABRIC_NETS` set in the hosts file.
- `--libs DIR`: take the three CUDA libraries from `DIR` on this machine.

## Container

The same install, with each rank in a container (Docker Compose, like the community recipes for these hosts):

```
./setup.sh my-sparks.conf --release URL --sha256 HEX --container     # URL and HEX: the README's Quick start
./setup.sh my-sparks.conf --container status | stop | start | smoke
./setup.sh my-sparks.conf --container stop check start
```

- **The image:** `inkling2-sparks:<release commit>`, from `container/Dockerfile`. It holds NVIDIA's CUDA 13.3 runtime
  for arm64 (the build's CUDA user space), the release's `bin/` and `lib/`, rdma-core's verbs library and the licence
  files. It holds no weights.
- **Pinned:** the base image by the digest of its linux/arm64 manifest
  (`nvcr.io/nvidia/cuda:13.3.1-runtime-ubuntu24.04@sha256:585155539dd4ff2ab653ee59d812aae81a9406a5804596d516c06e3a5719eb26`,
  `BASE_IMAGE` in `setup.sh`), and `libibverbs1` and `ibverbs-providers` at version `50.0-2ubuntu0.2` (`VERBS_VERSION`
  in the Dockerfile), both as gated. If Ubuntu's archive no longer lists that version, the build stops and names the
  versions it does list instead of installing another. The image carries both pins as labels (`inkling2.base`,
  `inkling2.verbs`); `setup.sh` rebuilds an image whose base label differs.
- **Where it is built:** on rank 0's host. `setup.sh` copies it to the other three with `docker save | docker load`,
  through the machine that runs `setup.sh`. It never goes to a registry.
- **Each rank:** `container/compose.yaml`, the same file on all four hosts, in `ROOT/container/`. It runs with host
  networking, `ipc: host`, the GPU, `/dev/infiniband` and unlimited memlock. It mounts the weights, plan, data and
  log folders at their host paths.
- **The rank's settings:** `setup.sh` writes `.env` (rank, image, folders, serve flags) and `rank.env` (the server's
  environment, the same as the host install's, from `release-defaults.sh`) next to `compose.yaml`.
- **By hand:** `docker compose up -d` in `ROOT/container`, on rank 0 first.

The hosts need Docker with Compose v2 and the NVIDIA Container Toolkit. If the user is not in the `docker` group,
`setup.sh` runs `sudo -n docker`. The host and container installs share `ROOT` and the weights. Run one or the
other, not both at once.

The numbered steps below (1 to 7) are what `setup.sh` does, for doing it by hand.

## Step 1. Binaries

**Download** from the Releases page of this repository `tensorfold-inkling2-tp4-VERSION-aarch64-sm121.tar.gz` and its `.sha256`, check
it, and unpack it on the control machine:

```
sha256sum -c tensorfold-inkling2-tp4-VERSION-aarch64-sm121.tar.gz.sha256
tar xzf tensorfold-inkling2-tp4-VERSION-aarch64-sm121.tar.gz
cd tensorfold-inkling2-tp4-VERSION-aarch64-sm121 && sha256sum -c SHA256SUMS
```

It holds:

- `bin/`: the aarch64 sm_121 build (`tensorfold-native`, `tf-inkling2-decode`, `tf-inkling2-plan`,
  `tf-inkling2-fold`, `tf-tp-test`, and `tp_mailbox.fatbin`, the RoCE wire's kernel image)
- `lib/`: NVIDIA's NCCL 2.30.7 and cuBLAS / cuBLASLt 13.1.1.3, unmodified, and `libtfgrammar.so` (structured output,
  built with xgrammar v0.2.8), with their licence texts
- `recipe/inkling2-tp4/`: this folder
- `tools/`: the weight copy tool and the benchmarks
- `plan/`: the rank ranges and the digest manifest
- `data/`: the decode check's prompt, the fp32 reference's replies with their top-two margins, the extended prompt
  set (`ext/`) and the structured-output cases (`so-cases/`)
- the licence files

**Or build** the engine from source: upstream TensorFold at `ENGINE_COMMIT` with this repository's `patches/` applied
(Zig 0.17.0 and CUDA 13.3.1, cross-compiled for aarch64 and sm_121). From the repository root, on an x86-64 or aarch64
Linux machine with git and Docker:

```
scripts/build-engine.sh        # build/dist/bin/: tensorfold-native, tf-inkling2-decode, tf-inkling2-plan,
                               # tf-inkling2-fold, tf-tp-test, tp_mailbox.fatbin; build/dist/lib/libtfgrammar.so
```

It runs the release build's commands in a pinned toolchain image (`scripts/build-engine.Dockerfile`), with xgrammar
v0.2.8 at a pinned commit for `libtfgrammar.so`; `docs/BUILD.md` at the repository root says how the result compares
with the release's binaries. In an engine checkout of your own (with `patches/` applied) the build is:

```
zig build native tf-inkling2-decode tf-inkling2-plan tf-inkling2-fold tf-tp-test -Doptimize=ReleaseFast \
  -Dtarget=aarch64-linux-gnu.2.28 -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dtp-nvcc=/usr/local/cuda/bin/nvcc \
  --prefix zig-out-arm
zig build grammar -Dxgrammar=/path/to/xgrammar -Doptimize=ReleaseFast -Dtarget=aarch64-linux-gnu.2.28 --prefix zig-out-arm
nvcc -fatbin -O3 -std=c++17 -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_121,code=sm_121 \
  -gencode=arch=compute_90,code=compute_90 -o zig-out-arm/tp_mailbox.fatbin zig/src/fabric/tp/mailbox.cu
```

Copy the binaries (and `tp_mailbox.fatbin`) to `ROOT/bin/` on every host, the same build on all four (the ranks
compare executable hashes at start-up), and `libtfgrammar.so` to `ROOT/lib/`. Put NCCL and cuBLAS in `ROOT/lib/`: the release's `lib/`, or NVIDIA's NCCL
2.30.7 for arm64 and CUDA 13 and cuBLAS 13.1.1.3 (what was measured). The
`data/` and `plan/` files stay on the control machine; `check.sh` and `weights.sh` copy what each host needs.

## Step 2. Fabric

See `fabric/`. In short, for the switch:

1. Configure the switch: `fabric/switch/crs812-fabric.rsc` for a MikroTik CRS812-class switch, or the notes in
   `fabric/switch/README.md` for others (jumbo frames, one L2 segment, no PFC needed).
2. On each host, as root: `fabric/host/install.sh RANK`. It installs a systemd unit that sets MTU 9000 and one
   address on each PCIe half of the 200G port, and a udev rule that redoes it if the driver recreates the port.
3. Check every port: `./status.sh my-sparks.conf --bw`.

**After any cable change, reboot the hosts whose cables moved.** A NIC re-probed at run time transmits at only about
13 Gb/s per PCIe half until the next boot. `status.sh` warns when it sees this.

The switchless ring setup is in `fabric/ring-experimental/`. It is not validated for this release.

## Step 3. Configure

```
cp config.example.sh my-sparks.conf
```

Fill in `HOSTS` (ssh names, in rank order), `MASTER` (rank 0's address on its first fabric port), `SOCKET_IFNAME`
(that port's interface, so NCCL's bootstrap stays clear of LAN firewalls), and `ROOT` and the paths under it.
(`setup.sh` writes this file from the hosts file.) Keep the switch `NCCL_ENV` as it is. The release's settings are
not in this file but in `release-defaults.sh` (the scripts read it before the config): `CONTEXT`, `PARALLEL`,
`MAX_STREAMS`, `MEMORY_RESERVE_GIB`, `PROMPT_CACHE_GIB`, `DSPARK`, `XROWS`, `ROCE_MULTI`, `MAILBOX_MAX_KB` and the folder name
`MODEL_DIRNAME` (`model-split`). Set one of them in the config only to differ from the release. `EXTRA_ENV` adds
more environment. Everything in `release-defaults.sh` and `EXTRA_ENV`, and every serve flag, must be the same on all
four ranks.

## Step 4. Weights

```
./weights.sh my-sparks.conf start ../../plan/        # each host fetches its rank's bytes (and the drafter's) from Hugging Face
./weights.sh my-sparks.conf status                   # until every log ends in "== done"
```

Each host fetches its rank's byte ranges from Hugging Face at the pinned revision, then verifies coverage, the
headers and every byte from disk, and checks a sample against Hugging Face. With DSpark on, it then builds the rank's
folds file with `ROOT/bin/tf-inkling2-fold` (when the release is unpacked in `ROOT` on the host). The digest manifest goes into the rank
folder, and the server checks every byte it loads against it. `weights/README.md` has the details, how to copy from a
full checkpoint instead (`weights.sh ... --server`), how to seed from a folder already on a host, and how to make the
plan files yourself.

## Step 5. Start, stop, status

```
./start.sh my-sparks.conf      # rank 0, then 1, 2, 3; returns once all four have loaded
./status.sh my-sparks.conf     # each rank's state, MemAvailable, last log lines, and the fabric check
./stop.sh my-sparks.conf       # SIGTERM to rank 0; the others end by themselves
```

`start.sh` copies `rank.sh`, `release-defaults.sh` and the config to `ROOT/recipe/` on every host and runs the ranks under `setsid nohup`.
Logs and pid files go to `RUN/rank-R.{log,pid}`. The served model id is `model`. HTTP listens on
`HTTP_HOST:HTTP_PORT` on rank 0 (loopback by default). If any rank dies, the others stop.

```
curl -s http://127.0.0.1:8890/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "model", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

## Step 6. Verify (the same checks are in [CORRECTNESS.md](CORRECTNESS.md))

**Decode check** (server stopped): each decode step must equal the whole-prompt forward at the same W bit for bit.

```
./stop.sh my-sparks.conf
./check.sh my-sparks.conf ../../data/long_4000.safetensors 3040 80    # prints PASS: ... bit-equal 80/80 ...
```

**The token gate**, **smoke** and the benchmarks run where the endpoint is reachable (rank 0's host when HTTP is on
loopback), from the unpacked release folder:

```
python3 recipe/inkling2-tp4/smoke.py http://127.0.0.1:8890 yardstick data/decode.json \
  chat_short=data/chat_short.ids.json tool_call=data/tool_call.ids.json --ext data/ext
                                      # v0.2 prints YARDSTICK: top-1 434/445 (< 438), in the oracle's top-2 443/445 (>= 443): FAIL
                                      # (top-1 below its bar: CORRECTNESS.md#agreement-with-the-fp32-reference)
python3 recipe/inkling2-tp4/smoke.py http://127.0.0.1:8890 oracle data/decode.json \
  chat_short=data/chat_short.ids.json tool_call=data/tool_call.ids.json --ext data/ext
                                      # free-running replies: EQUAL, or DEPARTS at j (information)
python3 recipe/inkling2-tp4/smoke.py http://127.0.0.1:8890 smoke --long-tokens 8000
```

The yardstick sends 445 one-token requests, one after another. The reference files come from
`tools/zig/inkling2_ref.py --attn fp32` (in the engine source: upstream TensorFold at `ENGINE_COMMIT` with `patches/`
applied, as `scripts/build-engine.sh` makes it): the checkpoint run through Transformers' own Inkling-2 modules in
PyTorch, with attention in fp32 as the kernels compute it ([CORRECTNESS.md](CORRECTNESS.md#the-attention-arithmetic)),
with TF32 off: `decode.json` and `ext/` in the tool's default mode (the stored expert weights decoded to bf16),
`so-cases/` in its checkpoint mode (activations quantized as the checkpoint's quantization config requires,
TokenSpeed's reference). Each token carries the reference's top-two margin at its position.

## Step 7. Benchmarks

`bench.sh` runs the release's measurements against the running server (Python 3.9 or later on rank 0's host,
standard library only) and writes JSON to `./bench-results/`:

```
./bench.sh my-sparks.conf check        # GPU-free install check on every host
./bench.sh my-sparks.conf sweep        # decode at 1/2/4 streams, the one-stream workloads, cold prompts
./bench.sh my-sparks.conf prompts      # the 16-row prompt set at temperature 0 and 0.7
./bench.sh my-sparks.conf needle       # concurrent needle sessions (8 at once need MAX_STREAMS=8, CONTEXT=32768)
./bench.sh my-sparks.conf ledger bench-results/sweep-*.json
```

What each one measures, and the other benchmark tools: [BENCHMARKS.md](BENCHMARKS.md#measuring-your-install).
