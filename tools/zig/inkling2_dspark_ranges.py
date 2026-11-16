#!/usr/bin/env python3
"""The byte ranges each rank's share of the DSpark drafter reads (cuda_dspark.zig load), in the ranges format
inkling2_rank_copy.py fetches: file, offset, len, stride, count, placement, tensor (offsets absolute in the file).

At W ranks, rank r reads from mtp.safetensors: the norms, mask_embed, norm, fuse_weight and the Markov head's
markov_w1 and rnn_proj whole; rows [r q, (r + 1) q) of q_proj, [r k, (r + 1) k) of k_proj, v_proj, k_proj_ctx and
v_proj_ctx, [r m, (r + 1) m) of gate_proj and up_proj; columns of the same cuts of o_proj and down_proj (strided
runs). The drafter shares the target's unembed rows [r V/W, (r + 1) V/W) when the rank holds them (the vocab-split
head); --unembed adds them for a rank folder without them. The engine does not read hidden_out (the folds file holds
it), but every rank gets markov_head.hidden_out.weight and .bias whole (2.3 MiB) so that tf-inkling2-fold can build
the rank's folds file from the rank folder itself. Standard library only.

  inkling2_dspark_ranges.py MODEL --world W --out PREFIX [--unembed] [--plan PLAN]
      writes PREFIX-r<rank>.tsv for every rank. --plan PLAN: instead, PLAN-r<rank>.tsv (tf-inkling2-plan --ranges R)
      followed by the drafter lines for tensors that file does not already list (the plan holds the whole drafter on
      its drafter rank): the release's ranges-w4-r<rank>.tsv, which `weights.sh --drafter` filters by prefix.

fetch them with `inkling2_rank_copy.py fetch ... --ranges PREFIX-r<rank>.tsv --prefix ''` (every line).
"""
import argparse
import json
import os
import struct

H, D, NQ, NKV, INTER = 4608, 128, 48, 8, 18432
WHOLE = ("fuse_weight", "mask_embed", "norm.weight", "markov_head.dflare_norm.weight",
         "markov_head.markov_w1.weight", "markov_head.rnn_proj.weight", "markov_head.rnn_proj.bias")
FOLD = ("markov_head.hidden_out.weight", "markov_head.hidden_out.bias")  # read by tf-inkling2-fold only
LAYER_WHOLE = ("input_layernorm.weight", "post_attention_layernorm.weight", "dflare_norm.weight",
               "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.k_norm_ctx.weight")


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--unembed", action="store_true", help="also the rank's unembed rows (a folder without the vocab-split head)")
    p.add_argument("--plan", help="merge into PLAN-r<rank>.tsv (tf-inkling2-plan --ranges) the drafter lines it lacks")
    a = p.parse_args()
    W = a.world
    if NQ % W or NKV % W or INTER % W:
        raise SystemExit(f"W = {W} does not split the drafter")
    base, mt = header(os.path.join(a.model, "mtp.safetensors"))
    idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    ufile = idx["model.llm.unembed.weight"]
    ubase, uh = header(os.path.join(a.model, ufile))
    ue = uh["model.llm.unembed.weight"]
    V = ue["shape"][0]
    if V % W or ue["shape"][1] != H or ue["dtype"] != "BF16":
        raise SystemExit("unexpected unembed")

    def ent(name, rows=None, cols=None):
        e = mt[name]
        if e["dtype"] != "BF16":
            raise SystemExit(f"{name}: {e['dtype']}")
        off = base + e["data_offsets"][0]
        size = e["data_offsets"][1] - e["data_offsets"][0]
        if rows is None and cols is None:
            return ("mtp.safetensors", off, size, size, 1, "drafter", name)
        k = e["shape"][1]
        if rows is not None:
            r0, n = rows
            return ("mtp.safetensors", off + r0 * k * 2, n * k * 2, n * k * 2, 1, "drafter", name)
        c0, n = cols
        return ("mtp.safetensors", off + c0 * 2, n * 2, k * 2, e["shape"][0], "drafter", name)

    for r in range(W):
        hq, hk, I = NQ // W, NKV // W, INTER // W
        q0, k0, m0 = r * hq * D, r * hk * D, r * I
        lines = [ent(n) for n in WHOLE + FOLD]
        for i in range(5):
            L = f"layers.{i}."
            lines += [ent(L + n) for n in LAYER_WHOLE]
            lines.append(ent(L + "self_attn.q_proj.weight", rows=(q0, hq * D)))
            for n in ("k_proj", "v_proj", "k_proj_ctx", "v_proj_ctx"):
                lines.append(ent(L + f"self_attn.{n}.weight", rows=(k0, hk * D)))
            lines.append(ent(L + "self_attn.o_proj.weight", cols=(q0, hq * D)))
            lines.append(ent(L + "mlp.gate_proj.weight", rows=(m0, I)))
            lines.append(ent(L + "mlp.up_proj.weight", rows=(m0, I)))
            lines.append(ent(L + "mlp.down_proj.weight", cols=(m0, I)))
        vn = V // W
        uoff = ubase + ue["data_offsets"][0] + r * vn * H * 2
        if a.unembed:
            lines.append((ufile, uoff, vn * H * 2, vn * H * 2, 1, "drafter_vocab_rows", "model.llm.unembed.weight"))
        path = f"{a.out}-r{r}.tsv"
        head = ["# file\toffset\tlen\tstride\tcount\tplacement\ttensor\n"]
        if a.plan:
            head = open(f"{a.plan}-r{r}.tsv").readlines()
            have = {ln.rstrip("\n").split("\t")[6] for ln in head if ln.strip() and not ln.startswith("#")}
            lines = [x for x in lines if x[6] not in have]
        with open(path, "w") as f:
            f.writelines(head)
            for x in lines:
                f.write("\t".join(str(v) for v in x) + "\n")
        total = sum(x[2] * x[4] for x in lines)
        print(f"{path}: {len(lines)} drafter runs{' added to the plan' if a.plan else ''}, {total / 2**30:.3f} GiB")


if __name__ == "__main__":
    main()
