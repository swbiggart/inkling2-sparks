#!/usr/bin/env python3
"""A prompt of exactly N token ids for bench_cache.py --prompt-file (a JSON array of raw token ids).

  prompt_ids.py MODEL_DIR N OUT.json [--seed S]

Tokenizes deterministic filler text (numbered inventory lines, varied by --seed so two files never share a prefix
and the server's prompt cache cannot help) with MODEL_DIR/tokenizer.json and keeps the first N ids. Needs the
`tokenizers` package (pip install tokenizers). MODEL_DIR can be a rank folder: it holds the tokenizer whole.
"""
import argparse
import json
import os
import random
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("n", type=int)
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=None, help="default: N")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
    rng = random.Random(a.n if a.seed is None else a.seed)
    colors = ["red", "blue", "green", "grey", "white", "black", "yellow", "orange"]
    kinds = ["box", "crate", "bag", "tin", "jar", "case", "roll", "pack"]
    ids, k = [], 0
    while len(ids) < a.n:
        lines = []
        for _ in range(200):
            k += 1
            lines.append(f"Item {k} ({rng.randrange(10**6):06d}) is a {rng.choice(colors)} {rng.choice(kinds)} on shelf "
                         f"{chr(65 + rng.randrange(26))}{rng.randrange(1, 100)}, weight {rng.randrange(1, 5000)} g.")
        ids.extend(tok.encode("\n".join(lines) + "\n", add_special_tokens=False).ids)
    with open(a.out, "w") as f:
        json.dump(ids[:a.n], f)
    print(f"{a.out}: {a.n} token ids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
