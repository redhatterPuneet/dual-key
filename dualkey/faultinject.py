"""Lab fault injection: flip bits in a model checkpoint to simulate a faulty GPU.

Works directly on .safetensors bytes (no torch needed), so it runs in the small
init container before the model server starts. Writes a ground-truth JSON next
to the model so test results can be graded later.

Usage:
  python -m dualkey.faultinject --model-dir /models/m --bit 13 --count 100 \
      [--tensor model.layers.14.mlp.down_proj.weight] [--seed 7]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import struct
import sys
import time

ELEMENT_SIZE = {"BF16": 2, "F16": 2, "F32": 4}


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (hdr_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hdr_len))
    return header, 8 + hdr_len


def list_tensors(model_dir: str) -> dict[str, str]:
    """tensor name -> safetensors file that holds it."""
    out = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        header, _ = read_header(path)
        for name in header:
            if name != "__metadata__":
                out[name] = path
    return out


def default_tensor(names: list[str]) -> str:
    """Pick the MLP down-projection of the middle decoder layer."""
    cands = []
    for n in names:
        m = re.search(r"layers\.(\d+)\..*down_proj\.weight$", n)
        if m:
            cands.append((int(m.group(1)), n))
    if not cands:
        weights = [n for n in names if n.endswith(".weight")]
        if not weights:
            raise ValueError("no weight tensors found")
        return weights[len(weights) // 2]
    cands.sort()
    return cands[len(cands) // 2][1]


def flip_bits(path: str, tensor: str, bit: int, count: int, seed: int) -> dict:
    header, base = read_header(path)
    info = header[tensor]
    dtype = info["dtype"]
    if dtype not in ELEMENT_SIZE:
        raise ValueError(f"unsupported dtype {dtype}")
    size = ELEMENT_SIZE[dtype]
    if not 0 <= bit < size * 8:
        raise ValueError(f"bit must be 0..{size * 8 - 1} for {dtype}")
    start, end = info["data_offsets"]
    n = (end - start) // size
    rng = random.Random(seed)
    idxs = rng.sample(range(n), min(count, n))
    with open(path, "r+b") as f:
        for i in idxs:
            off = base + start + i * size + bit // 8   # little-endian byte order
            f.seek(off)
            b = f.read(1)[0]
            f.seek(off)
            f.write(bytes([b ^ (1 << (bit % 8))]))
    return {"file": os.path.basename(path), "tensor": tensor, "dtype": dtype,
            "bit": bit, "count": len(idxs), "seed": seed}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--tensor", default="")
    p.add_argument("--bit", type=int, default=13, help="bf16: 0-6 mantissa, 7-14 exponent, 15 sign")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ground-truth", default="", help="where to write ground_truth.json")
    a = p.parse_args(argv)

    tensors = list_tensors(a.model_dir)
    if not tensors:
        print(f"no .safetensors files in {a.model_dir}", file=sys.stderr)
        return 1
    name = a.tensor or default_tensor(list(tensors))
    if name not in tensors:
        print(f"tensor {name} not found", file=sys.stderr)
        return 1
    result = flip_bits(tensors[name], name, a.bit, a.count, a.seed)
    result["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result["gpu"] = os.environ.get("DK_GPU_NAME", "")
    gt = a.ground_truth or os.path.join(a.model_dir, "..", "ground_truth.json")
    with open(gt, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
