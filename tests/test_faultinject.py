import json
import struct

from dualkey import faultinject as fi


def write_safetensors(path, tensors):
    """tensors: name -> list of uint16 raw bf16 values"""
    header, blob, off = {}, b"", 0
    for name, vals in tensors.items():
        data = struct.pack(f"<{len(vals)}H", *vals)
        header[name] = {"dtype": "BF16", "shape": [len(vals)], "data_offsets": [off, off + len(data)]}
        blob += data
        off += len(data)
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + blob)


def read_values(path, name):
    header, base = fi.read_header(path)
    s, e = header[name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + s)
        raw = f.read(e - s)
    return list(struct.unpack(f"<{len(raw) // 2}H", raw))


def test_flip_bits_changes_exactly_count_values_at_that_bit(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    names = {f"model.layers.{i}.mlp.down_proj.weight": [0x3F80] * 1000 for i in range(4)}
    names["model.embed_tokens.weight"] = [0x3F80] * 10
    write_safetensors(d / "model.safetensors", names)

    tensors = fi.list_tensors(str(d))
    target = fi.default_tensor(list(tensors))
    assert target == "model.layers.2.mlp.down_proj.weight"   # middle layer

    before = read_values(d / "model.safetensors", target)
    res = fi.flip_bits(tensors[target], target, bit=13, count=50, seed=1)
    after = read_values(d / "model.safetensors", target)
    changed = [(b, a) for b, a in zip(before, after) if b != a]
    assert res["count"] == 50 and len(changed) == 50
    assert all(a ^ b == (1 << 13) for b, a in changed)
    # other tensors untouched
    assert read_values(d / "model.safetensors", "model.layers.1.mlp.down_proj.weight") == [0x3F80] * 1000


def test_cli_writes_ground_truth(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    write_safetensors(d / "a.safetensors", {"model.layers.0.mlp.down_proj.weight": [1] * 100})
    gt = tmp_path / "gt.json"
    assert fi.main(["--model-dir", str(d), "--bit", "3", "--count", "5", "--ground-truth", str(gt)]) == 0
    data = json.loads(gt.read_text())
    assert data["count"] == 5 and data["bit"] == 3
