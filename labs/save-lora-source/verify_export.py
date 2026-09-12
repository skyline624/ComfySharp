"""Independent byte-level comparison of SaveLoRA exports; laboratory only, stdlib only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

WIDTHS = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "I32": 4, "I64": 8,
          "F16": 2, "BF16": 2, "F32": 4, "F64": 8}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def decode(path):
    if path.stat().st_size > 528 * 1024 * 1024:
        raise ValueError("Laboratory file allowance exceeded")
    raw = path.read_bytes()
    length, = struct.unpack("<Q", raw[:8])
    assert 0 < length <= 16 * 1024 * 1024 and length % 8 == 0
    header = json.loads(raw[8:8 + length], object_pairs_hook=unique_object)
    body = memoryview(raw)[8 + length:]
    tensors = {}
    spans = []
    for name, info in header.items():
        if name == "__metadata__":
            assert isinstance(info, dict) and all(isinstance(v, str) for v in info.values())
            continue
        shape = info["shape"]
        assert isinstance(shape, list) and all(type(x) is int and x >= 0 for x in shape)
        start, end = info["data_offsets"]
        assert type(start) is int and type(end) is int
        assert 0 <= start <= end <= len(body)
        assert end - start == math.prod(shape) * WIDTHS[info["dtype"]]
        spans.append((start, end))
        tensors[name] = (info["dtype"], shape, hashlib.sha256(body[start:end]).hexdigest())
    cursor = 0
    for start, end in sorted(spans):
        assert start == cursor
        cursor = end
    assert cursor == len(body)
    return hashlib.sha256(raw).hexdigest(), tensors


parser = argparse.ArgumentParser()
parser.add_argument("--original", type=Path, required=True)
parser.add_argument("--export", type=Path, required=True)
parser.add_argument("--host-report", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
original_sha, original = decode(args.original)
export_sha, exported = decode(args.export)
report = json.loads(args.host_report.read_text(encoding="utf-8"))
assert report["success"] and report["sourceProvided"]
assert original == exported
assert report["outputSha256"] == export_sha and report["tensorCount"] == len(exported)
assert report["parameterHashes"] == {key: value[2] for key, value in exported.items()}
with args.output.open("x", encoding="utf-8", newline="\n") as output:
    json.dump(dict(success=True, originalSha256=original_sha, exportSha256=export_sha,
                   tensorCount=len(exported), keysDtypesShapesAndPayloadsExact=True,
                   scope="Independent restricted safetensors byte decoder only; no training, generation or official safetensors-package execution."), output, indent=2)
    output.write("\n")
print(f"Independent decoder: {len(exported)} tensor keys, dtypes, shapes and payload hashes match.")
