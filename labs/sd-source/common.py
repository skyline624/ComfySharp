"""Utilities for the separate frozen-source SD laboratory, never imported by .NET.

Only verified upstream declarations execute. No checkpoint files or network are used.
"""
import ast
import hashlib
import json
import math
from pathlib import Path

COMMIT = "1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a"
PROFILE = "sd-components-native210-cpu-f32-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_symbols(source, relative, expected_sha256, names, namespace, evidence):
    path = (source / relative).resolve(strict=True)
    require(path.is_relative_to(source), "Source must remain within its snapshot.")
    raw = path.read_bytes()
    require(len(raw) <= 8 * 1024 * 1024, "Unexpected source size.")
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Frozen source SHA-256 mismatch: " + relative)
    selected = []
    for node in ast.parse(raw.decode("utf-8")).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            selected.append(node)
    require(len(selected) == len(names), "Selected source declarations do not match: " + relative)
    module = ast.Module(body=selected, type_ignores=[])
    evidence.append({"source": relative, "url": f"https://github.com/comfy-org/ComfyUI/blob/{COMMIT}/{relative}",
                     "fileSha256": expected_sha256, "symbols": names,
                     "astSha256": hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()})
    exec(compile(module, relative, "exec"), namespace)


def tensor_record(value):
    import torch
    flat = value.detach().cpu().contiguous()
    numbers = flat.reshape(-1).tolist()
    def number(v):
        if isinstance(v, float) and not math.isfinite(v):
            return "NaN" if math.isnan(v) else ("Infinity" if v > 0 else "-Infinity")
        return v
    return {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."),
            "sha256": hashlib.sha256(flat.numpy().tobytes()).hexdigest(), "values": [number(v) for v in numbers]}


def synthetic_tensor(name, shape, *, parameter=False):
    """Version 1 integer recipe, shared as input data only, with no Torch RNG.

    SHA256(name UTF8) first four bytes little endian seed. LCG high 16 bits give
    signed samples. All scaling is a power of two, exactly representable in F32.
    Source model state_dict independently defines parameter names and shapes.
    """
    import torch
    shape = tuple(shape)
    seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")
    indices = torch.arange(math.prod(shape), dtype=torch.int64, device="cpu")
    samples = (((indices * 1664525 + seed) & 0xffffffff) >> 16) - 32768
    if parameter and len(shape) == 1:
        if name.endswith(".weight"):
            values = 1.0 + samples.float() * (2.0 ** -18)
        else:
            values = samples.float() * (2.0 ** -20)
    elif parameter:
        fan = math.prod(shape[1:])
        ceil_log2 = (fan - 1).bit_length()
        values = samples.float() * (2.0 ** (-15 - (ceil_log2 + 1) // 2))
    else:
        values = samples.float() * (2.0 ** -15)
    return values.reshape(shape)


def fill_parameters(model):
    import torch
    entries = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = synthetic_tensor(name, parameter.shape, parameter=True)
            parameter.copy_(value)
            entries.append({"name": name, "shape": list(parameter.shape),
                            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()})
    return entries


def write_document(path, document):
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
