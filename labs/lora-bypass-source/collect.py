"""Frozen ordinary LoRA bypass forward and gradients on explicit CPU tensors; lab only."""
import argparse, ast, hashlib, json, math, subprocess, types
from typing import Callable
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
assert torch.__version__ == "2.10.0+cpu"
torch.set_num_threads(1)
commit = "1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a"
hashes = {}
namespace = dict(torch=torch, F=F, Callable=Callable)

def extract(path, class_name, names):
    raw = subprocess.check_output(["git", "-C", args.source, "show", commit + ":" + path])
    hashes[path] = hashlib.sha256(raw).hexdigest()
    cls = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    functions = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(functions) == len(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), path, "exec"), namespace)

extract("comfy/weight_adapter/lora.py", "LoRAAdapter", ["h"])
extract("comfy/weight_adapter/base.py", "WeightAdapterBase", ["g", "bypass_forward"])

def tensor(shape, shift=0):
    return (((torch.arange(math.prod(shape), dtype=torch.float32) + shift) % 17 - 8) / 31).reshape(shape).requires_grad_()

def pack(value):
    return dict(shape=list(value.shape), values=value.detach().flatten().tolist())

recipes = [
    ("linear", [2,3], [2,4], [4,2], [2,3], None, None, 1, 0, None, .5, False),
    ("linear-mid", [2,2,3], [2,2,4], [4,2], [2,3], [2,2], None, 1, 0, 2., -.75, False),
    ("conv-flat", [1,3,5,5], [1,4,3,3], [4,2], [2,27], None, [3,3], 2, 1, 1., .75, False),
    ("conv-kernel", [1,3,5,5], [1,4,5,5], [4,2,1,1], [2,3,3,3], None, [3,3], 1, 1, None, -.5, False),
    ("conv-mid", [1,3,5,5], [1,4,3,3], [4,2,1,1], [2,3,1,1], [2,2,3,3], [3,3], 2, 1, 1., .75, False),
    ("dora-field", [2,3], [2,4], [4,2], [2,3], None, None, 1, 0, 2., 1., True),
]
cases = []
for name, xs, bs, us, ds, ms, kernel, stride, padding, alpha, strength, has_dora in recipes:
    x, base, up, down = tensor(xs), tensor(bs, 3), tensor(us, 5), tensor(ds, 7)
    mid = None if ms is None else tensor(ms, 11)
    dora = None if not has_dora else torch.full([4,1], 3.)
    adapter = types.SimpleNamespace(weights=(up, down, alpha, mid, dora, None), multiplier=strength,
                                    is_conv=kernel is not None, conv_dim=2, kernel_size=kernel,
                                    in_channels=xs[1], kw_dict=dict(stride=stride, padding=padding))
    adapter.h = types.MethodType(namespace["h"], adapter)
    adapter.g = types.MethodType(namespace["g"], adapter)
    result = namespace["bypass_forward"](adapter, lambda unused: base, x)
    result.square().mean().backward()
    inputs = dict(input=x, baseOutput=base, up=up, down=down)
    if mid is not None: inputs["mid"] = mid
    cases.append(dict(name=name, tensors={k: pack(v) for k,v in inputs.items()},
                      gradients={k: pack(v.grad) for k,v in inputs.items()}, output=pack(result),
                      kernel=kernel, stride=stride, padding=padding, alpha=alpha, strength=strength, doraField=has_dora))
with open(args.output, "x", encoding="utf-8", newline="\n") as destination:
    json.dump(dict(sourceCommit=commit, sourceHashes=hashes, absTolerance=0.000008, relTolerance=0.000008,
                   cases=cases, scope="Frozen h/g/bypass_forward only; synthetic linear/Conv2d activations and gradients."), destination, indent=2)
    destination.write("\n")
print("Collected six frozen bypass forward/gradient cases.")
