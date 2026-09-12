"""Execute the frozen training node's output cast/detach loop on explicit small tensors."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert torch.__version__ == "2.10.0+cpu"
commit = "1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a"
raw = subprocess.check_output(["git", "-C", args.source, "show", commit + ":comfy_extras/nodes_train.py"])
node = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == "TrainLoraNode")
execute = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
loops = [n for n in ast.walk(execute) if isinstance(n, ast.For)
         and isinstance(n.target, ast.Name) and n.target.id == "param"
         and isinstance(n.iter, ast.Name) and n.iter.id == "lora_sd"]
assert len(loops) == 1
assert ast.unparse(loops[0]).strip() == "for param in lora_sd:\n    lora_sd[param] = lora_sd[param].to(lora_dtype).detach()"
function = ast.parse("def cast(lora_sd, lora_dtype):\n    return lora_sd").body[0]
function.body.insert(0, loops[0])
namespace = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "frozen-output", "exec"), namespace)
values = [0.1, -0.0, 1.00390625, 1.01171875, -2.34567, 65536.0]
cases = []
for name, dtype in [("bf16", torch.bfloat16), ("fp32", torch.float32)]:
    original = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    state = namespace["cast"]({"diffusion_model.norm.diff": original}, dtype)
    output = state["diffusion_model.norm.diff"]
    assert not output.requires_grad and original.requires_grad
    payload = bytes(output.view(torch.uint8).tolist())
    cases.append(dict(dtype=name, input=values, shape=list(output.shape), outputHex=payload.hex(),
                      outputSha256=hashlib.sha256(payload).hexdigest(), requiresGrad=output.requires_grad))
with args.output.open("x", encoding="utf-8", newline="\n") as destination:
    json.dump(dict(sourceCommit=commit, sourceSha256=hashlib.sha256(raw).hexdigest(),
                   sourceFile="comfy_extras/nodes_train.py", cases=cases,
                   scope="Exact frozen final cast/detach loop only; synthetic tensors, no training-node or model-family qualification."), destination, indent=2)
    destination.write("\n")
print("Collected bf16/fp32 output bits from the frozen final cast/detach loop.")
