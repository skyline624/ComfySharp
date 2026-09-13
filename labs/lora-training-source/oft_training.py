"""Laboratory only: frozen OFTDiff weight/output rotations, gradients and SGD updates."""
import argparse, ast, gzip, hashlib, json, logging, pathlib, subprocess
from typing import Callable, Optional
import torch
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
assert torch.__version__ == '2.10.0+cpu' and torch.version.cuda is None
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
commit = '1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a'
hashes = {}
ns = dict(torch=torch, nn=nn, logging=logging, Callable=Callable, Optional=Optional)

def extract(path, names):
    raw = subprocess.check_output(['git', '-C', a.source, 'show', commit + ':' + path])
    hashes[path] = hashlib.sha256(raw).hexdigest()
    nodes = [n for n in ast.parse(raw).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), ns)

extract('comfy/weight_adapter/base.py', ['WeightAdapterBase', 'WeightAdapterTrainBase'])
extract('comfy/weight_adapter/oft.py', ['OFTDiff'])

def values(shape, shift):
    count = 1
    for n in shape: count *= n
    return (((torch.arange(count, dtype=torch.float32) + shift) % 19 - 9) / 31).reshape(shape)

def record(value):
    if value is None: return None
    return dict(shape=list(value.shape), values=value.detach().float().contiguous().flatten().tolist())

cases = []
for mode in ['weight', 'output']:
    for dims in range(4):
        for alpha in [0., .05, 10., -.1]:
            for scale in ['none', 'scalar', 'channels']:
                for dtype in ([torch.float32, torch.float16, torch.bfloat16] if mode == 'weight' else [torch.float32]):
                    blocks = values([2, 3, 3], 5)
                    rescale = None if scale == 'none' else torch.tensor(.85) if scale == 'scalar' else .9 + values([6]+[1]*(dims+1), 11)
                    train = ns['OFTDiff']((blocks, rescale, alpha, None)).requires_grad_(True)
                    train.is_conv = dims > 0
                    train.multiplier = -.4 if alpha < 0 else .75
                    shape = [6, 4]+[2]*dims if mode == 'weight' else [2, 3, 6] if dims == 0 else [1, 6]+[3, 4, 2][:dims]
                    original = values(shape, 13).to(dtype)
                    # Exercise non-contiguous inputs, including 3D linear output with is_conv=False.
                    if len(shape) > 2: original = original.transpose(-1, -2).contiguous().transpose(-1, -2)
                    x = original.detach().clone().requires_grad_()
                    target = values(shape, 2)
                    initial = {k: record(v) for k, v in train.named_parameters()}
                    optimizer = torch.optim.SGD(train.parameters(), lr=.003)
                    output = train(x) if mode == 'weight' else train.g(x + train.h(x, x))
                    loss = (output.float()-target).square().mean()
                    loss.backward()
                    row = dict(mode=mode, dims=dims, alpha=alpha, scale=scale, dtype=str(dtype), multiplier=train.multiplier,
                        initial=initial, input=record(original), target=record(target), output=record(output), loss=loss.item(),
                        inputGradient=record(x.grad), gradients={k: record(v.grad) for k, v in train.named_parameters()})
                    optimizer.step()
                    row['updated'] = {k: record(v) for k, v in train.named_parameters()}
                    with torch.no_grad():
                        # OFTDiff captures float(alpha); updating the exported alpha leaf has no effect.
                        train.alpha.fill_(123.)
                        row['afterAlphaMutation'] = record(train(original) if mode == 'weight' else train.g(original + train.h(original, original)))
                    cases.append(row)

data = dict(sourceCommit=commit, sourceHashes=hashes, absoluteTolerance=3e-5, relativeTolerance=3e-5, cases=cases,
    scope='Frozen OFTDiff.__call__, h and g, 1D/2D/3D convolution and rank-three linear layouts, Float32 leaves, Float32/Float16/BFloat16 weights and Float32 bypass. Negative/zero/clamped/unclamped captured constraints, scalar/channel rescale, non-contiguous inputs, all gradients and SGD update. No inference loader, resume factory, complete model workflow or hardware qualification.')
raw = (json.dumps(data, indent=2, allow_nan=False)+'\n').encode()
with open(a.output, 'xb') as f: f.write(gzip.compress(raw, mtime=0))
print(json.dumps(dict(cases=len(cases), rawSha256=hashlib.sha256(raw).hexdigest(), compressedSha256=hashlib.sha256(pathlib.Path(a.output).read_bytes()).hexdigest(), rawBytes=len(raw), compressedBytes=pathlib.Path(a.output).stat().st_size)))
