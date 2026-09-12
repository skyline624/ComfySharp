"""Frozen SD training bypass oracle. Isolated CPU lab, never used by .NET tests."""
import argparse, ast, hashlib, json, logging, pathlib, subprocess, sys, tempfile
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
assert torch.__version__ == '2.10.0+cpu'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
repo = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'labs/sd-source'))
from unet import source_model_type, source_configuration
from common import fill_parameters, synthetic_tensor, tensor_record

commit = '1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a'
hashes, evidence = {}, []
ns = dict(torch=torch, nn=nn, F=F, logging=logging, Optional=Optional, Callable=Callable, BypassAdapter=object)

def extract(path, names):
    raw = subprocess.check_output(['git', '-C', a.source, 'show', commit + ':' + path])
    hashes[path] = hashlib.sha256(raw).hexdigest()
    selected = [n for n in ast.parse(raw).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, 'exec'), ns)

extract('comfy/weight_adapter/base.py', ['WeightAdapterBase', 'WeightAdapterTrainBase', 'tucker_weight_from_conv'])
extract('comfy/weight_adapter/lora.py', ['LoraDiff'])
extract('comfy/weight_adapter/bypass.py', ['get_module_type_info', 'BypassForwardHook'])
extract('comfy_extras/nodes_train.py', ['BiasDiff'])
with tempfile.TemporaryDirectory(prefix='comfysharp-training-bypass-source-') as directory:
    snapshot = pathlib.Path(directory)
    for relative in ['comfy/ldm/modules/diffusionmodules/util.py', 'comfy/ldm/modules/attention.py', 'comfy/ldm/modules/diffusionmodules/openaimodel.py']:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(subprocess.check_output(['git', '-C', a.source, 'show', commit + ':' + relative]))
    model_type, operations = source_model_type(snapshot.resolve(), evidence)

cases = []
for linear in [False, True]:
    model = model_type(**source_configuration(linear), dtype=torch.float32, device='meta', operations=operations)
    model.to_empty(device='cpu')
    model.requires_grad_(False)
    model.eval()
    fill_parameters(model)
    model.forward = model._forward
    parameters, modules = dict(model.named_parameters()), dict(model.named_modules())
    adapters, differences, initial, hooks = {}, {}, {}, []
    for prefix in ['time_embed.0', 'input_blocks.0.0', 'input_blocks.1.1.transformer_blocks.0.attn1.to_q',
                   'input_blocks.1.1.proj_in', 'input_blocks.3.0.op', 'out.2']:
        name = prefix + '.weight'
        weight = parameters[name]
        up = synthetic_tensor(name + '/up', [weight.shape[0], 1]) * .03125
        down = synthetic_tensor(name + '/down', [1, weight.numel() // weight.shape[0]]) * .03125
        adapter = ns['LoraDiff']((up, down, 1.25, None, None, None)).requires_grad_(True)
        adapters[name] = adapter
        initial[name] = dict(up=tensor_record(up), down=tensor_record(down), alpha=tensor_record(adapter.alpha))
        hook = ns['BypassForwardHook'](modules[prefix], adapter)
        # All objects already reside on CPU/F32. Device-dispatch portion of inject is outside this oracle.
        hook.original_forward = modules[prefix].forward
        modules[prefix].forward = hook._bypass_forward
        hooks.append(hook)
    for name in ['out.0.weight', 'out.2.bias']:
        value = synthetic_tensor(name + '/diff', list(parameters[name].shape)) * .03125
        differences[name] = ns['BiasDiff'](nn.Parameter(value.clone()))
        initial[name] = dict(difference=tensor_record(value))

    def values(grad=False):
        result = {}
        for name, adapter in adapters.items():
            row = dict(up=adapter.lora_up.weight, down=adapter.lora_down.weight, alpha=adapter.alpha)
            result[name] = {k: tensor_record(v.grad if grad else v) for k, v in row.items()}
        for name, adapter in differences.items():
            value = next(adapter.parameters())
            result[name] = dict(difference=tensor_record(value.grad if grad else value))
        return result

    latent = synthetic_tensor('training/latent', [1,4,8,8])
    context = synthetic_tensor('training/context', [1,3,16])
    times = torch.tensor([17.25])
    target = synthetic_tensor('training/target', [1,4,8,8])
    optimizer = torch.optim.SGD([v for adapter in [*adapters.values(), *differences.values()] for v in adapter.parameters()], lr=.01)
    steps = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        regular = {name: adapter(parameters[name]) for name, adapter in differences.items()}
        output = torch.func.functional_call(model, regular, (latent,), dict(timesteps=times, context=context, transformer_options={}), strict=False)
        loss = F.mse_loss(output, target)
        loss.backward()
        gradients = values(True)
        assert all(v.grad is None for v in parameters.values())
        optimizer.step()
        steps.append(dict(output=tensor_record(output), loss=loss.item(), gradients=gradients, updated=values()))
    cases.append(dict(linearProjection=linear, initial=initial, latent=tensor_record(latent), context=tensor_record(context),
                      timesteps=tensor_record(times), target=tensor_record(target), learningRate=.01, steps=steps))
helpers = {str(path.relative_to(repo)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
           for path in [repo/'labs/sd-source/unet.py', repo/'labs/sd-source/common.py']}
with open(a.output, 'x', encoding='utf-8', newline='\n') as destination:
    json.dump(dict(sourceCommit=commit, sourceHashes=hashes, sourceDeclarations=evidence, helperHashes=helpers,
                   torch=torch.__version__, absoluteTolerance=3e-5, relativeTolerance=3e-5, cases=cases,
                   scope='Frozen reduced-width SD1/SD2 _forward, LoraDiff.h and BypassForwardHook._bypass_forward, six factor targets with trainable alpha and two BiasDiff targets; two SGD steps. No full node, precision/offload or pretrained qualification.'), destination, indent=2, allow_nan=False)
    destination.write('\n')
print('Collected two source bypass topologies, eight targets each, two SGD updates.')
