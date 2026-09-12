"""Development-only collector. Distributed .NET tests consume JSON, never Python."""
import argparse, ast, hashlib, json, math, subprocess, types
import numpy as np
import torch

COMMIT = '1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a'
parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
assert torch.__version__ == '2.10.0+cpu'
namespace = {'torch': torch, 'np': np, 'math': math}
pins = {}
def declarations(path, names, owner=None):
    raw = subprocess.check_output(['git', '-C', args.source, 'show', COMMIT + ':' + path])
    pins[path] = hashlib.sha256(raw).hexdigest()
    tree = ast.parse(raw)
    body = tree.body if owner is None else next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    selected = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, 'exec'), namespace)

declarations('comfy/ldm/modules/diffusionmodules/util.py', ['make_beta_schedule'])
declarations('comfy/model_sampling.py', ['ModelSamplingDiscrete'])
declarations('comfy/k_diffusion/sampling.py', ['append_zero', 'get_sigmas_karras'])
declarations('comfy/samplers.py', ['set_steps'], 'KSampler')
declarations('comfy/samplers.py', ['max_denoise'], 'Sampler')
sampling = namespace['ModelSamplingDiscrete']()
cases = []
for steps, denoise in [(1, 1), (5, 1), (5, .99991), (5, .9999), (5, .5), (7, .3333), (3, .01), (5, 0)]:
    calls = []
    def calculate(count):
        calls.append(count)
        return namespace['get_sigmas_karras'](count, float(sampling.sigma_min), float(sampling.sigma_max))
    holder = types.SimpleNamespace(device='cpu', calculate_sigmas=calculate)
    namespace['set_steps'](holder, steps, denoise)
    maximum = None if holder.sigmas.numel() == 0 else namespace['max_denoise'](None,
        types.SimpleNamespace(inner_model=types.SimpleNamespace(model_sampling=sampling)), holder.sigmas)
    cases.append({'steps': steps, 'denoise': denoise, 'calculatedSteps': calls, 'sigmas': holder.sigmas.tolist(), 'maximumNoise': maximum})
with open(args.output, 'x', encoding='utf-8', newline='\n') as output:
    json.dump({'sourceCommit': COMMIT, 'sourceHashes': pins, 'torch': torch.__version__,
        'scope': 'KSampler.set_steps and Sampler.max_denoise with real source Karras and model sampling functions; no model-family qualification',
        'cases': cases}, output, indent=2)
    output.write('\n')
