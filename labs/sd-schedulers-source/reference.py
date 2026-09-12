"""Frozen scheduler functions; development laboratory only, no product Python dependency."""
import argparse, ast, hashlib, json, math, subprocess, sys, types
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--scipy-path', help='Optional existing Python 3.12 site-packages providing SciPy; no install or modification')
args = parser.parse_args()
assert torch.__version__ == '2.10.0+cpu'
if args.scipy_path: sys.path.append(args.scipy_path)
import scipy, scipy.stats
torch.set_num_threads(1); torch.set_num_interop_threads(1)
COMMIT = '1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a'
namespace = {'torch': torch, 'np': np, 'numpy': np, 'math': math, 'scipy': scipy}
pins = {}
def load(path, names, owner=None):
    raw = subprocess.check_output(['git', '-C', args.source, 'show', COMMIT + ':' + path])
    pins[path] = hashlib.sha256(raw).hexdigest()
    tree = ast.parse(raw)
    body = tree.body if owner is None else next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    selected = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, 'exec'), namespace)
load('comfy/ldm/modules/diffusionmodules/util.py', ['make_beta_schedule'])
load('comfy/model_sampling.py', ['ModelSamplingDiscrete'])
load('comfy/k_diffusion/sampling.py', ['append_zero', 'get_sigmas_karras', 'get_sigmas_exponential'])
load('comfy/samplers.py', ['simple_scheduler', 'ddim_scheduler', 'normal_scheduler', 'beta_scheduler', 'linear_quadratic_schedule', 'kl_optimal_scheduler'])
load('comfy/samplers.py', ['set_steps'], 'KSampler')
sampling = namespace['ModelSamplingDiscrete']()
routes = {'simple': lambda n: namespace['simple_scheduler'](sampling,n),
 'ddim_uniform':lambda n: namespace['ddim_scheduler'](sampling,n),
 'normal':lambda n: namespace['normal_scheduler'](sampling,n),
 'sgm_uniform':lambda n: namespace['normal_scheduler'](sampling,n,sgm=True),
 'beta':lambda n: namespace['beta_scheduler'](sampling,n),
 'linear_quadratic':lambda n: namespace['linear_quadratic_schedule'](sampling,n),
 'karras':lambda n: namespace['get_sigmas_karras'](n,float(sampling.sigma_min),float(sampling.sigma_max)),
 'exponential':lambda n: namespace['get_sigmas_exponential'](n,float(sampling.sigma_min),float(sampling.sigma_max)),
 'kl_optimal':lambda n: namespace['kl_optimal_scheduler'](n,float(sampling.sigma_min),float(sampling.sigma_max))}
cases=[]
for name, calculate in routes.items():
    options=[(1,1),(2,1),(7,1),(20,1),(7,.5),(3,.01),(20,0)]
    if name in ('ddim_uniform','beta'):options += [(333,1),(1001,1),(10000,1),(333,.5)]
    for steps,denoise in options:
        holder=types.SimpleNamespace(device='cpu',calculate_sigmas=calculate)
        namespace['set_steps'](holder,steps,denoise)
        valid=bool(holder.sigmas.isfinite().all())
        cases.append({'scheduler':name,'steps':steps,'denoise':denoise,'finite':valid,'sigmas':holder.sigmas.tolist() if valid else None})
with open(args.output,'x',encoding='utf-8',newline='\n') as output:
    json.dump({'sourceCommit':COMMIT,'sourceHashes':pins,'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,
        'absoluteTolerance':1e-6,'relativeTolerance':1e-6,'scope':'Nine KSampler schedulers and source set_steps with default SD discrete sampling. No model-family qualification. Nonfinite source results must be diagnosed.', 'cases':cases},output,indent=2,allow_nan=False)
    output.write('\n')
