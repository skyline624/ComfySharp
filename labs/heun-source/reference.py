"""Development-only frozen-function collector; .NET tests consume only its JSON."""
import argparse, ast, hashlib, json, subprocess, types
import torch

COMMIT = '1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a'
parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
assert torch.__version__ == '2.10.0+cpu'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
namespace = {'torch': torch, 'trange': lambda n, **kw: range(n)}
pins = {}
def load(path, names, scope):
    raw = subprocess.check_output(['git', '-C', args.source, 'show', COMMIT + ':' + path])
    pins[path] = hashlib.sha256(raw).hexdigest()
    selected = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, 'exec'), scope)
utils = {'torch': torch}
load('comfy/k_diffusion/utils.py', ['append_dims'], utils)
namespace['utils'] = types.SimpleNamespace(append_dims=utils['append_dims'])
load('comfy/k_diffusion/sampling.py', ['to_d', 'sample_heun'], namespace)
cases = []
for sigmas in ([1.0, 0.0], [2.0, 0.75, 0.125, 0.0], [1.0, 1.0, 0.5, 0.0], [0.01, 0.005, 0.0]):
    initial = torch.tensor([0.125, -0.75, 2.5, -3.125, 0.0, 0.25, -0.5, 1.25], dtype=torch.float32).reshape(2, 4, 1, 1)
    calls = []
    def model(x, sigma):
        calls.append({'x': x.flatten().tolist(), 'sigma': sigma.tolist()})
        # Analytical denoiser, not a substituted neural network or a model qualification.
        return x * 0.25 + sigma.reshape(2, 1, 1, 1) * 0.125
    output = namespace['sample_heun'](model, initial, torch.tensor(sigmas), disable=True)
    cases.append({'sigmas': sigmas, 'initial': initial.flatten().tolist(), 'calls': calls, 'output': output.flatten().tolist()})
with open(args.output, 'x', encoding='utf-8', newline='\n') as output:
    json.dump({'sourceCommit': COMMIT, 'sourceHashes': pins, 'torch': torch.__version__,
        'absoluteTolerance': 1e-6, 'relativeTolerance': 1e-6,
        'scope': 'Heun no-churn ODE arithmetic and model evaluation inputs with an analytical denoiser; no model-family qualification', 'cases': cases}, output, indent=2)
    output.write('\n')
