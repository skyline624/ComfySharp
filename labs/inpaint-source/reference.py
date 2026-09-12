"""Development-only frozen SD mask/preprocessing reference collector; no model loading."""
import argparse, ast, hashlib, json, math, subprocess, types
import torch

p = argparse.ArgumentParser()
p.add_argument('--source', required=True); p.add_argument('--output', required=True)
a = p.parse_args()
assert torch.__version__ == '2.10.0+cpu'
torch.set_num_threads(1); torch.set_num_interop_threads(1)
commit = '1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a'
hashes = {}
ns = {'torch': torch, 'math': math}
def extract(path, names):
    raw = subprocess.check_output(['git', '-C', a.source, 'show', commit + ':' + path])
    hashes[path] = hashlib.sha256(raw).hexdigest()
    tree = ast.parse(raw)
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(defs) == len(names)
    exec(compile(ast.Module(body=defs, type_ignores=[]), path, 'exec'), ns)
extract('comfy/utils.py', ['reshape_mask', 'repeat_to_batch_size'])
extract('comfy/model_sampling.py', ['reshape_sigma', 'EPS'])
extract('comfy/samplers.py', ['KSamplerX0Inpaint'])
extract('nodes.py', ['VAEEncodeForInpaint', 'SetLatentNoiseMask'])
def flat(t): return t.flatten().tolist()
def packed(t): return {'shape': list(t.shape), 'values': flat(t)}
mask = torch.tensor([0, .25, .5, 1, 1.25, .75, .5, -.25], dtype=torch.float32).reshape(2, 2, 2)
prepared = []
for shape in ([3, 4, 2, 3], [1, 4, 3, 1]):
    prepared.append({'mask': packed(mask), 'shape': shape, 'output': packed(ns['reshape_mask'](mask, shape))})
calls = []
for mask in [torch.zeros(2, 2, 3), torch.ones(2, 2, 3), torch.arange(12).reshape(2, 2, 3).float() / 11]:
    latent = torch.arange(48).reshape(2, 4, 2, 3).float() / 32 - .5
    noise = latent.flip(-1) * .75
    current = latent.flip(-2) * 1.25
    recorded = []
    sampling = ns['EPS']()
    class Model:
        def __init__(self): self.inner_model = self
        def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
            return sampling.noise_scaling(sigma.reshape(-1, 1, 1, 1), noise, latent_image)
        def __call__(self, x, sigma, **kwargs):
            recorded.append(packed(x)); return x * .25 + sigma.reshape(-1, 1, 1, 1) * .125
    wrapper = ns['KSamplerX0Inpaint'](Model(), None)
    wrapper.noise = noise; wrapper.latent_image = latent
    ready = ns['reshape_mask'](mask, list(latent.shape))
    outputs = []
    for sigma in [torch.tensor([2., 1.]), torch.tensor([.75, .1]), torch.tensor([0., 0.])]:
        outputs.append({'sigma': packed(sigma), 'output': packed(wrapper(current, sigma, ready))})
    calls.append({'mask': packed(mask), 'latent': packed(latent), 'noise': packed(noise), 'current': packed(current), 'modelInputs': recorded, 'outputs': outputs})
images = []
for grow in [0, 1, 2, 6, 64]:
    pixels = (torch.arange(2*9*11*4) % 37).float().reshape(2, 9, 11, 4) / 36
    mask = torch.tensor([0., .5, 1., .25, .75, 1.]).reshape(1, 2, 3)
    class CaptureVae:
        def spacial_compression_encode(self): return 8
        def encode(self, pixels): self.pixels = pixels.clone(); return torch.empty(0)
    vae = CaptureVae()
    out = ns['VAEEncodeForInpaint']().encode(vae, pixels, mask, grow)[0]
    images.append({'grow': grow, 'pixels': packed(pixels), 'mask': packed(mask), 'preparedPixels': packed(vae.pixels), 'noiseMask': packed(out['noise_mask'])})
with open(a.output, 'x', encoding='utf-8', newline='\n') as f:
    json.dump({'sourceCommit': commit, 'sourceHashes': hashes, 'torch': torch.__version__,
        'absoluteTolerance': 1e-6, 'relativeTolerance': 1e-6,
        'scope': 'Actual source mask transforms and inpaint wrapper with analytical prediction and capture-only VAE; not pretrained parity.',
        'prepared': prepared, 'calls': calls, 'images': images}, f, indent=2, allow_nan=False)
    f.write('\n')
