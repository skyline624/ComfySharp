"""Separate square SD15 operator diagnostic; never generates accepted references."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    source = args.source_directory.resolve(strict=True)
    require(source.is_dir(), 'Source snapshot must be a directory.')
    require(args.output.is_absolute(), 'Output must be absolute.')
    output = args.output.resolve()
    require(not output.exists(), 'Output must be absent; evidence is never overwritten.')
    for protected in (repo, source):
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output),
                'Output overlaps protected input.')
    lab = repo / 'labs/sd-source'
    accepted = json.loads((repo / 'tests/ComfySharp.Inference.Tests/Fixtures/sd-components.linux-x64.manifest.json').read_text())
    scripts = {p.name: digest(p) for p in lab.glob('*.py')}
    require(scripts == accepted['scripts'], 'Accepted source scripts changed.')
    require(sys.version_info[:3] == (3, 12, 10), 'Use Python 3.12.10.')
    require(importlib.metadata.version('torch') in ('2.10.0', '2.10.0+cpu'), 'Use the pinned PyTorch 2.10 CPU laboratory.')
    # Import only the verified, unchanged laboratory utilities and exact AST loader.
    sys.path.insert(0, str(lab))
    import torch
    from common import COMMIT, PROFILE, fill_parameters, synthetic_tensor, tensor_record
    from unet import source_configuration, source_model_type
    require(torch.version.cuda is None, 'CUDA wheels are outside this diagnostic.')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target = {('Linux', 'x86_64'): 'linux-x64', ('Windows', 'AMD64'): 'win-x64'}.get((platform.system(), platform.machine()))
    require(target is not None, 'Use Linux x64 or the declared Windows development laboratory.')
    evidence = []
    model_type, operations = source_model_type(source, evidence)
    configuration = source_configuration(False)
    captures = {}

    def record(value):
        result = tensor_record(value)
        result.update(stride=list(value.stride()), aligned64=value.data_ptr() % 64 == 0)
        return result

    with torch.no_grad():
        model = model_type(**configuration, dtype=torch.float32, device='meta', operations=operations)
        model.to_empty(device='cpu')
        model.requires_grad_(False)
        model.eval()
        parameters = fill_parameters(model)
        require(len(parameters) == 686, 'Full plain U-Net parameter count changed.')
        parameter_layouts = {name: {'shape': list(p.shape), 'stride': list(p.stride()), 'aligned64': p.data_ptr() % 64 == 0}
                             for name, p in model.named_parameters()}
        identifier = 'sd15-reduced/square'
        latent = synthetic_tensor(identifier + '/latent', (1, 4, 8, 8))
        context = synthetic_tensor(identifier + '/context', (1, 3, 16))
        timesteps = torch.tensor([0.125], dtype=torch.float32, device='cpu')

        def forward():
            return model._forward(latent, timesteps=timesteps, context=context, y=None, control=None,
                                  transformer_options={})

        # Hold returned tensors so snapshots cannot be affected by later storage reuse.
        before = forward()
        before_record = record(before)
        hooks = []

        def capture(name, value):
            require(name not in captures, 'Operator boundary executed more than once: ' + name)
            captures[name] = record(value)

        def hook_output(module_name, label):
            hooks.append(model.get_submodule(module_name).register_forward_hook(
                lambda _module, _args, value, name=label: capture(name, value)))

        def hook_input(module_name, label, argument=0):
            hooks.append(model.get_submodule(module_name).register_forward_pre_hook(
                lambda _module, args, name=label, arg=argument: capture(name, args[arg])))

        hook_output('time_embed', 'timeEmbedding')
        hook_output('input_blocks.8.1', 'down2')
        hook_input('input_blocks.9.0.op', 'input_blocks.9.0.op.input')
        hook_output('input_blocks.9.0.op', 'input_blocks.9.0.op.output')
        for index in (10, 11):
            prefix = f'input_blocks.{index}.0'
            hook_input(prefix, prefix + '.input')
            hook_input(prefix, prefix + '.embedding', 1)
            for child in ('in_layers.0', 'in_layers.1', 'in_layers.2', 'emb_layers.0', 'emb_layers.1',
                          'out_layers.0', 'out_layers.1', 'out_layers.3'):
                hook_input(prefix + '.' + child, prefix + '.' + child + '.input')
                hook_output(prefix + '.' + child, prefix + '.' + child + '.output')
            # out_layers.0.input is the exact functional time-add result, not reconstructed.
            hook_output(prefix, prefix + '.output')
        try:
            observed = forward()
            observed_record = record(observed)
        finally:
            for hook in hooks:
                hook.remove()
        after = forward()
        after_record = record(after)
        hashes = [r['sha256'] for r in (before_record, observed_record, after_record)]
        stable = len(set(hashes)) == 1
        captures['output'] = observed_record
        require(len(captures) == 43, 'An expected source operator boundary was not captured.')
        inputs = {'latent': record(latent), 'timesteps': record(timesteps), 'context': record(context)}
        actual_parameters = dict(model.named_parameters())
        for entry in parameters:
            raw = actual_parameters[entry['name']].detach().contiguous().numpy().tobytes()
            require(hashlib.sha256(raw).hexdigest() == entry['sha256'], 'Actual parameter bytes changed.')

    libraries = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)}
                 for p in sorted((Path(torch.__file__).resolve().parent / 'lib').iterdir())
                 if p.is_file() and (p.suffix.lower() in ('.dll', '.so', '.dylib') or '.so.' in p.name)]
    build = torch.__config__.show()
    for private in (str(Path.home()), str(Path(sys.prefix)), str(source), str(output), str(repo)):
        build = build.replace(private, '<local>')
    document = {'id': identifier, 'diagnosticOnly': True, 'synthetic': True,
                'backendCommit': COMMIT, 'profile': PROFILE, 'target': target, 'sources': evidence,
                'sourceScripts': scripts, 'diagnosticScriptSha256': digest(Path(__file__)),
                'sourceConfig': configuration, 'parameters': parameters, 'parameterLayouts': parameter_layouts,
                'inputs': inputs, 'tensors': captures,
                'observer': {'sequence': ['off', 'on', 'off'], 'outputHashes': hashes, 'bitIdentical': stable},
                'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                            'torchGit': torch.version.git_version, 'threads': torch.get_num_threads(),
                            'interopThreads': torch.get_num_interop_threads(), 'build': build,
                            'requestedAtenCpuCapability': os.environ.get('ATEN_CPU_CAPABILITY'),
                            'cpuCapability': torch.backends.cpu.get_cpu_capability(), 'nativeLibraries': libraries},
                'capture': 'Exact module pre/post hooks; functional time-add observed at out_layers input; no replacement math.'}
    require({p.name: digest(p) for p in lab.glob('*.py')} == scripts, 'Accepted source scripts changed during capture.')
    output.mkdir(parents=True)
    (output / 'operators.json').write_text(json.dumps(document, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    require(stable, 'Observer off/on/off altered source output; evidence is diagnostic-invalid.')
    print('Source square operator capture complete; observer off/on/off bit-identical.')


if __name__ == '__main__':
    main()
