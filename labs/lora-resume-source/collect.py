"""Independent platform source oracle for the frozen resume factory; no .NET inputs."""
import argparse, hashlib, json, os, pathlib, platform, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'labs/lora-resume-source/protocol.json'

def sha(data): return hashlib.sha256(data).hexdigest()
def git(source, *args): return subprocess.check_output(['git', '-C', str(source), *args])

def preflight(source):
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    if platform.python_version() != protocol['python']: raise RuntimeError('Incorrect laboratory Python version')
    for path, expected in protocol['inputs'].items():
        if sha((ROOT / path).read_bytes()) != expected: raise RuntimeError('Collector input differs: ' + path)
    for path, expected in protocol['sources'].items():
        if sha(git(source, 'show', protocol['backendCommit'] + ':' + path)) != expected:
            raise RuntimeError('Frozen source differs: ' + path)
    return protocol

def collect(source, output):
    protocol = preflight(source)
    targets = {('Windows','amd64'):'win-x64', ('Linux','x86_64'):'linux-x64', ('Darwin','arm64'):'osx-arm64'}
    target = targets[(platform.system(), platform.machine().lower())]
    import torch, numpy, einops
    if torch.__version__ != ('2.10.0' if target == 'osx-arm64' else '2.10.0+cpu') or torch.version.cuda is not None:
        raise RuntimeError('Incorrect CPU runtime')
    if numpy.__version__ != '2.2.6' or einops.__version__ != '0.8.1': raise RuntimeError('Incorrect source dependencies')
    output.mkdir(parents=True, exist_ok=False)
    destination = output / 'resume.json'
    subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'labs/lora-training-source/resume.py'),
                    '--source', str(source), '--output', str(destination)], check=True)
    raw = destination.read_bytes(); data = json.loads(raw)
    if data['sourceCommit'] != protocol['backendCommit'] or len(data['cases']) != 4:
        raise RuntimeError('Resume identity or case count differs')
    for path, digest in data['sourceHashes'].items():
        if protocol['sources'].get(path) != digest: raise RuntimeError('Unpinned executed source: ' + path)
    if any(c['targets'] != 686 or c['parameterCount'] != 1250 or len(c['resumed']) != 7 for c in data['cases']):
        raise RuntimeError('Resume recipe differs')
    manifest = dict(profile=protocol['profile'], target=target, backendCommit=protocol['backendCommit'],
                    collectorCommit=git(ROOT, 'rev-parse', 'HEAD').decode().strip(), runId=os.environ.get('GITHUB_RUN_ID'),
                    runAttempt=os.environ.get('GITHUB_RUN_ATTEMPT'), protocolSha256=sha(PROTOCOL.read_bytes()),
                    inputs=protocol['inputs'], sources=protocol['sources'],
                    artifact=dict(file='resume.json', sha256=sha(raw), bytes=len(raw), cases=4),
                    runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=numpy.__version__,
                                 einops=einops.__version__, device='cpu', dtype='float32', threads=1,
                                 system=platform.system(), machine=platform.machine(), torchBuild=torch.__config__.show()),
                    scope='Independent frozen Python source only. No model weights or .NET outputs. Not product qualification.')
    with (output / 'manifest.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(target=target, artifact=manifest['artifact'],
                         hashes=[dict(parameters=c['allParameterSha256'], rng=c['randomStateSha256']) for c in data['cases']])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path); parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only: preflight(args.source); print('Resume source preflight passed')
    elif args.output is None: parser.error('--output is required')
    else: collect(args.source, args.output)
