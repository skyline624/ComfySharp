"""Run four isolated square-operator processes; preserve committed-test failures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[2]
# Read-only stdlib helpers, not a source model or an accepted-reference producer.
HELPERS = runpy.run_path(str(REPO / 'labs/sd-diagnostic/run.py'))
require, load, digest, write, compare = (HELPERS[k] for k in ('require', 'load', 'digest', 'write', 'compare'))
MODES = ('auto', 'default')
ENV = HELPERS['ENVIRONMENT']


def labels():
    names = {'timeEmbedding', 'down2', 'output', 'input_blocks.9.0.op.input', 'input_blocks.9.0.op.output'}
    for block in (10, 11):
        prefix = f'input_blocks.{block}.0'
        names.update(prefix + '.' + suffix for suffix in ('input', 'embedding', 'output'))
        for child in ('in_layers.0', 'in_layers.1', 'in_layers.2', 'emb_layers.0', 'emb_layers.1',
                      'out_layers.0', 'out_layers.1', 'out_layers.3'):
            names.update(prefix + '.' + child + '.' + suffix for suffix in ('input', 'output'))
    return names


def verify(document):
    require(document['id'] == 'sd15-reduced/square' and document['synthetic'], 'Wrong synthetic case.')
    require(set(document['tensors']) == labels(), 'Required fine tensor labels differ.')
    for r in (*document['tensors'].values(), *document['inputs'].values()):
        HELPERS['numbers'](r)
        require(len(r['stride']) == len(r['shape']) and isinstance(r['aligned64'], bool), 'Missing layout evidence.')
    observer = document['observer']
    require(observer['sequence'] == ['off', 'on', 'off'] and observer['bitIdentical'], 'Observer neutrality failed.')
    require(observer['outputHashes'] == [document['tensors']['output']['sha256']] * 3,
            'Observer off/on/off output hashes differ.')
    parameters = {p['name']: p for p in document['parameters']}
    require(len(parameters) == len(document['parameters']) == 686, 'Wrong or duplicate parameter identities.')
    require(parameters.keys() == document['parameterLayouts'].keys(), 'Parameter layout evidence incomplete.')
    for name, p in parameters.items():
        layout = document['parameterLayouts'][name]
        require(p['shape'] == layout['shape'] and len(layout['stride']) == len(p['shape'])
                and isinstance(layout['aligned64'], bool), 'Parameter layout shape is invalid.')
    require(set(document['inputs']) == {'latent', 'timesteps', 'context'}, 'Input identities incomplete.')


def comparison(actual, expected):
    require({p['name']: p for p in actual['parameters']} == {p['name']: p for p in expected['parameters']},
            'Parameter names, shapes or bytes differ.')
    require({k: (v['shape'], v['sha256']) for k, v in actual['inputs'].items()} ==
            {k: (v['shape'], v['sha256']) for k, v in expected['inputs'].items()}, 'Input bytes or shapes differ.')
    def record(a, b):
        return {'data': compare(a, b), 'shapeEqual': a['shape'] == b['shape'],
                'strideEqual': a['stride'] == b['stride'],
                'actualStride': a['stride'], 'expectedStride': b['stride'],
                'actualAligned64': a['aligned64'], 'expectedAligned64': b['aligned64']}
    tensors = {k: record(actual['tensors'][k], expected['tensors'][k]) for k in expected['tensors']}
    return {'tensors': tensors,
            'firstDataDifferenceInSourceCaptureOrder': next((k for k,v in tensors.items() if not v['data']['byteIdentical']), None),
            'causality': 'not_assessed; an exact primitive input and comparable parameter/layout evidence are required',
            'inputs': {k: record(actual['inputs'][k], expected['inputs'][k]) for k in expected['inputs']},
            'parameterLayoutsEqual': actual['parameterLayouts'] == expected['parameterLayouts']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require((platform.system(), platform.machine()) == ('Linux', 'x86_64'), 'Use the Linux x64 diagnostic runner.')
    require(sys.version_info[:3] == (3, 12, 10), 'Use the pinned Python runtime.')
    source = args.source_directory.resolve(strict=True)
    output = args.output.resolve()
    require(args.output.is_absolute() and not output.exists(), 'Output must be absolute and absent.')
    for protected in (REPO, source):
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output), 'Output overlaps input.')
    fixtures = REPO / 'tests/ComfySharp.Inference.Tests/Fixtures'
    fixture_hashes = lambda: {str(p.relative_to(fixtures)): digest(p) for p in fixtures.rglob('*') if p.is_file()}
    before = fixture_hashes()
    accepted = load(fixtures / 'sd-components.linux-x64.manifest.json')
    scripts = lambda: {p.name: digest(p) for p in (REPO / 'labs/sd-source').glob('*.py')}
    require(scripts() == accepted['scripts'], 'Accepted source scripts changed.')
    output.mkdir(parents=True)
    cpu = {}
    for line in Path('/proc/cpuinfo').read_text().split('\n\n', 1)[0].splitlines():
        key, _, value = line.partition(':')
        if key.strip() in ('vendor_id', 'cpu family', 'model', 'model name', 'stepping', 'flags'):
            cpu[key.strip()] = value.strip()
    native = REPO / 'tests/ComfySharp.Inference.Tests/bin/native/linux-x64/cpu/Release/net10.0'
    binaries = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)}
                for p in sorted(native.rglob('*.so*')) if p.is_file()]
    require(binaries, 'Built product native libraries missing.')
    write(output / 'host.json', {'cpu': cpu, 'kernel': platform.release(),
          'runnerImage': {k: os.environ.get(k) for k in ('ImageOS', 'ImageVersion')},
          'environment': {k: os.environ.get(k) for k in ENV}, 'productNativeBinaries': binaries,
          'sourceScripts': accepted['scripts'], 'requirementsSha256': digest(REPO / 'labs/clip-source/requirements-linux-x64.txt'),
          'diagnosticScripts': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
          'comparisonHelperSha256': digest(REPO / 'labs/sd-diagnostic/run.py'),
          'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()})
    runs, errors = [], []
    for mode in MODES:
        env = os.environ.copy()
        for key in ('ATEN_CPU_CAPABILITY', 'COMFYSHARP_SD_UNET_FINE_TRACE_DIR', 'COMFYSHARP_SD_UNET_TRACE_DIR',
                    'COMFYSHARP_SD_LATENT_OFFSET', 'COMFYSHARP_SD_GUIDANCE_TRACE_DIR'):
            env.pop(key, None)
        if mode != 'auto': env['ATEN_CPU_CAPABILITY'] = mode
        commands = [('source', [sys.executable, '-I', '-B', str(Path(__file__).parent / 'source.py'),
                               '--source-directory', str(source), '--output', str(output / mode / 'source')]),
                    ('product', ['dotnet', 'test', 'tests/ComfySharp.Inference.Tests', '--no-build', '--no-restore',
                                 '-c', 'Release', '-p:NativeRuntime=linux-x64', '--filter',
                                 'FullyQualifiedName~SdUnetReferenceTests&DisplayName~sd15-reduced&DisplayName~square',
                                 '--logger', 'trx', '--results-directory', str(output / mode / 'product/results')])]
        for kind, command in commands:
            directory = output / mode / (kind + '-process' if kind == 'source' else kind)
            directory.mkdir(parents=True)
            child_env = env.copy()
            if kind == 'product': child_env['COMFYSHARP_SD_UNET_FINE_TRACE_DIR'] = str(directory / 'traces')
            start = time.perf_counter()
            print(mode + '/' + kind + ': start', flush=True)
            try:
                result = subprocess.run(command, cwd=REPO, env=child_env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', timeout=300)
                code, text = result.returncode, result.stdout
            except subprocess.TimeoutExpired as error:
                code, text = 124, error.stdout or ''
                if isinstance(text, bytes): text = text.decode('utf-8', errors='replace')
                text += '\nDiagnostic process exceeded 300 seconds.\n'
            except OSError as error:
                code, text = 127, str(error)
            for private in (str(REPO), str(source), str(output), str(Path.home()), str(Path(sys.prefix))):
                text = text.replace(private, '<local>')
            (directory / 'process.log').write_text(text, encoding='utf-8')
            runs.append({'label': mode + '/' + kind, 'exitCode': code, 'seconds': time.perf_counter()-start,
                         'environment': {k: child_env.get(k) for k in ENV}})
            write(output / 'runs.json', runs)
            print(mode + '/' + kind + ': exit ' + str(code), flush=True)
    documents = {}
    for mode in MODES:
        try:
            result_files = list((output / mode / 'product/results').glob('*.trx'))
            require(len(result_files) == 1, 'Exactly one product TRX is required.')
            tree = ET.parse(result_files[0]).getroot()
            ns = {'t': tree.tag.split('}')[0].lstrip('{')}
            tests = tree.findall('.//t:UnitTestResult', ns)
            require(len(tests) == 1 and tests[0].attrib['outcome'] in ('Passed', 'Failed'),
                    'The operator filter must execute exactly one existing test, without skips.')
            name = tests[0].attrib['testName']
            require('FullGraphMatchesSamePlatformFrozenSource' in name and 'sd15-reduced' in name and 'square' in name,
                    'The selected product test is not the existing SD15 square case.')
        except Exception as error:
            errors.append(mode + '/product-trx: ' + str(error))
        for kind, relative in (('source', 'source/operators.json'), ('product', 'product/traces/sd15-reduced--square.json')):
            try:
                doc = load(output / mode / relative)
                verify(doc)
                if kind == 'source':
                    require(doc['sourceScripts'] == accepted['scripts'], 'Source script identities differ.')
                    require(doc['backendCommit'] == accepted['backendCommit'] and doc['profile'] == accepted['profile'],
                            'Source provenance differs.')
                    require(doc['runtime']['threads'] == doc['runtime']['interopThreads'] == 1, 'Source threads differ.')
                    require(doc['runtime']['requestedAtenCpuCapability'] == (None if mode == 'auto' else mode),
                            'Requested source capability differs.')
                    if mode == 'default': require(doc['runtime']['cpuCapability'] in ('DEFAULT', 'NO AVX'), 'DEFAULT not observed.')
                else:
                    require(doc['native']['threads'] == doc['native']['interopThreads'] == 1, 'Product threads differ.')
                    require(doc['native']['requestedCapability'] == mode, 'Requested product capability differs.')
                    built = {p['name']: (p['bytes'], p['sha256']) for p in binaries}
                    require(doc['native']['libraries'], 'Loaded native product evidence missing.')
                    require(all(built.get(p['name']) == (p['bytes'], p['sha256']) for p in doc['native']['libraries']),
                            'Loaded product native library differs from build inventory.')
                documents[mode + '/' + kind] = doc
            except Exception as error:
                errors.append(mode + '/' + kind + ': ' + str(error))
    comparisons = {'diagnosticOnly': True, 'qualification': 'none', 'productVsSource': {}, 'sourceVsSource': {}}
    for pm in MODES:
        for sm in MODES:
            try:
                comparisons['productVsSource'][pm + '-product_vs_' + sm + '-source'] = comparison(
                    documents[pm + '/product'], documents[sm + '/source'])
            except Exception as error: errors.append(pm + '/' + sm + ': ' + str(error))
    try:
        a,b=documents['auto/source'],documents['default/source']
        require(a['sources'] == b['sources'] and a['sourceConfig'] == b['sourceConfig'], 'Exact source AST/config differ.')
        for key in ('python','torch','torchGit','threads','interopThreads','nativeLibraries'):
            require(a['runtime'][key] == b['runtime'][key], 'Non-dispatch runtime changed: ' + key)
        require(HELPERS['invariant_build'](a['runtime']['build']) == HELPERS['invariant_build'](b['runtime']['build']),
                'Non-dispatch source build changed.')
        comparisons['sourceVsSource']['auto-expected_vs_default'] = comparison(b,a)
    except Exception as error: errors.append('source/source: ' + str(error))
    try:
        a,b=documents['auto/product']['native'],documents['default/product']['native']
        for key in ('torchSharp','declaredLibtorchPackage','threads','interopThreads','os','architecture','dotnet'):
            require(a[key] == b[key], 'Product runtime identity changed: ' + key)
        require({p['name']:(p['bytes'],p['sha256']) for p in a['libraries']} ==
                {p['name']:(p['bytes'],p['sha256']) for p in b['libraries']}, 'Product native libraries changed.')
    except Exception as error: errors.append('product/runtime: ' + str(error))
    if fixture_hashes() != before or scripts() != accepted['scripts']:
        errors.append('Protected inputs changed.')
    write(output / 'comparisons.json', comparisons)
    write(output / 'status.json', {'runs': runs, 'integrityErrors': errors,
          'qualification': 'none; committed test failures remain failures',
          'failedProcesses': sum(r['exitCode'] != 0 for r in runs)})
    return 1 if errors or any(r['exitCode'] for r in runs) else 0


if __name__ == '__main__':
    raise SystemExit(main())
