"""Qualify a native-only Linux training candidate against unchanged same-host source.

Only independent copies in runner temporary storage are modified. Original test
verdicts remain in TRX; source comparison is a separate, explicitly scoped gate.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def trx(directory, expected):
    paths = list(directory.glob('*.trx'))
    require(len(paths) == 1, 'Missing or ambiguous TRX.')
    root = ET.parse(paths[0]).getroot()
    ns = {'t': root.tag.split('}')[0].lstrip('{')}
    tests = root.findall('.//t:UnitTestResult', ns)
    require(len(tests) == expected if expected is not None else len(tests) >= 865,
            'Test count changed or tests did not execute.')
    require(all(t.attrib['outcome'] in ('Passed', 'Failed') for t in tests), 'Tests were skipped or incomplete.')
    failures = []
    for test in tests:
        if test.attrib['outcome'] == 'Failed':
            failures.append({'name': test.attrib['testName'], 'message': test.findtext('.//t:Message', default='', namespaces=ns)})
    return {'executed': len(tests), 'passed': len(tests)-len(failures), 'failed': failures}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--staging', type=Path, required=True)
    a = p.parse_args()
    require((platform.system(), platform.machine()) == ('Linux', 'x86_64'), 'Linux x64 required.')
    require(sys.version_info[:3] == (3, 12, 10), 'Pinned laboratory Python required.')
    require(importlib.metadata.version('torch') == '2.10.0+cpu', 'Pinned source CPU wheel required.')
    protocol = json.loads((ROOT/'labs/training-trace/protocol.json').read_text())
    for name, sha in protocol['inputs'].items():
        require(digest(ROOT/name) == sha, 'Protocol input changed: '+name)
    # These established controls validate ELF closure, independent copies and actual loaded images.
    stage = runpy.run_path(str(ROOT/'labs/sd-native-copy-diagnostic/staging.py'))
    native = runpy.run_path(str(ROOT/'labs/sd-native-diagnostic/run.py'))
    wheel = Path(importlib.metadata.distribution('torch').locate_file('torch/lib')).resolve(strict=True)
    source = a.source_evidence.resolve(strict=True)
    built = ROOT/'tests/ComfySharp.Inference.Tests/bin/native/linux-x64/cpu/Release/net10.0'
    output, staging = stage['protect_destinations'](a.output, a.staging, (ROOT, built, wheel, source, Path(sys.prefix)))
    output.mkdir(parents=True)
    precondition = native['validate_loader_environment'](os.environ, sys.base_prefix)
    inventory = stage['inventory']
    before = inventory(built)
    bridge = built/'libComfySharp.Native.so'
    require(bridge.is_file(), 'Runtime identity bridge is missing.')
    bridge_elf = subprocess.check_output(['readelf','-d',str(bridge)],text=True)
    require('[$ORIGIN]' in bridge_elf and str(wheel) not in bridge_elf,
            'Runtime bridge must resolve native siblings without an SDK runpath.')
    wheel_before, source_before = inventory(wheel), inventory(source)
    probes, omps = stage['probe_paths'](before)
    record = native['library_record']
    original = {name: record(built/paths[0]) for name, paths in probes.items()}
    original_omp = record(built/omps[0])
    core = {name: record(wheel/name) for name in stage['CORE']}
    omp = [record(path) for path in sorted(wheel.iterdir()) if path.name.startswith('libgomp') and path.is_file()]
    require(len(omp) == 1, 'Expected one source OpenMP library.')
    omp = omp[0]
    stage['validate_elf'](core, original[stage['BINDING']], omp)
    source_manifest = json.loads((source/'manifest.json').read_text())
    require(source_manifest['sourceUnchangedByObservation'], 'Source observation was not neutral.')
    require(source_manifest['protocolSha256'] == digest(ROOT/'labs/training-trace/protocol.json'), 'Source collection protocol differs.')
    for image in (*core.values(), omp):
        loaded = [r for r in source_manifest['nativeLibraries'] if r['name'] == image['name']]
        require(len(loaded) == 1 and all(loaded[0][k] == image[k] for k in ('bytes', 'sha256')), 'Source loaded a different native image.')
    staging.mkdir(parents=True)
    builds = {'original': built, 'nuget-copy': staging/'nuget', 'source-native-copy': staging/'source-native'}
    for name in ('nuget-copy', 'source-native-copy'):
        stage['copy_original'](built, builds[name], before)
    candidate_inventory, aliases = stage['wheel_substitution'](builds['source-native-copy'], before, probes, omps, wheel, core, omp)
    write(output/'native-inputs.json', {'original': original, 'originalOpenMp': original_omp, 'candidate': core,
        'candidateOpenMp': omp, 'aliases': aliases, 'loaderPrecondition': precondition,
        'protocolSha256': digest(ROOT/'labs/training-trace/protocol.json'), 'sourceManifestSha256': digest(source/'manifest.json')})
    runs, documents, comparisons = {}, {}, {}
    for origin, build in builds.items():
        directory = output/origin
        directory.mkdir()
        env = os.environ.copy()
        env.update(COMFYSHARP_TRAINING_TRACE_DIR=str(directory/'traces'), COMFYSHARP_TRAINING_TRACE_COMPLETE='1',
                   COMFYSHARP_TRAINING_INTEROP_ONE='1', COMFYSHARP_TRAINING_NATIVE_IDENTITY='1')
        command = ['dotnet', 'vstest', str(build/'ComfySharp.Inference.Tests.dll'),
                   '--TestCaseFilter:FullyQualifiedName~SdAdapterInitializationTests', '--logger:trx',
                   '--ResultsDirectory:'+str(directory/'results'), '--TestAdapterPath:'+str(build)]
        print(origin+': start', flush=True)
        start = time.perf_counter()
        with (directory/'process.log').open('x', encoding='utf-8') as log:
            child = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        runs[origin] = {'exitCode': child.returncode, 'seconds': time.perf_counter()-start, **trx(directory/'results', 3)}
        require(child.returncode in (0, 1), 'Native test process crashed.')
        documents[origin] = []
        for case in range(2):
            document = json.loads((directory/'traces'/f'case-{case}.json').read_text())
            require(document['completed'], 'Training stopped before both updates and the unchanged base check.')
            require(document['threads'] == document['interopThreads'] == 1, 'Thread control differs.')
            require(document['native']['cpuCapability'], 'Effective libtorch CPU dispatch is missing.')
            require(document['native']['buildConfiguration'], 'Loaded libtorch build configuration is missing.')
            if origin == 'source-native-copy':
                reference = json.loads((source/f'case-{case}.json').read_text())
                require(document['native']['cpuCapability'] == reference['cpuCapability'], 'Candidate/source CPU dispatch differs.')
                require(document['native']['buildConfiguration'] == source_manifest['torchBuild'], 'Candidate/source build configuration differs.')
            require(not any('python' in r['name'].lower() for r in document['native']['libraries']), 'Python loaded in .NET.')
            native['verify_origin'](document, core if origin == 'source-native-copy' else original, original,
                original[stage['BINDING']], omp if origin == 'source-native-copy' else original_omp, original_omp,
                'wheel' if origin == 'source-native-copy' else 'nuget')
            documents[origin].append(document)
        compare_command = [sys.executable, '-I', '-S', '-B', str(ROOT/'labs/training-trace/compare.py'),
            '--source', str(source), '--actual', str(directory/'traces'), '--output', str(directory/'comparison.json'), '--require-complete']
        subprocess.run(compare_command, cwd=ROOT, check=True)
        comparisons[origin] = json.loads((directory/'comparison.json').read_text())
        print(origin+': original test exit '+str(child.returncode), flush=True)
    # The copied NuGet build must produce exactly the same evidence before attributing native differences.
    require(documents['original'] == documents['nuget-copy'], 'Copy/control invocation changed observed results.')
    for origin in ('original', 'source-native-copy'):
        build = builds[origin]
        directory = output/(origin+'-ordinary')
        directory.mkdir()
        print(origin+': ordinary inference suite', flush=True)
        command = ['dotnet', 'vstest', str(build/'ComfySharp.Inference.Tests.dll'),
                   '--TestCaseFilter:FullyQualifiedName!~ClipStockReferenceTests', '--logger:trx',
                   '--ResultsDirectory:'+str(directory/'results'), '--TestAdapterPath:'+str(build)]
        with (directory/'process.log').open('x', encoding='utf-8') as log:
            child = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, timeout=420)
        require(child.returncode in (0, 1), 'Ordinary inference process crashed.')
        expected_count = None if origin == 'original' else runs['original-ordinary']['executed']
        runs[origin+'-ordinary'] = {'exitCode': child.returncode, **trx(directory/'results', expected_count)}
    for name, build in builds.items():
        require(inventory(build) == (candidate_inventory if name == 'source-native-copy' else before), 'Build inventory changed.')
    require(inventory(wheel) == wheel_before and inventory(source) == source_before, 'Protected source inputs changed.')
    stage['verify_aliases'](builds['source-native-copy'], aliases)
    candidate = comparisons['source-native-copy']['cases']
    accepted = all(not c['baseWeightDifferences'] and not c['missingCaptures'] and c['completed'] and
                   all(not row['outsideOriginalTolerance'] for row in c['comparisons']) for c in candidate)
    write(output/'result.json', {'scope': 'Reduced SD1/SD2 all-target training only; immutable cross-host oracle verdicts remain separate.',
        'effectiveCpuDispatch':{name:[d['native']['cpuCapability'] for d in docs] for name,docs in documents.items()},
        'nativeIdentityVerified': True, 'copyControlExact': True, 'protectedInputsUnchanged': True,
        'candidateMatchesSameHostSourceTolerance': accepted, 'runs': runs, 'candidateComparison': candidate})
    # Preserve original verdicts, but gate this candidate experiment on its independently observed same-host source.
    return 0 if accepted else 1


if __name__ == '__main__':
    raise SystemExit(main())
