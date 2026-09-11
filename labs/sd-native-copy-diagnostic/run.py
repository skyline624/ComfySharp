"""Eight isolated processes: source and identical VSTest launches from three origins."""
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

REPO = Path(__file__).resolve().parents[2]
PINS = {
    'labs/sd-native-diagnostic/run.py': '6ffa862eb73757e6c9d8fdf664a54724e414bf10fd648007790c4713ca13a7f1',
    'labs/sd-operator-diagnostic/source.py': 'd269a0fe63530e10f7397d0071bb4391518ff6200d1b892b842b9380f58237ab',
    'labs/sd-operator-diagnostic/run.py': 'a9bce23f9c2ac1d600f0b297a8f38e38cd2effbfd83caba0dde52d8d680d435a',
    'labs/sd-diagnostic/run.py': 'ddd56f206e1642156d8748d23409a504baabfe218cc3e4979797f46c29544e57',
}
MODES = ('auto', 'default')
ORIGINS = ('original', 'nuget-copy', 'wheel-copy')
FILTER = 'FullyQualifiedName~SdUnetReferenceTests&DisplayName~sd15-reduced&DisplayName~square'
TEST_DLL = 'ComfySharp.Inference.Tests.dll'


def require(value, message):
    if not value: raise ValueError(message)


def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def redact(text, roots):
    for value in sorted({str(p) for p in roots},key=len,reverse=True):
        text = text.replace(value,'<local>')
    return text


def helpers():
    for name, sha in PINS.items(): require(digest(REPO/name) == sha, 'Pinned helper changed: '+name)
    native = runpy.run_path(str(REPO/'labs/sd-native-diagnostic/run.py'))
    return native, native['pinned_helpers'](), runpy.run_path(str(Path(__file__).with_name('staging.py')))


def vstest_command(build, results):
    return ['dotnet','vstest',str(build/TEST_DLL),'--TestCaseFilter:'+FILTER,
            '--logger:trx','--ResultsDirectory:'+str(results),'--TestAdapterPath:'+str(build)]


def verify_trx(directory):
    paths = list(directory.glob('*.trx'))
    require(len(paths) == 1, 'Expected exactly one TRX.')
    tree = ET.parse(paths[0]).getroot()
    ns = {'t':tree.tag.split('}')[0].lstrip('{')}
    tests = tree.findall('.//t:UnitTestResult',ns)
    require(len(tests) == 1 and tests[0].attrib['outcome'] in ('Passed','Failed'), 'Expected one executed test, without skips.')
    name = tests[0].attrib['testName']
    require(all(x in name for x in ('FullGraphMatchesSamePlatformFrozenSource','sd15-reduced','square')), 'Unexpected test selected.')


def exact_copy_control(comparison):
    return comparison['parameterLayoutsEqual'] and all(
        r['data']['byteIdentical'] and r['shapeEqual'] and r['strideEqual'] and
        r['actualAligned64'] == r['expectedAligned64']
        for category in ('inputs','tensors') for r in comparison[category].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--staging-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    require((platform.system(),platform.machine()) == ('Linux','x86_64'), 'Use Linux x64.')
    require(sys.version_info[:3] == (3,12,10), 'Use pinned Python.')
    require(importlib.metadata.version('torch') == '2.10.0+cpu', 'Use pinned CPU wheel.')
    native, op, stage = helpers()
    source = args.source_directory.resolve(strict=True)
    wheel = Path(importlib.metadata.distribution('torch').locate_file('torch/lib')).resolve(strict=True)
    built = REPO/'tests/ComfySharp.Inference.Tests/bin/native/linux-x64/cpu/Release/net10.0'
    require(built.is_dir(), 'Original build is missing.')
    output, staging = stage['protect_destinations'](args.output,args.staging_directory,
        (REPO,source,built,wheel,Path(sys.prefix),Path(sys.base_prefix)))
    output.mkdir(parents=True)
    private_roots = (REPO,source,output,staging,wheel,Path.home(),Path(sys.prefix),Path(sys.base_prefix))
    runs, errors = [], []
    def status(phase, complete=False, **extra):
        write(output/'status.json',{'phase':phase,'complete':complete,'plannedProcesses':8,'runs':runs,
              'integrityErrors':[redact(e,private_roots) for e in errors],
              'qualification':'none; acceptance references unchanged',**extra})
    status('preflight')
    try:
        precondition = native['validate_loader_environment'](os.environ,sys.base_prefix)
        write(output/'precondition.json',{'valid':True,**precondition})
        inventory = stage['inventory']
        original_before = inventory(built)
        wheel_before = inventory(wheel)
        source_before = inventory(source)
        fixtures = REPO/'tests/ComfySharp.Inference.Tests/Fixtures'
        fixtures_before = inventory(fixtures)
        scripts = lambda: {p.name:digest(p) for p in (REPO/'labs/sd-source').glob('*.py')}
        accepted = json.loads((fixtures/'sd-components.linux-x64.manifest.json').read_text())
        require(scripts() == accepted['scripts'], 'Accepted six source scripts changed.')
        script_before = {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
        probes, openmp_paths = stage['probe_paths'](original_before)
        record = native['library_record']
        core_names, binding_name = stage['CORE'],stage['BINDING']
        original = {n:record(built/probes[n][0]) for n in (*core_names,binding_name)}
        original_openmp = record(built/openmp_paths[0])
        wheel_core = {n:record(wheel/n) for n in core_names}
        wheel_libraries = [record(p) for p in sorted(wheel.iterdir()) if p.is_file() and '.so' in p.name]
        openmps = [p for p in wheel_libraries if p['soname'] == 'libgomp.so.1']
        require(len(openmps) == 1, 'Expected exactly one wheel OpenMP ELF image.')
        wheel_openmp = openmps[0]
        for core, omp in (({n:original[n] for n in core_names},original_openmp),(wheel_core,wheel_openmp)):
            stage['validate_elf'](core,original[binding_name],omp)
        staging.mkdir(parents=True)
        builds = {'original':built,'nuget-copy':staging/'nuget','wheel-copy':staging/'wheel'}
        for origin in ORIGINS[1:]: stage['copy_original'](built,builds[origin],original_before)
        wheel_expected, aliases = stage['wheel_substitution'](builds['wheel-copy'],original_before,probes,openmp_paths,
                                                              wheel,wheel_core,wheel_openmp)
        before = {'original':original_before,'nuget-copy':original_before,'wheel-copy':wheel_expected}
        write(output/'inventories-before.json',{'builds':before,'wheelPackage':wheel_before,'source':source_before,
                                               'fixtures':fixtures_before,'aliases':aliases})
        for origin in ORIGINS:
            require((builds[origin]/TEST_DLL).is_file(), 'Test assembly missing.')
            require((builds[origin]/'ComfySharp.Inference.Tests.deps.json').is_file(), 'Test dependency manifest missing.')
            require((builds[origin]/'ComfySharp.Inference.Tests.runtimeconfig.json').is_file(), 'Runtime configuration missing.')
            require(any('testadapter' in Path(p).name.lower() and p.endswith('.dll') for p in before[origin]['files']),
                    'Copied test adapter missing.')
        cpu = {}
        for line in Path('/proc/cpuinfo').read_text().split('\n\n',1)[0].splitlines():
            key,_,value = line.partition(':')
            if key.strip() in ('vendor_id','cpu family','model','model name','stepping','flags'): cpu[key.strip()] = value.strip()
        original_env = os.environ.copy()
        write(output/'host.json',{'cpu':cpu,'kernel':platform.release(),'pinnedHelpers':PINS,
              'diagnosticScripts':script_before,'sourceScripts':accepted['scripts'],
              'requirementsSha256':digest(REPO/'labs/clip-source/requirements-linux-x64.txt'),
              'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
              'runnerImage':{k:os.environ.get(k) for k in ('ImageOS','ImageVersion')},
              'readelfVersion':subprocess.check_output(['readelf','--version'],text=True).splitlines()[0],
              'originalCoreAndBinding':original,'originalOpenMp':original_openmp,
              'wheelCore':wheel_core,'wheelOpenMp':wheel_openmp,'wheelNativeLibraries':wheel_libraries,
              'probePaths':probes,'openMpPaths':openmp_paths,'aliases':aliases,
              'loaderPrecondition':precondition,'originalPathSha256':hashlib.sha256(original_env.get('PATH','').encode()).hexdigest(),
              'invocation':{'command':'dotnet vstest','testAssembly':TEST_DLL,'filter':FILTER,'sameArgumentsExceptBuildAndEvidencePaths':True}})
    except Exception as error:
        errors.append(str(error));status('precondition-or-copy',False)
        print('Precondition/copy failed; see status.json.',flush=True)
        return 1
    clear = {'ATEN_CPU_CAPABILITY','COMFYSHARP_SD_UNET_FINE_TRACE_DIR','COMFYSHARP_SD_UNET_TRACE_DIR',
             'COMFYSHARP_SD_LATENT_OFFSET','COMFYSHARP_SD_GUIDANCE_TRACE_DIR'}
    for mode in MODES:
        for origin in ('source',*ORIGINS):
            directory = output/mode/(origin+'-process' if origin == 'source' else origin)
            directory.mkdir(parents=True)
            env = original_env.copy()
            for key in clear: env.pop(key,None)
            if mode == 'default': env['ATEN_CPU_CAPABILITY'] = mode
            if origin == 'source':
                command = [sys.executable,'-I','-B',str(REPO/'labs/sd-operator-diagnostic/source.py'),
                           '--source-directory',str(source),'--output',str(output/mode/'source')]
            else:
                env['COMFYSHARP_SD_UNET_FINE_TRACE_DIR'] = str(directory/'traces')
                command = vstest_command(builds[origin],directory/'results')
            require({k:v for k,v in env.items() if k not in clear} == {k:v for k,v in original_env.items() if k not in clear},
                    'Unrelated environment changed.')
            label = mode+'/'+origin
            print(label+': start',flush=True);start = time.perf_counter()
            try:
                child = subprocess.run(command,cwd=REPO,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                       text=True,encoding='utf-8',errors='replace',timeout=300)
                code, text = child.returncode,child.stdout
            except subprocess.TimeoutExpired as error:
                code,text = 124,error.stdout or ''
                if isinstance(text,bytes): text = text.decode('utf-8',errors='replace')
                text += '\nDiagnostic exceeded 300 seconds.\n'
            except OSError as error: code,text = 127,str(error)
            text = redact(text,private_roots)
            (directory/'process.log').write_text(text,encoding='utf-8')
            runs.append({'label':label,'exitCode':code,'seconds':time.perf_counter()-start,
                         'environment':{k:env.get(k) for k in native['ENVIRONMENT']},
                         'loader':{'preloadBasenames':[],'libraryPath':'unchanged baseline','pathUnchanged':True,
                                   'unrelatedEnvironmentUnchanged':True}})
            write(output/'runs.json',runs);status('collecting');print(label+': exit '+str(code),flush=True)
    docs, identities = {}, {}
    for mode in MODES:
        for origin in ('source',*ORIGINS):
            key = mode+'/'+origin
            try:
                relative = 'source/operators.json' if origin == 'source' else origin+'/traces/sd15-reduced--square.json'
                doc = json.loads((output/mode/relative).read_text());op['verify'](doc)
                if origin == 'source':
                    require(doc['sourceScripts'] == accepted['scripts'] and doc['backendCommit'] == accepted['backendCommit']
                            and doc['profile'] == accepted['profile'], 'Source provenance differs.')
                    require(doc['diagnosticScriptSha256'] == PINS['labs/sd-operator-diagnostic/source.py'], 'Source helper changed.')
                    runtime = doc['runtime']
                    require(runtime['threads'] == runtime['interopThreads'] == 1, 'Source thread count differs.')
                    require(runtime['requestedAtenCpuCapability'] == (None if mode == 'auto' else mode), 'Source requested mode differs.')
                    if mode == 'default': require(runtime['cpuCapability'] in ('DEFAULT','NO AVX'), 'DEFAULT source not observed.')
                    loaded = {p['name']:p for p in runtime['nativeLibraries']}
                    for record in (*wheel_core.values(),wheel_openmp):
                        require((loaded[record['name']]['bytes'],loaded[record['name']]['sha256']) ==
                                (record['bytes'],record['sha256']), 'Source wheel identity differs.')
                else:
                    require(doc['native']['threads'] == doc['native']['interopThreads'] == 1, 'Product threads differ.')
                    require(doc['native']['requestedCapability'] == mode, 'Product requested mode differs.')
                    is_wheel = origin == 'wheel-copy'
                    identities[key] = native['verify_origin'](doc,wheel_core if is_wheel else original,original,original[binding_name],
                        wheel_openmp if is_wheel else original_openmp,original_openmp,'wheel' if is_wheel else 'nuget')
                    verify_trx(output/mode/origin/'results')
                docs[key] = doc
            except Exception as error:
                errors.append(key+': '+str(error));identities[key] = {'verified':False,'diagnosticInvalid':True}
    report = {'diagnosticOnly':True,'qualification':'none','loadedOrigin':identities,'productVsSource':{},
              'originalVsNugetCopy':{},'wheelVsNugetCopy':{},'sourceVsSource':{},'attribution':{}}
    for mode in MODES:
        for origin in ORIGINS:
            try: report['productVsSource'][mode+'/'+origin] = op['comparison'](docs[mode+'/'+origin],docs[mode+'/source'])
            except Exception as error: errors.append(mode+'/'+origin+'/source-comparison: '+str(error))
        try:
            control = op['comparison'](docs[mode+'/nuget-copy'],docs[mode+'/original'])
            report['originalVsNugetCopy'][mode] = control
            report['wheelVsNugetCopy'][mode] = op['comparison'](docs[mode+'/wheel-copy'],docs[mode+'/nuget-copy'])
            valid = exact_copy_control(control)
            report['attribution'][mode] = {'copyControlBitExactAndLayoutsEqual':valid,'nativeOriginContrastEligible':valid,
                'causality':'native package contrast only; primitive cause not established' if valid else 'deferred; invocation/path/allocation control differs'}
        except Exception as error:
            errors.append(mode+'/copy-comparison: '+str(error))
            report['attribution'][mode] = {'nativeOriginContrastEligible':False,'causality':'invalid or incomplete evidence'}
    try:
        a,b = docs['auto/source'],docs['default/source']
        require(a['sources'] == b['sources'] and a['sourceConfig'] == b['sourceConfig'], 'Source AST/config changed.')
        for name in ('python','torch','torchGit','threads','interopThreads','nativeLibraries'):
            require(a['runtime'][name] == b['runtime'][name], 'Non-dispatch runtime differs: '+name)
        require(op['HELPERS']['invariant_build'](a['runtime']['build']) == op['HELPERS']['invariant_build'](b['runtime']['build']),
                'Non-dispatch source build differs.')
        report['sourceVsSource']['auto-expected_vs_default'] = op['comparison'](b,a)
    except Exception as error: errors.append('source/source: '+str(error))
    try:
        reference = docs['auto/original']['native']
        for mode in MODES:
            for origin in ORIGINS:
                runtime = docs[mode+'/'+origin]['native']
                for name in ('torchSharp','declaredLibtorchPackage','threads','interopThreads','os','architecture','dotnet'):
                    require(runtime[name] == reference[name], 'Non-native product runtime differs: '+name)
    except Exception as error: errors.append('product/runtime: '+str(error))
    unchanged = {}
    after = {}
    for name, directory, expected in [(n,builds[n],before[n]) for n in ORIGINS] + [
            ('wheelPackage',wheel,wheel_before),('source',source,source_before),('fixtures',fixtures,fixtures_before)]:
        try:
            after[name] = inventory(directory);unchanged[name] = after[name] == expected
            require(unchanged[name], 'Protected/staged inventory changed: '+name)
        except Exception as error: errors.append(str(error));unchanged[name] = False
    try:
        stage['verify_aliases'](builds['wheel-copy'],aliases)
        require(scripts() == accepted['scripts'], 'Accepted scripts changed.')
        require({p.name:digest(p) for p in Path(__file__).parent.glob('*.py')} == script_before, 'Diagnostic scripts changed.')
        helpers()
    except Exception as error: errors.append(str(error))
    if errors:
        for value in report['attribution'].values():
            value['nativeOriginContrastEligible'] = False
            value['integrityGate'] = 'invalid; inspect status integrityErrors'
    write(output/'inventories-after.json',after)
    write(output/'comparisons.json',report)
    status('complete',True,protectedAndStagedUnchanged=unchanged,failedProcesses=sum(r['exitCode'] != 0 for r in runs))
    return 1 if errors or any(r['exitCode'] for r in runs) else 0


if __name__ == '__main__': raise SystemExit(main())
