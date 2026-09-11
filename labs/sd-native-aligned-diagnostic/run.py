"""Eight isolated processes for eight reduced U-Nets and six CFG policy outputs."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import runpy
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[2]
PINS = {
    'labs/sd-native-copy-diagnostic/run.py': '7a0741233c1e9b36a4d9ab0d4d4d4570e357745be03bb1e93864b4f93b669c73',
    'labs/sd-native-copy-diagnostic/staging.py': '0ee6a1c95847d0d76d7b99b33e21497096fbe54b0f92a53f2ef943f3ebf326fd',
    'labs/sd-native-diagnostic/run.py': '6ffa862eb73757e6c9d8fdf664a54724e414bf10fd648007790c4713ca13a7f1',
    'labs/sd-operator-diagnostic/source.py': 'd269a0fe63530e10f7397d0071bb4391518ff6200d1b892b842b9380f58237ab',
    'labs/sd-operator-diagnostic/run.py': 'a9bce23f9c2ac1d600f0b297a8f38e38cd2effbfd83caba0dde52d8d680d435a',
    'labs/sd-diagnostic/run.py': 'ddd56f206e1642156d8748d23409a504baabfe218cc3e4979797f46c29544e57',
}
MODES = ('auto', 'default')
ORIGINS = ('original', 'nuget-copy', 'wheel-copy')
FILTER = 'FullyQualifiedName~SdUnetReferenceTests|FullyQualifiedName~SdDenoiserReferenceTests'
TEST_DLL = 'ComfySharp.Inference.Tests.dll'


def require(value, message):
    if not value: raise ValueError(message)


def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def raw_record(path):
    return {'bytes':path.stat().st_size,'sha256':digest(path)}


def unchanged_record(after, before):
    require(after == before, 'Dependency lock bytes changed.')
    return after


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def redact(text, roots):
    for value in sorted({str(p) for p in roots},key=len,reverse=True):
        text = text.replace(value,'<local>')
    return text


def helpers():
    for name, sha in PINS.items(): require(digest(REPO/name) == sha, 'Pinned helper changed: '+name)
    parent = runpy.run_path(str(REPO/'labs/sd-native-copy-diagnostic/run.py'))
    native, op, stage = parent['helpers']()
    sys.path.insert(0,str(Path(__file__).parent))
    suite = runpy.run_path(str(Path(__file__).with_name('compare.py')))
    return native, op, stage, suite


def vstest_command(build, results):
    return ['dotnet','vstest',str(build/TEST_DLL),'--TestCaseFilter:'+FILTER,
            '--logger:trx','--ResultsDirectory:'+str(results),'--TestAdapterPath:'+str(build)]


def verify_trx(directory, expected_ids, exit_code):
    paths = list(directory.glob('*.trx'))
    require(len(paths) == 1, 'Expected exactly one TRX.')
    tree = ET.parse(paths[0]).getroot();ns = {'t':tree.tag.split('}')[0].lstrip('{')}
    tests = tree.findall('.//t:UnitTestResult',ns)
    require(len(tests) == 14 and all(t.attrib['outcome'] in ('Passed','Failed') for t in tests),
            'Expected fourteen executed tests, without skips.')
    selected = []
    for test in tests:
        name = test.attrib['testName']
        matches = [identifier for identifier in expected_ids if all(piece in name for piece in identifier.split('/'))]
        require(len(matches) == 1, 'Unexpected or ambiguous test selection.')
        expected_method = 'FullGraphMatchesSamePlatformFrozenSource' if matches[0].startswith(('sd15-','sd2-')) else 'PolicyMatchesItsSamePlatformSource'
        require(expected_method in name, 'Wrong test method.')
        selected.append(matches[0])
    require(set(selected) == set(expected_ids), 'Required reference test missing or duplicated.')
    passed = sum(t.attrib['outcome'] == 'Passed' for t in tests)
    failed = len(tests)-passed
    require((exit_code == 0 and passed == 14) or (exit_code == 1 and failed > 0),
            'Child exit code contradicts fourteen completed reference tests.')
    summaries = tree.findall('./t:ResultSummary',ns)
    require(len(summaries) == 1, 'Exactly one TRX ResultSummary required.')
    summary = summaries[0]
    require(summary.attrib.get('outcome') == ('Failed' if failed else 'Completed'), 'TRX summary outcome differs.')
    counters = summary.findall('./t:Counters',ns)
    require(len(counters) == 1, 'TRX summary counters missing or duplicated.')
    counts = {k:int(v) for k,v in counters[0].attrib.items()}
    require(all(counts.get(k) == v for k,v in {'total':14,'executed':14,'passed':passed,'failed':failed}.items())
            and all(v == 0 for k,v in counts.items() if k not in ('total','executed','passed','failed')),
            'TRX counters contain incomplete or infrastructure outcomes.')
    require(not summary.findall('.//t:ErrorInfo',ns), 'Global TRX error information invalidates attribution.')
    failed_names = {t.attrib['testName'] for t in tests if t.attrib['outcome'] == 'Failed'}
    notifications = 0
    for info in summary.findall('.//t:RunInfo',ns):
        outcome = info.attrib.get('outcome')
        if outcome in ('Information','Warning'): continue
        message = info.findtext('./t:Text',default='',namespaces=ns).strip()
        match = re.fullmatch(r'\[xUnit\.net \d+:\d+:\d+(?:\.\d+)?\]\s+(.+) \[FAIL\]',message)
        require(outcome == 'Error' and match is not None and match.group(1) in failed_names,
                'Unmatched TRX runner error or infrastructure outcome.')
        notifications += 1
    starts=[t.attrib.get('startTime') for t in tests]
    require(all(starts) and len(set(starts))==14,'Exact case execution order missing or ambiguous.')
    order=[identifier for _,identifier in sorted(zip(starts,selected))]
    return {'verified':True,'exitCode':exit_code,'passed':passed,'failed':failed,'caseOrder':order,
            'xunitFailureNotifications':notifications,'globalErrors':0}


def source_origin(document, wheel_core, wheel_openmp, original):
    libraries = document['runtime']['nativeLibraries']
    for name, expected in wheel_core.items():
        matches = [p for p in libraries if p['name'] == name]
        require(len(matches) == 1 and (matches[0]['bytes'],matches[0]['sha256']) == (expected['bytes'],expected['sha256']),
                'Source actual core origin differs: '+name)
    omp = [p for p in libraries if any(word in p['name'].lower() for word in ('gomp','iomp','libomp'))]
    require(len(omp) == 1 and (omp[0]['bytes'],omp[0]['sha256']) == (wheel_openmp['bytes'],wheel_openmp['sha256']), 'Source OpenMP origin differs.')
    forbidden = {v['sha256'] for n,v in original.items() if n in wheel_core} - {v['sha256'] for v in wheel_core.values()}
    require(not any(p['sha256'] in forbidden for p in libraries), 'Source contains NuGet core image.')
    wheel = {p['name']:(p['bytes'],p['sha256']) for p in document['runtime']['wheelLibraries']}
    require(all(wheel.get(p['name']) == (p['bytes'],p['sha256']) for p in libraries), 'Source maps an unselected native package image.')
    return {'verified':True,'origin':'wheel source','loadedLibraries':libraries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--staging-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    require((platform.system(),platform.machine()) == ('Linux','x86_64'), 'Use Linux x64.')
    require(sys.version_info[:3] == (3,12,10), 'Use pinned Python.')
    require(importlib.metadata.version('torch') == '2.10.0+cpu', 'Use pinned CPU wheel.')
    protocol_path=Path(__file__).with_name('protocol.json')
    protocol_before=protocol_path.read_bytes()
    require(hashlib.sha256(protocol_before).hexdigest()=='27f4b82a8b26cbdfc2ac384d48a0433c2e351c9142efb7592f8f8ccac44ca3e2','Prospective protocol changed.')
    for name,pin in json.loads(protocol_before)['parentFiles'].items():
        require(digest(REPO/name)==pin,'Parent/input pin differs: '+name)
    native, op, stage, suite = helpers()
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
        lock_path = REPO/'labs/clip-source/requirements-linux-x64.txt'
        lock_before = raw_record(lock_path)
        write(output/'dependency-lock-before.json',lock_before)
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
              'diagnosticScripts':script_before,'sourceScripts':accepted['scripts'],'inputAllocationProtocol':json.loads(protocol_before),'protocolSha256':hashlib.sha256(protocol_before).hexdigest(),
              'requirementsSha256':lock_before['sha256'],'requirementsRecord':lock_before,
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
    clear = {'COMFYSHARP_SD_NATIVE_SUITE_TRACE_DIR','COMFYSHARP_SD_ALIGNED_SUITE_TRACE_DIR','ATEN_CPU_CAPABILITY','COMFYSHARP_SD_UNET_FINE_TRACE_DIR','COMFYSHARP_SD_UNET_TRACE_DIR',
             'COMFYSHARP_SD_LATENT_OFFSET','COMFYSHARP_SD_GUIDANCE_TRACE_DIR'}
    for mode in MODES:
        for origin in ('source',*ORIGINS):
            directory = output/mode/(origin+'-process' if origin == 'source' else origin)
            directory.mkdir(parents=True)
            env = original_env.copy()
            for key in clear: env.pop(key,None)
            if mode == 'default': env['ATEN_CPU_CAPABILITY'] = mode
            if origin == 'source':
                command = [sys.executable,'-I','-B',str(Path(__file__).with_name('source.py')),
                           '--source-directory',str(source),'--output',str(output/mode/'source')]
            else:
                env['COMFYSHARP_SD_ALIGNED_SUITE_TRACE_DIR'] = str(directory/'traces')
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
    docs, identities, sources = {}, {}, {}
    process_codes = {r['label']:r['exitCode'] for r in runs}
    unet = json.loads((fixtures/'sd-components.linux-x64.unet.json').read_text())
    guidance = json.loads((fixtures/'sd-components.linux-x64.guidance.json').read_text())
    for mode in MODES:
        key = mode+'/source'
        try:
            require(process_codes[key] == 0, 'Source process did not exit successfully.')
            source_doc = json.loads((output/mode/'source/suite.json').read_text())
            require(source_doc['sourceScripts'] == accepted['scripts'] and source_doc['backendCommit'] == accepted['backendCommit']
                    and source_doc['profile'] == accepted['profile'], 'Source provenance differs.')
            require(source_doc['diagnosticScriptSha256'] == script_before['source.py'], 'Source adapter identity differs.')
            require(source_doc['inputAllocationPolicy']=='sd-native-aligned-inputs-v1' and source_doc['protocolSha256']==hashlib.sha256(protocol_before).hexdigest(),'Source allocation protocol differs.')
            runtime = source_doc['runtime'];versions = source_doc['operationVersions']
            require(versions['python'] == '3.12.10' and versions['torch'] == '2.10.0+cpu', 'Actual source operation runtime differs.')
            require(versions['torchGit'] == accepted['runtime']['torchGit'], 'Source operation commit differs from pinned runtime.')
            require(versions['numpy'] == '2.2.6' and versions['einops'] == '0.8.1', 'Source auxiliary operation versions differ.')
            require(runtime['threads'] == runtime['interopThreads'] == 1 and runtime['requestedCapability'] == mode, 'Source threads/mode differ.')
            if mode == 'default': require(runtime['cpuCapability'] in ('DEFAULT','NO AVX'), 'DEFAULT source not observed.')
            identities[key] = source_origin(source_doc,wheel_core,wheel_openmp,original)
            require(len(source_doc['cases']) == 14 and {c['id'] for c in source_doc['cases']} == set(suite['IDS']), 'Incomplete source case set.')
            for doc in source_doc['cases']:
                suite['verify'](doc,source=True);suite['check_corpus_inputs'](doc,unet,guidance)
            sources[mode] = source_doc;docs[key] = {c['id']:c for c in source_doc['cases']}
        except Exception as error:
            errors.append(key+': '+str(error));identities[key] = {'verified':False,'diagnosticInvalid':True}
        for origin in ORIGINS:
            key = mode+'/'+origin
            try:
                trace_paths = list((output/mode/origin/'traces').glob('*.json'))
                require(len(trace_paths) == 14, 'Fourteen product traces required.')
                cases = [json.loads(p.read_text()) for p in trace_paths]
                require({c['id'] for c in cases} == set(suite['IDS']), 'Product case IDs differ.')
                snapshots = []
                for doc in cases:
                    suite['verify'](doc);suite['check_corpus_inputs'](doc,unet,guidance)
                    runtime = doc['native']
                    require(runtime['threads'] == runtime['interopThreads'] == 1 and runtime['requestedCapability'] == mode, 'Product threads/mode differ.')
                    require(runtime['torchSharp'] == '0.107.0.0' and runtime['declaredLibtorchPackage'] == '2.10.0', 'Product operation version differs.')
                    managed = runtime['managedAssemblies']
                    require(len(managed) == 1 and managed[0]['name'] == 'TorchSharp.dll' and
                            (managed[0]['bytes'],managed[0]['sha256']) ==
                            (original_before['files']['TorchSharp.dll']['bytes'],original_before['files']['TorchSharp.dll']['sha256']),
                            'Actual managed TorchSharp assembly differs from the unchanged build.')
                    require(len(runtime['libraries']) == 5, 'Unexpected extra or missing loaded native image.')
                    is_wheel = origin == 'wheel-copy'
                    native['verify_origin'](doc,wheel_core if is_wheel else original,original,original[binding_name],
                        wheel_openmp if is_wheel else original_openmp,original_openmp,'wheel' if is_wheel else 'nuget')
                    snapshots.append(sorted(runtime['libraries'],key=lambda p:p['name']))
                require(all(s == snapshots[0] for s in snapshots), 'Loaded native maps changed during the process.')
                test_status = verify_trx(output/mode/origin/'results',suite['IDS'],process_codes[key])
                identities[key] = {'verified':True,'caseSnapshots':14,'stableLoadedLibraries':snapshots[0],
                                   'testExecution':test_status}
                docs[key] = {c['id']:c for c in cases}
            except Exception as error:
                errors.append(key+': '+str(error));identities[key] = {'verified':False,'diagnosticInvalid':True}
    report = {'diagnosticOnly':True,'qualification':'none','loadedOrigin':identities,'productVsSource':{},
              'originalVsNugetCopy':{},'wheelVsNugetCopy':{},'sourceVsSource':{},'attribution':{}}
    for mode in MODES:
        for origin in ORIGINS:
            key = mode+'/'+origin
            try: report['productVsSource'][key] = {i:suite['compare'](docs[key][i],docs[mode+'/source'][i]) for i in suite['IDS']}
            except Exception as error: errors.append(key+'/source-comparison: '+str(error))
        try:
            controls = {i:suite['compare'](docs[mode+'/nuget-copy'][i],docs[mode+'/original'][i]) for i in suite['IDS']}
            report['originalVsNugetCopy'][mode] = controls
            report['wheelVsNugetCopy'][mode] = {i:suite['compare'](docs[mode+'/wheel-copy'][i],docs[mode+'/nuget-copy'][i]) for i in suite['IDS']}
            report['attribution'][mode] = {i:{
                'nativeOriginContrastEligible':suite['exact_control'](c) and
                    suite['input_layout_control'](report['wheelVsNugetCopy'][mode][i]),
                'copyControlBitExactAndLayoutsEqual':suite['exact_control'](c),
                'wheelNugetInputAndParameterLayoutsEqual':suite['input_layout_control'](report['wheelVsNugetCopy'][mode][i]),
                'causality':'package contrast only when gates pass; differing inputs/layouts confound attribution; no primitive attribution'}
                for i,c in controls.items()}
        except Exception as error:
            errors.append(mode+'/copy-comparison: '+str(error))
            report['attribution'][mode] = {'incomplete':{'nativeOriginContrastEligible':False}}
    try:
        a,b = sources['auto'],sources['default']
        require(a['sources'] == b['sources'] and a['sourceConfigurations'] == b['sourceConfigurations'], 'Source AST/config changed.')
        require(a['operationVersions'] == b['operationVersions'], 'Source operation versions changed.')
        require(sorted(a['runtime']['nativeLibraries'],key=lambda p:p['name']) == sorted(b['runtime']['nativeLibraries'],key=lambda p:p['name']), 'Source actual libraries changed.')
        require(op['HELPERS']['invariant_build'](a['runtime']['build']) == op['HELPERS']['invariant_build'](b['runtime']['build']), 'Source non-dispatch build differs.')
        report['sourceVsSource'] = {i:suite['compare'](docs['default/source'][i],docs['auto/source'][i]) for i in suite['IDS']}
        reference_order=identities['auto/original']['testExecution']['caseOrder']
        require(all(identities[mode+'/'+origin]['testExecution']['caseOrder']==reference_order for mode in MODES for origin in ORIGINS),'Product case execution order changed.')
        reference = next(iter(docs['auto/original'].values()))['native']
        for mode in MODES:
            for origin in ORIGINS:
                for doc in docs[mode+'/'+origin].values():
                    for name in ('torchSharp','declaredLibtorchPackage','threads','interopThreads','os','architecture','dotnet','operationBoundary','managedAssemblies'):
                        require(doc['native'][name] == reference[name], 'Product operation/runtime changed: '+name)
    except Exception as error: errors.append('cross-process runtime: '+str(error))
    unchanged = {}
    after = {}
    for name, directory, expected in [(n,builds[n],before[n]) for n in ORIGINS] + [
            ('wheelPackage',wheel,wheel_before),('source',source,source_before),('fixtures',fixtures,fixtures_before)]:
        try:
            after[name] = inventory(directory);unchanged[name] = after[name] == expected
            require(unchanged[name], 'Protected/staged inventory changed: '+name)
        except Exception as error: errors.append(str(error));unchanged[name] = False
    try:
        lock_after = raw_record(lock_path)
        write(output/'dependency-lock-after.json',lock_after)
        unchanged_record(lock_after,lock_before)
    except Exception as error: errors.append(str(error))
    try:
        stage['verify_aliases'](builds['wheel-copy'],aliases)
        require(scripts() == accepted['scripts'], 'Accepted scripts changed.')
        require({p.name:digest(p) for p in Path(__file__).parent.glob('*.py')} == script_before, 'Diagnostic scripts changed.')
        helpers()
        require(protocol_path.read_bytes()==protocol_before,'Prospective protocol mutated.')
        for name,pin in json.loads(protocol_before)['parentFiles'].items():
            require(digest(REPO/name)==pin,'Parent/input changed during calculation: '+name)
    except Exception as error: errors.append(str(error))
    if errors:
        for cases in report['attribution'].values():
            for value in cases.values():
                value['nativeOriginContrastEligible'] = False
                value['integrityGate'] = 'invalid; inspect status integrityErrors'
    write(output/'inventories-after.json',after)
    write(output/'comparisons.json',report)
    ineligible = [mode+'/'+case for mode,cases in report['attribution'].items() for case,value in cases.items()
                  if not value['nativeOriginContrastEligible']]
    status('complete',True,protectedAndStagedUnchanged=unchanged,failedProcesses=sum(r['exitCode'] != 0 for r in runs),
           attributionIneligible=ineligible)
    return 1 if errors or ineligible or any(r['exitCode'] for r in runs) else 0


if __name__ == '__main__': raise SystemExit(main())
