"""Isolated native-origin experiment; never changes the product build or references."""
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
    'labs/sd-operator-diagnostic/source.py': 'd269a0fe63530e10f7397d0071bb4391518ff6200d1b892b842b9380f58237ab',
    'labs/sd-operator-diagnostic/run.py': 'a9bce23f9c2ac1d600f0b297a8f38e38cd2effbfd83caba0dde52d8d680d435a',
    'labs/sd-diagnostic/run.py': 'ddd56f206e1642156d8748d23409a504baabfe218cc3e4979797f46c29544e57',
}
CORE = ('libc10.so', 'libtorch_cpu.so', 'libtorch.so')
BINDING = 'libLibTorchSharp.so'
MODES = ('auto', 'default')
ENVIRONMENT = ('ATEN_CPU_CAPABILITY', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'MKL_CBWR',
               'ONEDNN_MAX_CPU_ISA', 'DNNL_MAX_CPU_ISA')


def require(value, message):
    if not value: raise ValueError(message)


def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def pinned_helpers():
    for relative, expected in PINS.items():
        require(digest(REPO/relative)==expected, 'Pinned diagnostic helper changed: '+relative)
    return runpy.run_path(str(REPO/'labs/sd-operator-diagnostic/run.py'))


def library_record(path):
    raw = subprocess.check_output(['readelf','-d',str(path)], text=True, encoding='utf-8')
    needed = re.findall(r'\(NEEDED\).*?\[([^\]]+)\]',raw)
    sonames = re.findall(r'\(SONAME\).*?\[([^\]]+)\]',raw)
    require(all(Path(n).name==n for n in needed+sonames), 'Unexpected absolute ELF dependency.')
    return {'name':path.name,'bytes':path.stat().st_size,'sha256':digest(path),
            'soname':sonames[0] if len(sonames)==1 else None,'needed':needed}


def validate_loader_environment(environment, base_prefix):
    """Admit only setup-python's single interpreter lib directory, without rewriting it."""
    require(not environment.get('LD_PRELOAD'), 'Nonempty LD_PRELOAD invalidates the controlled loader experiment.')
    value=environment.get('LD_LIBRARY_PATH','')
    if not value:
        return {'classification':'empty','baselinePythonLibraryPreserved':False,
                'libraryPathPresent':'LD_LIBRARY_PATH' in environment}
    require(len(value.split(':'))==1 and bool(value), 'LD_LIBRARY_PATH must be exactly one nonempty directory.')
    provided=Path(value)
    require(provided.is_absolute(), 'LD_LIBRARY_PATH must be an absolute interpreter library directory.')
    try:
        expected=(Path(base_prefix)/'lib').resolve(strict=True)
        actual=provided.resolve(strict=True)
        require(expected.is_dir() and actual.is_dir(), 'Interpreter library directory is unavailable.')
    except OSError:
        raise ValueError('Interpreter library directory cannot be resolved.') from None
    require(actual==expected, 'LD_LIBRARY_PATH is not the active interpreter base-prefix lib directory.')
    try:
        entries=list(actual.iterdir())
        # The ELF loader searches these direct basenames, not nested site-packages.
        for entry in entries:
            name=entry.name.casefold()
            forbidden=any(name==n.casefold() or name.startswith(n.casefold()+'.') for n in (*CORE,BINDING))
            forbidden=forbidden or any(name.startswith(n) for n in ('libgomp','libiomp','libomp'))
            require(not forbidden, 'Interpreter library directory contains a core, binding or OpenMP loader candidate.')
        python_libraries=[{'name':p.name,'sha256':digest(p),'bytes':p.stat().st_size}
                          for p in entries if p.name.startswith('libpython') and p.is_file()]
        require(python_libraries, 'Interpreter library directory has no verifiable libpython identity.')
    except OSError:
        raise ValueError('Interpreter library identities cannot be inspected.') from None
    return {'classification':'verifiedInterpreterLib','directoryName':expected.name,
            'matchesInterpreterBasePrefixLib':True,'baselinePythonLibraryPreserved':True,
            'directEntryCount':len(entries),'directEntryNamesSha256':hashlib.sha256(
                json.dumps(sorted(p.name for p in entries)).encode()).hexdigest(),
            'pythonLibraries':sorted(python_libraries,key=lambda p:p['name'])}


def verify_origin(document, expected_core, original_core, original_binding, expected_openmp, original_openmp, origin):
    """Attribution is based on actual Process.Modules, not environment intent."""
    libraries = document['native']['libraries']
    require(libraries, 'Loaded native-library evidence missing.')
    core_hashes = set()
    for name in CORE:
        found=[p for p in libraries if p['name']==name]
        require(len(found)==1, 'Missing or duplicate loaded core library: '+name)
        require((found[0]['bytes'],found[0]['sha256'])==
                (expected_core[name]['bytes'],expected_core[name]['sha256']), 'Wrong loaded core origin: '+name)
        core_hashes.add(found[0]['sha256'])
    binding=[p for p in libraries if p['name']==BINDING]
    require(len(binding)==1 and (binding[0]['bytes'],binding[0]['sha256'])==
            (original_binding['bytes'],original_binding['sha256']), 'TorchSharp binding changed or duplicated.')
    if origin=='wheel':
        forbidden={original_core[n]['sha256'] for n in CORE}-core_hashes
        require(not any(p['sha256'] in forbidden for p in libraries), 'A second NuGet core remains loaded.')
    openmp=[p for p in libraries if 'gomp' in p['name'].lower() or 'iomp' in p['name'].lower()]
    require(len(openmp)==1 and (openmp[0]['bytes'],openmp[0]['sha256'])==
            (expected_openmp['bytes'],expected_openmp['sha256']), 'OpenMP origin is wrong, missing or mixed.')
    if origin=='wheel' and original_openmp['sha256']!=expected_openmp['sha256']:
        require(not any(p['sha256']==original_openmp['sha256'] for p in libraries), 'NuGet OpenMP remains loaded.')
    require(not any('libtorch_python' in p['name'].lower() for p in libraries), 'Python binding must not be loaded into dotnet.')
    return {'verified':True,'origin':origin,'loadedCoreHashes':sorted(core_hashes),
            'bindingSha256':binding[0]['sha256'],'openmpSha256':openmp[0]['sha256']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require((platform.system(),platform.machine())==('Linux','x86_64'), 'Use Linux x64.')
    require(sys.version_info[:3]==(3,12,10), 'Use the pinned Python runtime.')
    require(importlib.metadata.version('torch')=='2.10.0+cpu', 'Use the pinned CPU wheel.')
    helpers=pinned_helpers()
    source=args.source_directory.resolve(strict=True);output=args.output.resolve()
    require(args.output.is_absolute() and not output.exists(), 'Output must be absolute and absent.')
    for protected in (REPO,source):
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output), 'Output overlaps protected input.')
    output.mkdir(parents=True)
    try:
        loader_precondition=validate_loader_environment(os.environ,sys.base_prefix)
    except ValueError as error:
        write(output/'status.json',{'phase':'loader-precondition','complete':False,'runs':[],
              'plannedProcesses':6,'integrityErrors':[str(error)],'qualification':'none; no child process started'})
        print('Loader precondition rejected; see status.json.',flush=True)
        return 1
    write(output/'precondition.json',{'valid':True,**loader_precondition})
    write(output/'status.json',{'phase':'preflight','complete':False,'runs':[],'plannedProcesses':6,
          'integrityErrors':[],'qualification':'none; collection not complete'})
    wheel=Path(importlib.metadata.distribution('torch').locate_file('torch/lib')).resolve(strict=True)
    built=REPO/'tests/ComfySharp.Inference.Tests/bin/native/linux-x64/cpu/Release/net10.0'
    require(built.is_dir(), 'Original product build missing.')
    def find(name):
        matches=list(built.rglob(name));require(len(matches)==1,'Expected one original native library: '+name)
        return matches[0]
    originals={name:library_record(find(name)) for name in (*CORE,BINDING)}
    wheel_core={name:library_record(wheel/name) for name in CORE}
    wheel_libraries=[library_record(p) for p in sorted(wheel.iterdir()) if p.is_file() and '.so' in p.name]
    require(all(wheel_core[n]['soname']==n for n in CORE), 'Wheel core SONAMEs differ from binding dependencies.')
    wheel_openmp_candidates=[p for p in wheel_libraries if p['soname']=='libgomp.so.1']
    original_openmp_candidates=list(built.rglob('libgomp*.so*'))
    require(len(wheel_openmp_candidates)==len(original_openmp_candidates)==1, 'Expected one OpenMP library per origin.')
    wheel_openmp=wheel_openmp_candidates[0];original_openmp=library_record(original_openmp_candidates[0])
    require(original_openmp['soname']=='libgomp.so.1', 'Unexpected original OpenMP SONAME.')
    fixtures=REPO/'tests/ComfySharp.Inference.Tests/Fixtures'
    hashes=lambda directory:{str(p.relative_to(directory)):digest(p) for p in directory.rglob('*') if p.is_file()}
    fixture_before=hashes(fixtures); build_before=hashes(built)
    source_script_hashes=lambda:{p.name:digest(p) for p in (REPO/'labs/sd-source').glob('*.py')}
    accepted_scripts=json.loads((fixtures/'sd-components.linux-x64.manifest.json').read_text())['scripts']
    require(source_script_hashes()==accepted_scripts, 'Accepted source script identity changed.')
    cpu={}
    for line in Path('/proc/cpuinfo').read_text().split('\n\n',1)[0].splitlines():
        key,_,value=line.partition(':')
        if key.strip() in ('vendor_id','cpu family','model','model name','stepping','flags'):cpu[key.strip()]=value.strip()
    original_environment=os.environ.copy()
    write(output/'host.json',{'cpu':cpu,'kernel':platform.release(),
          'runnerImage':{k:os.environ.get(k) for k in ('ImageOS','ImageVersion')},
          'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
          'pinnedDiagnosticHelpers':PINS,'diagnosticScriptSha256':digest(Path(__file__)),
          'requirementsSha256':digest(REPO/'labs/clip-source/requirements-linux-x64.txt'),
          'sourceScripts':accepted_scripts,
          'originalBuildManifestSha256':hashlib.sha256(json.dumps(build_before,sort_keys=True).encode()).hexdigest(),
          'originalBuildFileCount':len(build_before),
          'originalManagedAssemblies':[{'name':Path(n).name,'sha256':h} for n,h in build_before.items() if n.lower().endswith('.dll')],
          'readelfVersion':subprocess.check_output(['readelf','--version'],text=True).splitlines()[0],
          'originalCoreAndBinding':originals,'wheelCore':wheel_core,'wheelNativeLibraries':wheel_libraries,
          'originalOpenMp':original_openmp,'wheelOpenMp':wheel_openmp,
          'loaderPrecondition':loader_precondition,
          'originalPathSha256':hashlib.sha256(original_environment.get('PATH','').encode()).hexdigest(),
          'originalLoaderOverrides':{k:{'present':k in original_environment,'empty':not original_environment.get(k)}
                                     for k in ('LD_PRELOAD','LD_LIBRARY_PATH')}})
    runs=[];errors=[]
    clear={'ATEN_CPU_CAPABILITY','COMFYSHARP_SD_UNET_FINE_TRACE_DIR','COMFYSHARP_SD_UNET_TRACE_DIR',
           'COMFYSHARP_SD_LATENT_OFFSET','COMFYSHARP_SD_GUIDANCE_TRACE_DIR'}

    def execute(mode,kind,origin=None):
        label=mode+'/'+kind+('/'+origin if origin else '')
        directory=output/mode/(kind+'-process' if kind=='source' else origin)
        directory.mkdir(parents=True)
        env=original_environment.copy()
        for k in clear:env.pop(k,None)
        if mode=='default':env['ATEN_CPU_CAPABILITY']='default'
        if kind=='source':
            command=[sys.executable,'-I','-B',str(REPO/'labs/sd-operator-diagnostic/source.py'),
                     '--source-directory',str(source),'--output',str(output/mode/'source')]
        else:
            env['COMFYSHARP_SD_UNET_FINE_TRACE_DIR']=str(directory/'traces')
            if origin=='wheel':
                env['LD_PRELOAD']=' '.join(str(wheel/name) for name in CORE)
                baseline_library_path=original_environment.get('LD_LIBRARY_PATH','')
                env['LD_LIBRARY_PATH']=str(wheel)+(':'+baseline_library_path if baseline_library_path else '')
            command=['dotnet','test','tests/ComfySharp.Inference.Tests','--no-build','--no-restore','-c','Release',
                     '-p:NativeRuntime=linux-x64','--filter',
                     'FullyQualifiedName~SdUnetReferenceTests&DisplayName~sd15-reduced&DisplayName~square',
                     '--logger','trx','--results-directory',str(directory/'results')]
        allowed=clear|({'LD_PRELOAD','LD_LIBRARY_PATH'} if origin=='wheel' else set())
        require({k:v for k,v in env.items() if k not in allowed}==
                {k:v for k,v in original_environment.items() if k not in allowed}, 'An unrelated child environment value changed.')
        print(label+': start',flush=True);start=time.perf_counter()
        try:
            p=subprocess.run(command,cwd=REPO,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,encoding='utf-8',errors='replace',timeout=300)
            code,text=p.returncode,p.stdout
        except subprocess.TimeoutExpired as error:
            code,text=124,error.stdout or ''
            if isinstance(text,bytes):text=text.decode('utf-8',errors='replace')
            text+='\nNative diagnostic exceeded 300 seconds.\n'
        except OSError as error:code,text=127,str(error)
        for private in (str(REPO),str(source),str(output),str(wheel),str(Path.home()),str(Path(sys.prefix))):
            text=text.replace(private,'<local>')
        (directory/'process.log').write_text(text,encoding='utf-8')
        runs.append({'label':label,'exitCode':code,'seconds':time.perf_counter()-start,
                     'environment':{k:env.get(k) for k in ENVIRONMENT},
                     'loader':{'preloadBasenames':list(CORE) if origin=='wheel' else [],
                               'libraryPath':'wheel followed by validated original Python library path' if origin=='wheel' else 'original',
                               'baselinePythonLibraryPreserved':loader_precondition['baselinePythonLibraryPreserved'],
                               'unrelatedEnvironmentUnchanged':True,'pathUnchanged':env.get('PATH')==original_environment.get('PATH')}})
        write(output/'runs.json',runs);print(label+': exit '+str(code),flush=True)

    for mode in MODES:
        execute(mode,'source')
        execute(mode,'product','nuget')
        execute(mode,'product','wheel')

    docs={};identities={}
    for mode in MODES:
        for origin in ('source','nuget','wheel'):
            key=mode+'/'+origin
            try:
                relative='source/operators.json' if origin=='source' else origin+'/traces/sd15-reduced--square.json'
                doc=json.loads((output/mode/relative).read_text());helpers['verify'](doc)
                if origin=='source':
                    require(doc['diagnosticScriptSha256']==PINS['labs/sd-operator-diagnostic/source.py'], 'Source helper identity changed.')
                    require(doc['runtime']['threads']==doc['runtime']['interopThreads']==1, 'Source threads differ.')
                    require(doc['runtime']['requestedAtenCpuCapability']==(None if mode=='auto' else mode), 'Source mode differs.')
                    loaded={p['name']:p for p in doc['runtime']['nativeLibraries']}
                    require(all((loaded[n]['bytes'],loaded[n]['sha256'])==(wheel_core[n]['bytes'],wheel_core[n]['sha256']) for n in CORE),
                            'Source core hashes differ from selected wheel files.')
                else:
                    native=doc['native'];require(native['threads']==native['interopThreads']==1,'Product threads differ.')
                    require(native['requestedCapability']==mode,'Product dispatch request differs.')
                    identities[key]=verify_origin(doc,wheel_core if origin=='wheel' else originals,originals,originals[BINDING],
                                                   wheel_openmp if origin=='wheel' else original_openmp,original_openmp,origin)
                    trx=list((output/mode/origin/'results').glob('*.trx'));require(len(trx)==1,'Expected one TRX.')
                    tree=ET.parse(trx[0]).getroot();ns={'t':tree.tag.split('}')[0].lstrip('{')}
                    tests=tree.findall('.//t:UnitTestResult',ns)
                    require(len(tests)==1 and tests[0].attrib['outcome'] in ('Passed','Failed'), 'Expected one executed reference test.')
                    name=tests[0].attrib['testName']
                    require('FullGraphMatchesSamePlatformFrozenSource' in name and 'sd15-reduced' in name and 'square' in name,
                            'Unexpected product test selection.')
                docs[key]=doc
            except Exception as error:
                errors.append(key+': '+str(error))
                identities[key]={'verified':False,'origin':origin,'diagnosticInvalid':True}
    report={'diagnosticOnly':True,'qualification':'none','loadedOrigin':identities,
            'productVsSource':{},'productVsProduct':{},'sourceVsSource':{}}
    for mode in MODES:
        for origin in ('nuget','wheel'):
            try:report['productVsSource'][mode+'/'+origin]=helpers['comparison'](docs[mode+'/'+origin],docs[mode+'/source'])
            except Exception as error:errors.append(mode+'/'+origin+'/compare: '+str(error))
        try:report['productVsProduct'][mode]=helpers['comparison'](docs[mode+'/wheel'],docs[mode+'/nuget'])
        except Exception as error:errors.append(mode+'/origins/compare: '+str(error))
    try:
        a,b=docs['auto/source'],docs['default/source']
        require(a['sources']==b['sources'] and a['sourceConfig']==b['sourceConfig'], 'Source AST/config changed.')
        require(a['runtime']['nativeLibraries']==b['runtime']['nativeLibraries'], 'Source binaries changed.')
        require(b['runtime']['cpuCapability'] in ('DEFAULT','NO AVX'), 'Source DEFAULT not observed.')
        report['sourceVsSource']['auto-expected_vs_default']=helpers['comparison'](b,a)
    except Exception as error:errors.append('source/source: '+str(error))
    fixtures_unchanged=hashes(fixtures)==fixture_before
    build_unchanged=hashes(built)==build_before
    if not fixtures_unchanged:errors.append('Accepted fixtures changed.')
    if not build_unchanged:errors.append('Original product build changed.')
    if source_script_hashes()!=accepted_scripts:errors.append('Accepted source scripts changed.')
    try:pinned_helpers()
    except Exception as error:errors.append(str(error))
    write(output/'comparisons.json',report)
    write(output/'status.json',{'complete':True,'runs':runs,'integrityErrors':errors,
          'originalBuildUnchanged':build_unchanged,'acceptedFixturesUnchanged':fixtures_unchanged,
          'failedProcesses':sum(r['exitCode']!=0 for r in runs),'qualification':'none; no adoption or replaced acceptance'})
    return 1 if errors or any(r['exitCode'] for r in runs) else 0


if __name__=='__main__':raise SystemExit(main())
