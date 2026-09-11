"""Diagnostic metadata around unchanged, pinned U-Net and CFG source generators."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import weakref

REPO=Path(__file__).resolve().parents[2]


def require(value,message):
    if not value: raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def loaded_libraries():
    paths=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        columns=line.split(maxsplit=5)
        if len(columns)!=6 or not columns[5].startswith('/'):continue
        path=Path(columns[5]);name=path.name.lower()
        if not any(word in name for word in ('torch','c10','gomp','iomp','libomp')):continue
        require(path.is_file(),'A mapped native image cannot be hashed.')
        paths.add(path)
    return [{'name':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(paths)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require((platform.system(),platform.machine())==('Linux','x86_64'),'Use Linux x64.')
    require(sys.version_info[:3]==(3,12,10),'Use pinned Python.')
    require(importlib.metadata.version('torch')=='2.10.0+cpu','Use pinned CPU wheel.')
    source=args.source_directory.resolve(strict=True);output=args.output.resolve()
    require(args.output.is_absolute() and not output.exists(),'Output must be absolute and absent.')
    for protected in (REPO,source):
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output),'Output overlaps input.')
    accepted=json.loads((REPO/'tests/ComfySharp.Inference.Tests/Fixtures/sd-components.linux-x64.manifest.json').read_text())
    scripts=lambda:{p.name:digest(p) for p in (REPO/'labs/sd-source').glob('*.py')}
    require(scripts()==accepted['scripts'],'Accepted source scripts changed.')
    protocol=Path(__file__).with_name('protocol.json')
    require(digest(protocol)=='27f4b82a8b26cbdfc2ac384d48a0433c2e351c9142efb7592f8f8ccac44ca3e2','Prospective protocol differs.')
    protocol_before=protocol.read_bytes()
    for name,pin in json.loads(protocol_before)['parentFiles'].items():
        require(digest(REPO/name)==pin,'Pinned parent/input changed: '+name)
    output.mkdir(parents=True)
    sys.path.insert(0,str(REPO/'labs/sd-source'))
    import torch
    require(torch.__version__=='2.10.0+cpu' and torch.version.cuda is None,'Actual torch CPU runtime differs.')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    common=importlib.import_module('common')
    sys.path.insert(0,str(Path(__file__).parent))
    from generation import generate
    cases,documents=generate(source,common,torch,REPO)
    build=torch.__config__.show()
    for p in (Path.home(),Path(sys.prefix),source,output):build=build.replace(str(p),'<local>')
    wheel=Path(torch.__file__).resolve().parent/'lib'
    report={'backendCommit':common.COMMIT,'profile':common.PROFILE,'sourceScripts':scripts(),'diagnosticScriptSha256':digest(Path(__file__)),
        'operationVersions':{'python':platform.python_version(),'torch':torch.__version__,'torchGit':torch.version.git_version,
            'numpy':importlib.metadata.version('numpy'),'einops':importlib.metadata.version('einops'),
            'implementation':'unchanged frozen UNetModel._forward, EPS, ModelSamplingDiscrete, cfg_function, CONDRegular/CONDCrossAttn; prospective aligned-input orchestration'},
        'runtime':{'threads':torch.get_num_threads(),'interopThreads':torch.get_num_interop_threads(),'build':build,
            'requestedCapability':os.environ.get('ATEN_CPU_CAPABILITY','auto'),'cpuCapability':torch.backends.cpu.get_cpu_capability(),
            'nativeLibraries':loaded_libraries(),
            'wheelLibraries':[{'name':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(wheel.iterdir()) if p.is_file() and '.so' in p.name]},
        'sources':{name:doc['sources'] for name,doc in documents.items()},
        'sourceConfigurations':{name:model['sourceConfig'] for name,model in documents['unet']['models'].items()},
        'cases':cases,'qualification':'none; diagnostic metadata around unchanged source generators'}
    require(report['sourceScripts']==accepted['scripts'],'Accepted source script changed during execution.')
    require(len(cases)==14 and len({c['id'] for c in cases})==14,'Source case set differs.')
    require(protocol.read_bytes()==protocol_before,'Protocol changed during source calculation.')
    report['inputAllocationPolicy']='sd-native-aligned-inputs-v1'
    report['protocolSha256']=digest(protocol)
    write(output/'suite.json',report)


if __name__=='__main__':raise SystemExit(main())
