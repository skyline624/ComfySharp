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
    output.mkdir(parents=True)
    sys.path.insert(0,str(REPO/'labs/sd-source'))
    import torch
    require(torch.__version__=='2.10.0+cpu' and torch.version.cuda is None,'Actual torch CPU runtime differs.')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    common=importlib.import_module('common')
    documents={};parameter_states={}

    def parameters(model):
        records=[];layouts={}
        for name,p in model.named_parameters():
            records.append({'name':name,'shape':list(p.shape),'sha256':hashlib.sha256(p.detach().contiguous().numpy().tobytes()).hexdigest()})
            layouts[name]={'shape':list(p.shape),'stride':list(p.stride()),'aligned64':p.data_ptr()%64==0}
        require(len(records)==686,'Parameter set incomplete.')
        return records,layouts

    def observe_fill(states):
        def fill(model):
            expected=common.fill_parameters(model)
            before,layouts=parameters(model)
            require(before==expected,'Actual filled source parameters differ from recipe.')
            state={'parameters':before,'parameterLayouts':layouts,'forwardCount':0}
            states.append(state)
            # The weak bound method does not retain the model beyond the original
            # generator's lifetime. Its exact compiled AST body is unchanged.
            original=weakref.WeakMethod(model._forward)
            def forward(*args,**kwargs):
                method=original();require(method is not None,'Source model lifetime ended early.')
                result=method(*args,**kwargs)
                after,after_layout=parameters(method.__self__)
                state['parametersAfter']=after;state['parameterLayoutsAfter']=after_layout
                state['forwardCount']+=1
                require(after==before and after_layout==layouts,'Source forward changed parameters/layouts.')
                return result
            model._forward=forward
            return expected
        return fill

    def record(value):
        result=common.tensor_record(value)
        return {**result,'stride':list(value.stride()),'aligned64':value.data_ptr()%64==0}

    with torch.no_grad():
        for component in ('unet','guidance'):
            module=importlib.import_module(component)
            original_record,original_fill=module.tensor_record,module.fill_parameters
            states=[];evidence=[]
            try:
                module.tensor_record=record;module.fill_parameters=observe_fill(states)
                document=module.generate(source,evidence)
            finally:
                module.tensor_record=original_record;module.fill_parameters=original_fill
            documents[component]={'sources':evidence,**document};parameter_states[component]=states
    require(len(parameter_states['unet'])==2 and len(parameter_states['guidance'])==1,'Source model count differs.')
    require([p['forwardCount'] for p in parameter_states['unet']]==[12,12],'Source U-Net forward count differs.')
    require(parameter_states['guidance'][0]['forwardCount']==30,'Source CFG forward count differs.')
    cases=[]
    for case in documents['unet']['cases']:
        model=documents['unet']['models'][case['model']]
        state=parameter_states['unet'][0 if case['model']=='sd15-reduced' else 1]
        cases.append({'id':case['id'],'kind':'unet','config':model['config'],'options':{},**state,
            'inputs':{n:case[n] for n in ('latent','timesteps','context')},
            'tensors':{**case['intermediates'],'output':case['output']},
            'observer':{'sequence':['on','off','off'],'outputHashes':case['repeatHashes'],'bitIdentical':len(set(case['repeatHashes']))==1,
                'scope':'existing frozen generator hooks in first forward; remaining two unobserved'}})
    for case in documents['guidance']['cases']:
        for policy in ('separate','concatenateCompatible'):
            hashes=[r[policy] for r in case['repeatHashes']]
            cases.append({'id':case['id']+'/'+policy,'kind':'guidance','config':documents['guidance']['config'],
                'options':{'policy':policy,'scale':case['scale'],'predictionKind':documents['guidance']['predictionKind'],
                    'concatEligible':case['concatEligible'],'commonLength':case['commonLength']},
                **parameter_states['guidance'][0],
                'inputs':{n:case[n] for n in ('latent','sigma','positive','negative')},'tensors':{'output':case['outputs'][policy]},
                'observer':{'sequence':['off','off','off'],'outputHashes':hashes,'bitIdentical':len(set(hashes))==1,
                    'scope':'no module hook; output metadata after forward'}})
    build=torch.__config__.show()
    for p in (Path.home(),Path(sys.prefix),source,output):build=build.replace(str(p),'<local>')
    wheel=Path(torch.__file__).resolve().parent/'lib'
    report={'backendCommit':common.COMMIT,'profile':common.PROFILE,'sourceScripts':scripts(),'diagnosticScriptSha256':digest(Path(__file__)),
        'operationVersions':{'python':platform.python_version(),'torch':torch.__version__,'torchGit':torch.version.git_version,
            'numpy':importlib.metadata.version('numpy'),'einops':importlib.metadata.version('einops'),
            'implementation':'unchanged frozen UNetModel._forward, EPS, ModelSamplingDiscrete, cfg_function, CONDRegular/CONDCrossAttn'},
        'runtime':{'threads':torch.get_num_threads(),'interopThreads':torch.get_num_interop_threads(),'build':build,
            'requestedCapability':os.environ.get('ATEN_CPU_CAPABILITY','auto'),'cpuCapability':torch.backends.cpu.get_cpu_capability(),
            'nativeLibraries':loaded_libraries(),
            'wheelLibraries':[{'name':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(wheel.iterdir()) if p.is_file() and '.so' in p.name]},
        'sources':{name:doc['sources'] for name,doc in documents.items()},
        'sourceConfigurations':{name:model['sourceConfig'] for name,model in documents['unet']['models'].items()},
        'cases':cases,'qualification':'none; diagnostic metadata around unchanged source generators'}
    require(report['sourceScripts']==accepted['scripts'],'Accepted source script changed during execution.')
    require(len(cases)==14 and len({c['id'] for c in cases})==14,'Source case set differs.')
    write(output/'suite.json',report)


if __name__=='__main__':raise SystemExit(main())
