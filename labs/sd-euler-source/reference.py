"""Prospective frozen-source Euler laboratory; never imported by .NET."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import sys
import time
import types

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL_SHA256 = '7868368a38fdf895fbe8698d4ab5bbbea1352a11df6dd7b3bd6c1e776c55cd05'


def require(value, message):
    if not value: raise ValueError(message)


def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, document):
    require(not path.exists(), 'Evidence is never overwritten.')
    path.write_text(json.dumps(document, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def preflight(source, output):
    require(digest(HERE/'protocol.json') == PROTOCOL_SHA256, 'Prospective protocol changed.')
    protocol = json.loads((HERE/'protocol.json').read_text())
    source = source.resolve(strict=True)
    require(source.is_dir() and output.is_absolute(), 'Explicit source and absolute output required.')
    output = output.resolve()
    for protected in (REPO, source, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()):
        require(not (output.is_relative_to(protected) or protected.is_relative_to(output)), 'Output overlaps protected input.')
    require(not output.exists(), 'Output must be absent.')
    for name, sha in protocol['sources'].items():
        path = (source/name).resolve(strict=True)
        require(path.is_relative_to(source) and path.stat().st_size <= 8*1024*1024, 'Source escapes snapshot or exceeds bound.')
        require(digest(path) == sha, 'Source SHA differs: '+name)
        ast.parse(path.read_text(encoding='utf-8'))
    for name, sha in protocol['helpers'].items():
        require(digest(REPO/name) == sha, 'Audited helper changed: '+name)
    for name, sha in protocol['locks'].items():
        require(digest(REPO/'labs/clip-source'/name) == sha, 'Canonical lock bytes differ: '+name)
    for case in protocol['cases']:
        sigmas = case['sigmas']
        require(len(sigmas)>=2 and sigmas[-1]==0 and all(math.isfinite(x) and x>0 for x in sigmas[:-1])
                and all(a>=b for a,b in zip(sigmas,sigmas[1:])), 'Invalid prospective sigma sequence.')
    return source, output, protocol


def generate(source, protocol, evidence):
    import torch
    import numpy as np
    from common import load_symbols, synthetic_tensor, fill_parameters, tensor_record
    from unet import source_model_type, source_configuration
    ns = {'torch':torch,'np':np,'math':math,'logging':logging}
    def load(relative, names, namespace=ns):
        load_symbols(source,relative,protocol['sources'][relative],names,namespace,evidence)
    load('comfy/ldm/modules/diffusionmodules/util.py',['make_beta_schedule'])
    load('comfy/model_sampling.py',['reshape_sigma','EPS','V_PREDICTION','ModelSamplingDiscrete'])
    load('comfy/samplers.py',['cfg_function','sampling_function'])
    load('comfy/conds.py',['CONDRegular','CONDCrossAttn'])
    utility = {}
    load('comfy/k_diffusion/utils.py',['append_dims'],utility)
    ns['utils'] = types.SimpleNamespace(append_dims=utility['append_dims'])
    # Progress-only adapter; no tensor or model operation is replaced.
    ns['trange'] = lambda count, disable=None: range(count)
    load('comfy/k_diffusion/sampling.py',['to_d','sample_euler'])
    model_type, operations = source_model_type(source,evidence)
    configuration = source_configuration(False)
    model = model_type(**configuration,dtype=torch.float32,device='meta',operations=operations)
    model.to_empty(device='cpu');model.requires_grad_(False);model.eval()
    parameters = fill_parameters(model)
    require(len(parameters)==protocol['parameterCount'], 'Source parameter schema differs.')
    schedule = ns['ModelSamplingDiscrete']()
    regular,cross = ns['CONDRegular'],ns['CONDCrossAttn']
    def record(tensor):
        result = tensor_record(tensor)
        require(result['dtype']=='float32' and all(isinstance(x,(int,float)) and math.isfinite(x) for x in result['values']),
                'Finite CPU/F32 trajectory required.')
        result['stride']=list(tensor.stride());result['aligned64']=tensor.data_ptr()%64==0
        return result
    def parameter_hashes():
        return [{'name':n,'shape':list(p.shape),'sha256':tensor_record(p)['sha256']} for n,p in model.named_parameters()]
    cases=[]
    for spec in protocol['cases']:
        inputs={n:(torch.tensor(spec['sigmas'],dtype=torch.float32,device='cpu') if n=='sigmas'
                  else synthetic_tensor(value['name'],value['shape'])) for n,value in spec['inputs'].items()}
        before={n:record(v) for n,v in inputs.items()}
        for name,value in before.items():
            require((value['shape'],value['sha256'])==(spec['inputs'][name]['shape'],spec['inputs'][name]['sha256']),
                    'Prospectively pinned input differs: '+name)
        predictor=ns['EPS' if spec['predictionKind']=='epsilon' else 'V_PREDICTION']();predictor.sigma_data=1.0
        calls=[]
        def denoise(latent,sigma,context):
            scaled=predictor.calculate_input(sigma,latent)
            timestep=schedule.timestep(sigma).float().reshape(-1)
            prediction=model._forward(scaled,timesteps=timestep,context=context,y=None,control=None,transformer_options={})
            return predictor.calculate_denoised(sigma,prediction,latent)
        def calc(args):
            # Full-image resident condition adapter, same audited policy composition as guidance.py.
            positive,negative=args['conds'];latent=args['input'];sigma=args['sigma']
            eligible=negative is not None and cross(positive).can_concat(cross(negative))
            if spec['policy']=='concatenateCompatible' and eligible:
                context=cross(positive).concat([cross(negative)])
                both=denoise(regular(latent).concat([regular(latent)]),regular(sigma).concat([regular(sigma)]),context)
                return list(both.chunk(2))
            return [denoise(latent,sigma,positive),torch.zeros_like(latent) if negative is None else denoise(latent,sigma,negative)]
        options={'disable_cfg1_optimization':spec['disableScaleOneOptimization'],'sampler_calc_cond_batch_function':calc}
        def guided(latent,sigma):
            calls.append(tuple(sigma.shape))
            return ns['sampling_function'](None,latent,sigma,inputs.get('negative'),inputs['positive'],spec['scale'],model_options=options)
        captures=[];hashes=[];output=None
        def callback(step):
            captures.append({'index':step['i'],'x':record(step['x']),'denoised':record(step['denoised']),
                             'sigma':record(step['sigma']),'sigmaHat':record(step['sigma_hat'])})
        for iteration in range(3):
            calls.clear()
            result=ns['sample_euler'](guided,inputs['latent'],inputs['sigmas'],callback=callback if iteration==1 else None,
                                      disable=True,s_churn=0.)
            output=record(result);hashes.append(output['sha256'])
            require(calls==[(inputs['latent'].shape[0],)]*(len(spec['sigmas'])-1), 'Wrong Euler model call count or sigma batch.')
            require(not result.requires_grad and not torch.is_grad_enabled(), 'No-grad source contract differs.')
            del result
        require(len(set(hashes))==1, 'Source observer/repeated trajectories differ.')
        require(len(captures)==len(spec['sigmas'])-1, 'Euler callback trajectory incomplete.')
        after={n:record(v)['sha256'] for n,v in inputs.items()}
        require(after=={n:v['sha256'] for n,v in before.items()}, 'Source trajectory mutated an input.')
        require(parameter_hashes()==parameters, 'Source model parameters changed.')
        cases.append({k:spec[k] for k in ('id','predictionKind','policy','scale','disableScaleOneOptimization')} |
                     {**before,'negative':before.get('negative'),'output':output,'steps':captures,'repeatHashes':hashes,
                      'observerSequence':['off','on','off'],'inputsUnchanged':True,'parametersUnchanged':True,
                      'modelCallsPerRun':len(calls),'modelSigmaShape':[inputs['latent'].shape[0]]})
    return {'config':protocol['config'],'sourceConfig':configuration,'parameters':parameters,'cases':cases}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();source,output,protocol=preflight(args.source_directory,args.output)
    if args.validate_only:
        print('Validated prospective protocol, frozen sources, helpers and locks; no native imports or outputs.');return
    require(sys.version_info[:3]==(3,12,10), 'Pinned Python required.')
    for name,version in [('torch','2.10.0'),('numpy','2.2.6'),('einops','0.8.1')]:
        actual=importlib.metadata.version(name)
        require(actual==version or (name=='torch' and actual==version+'+cpu'), 'Pinned package required: '+name)
    require(not os.environ.get('ATEN_CPU_CAPABILITY'), 'Prospective collection requires unforced CPU dispatch.')
    sys.path.insert(0,str(REPO/'labs/sd-source'))
    import torch
    require(torch.version.cuda is None, 'CPU wheel required.')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    target={('Windows','AMD64'):'win-x64',('Linux','x86_64'):'linux-x64',('Darwin','arm64'):'osx-arm64'}.get((platform.system(),platform.machine()))
    require(target is not None, 'Unsupported target.')
    output.mkdir(parents=True)
    scripts={p.name:digest(p) for p in HERE.glob('*') if p.is_file()}
    start=time.perf_counter();evidence=[]
    write(output/'prospective-protocol.json',protocol)
    with torch.no_grad():document=generate(source,protocol,evidence)
    source_after={name:digest(source/name) for name in protocol['sources']}
    require(source_after==protocol['sources'], 'Source changed during collection.')
    require({p.name:digest(p) for p in HERE.glob('*') if p.is_file()}==scripts, 'Laboratory changed during collection.')
    for name,sha in protocol['helpers'].items():require(digest(REPO/name)==sha,'Helper changed during collection.')
    for name,sha in protocol['locks'].items():require(digest(REPO/'labs/clip-source'/name)==sha,'Lock changed during collection.')
    build=torch.__config__.show();parallel=torch.__config__.parallel_info()
    for private in (str(Path.home()),str(source),str(output),str(REPO),sys.prefix,sys.base_prefix):
        build=build.replace(private,'<local>');parallel=parallel.replace(private,'<local>')
    libraries=[{'name':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted((Path(torch.__file__).resolve().parent/'lib').iterdir())
               if p.is_file() and (p.suffix.lower() in ('.dll','.so','.dylib') or '.so.' in p.name)]
    document={'backendCommit':protocol['backendCommit'],'profile':protocol['profile'],'target':target,'sources':evidence,**document}
    write(output/'euler.json',document)
    write(output/'manifest.json',{'backendCommit':protocol['backendCommit'],'profile':protocol['profile'],'target':target,
          'protocolSha256':PROTOCOL_SHA256,'scripts':scripts,'helpers':protocol['helpers'],'locks':protocol['locks'],
          'sources':evidence,'comparison':protocol['comparison'],'parameterRecipe':protocol['parameterRecipe'],
          'runtime':{'python':platform.python_version(),'torch':torch.__version__,'torchGit':torch.version.git_version,
                     'threads':torch.get_num_threads(),'interopThreads':torch.get_num_interop_threads(),
                     'cpuCapability':torch.backends.cpu.get_cpu_capability(),'build':build,'parallelInfo':parallel,
                     'wheelPackageLibraries':libraries,'libraryEvidence':'package inventory; not actual loaded mappings'},
          'syntheticWeights':True,'pretrainedWeightsUsed':False,'qualification':'none; prospective source collection only',
          'component':{'name':'euler','file':'euler.json','bytes':(output/'euler.json').stat().st_size,
                       'sha256':digest(output/'euler.json'),'elapsedSeconds':time.perf_counter()-start}})
    print('Eight frozen-source Euler trajectories collected; no acceptance fixtures changed.')


if __name__=='__main__':main()
