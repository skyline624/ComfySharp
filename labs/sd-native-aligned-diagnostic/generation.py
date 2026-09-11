"""New orchestration only; frozen native operations and source input corpus stay unchanged."""
import ast
import hashlib
import importlib
import json
import logging
import math
from pathlib import Path

from allocation import Prepared, layout

def generate(source, common, torch, repo):
    import numpy as np
    unet=importlib.import_module('unet')
    fixtures=repo/'tests/ComfySharp.Inference.Tests/Fixtures'
    # Expected outputs are not selected or used in this generator.
    u=json.loads((fixtures/'sd-components.linux-x64.unet.json').read_text())
    g=json.loads((fixtures/'sd-components.linux-x64.guidance.json').read_text())
    evidence=[]
    model_type,operations=unet.source_model_type(source,evidence)
    documents={'unet':{'sources':evidence,'models':{}},'guidance':{'sources':[]}}
    boundaries={'time_embed':'timeEmbedding','input_blocks.2.1':'down0','input_blocks.5.1':'down1',
                'input_blocks.8.1':'down2','input_blocks.11.0':'down3','middle_block.2':'middle',
                'output_blocks.2.1':'up0','output_blocks.5.2':'up1','output_blocks.8.2':'up2'}
    def record(value):
        return {**common.tensor_record(value),**layout(value)}
    def parameters(model):
        entries=[];layouts={}
        for name,value in model.named_parameters():
            r=record(value)
            entries.append({k:r[k] for k in ('shape','sha256')}|{'name':name})
            layouts[name]={k:r[k] for k in ('shape','stride','aligned64','storageOffset','addressModulo64','dtype')}
        common.require(len(entries)==686,'Unexpected parameters.')
        return entries,layouts
    def input_value(record):
        common.require(record['dtype']=='float32','Input dtype differs.')
        value=torch.tensor(record['values'],dtype=torch.float32,device='cpu').reshape(record['shape'])
        common.require(common.tensor_record(value)['sha256']==record['sha256'],'Input payload SHA differs.')
        return value
    def run_case(identifier,kind,config,options,model,borrowed,forward):
        prepared=Prepared(torch,kind,borrowed)
        before,layouts=parameters(model);input_records={n:record(v) for n,v in prepared.values.items()}
        captures={};outputs=[];hashes=[]
        for repeat in range(3):
            hooks=[]
            if kind=='unet' and repeat==1:
                for path,name in boundaries.items():
                    hooks.append(model.get_submodule(path).register_forward_hook(
                        lambda _module,_args,value,label=name:captures.__setitem__(label,record(value))))
            try:
                prepared.observe()
                result=forward(prepared.values)
                prepared.observe()
                outputs.append(result);hashes.append(record(result)['sha256'])
                if repeat==1:captures['output']=record(result)
            finally:
                for hook in hooks:hook.remove()
        after,after_layouts=parameters(model)
        common.require(before==after and layouts==after_layouts and len(set(hashes))==1,'Parameter/repeat integrity failed.')
        common.require(set(captures)==(set(boundaries.values())|{'output'} if kind=='unet' else {'output'}),'Incomplete captures.')
        return {'id':identifier,'kind':kind,'config':config,'options':options,'parameters':before,'parameterLayouts':layouts,
                'parametersAfter':after,'parameterLayoutsAfter':after_layouts,'inputs':input_records,'tensors':captures,
                'inputAllocation':prepared.evidence(),
                'observer':{'sequence':['off','on','off'] if kind=='unet' else ['off','off','off'],
                            'outputHashes':hashes,'bitIdentical':True,'scope':'new aligned-input diagnostic; hooks only middle forward'}}
    cases=[]
    with torch.no_grad():
        for linear in (False,True):
            ident='sd2-reduced' if linear else 'sd15-reduced';configuration=unet.source_configuration(linear)
            model=model_type(**configuration,dtype=torch.float32,device='meta',operations=operations)
            model.to_empty(device='cpu');model.requires_grad_(False);model.eval();common.fill_parameters(model)
            documents['unet']['models'][ident]={'sourceConfig':configuration}
            for spec in (s for s in u['cases'] if s['model']==ident):
                borrowed={n:input_value(spec[n]) for n in ('latent','timesteps','context')}
                def forward(v):return model._forward(v['latent'],timesteps=v['timesteps'],context=v['context'],y=None,control=None,transformer_options={})
                cases.append(run_case(spec['id'],'unet',u['models'][ident]['config'],{},model,borrowed,forward))
            del model
        ns={'torch':torch,'np':np,'math':math,'logging':logging};ev=documents['guidance']['sources']
        for file,pin,names in (
            ('comfy/ldm/modules/diffusionmodules/util.py','fb58652a35521fc23bdcb75d91adace8e4cc79e2d5b13af1617a38d0c0f7142e',['make_beta_schedule']),
            ('comfy/model_sampling.py','173346f1975f6ddedee08505915ed97d152c3951747ebd213be0434f8d7011bd',['reshape_sigma','EPS','ModelSamplingDiscrete']),
            ('comfy/samplers.py','f2c264ca9d394612f828e3ffe167c856a278a10e1711b269f0ba65dccb66393f',['cfg_function']),
            ('comfy/conds.py','72058e9a22c972a9c875819e59d432d30d367fd2f7092ee6c6c45e5a60c959b0',['CONDRegular','CONDCrossAttn'])):
            common.load_symbols(source,file,pin,names,ns,ev)
        model=model_type(**unet.source_configuration(False),dtype=torch.float32,device='meta',operations=operations)
        model.to_empty(device='cpu');model.requires_grad_(False);model.eval();common.fill_parameters(model)
        predictor=ns['EPS']();predictor.sigma_data=1.0
        # Execute the exact already-audited adapter function declarations, without its generator.
        helper=repo/'labs/sd-source/guidance.py';tree=ast.parse(helper.read_bytes())
        generate_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
        nodes=[n for n in generate_node.body if isinstance(n,ast.FunctionDef) and n.name in ('denoise','separate')]
        common.require(len(nodes)==2,'Guidance adapter declarations differ.')
        module=ast.Module(body=nodes,type_ignores=[])
        ev.append({'adapter':'labs/sd-source/guidance.py:denoise,separate','fileSha256':hashlib.sha256(helper.read_bytes()).hexdigest(),
                   'astSha256':hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest()})
        ns.update(model=model,predictor=predictor,schedule=ns['ModelSamplingDiscrete'](),cfg=ns['cfg_function'])
        exec(compile(module,str(helper),'exec'),ns)
        regular,cross=ns['CONDRegular'],ns['CONDCrossAttn']
        for spec in g['cases']:
            for policy in ('separate','concatenateCompatible'):
                borrowed={n:input_value(spec[n]) for n in ('latent','sigma','positive','negative')}
                ns['scale']=spec['scale']
                def forward(v):
                    if policy=='separate':return ns['separate'](v['latent'],v['sigma'],v['positive'],v['negative'])
                    positive,negative=cross(v['positive']),cross(v['negative'])
                    if not positive.can_concat(negative):return ns['separate'](v['latent'],v['sigma'],v['positive'],v['negative'])
                    context=positive.concat([negative])
                    latents=regular(v['latent']).concat([regular(v['latent'])])
                    sigmas=regular(v['sigma']).concat([regular(v['sigma'])])
                    prediction=ns['denoise'](latents,sigmas,context)
                    conditional,unconditional=prediction.chunk(2)
                    return ns['cfg'](None,conditional,unconditional,spec['scale'],v['latent'],v['sigma'])
                options={'policy':policy,'scale':spec['scale'],'predictionKind':g['predictionKind'],
                         'concatEligible':spec['concatEligible'],'commonLength':spec['commonLength']}
                cases.append(run_case(spec['id']+'/'+policy,'guidance',g['config'],options,model,borrowed,forward))
        del model
    return cases,documents
