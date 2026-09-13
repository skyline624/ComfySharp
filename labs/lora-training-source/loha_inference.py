"""Collect frozen LoHaAdapter calculate_weight/load/h using the separate CPU source lab."""
import argparse, ast, functools, hashlib, json, logging, pathlib, subprocess, types
from typing import Callable, Optional
import torch

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert torch.__version__=='2.10.0+cpu' and torch.version.cuda is None
torch.set_num_threads(1);torch.set_num_interop_threads(1)
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={}
# Only device conversion is stubbed; the adapter, decomposition, loader and bypass are source AST.
comfy=types.SimpleNamespace(model_management=types.SimpleNamespace(cast_to_device=lambda t,device,dtype:t.to(device=device,dtype=dtype)))
ns=dict(torch=torch,nn=torch.nn,F=torch.nn.functional,Callable=Callable,Optional=Optional,
        comfy=comfy,cache=functools.cache,logging=logging)
def extract(path,names):
    raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
    nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
extract('comfy/weight_adapter/base.py',['WeightAdapterBase','weight_decompose','pad_tensor_to_shape'])
extract('comfy/weight_adapter/lora.py',['LoRAAdapter'])
extract('comfy/weight_adapter/loha.py',['_warn_loha_bypass_inefficient','LoHaAdapter'])
ns['weight_adapter']=types.SimpleNamespace(adapters=[ns['LoRAAdapter'],ns['LoHaAdapter']])
extract('comfy/lora.py',['load_lora'])
def values(shape,offset):
    count=1
    for n in shape:count*=n
    return (((torch.arange(count,dtype=torch.float32)+offset)%19-9)/17).reshape(shape)
def record(t):
    return None if t is None else dict(shape=list(t.shape),values=t.detach().float().flatten().tolist())
cases=[]
for kind in ['matrix','conv2d','tucker2d','broadcast']:
    conv=kind in ['conv2d','tucker2d'];tucker=kind=='tucker2d';shape=[3,5]+([2,2] if conv else [])
    for axis in ['none','output','input']:
        for strength,alpha,dtype in [(1.,None,torch.float32),(.75,1.25,torch.float16),(-.4,2.25,torch.bfloat16),(0.,1.75,torch.float32)]:
            factors=dict(hada_w1_a=values([2,3] if tucker else [3,2],0),
                hada_w1_b=values([2,5 if tucker or not conv else 20],3),
                hada_w2_a=values([1,3] if tucker else [1 if kind=='broadcast' else 3,3],5),
                hada_w2_b=values([3,5 if tucker or not conv else 20],7))
            if tucker:factors.update(hada_t1=values([2,2,2,2],11),hada_t2=values([1,3,2,2],13))
            factors={k:v.to(dtype) for k,v in factors.items()}
            if alpha is not None:factors['alpha']=torch.tensor(alpha)
            dora=None if axis=='none' else values([3,1] if axis=='output' else [1,5],2).abs()+.5
            if dora is not None:
                if conv:dora=dora.reshape(*dora.shape,1,1)
                factors['dora_scale']=dora.to(dtype)
            adapter=ns['LoHaAdapter'].load('layer',{'layer.'+k:v for k,v in factors.items()},alpha,factors.get('dora_scale'))
            weight=values(shape,15)
            output=adapter.calculate_weight(weight.clone(),'layer.weight',strength,1,None,lambda x:x)
            x=values([1,5,4,5] if conv else [2,4,5],1).requires_grad_(True)
            adapter.multiplier=strength;adapter.is_conv=conv;adapter.conv_dim=2 if conv else 0
            adapter.kernel_size=(2,2);adapter.in_channels=5;adapter.kw_dict=dict(stride=2,padding=1) if conv else {}
            forward=(lambda x:torch.nn.functional.conv2d(x,weight,stride=2,padding=1)) if conv else (lambda x:torch.nn.functional.linear(x,weight))
            bypass=adapter.bypass_forward(forward,x);bypass.square().mean().backward()
            cases.append(dict(kind=kind,axis=axis,dtype=str(dtype),strength=strength,alpha=alpha,
                factors={k:record(v) for k,v in factors.items()},loadedKeys=sorted(adapter.loaded_keys),weight=record(weight),output=record(output),
                input=record(x),baseOutput=record(forward(x)),bypass=record(bypass),inputGradient=record(x.grad)))
loader_cases=[]
for kind in ['provider-overwrite','later-alias','norm-overwrite','diff-overwrite']:
    tensors={'layer.hada_w1_a':values([3,2],0),'layer.hada_w1_b':values([2,5],3),
             'layer.hada_w2_a':values([3,1],5),'layer.hada_w2_b':values([1,5],7)}
    mapping={'layer':'layer.weight'}
    if kind in ['provider-overwrite','later-alias']:
        prefix='layer' if kind=='provider-overwrite' else 'later'
        tensors.update({prefix+'.lora_up.weight':values([3,1],1),prefix+'.lora_down.weight':values([1,5],2)})
        mapping[prefix]='layer.weight'
    else:
        tensors['layer.w_norm']=values([3,5],3)
        if kind=='diff-overwrite':tensors['layer.diff']=values([3,5],8)
    selected=ns['load_lora'](tensors,mapping,log_missing=False)['layer.weight'];weight=values([3,5],11)
    if isinstance(selected,tuple):result=weight+selected[1][0];selected_kind='diff'
    else:result=selected.calculate_weight(weight.clone(),'layer.weight',1,1,None,lambda x:x);selected_kind=selected.name
    loader_cases.append(dict(kind=kind,tensors={k:record(v) for k,v in tensors.items()},aliases=list(mapping),selectedKind=selected_kind,weight=record(weight),output=record(result)))
data=dict(sourceCommit=commit,sourceHashes=hashes,torch=torch.__version__,absoluteTolerance=3e-5,relativeTolerance=3e-5,cases=cases,loaderCases=loader_cases,
    scope='Frozen LoHa inference load, weight decomposition, linear/Conv2d bypass and activation gradients; F32 computation with F32/F16/BF16 factors. Not trainable bypass, higher dimensional convolution, model-family or GPU qualification.')
with open(a.output,'x',encoding='utf-8',newline='\n') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
print('Collected',len(cases),'LoHa inference cases.')
