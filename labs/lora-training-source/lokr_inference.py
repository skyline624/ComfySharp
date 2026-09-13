"""Independent frozen LoKr inference reference; Python is laboratory-only."""
import argparse, ast, functools, hashlib, json, logging, pathlib, subprocess, types
from typing import Callable, Optional
import torch

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert torch.__version__=='2.10.0+cpu' and torch.version.cuda is None
torch.set_num_threads(1);torch.set_num_interop_threads(1)
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={}
comfy=types.SimpleNamespace(model_management=types.SimpleNamespace(cast_to_device=lambda t,device,dtype:t.to(device=device,dtype=dtype)))
ns=dict(torch=torch,nn=torch.nn,F=torch.nn.functional,Callable=Callable,Optional=Optional,comfy=comfy,cache=functools.cache,logging=logging)
def extract(path,names):
    raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
    nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
extract('comfy/weight_adapter/base.py',['WeightAdapterBase','weight_decompose','pad_tensor_to_shape'])
extract('comfy/weight_adapter/lora.py',['LoRAAdapter'])
extract('comfy/weight_adapter/loha.py',['_warn_loha_bypass_inefficient','LoHaAdapter'])
extract('comfy/weight_adapter/lokr.py',['LoKrAdapter'])
ns['weight_adapter']=types.SimpleNamespace(adapters=[ns['LoRAAdapter'],ns['LoHaAdapter'],ns['LoKrAdapter']])
extract('comfy/lora.py',['load_lora'])
def values(shape,offset):
    count=1
    for n in shape:count*=n
    return (((torch.arange(count,dtype=torch.float32)+offset)%19-9)/17).reshape(shape)
def record(t):return dict(shape=list(t.shape),values=t.detach().float().flatten().tolist())
class Errors(logging.Handler):
    def __init__(self):super().__init__(logging.ERROR);self.messages=[]
    def emit(self,entry):self.messages.append(entry.getMessage())
errors=Errors();logging.getLogger().addHandler(errors)
cases=[]
kinds=['direct2','direct3','direct4','direct5','first','second','both','inactive','tucker-single','tucker-multichannel']
for kind in kinds:
    kernel=[2]*(int(kind[-1])-2) if kind.startswith('direct') else ([2,2] if kind.startswith('tucker') else [])
    out2,in2=(1,1) if kind=='tucker-single' else (3,2)
    for axis in ['none','output','input']:
        for strength,alpha,dtype in [(1.,None,torch.float32),(.75,1.25,torch.float16),(-.4,2.25,torch.bfloat16),(0.,1.75,torch.float32),(1.,0.,torch.float32)]:
            factors=dict(lokr_w1=values([2,2],0),lokr_w2=values([out2,in2]+kernel,3))
            if kind in ['first','both','inactive']:
                factors.update(lokr_w1_a=values([2,3],5),lokr_w1_b=values([3,2],7))
                if kind!='inactive':del factors['lokr_w1']
            if kind in ['second','both','inactive'] or kind.startswith('tucker'):
                factors.update(lokr_w2_a=values([1,out2] if kernel else [out2,1],11),lokr_w2_b=values([1,in2],13))
                if kernel:factors['lokr_t2']=values([1,1]+kernel,17)
                if kind!='inactive':del factors['lokr_w2']
            factors={k:v.to(dtype) for k,v in factors.items()}
            if alpha is not None:factors['alpha']=torch.tensor(alpha)
            shape=[2*out2,2*in2]+kernel
            if axis!='none':
                ds=[shape[0],1] if axis=='output' else [1,shape[1]]
                factors['dora_scale']=(values(ds,2).abs()+.5).reshape(ds+[1]*len(kernel)).to(dtype)
            adapter=ns['LoKrAdapter'].load('layer',{'layer.'+k:v for k,v in factors.items()},alpha,factors.get('dora_scale'))
            weight=values(shape,15);errors.messages.clear()
            output=adapter.calculate_weight(weight.clone(),'layer.weight',strength,1,None,lambda x:x)
            if errors.messages:assert torch.equal(output,weight)
            cases.append(dict(kind=kind,axis=axis,strength=strength,alpha=alpha,dtype=str(dtype),factors={k:record(v) for k,v in factors.items()},
                weight=record(weight),output=record(output),loadedKeys=sorted(adapter.loaded_keys),sourceErrors=list(errors.messages)))
loader_cases=[]
for kind in ['provider-overwrite','later-alias','norm-overwrite','diff-overwrite']:
    tensors={'layer.lokr_w1':values([2,2],0),'layer.lokr_w2':values([3,2],3)};mapping={'layer':'layer.weight'}
    if kind=='provider-overwrite':
        tensors.update({'layer.lora_up.weight':values([6,1],1),'layer.lora_down.weight':values([1,4],2),
            'layer.hada_w1_a':values([6,2],1),'layer.hada_w1_b':values([2,4],2),
            'layer.hada_w2_a':values([6,1],3),'layer.hada_w2_b':values([1,4],4)})
    elif kind=='later-alias':
        tensors.update({'later.lora_up.weight':values([6,1],1),'later.lora_down.weight':values([1,4],2)});mapping['later']='layer.weight'
    else:
        tensors['layer.w_norm']=values([6,4],3)
        if kind=='diff-overwrite':tensors['layer.diff']=values([6,4],8)
    selected=ns['load_lora'](tensors,mapping,log_missing=False)['layer.weight'];weight=values([6,4],11)
    if isinstance(selected,tuple):result=weight+selected[1][0];selected_kind='diff'
    else:result=selected.calculate_weight(weight.clone(),'layer.weight',1,1,None,lambda x:x);selected_kind=selected.name
    loader_cases.append(dict(kind=kind,tensors={k:record(v) for k,v in tensors.items()},aliases=list(mapping),selectedKind=selected_kind,weight=record(weight),output=record(result)))
data=dict(sourceCommit=commit,sourceHashes=hashes,torch=torch.__version__,absoluteTolerance=3e-5,relativeTolerance=3e-5,cases=cases,loaderCases=loader_cases,
    scope='Frozen weight inference only. Source logged failures remain explicit failures, not accepted unchanged-weight success. No bypass, training, real-model or GPU qualification.')
with open(a.output,'x',encoding='utf-8',newline='\n') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
print('Collected',len(cases),'cases;',sum(bool(c['sourceErrors']) for c in cases),'source logged failures.')
