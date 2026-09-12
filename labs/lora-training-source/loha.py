"""Frozen LohaDiff forward, custom backward and optimizer oracle. Separate CPU source lab."""
import argparse, ast, hashlib, json, pathlib, subprocess
from typing import Callable, Optional
import torch

p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--output',required=True); a=p.parse_args()
assert torch.__version__=='2.10.0+cpu' and torch.version.cuda is None
torch.set_num_threads(1); torch.set_num_interop_threads(1)
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a'; hashes={}
ns=dict(torch=torch,nn=torch.nn,Callable=Callable,Optional=Optional)
def extract(path,names):
    raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]); hashes[path]=hashlib.sha256(raw).hexdigest()
    nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
extract('comfy/weight_adapter/base.py',['WeightAdapterTrainBase'])
extract('comfy/weight_adapter/loha.py',['HadaWeight','HadaWeightTucker','LohaDiff'])

def values(shape,offset):
    count=1
    for n in shape: count*=n
    return (((torch.arange(count,dtype=torch.float32)+offset)%17-8)/19).reshape(shape)
def record(t):
    if t is None:return None
    return dict(shape=list(t.shape),values=t.detach().flatten().tolist())
cases=[]
for kind in ['matrix','conv2d','tucker2d','tucker1d']:
    tucker=kind.startswith('tucker'); spatial=[2,2] if kind.endswith('2d') else ([3] if kind.endswith('1d') else [])
    columns=5
    if not tucker:
        for n in spatial:columns*=n
    for optimizer in ['SGD','Adam','AdamW','RMSprop']:
        w1a=values([2,4] if tucker else [4,2],0);w1b=values([3 if tucker else 2,columns],3)
        w2a=values([2,4] if tucker else [4,3],5);w2b=values([1 if tucker else 3,columns],7)
        t1=values([2,3]+spatial,2) if tucker else None
        t2=values([2,1]+spatial,8) if tucker else None
        adapter=ns['LohaDiff']((w1a,w1b,1.75,w2a,w2b,t1,t2,None)).requires_grad_(True)
        initial={k:record(v) for k,v in adapter.named_parameters()}
        weight=values([4,5]+spatial,11);target=values(list(weight.shape),13)
        optim=getattr(torch.optim,optimizer)(adapter.parameters(),lr=.003)
        steps=[]
        for i in range(2):
            optim.zero_grad(set_to_none=True); output=adapter(weight)
            loss=torch.nn.functional.mse_loss(output,target);loss.backward()
            gradients={k:record(v.grad) for k,v in adapter.named_parameters()}
            assert gradients['alpha'] is None
            optim.step()
            steps.append(dict(output=record(output),loss=loss.item(),gradients=gradients,updated={k:record(v) for k,v in adapter.named_parameters()}))
        try:adapter.h(torch.ones([1]),torch.ones([1]))
        except NotImplementedError:bypass='NotImplementedError'
        else:raise RuntimeError('Source trainable bypass unexpectedly exists')
        cases.append(dict(kind=kind,optimizer=optimizer,learningRate=.003,initial=initial,weight=record(weight),target=record(target),steps=steps,bypass=bypass))
data=dict(sourceCommit=commit,sourceHashes=hashes,torch=torch.__version__,absoluteTolerance=3e-5,relativeTolerance=3e-5,cases=cases,
          scope='Frozen first-order LohaDiff custom backward, including cross-side Tucker a-gradients and null alpha gradient. Four optimizer updates on fixed small weights; not fresh factory, inference loader, pretrained workflow or platform qualification.')
with open(a.output,'x',encoding='utf-8',newline='\n') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
print('Collected sixteen frozen LoHa cases and 32 optimizer updates.')
