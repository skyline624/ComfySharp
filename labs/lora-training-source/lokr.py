"""Frozen LokrDiff reconstruction, alpha rules and optimizer/gradient reference."""
import argparse,ast,hashlib,json,pathlib,subprocess
from typing import Callable,Optional
import torch
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert torch.__version__=='2.10.0+cpu' and torch.version.cuda is None
torch.set_num_threads(1);torch.set_num_interop_threads(1)
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={};ns=dict(torch=torch,nn=torch.nn,F=torch.nn.functional,Callable=Callable,Optional=Optional)
for path,names in [('comfy/weight_adapter/base.py',['WeightAdapterTrainBase']),('comfy/weight_adapter/lokr.py',['LokrDiff'])]:
 raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
 nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
def values(shape,offset):
 n=1
 for size in shape:n*=size
 return (((torch.arange(n,dtype=torch.float32)+offset)%17-8)/19).reshape(shape)
def record(t):return None if t is None else dict(shape=list(t.shape),values=t.detach().flatten().tolist())
cases=[]
for kind in ['direct','first-rebuilt','second-rebuilt','both-rebuilt','tucker','direct-overrides-rebuilt','direct-conv1d','direct-conv2d','direct-conv3d','tucker-single-channel']:
 for optimizer in ['SGD','Adam','AdamW','RMSprop']:
  tucker=kind.startswith('tucker');single=kind=='tucker-single-channel';direct=kind.startswith('direct');both=kind in ['both-rebuilt','direct-overrides-rebuilt']
  spatial=[2]*int(kind[-2]) if kind.startswith('direct-conv') else ([2,2] if tucker else [])
  a1=values([2,3],1) if both or kind=='first-rebuilt' else None;b1=values([3,2],2) if a1 is not None else None
  a2=values([2,1 if single else 3] if tucker else [3,2],3) if both or kind=='second-rebuilt' or tucker else None
  b2=values([2,1 if single else 4],4) if a2 is not None else None;t2=values([2,2,2,2],5) if tucker else None
  w1=values([2,2],6) if direct or a1 is None else None
  w2=values([3,4]+spatial,7) if direct or a2 is None else None
  adapter=ns['LokrDiff']((w1,w2,1.75,a1,b1,a2,b2,t2,None)).requires_grad_(True)
  initial={k:record(v) for k,v in adapter.named_parameters()}
  shape=([2,2] if single else [6,8])+spatial;weight=values(shape,11);target=values(shape,13)
  optim=getattr(torch.optim,optimizer)(adapter.parameters(),lr=.003);steps=[];error=None
  for step in range(2):
   optim.zero_grad(set_to_none=True)
   try:output=adapter(weight)
   except RuntimeError as failure:
    if step!=0:raise
    error=dict(type=type(failure).__name__,message=str(failure));break
   loss=torch.nn.functional.mse_loss(output,target);loss.backward()
   gradients={k:record(v.grad) for k,v in adapter.named_parameters()};optim.step()
   steps.append(dict(output=record(output),loss=loss.item(),gradients=gradients,updated={k:record(v) for k,v in adapter.named_parameters()}))
  cases.append(dict(kind=kind,optimizer=optimizer,initial=initial,weight=record(weight),target=record(target),steps=steps,error=error))
data=dict(sourceCommit=commit,sourceHashes=hashes,absoluteTolerance=3e-5,relativeTolerance=3e-5,cases=cases,
 scope='Original LokrDiff direct/decomposed/Tucker reconstruction and first-order gradients, two updates with each of four optimizers. Not inference loader, bypass, fresh factory, pretrained model or platform qualification.')
with open(a.output,'x',encoding='utf-8',newline='\n') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
print(hashlib.sha256(pathlib.Path(a.output).read_bytes()).hexdigest())
