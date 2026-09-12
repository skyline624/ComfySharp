"""Frozen LoRA arithmetic and actual source LoraDiff gradients; development lab only."""
import argparse,ast,hashlib,json,logging,subprocess,types
from typing import Callable,Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert torch.__version__=='2.10.0+cpu'
torch.set_num_threads(1);torch.set_num_interop_threads(1)
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={}
ns=dict(torch=torch,nn=nn,F=F,logging=logging,Optional=Optional,Callable=Callable)
ns['comfy']=types.SimpleNamespace(model_management=types.SimpleNamespace(cast_to_device=lambda t,device,dtype:t.to(device=device,dtype=dtype)))
def extract(path,names):
    raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
    selected=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(selected)==len(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),path,'exec'),ns)
extract('comfy/weight_adapter/base.py',['WeightAdapterBase','WeightAdapterTrainBase','weight_decompose','pad_tensor_to_shape','tucker_weight_from_conv'])
extract('comfy/weight_adapter/lora.py',['LoRAAdapter','LoraDiff'])
def values(shape,offset=0):
    n=1
    for d in shape:n*=d
    return ((torch.arange(n,dtype=torch.float32)+offset)%17-8).reshape(shape)/16
def pack(t):return None if t is None else {'shape':list(t.shape),'values':t.flatten().tolist()}
cases=[]
for kind in ['dense','conv','mid','dora-output','dora-input']:
    shape=(3,4) if kind!='conv' and kind!='mid' else (3,2,3,3)
    w=values(shape,1);up=values((3,2) if len(shape)==2 else (3,2,1,1),2)
    down=values((2,4) if len(shape)==2 else (2,2,1,1) if kind=='mid' else (2,2,3,3),3)
    mid=values((2,2,3,3),5) if kind=='mid' else None
    dora=torch.tensor([.2,.4,.6]).reshape(3,1) if kind=='dora-output' else torch.tensor([.2,.4,.6,.8]).reshape(1,4) if kind=='dora-input' else None
    for strength,alpha in [(1,None),(-.75,3.),(0.,2.)]:
        adapter=ns['LoRAAdapter'](set(),(up,down,alpha,mid,dora,None))
        actual=adapter.calculate_weight(w.clone(),'fixture',strength,1,None,lambda x:x)
        cases.append({'kind':kind,'weight':pack(w),'up':pack(up),'down':pack(down),'mid':pack(mid),'dora':pack(dora),'strength':strength,'alpha':alpha,'output':pack(actual)})
w=values((3,4),1);up=values((3,2),2);down=values((2,4),3);alpha=2.5
train=ns['LoraDiff']((up,down,alpha,None,None,None))
result=train(w);loss=(result**2).mean();loss.backward()
training={'weight':pack(w),'up':pack(up),'down':pack(down),'alpha':alpha,'output':pack(result.detach()),'loss':loss.item(),
    'upGradient':pack(train.lora_up.weight.grad),'downGradient':pack(train.lora_down.weight.grad),'learningRate':.05}
optimizer=torch.optim.SGD([train.lora_up.weight,train.lora_down.weight],lr=.05);optimizer.step()
training.update(upUpdated=pack(train.lora_up.weight.detach()),downUpdated=pack(train.lora_down.weight.detach()),outputUpdated=pack(train(w).detach()))
with open(a.output,'x',encoding='utf-8',newline='\n') as f:
    json.dump({'sourceCommit':commit,'sourceHashes':hashes,'torch':torch.__version__,'absoluteTolerance':1e-6,'relativeTolerance':1e-6,
        'scope':'Actual LoRAAdapter.calculate_weight and LoraDiff forward/gradients with explicit small tensors. Native cast shim is Tensor.to. No model-family or training-node qualification.',
        'cases':cases,'training':training},f,indent=2,allow_nan=False);f.write('\n')
