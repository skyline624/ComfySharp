"""Capture frozen LoRA suffix and alias precedence. No checkpoint or application imports."""
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
    definitions=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(definitions)==len(names)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),path,'exec'),ns)
extract('comfy/weight_adapter/base.py',['WeightAdapterBase','weight_decompose','pad_tensor_to_shape'])
extract('comfy/weight_adapter/lora.py',['LoRAAdapter'])
ns['weight_adapter']=types.SimpleNamespace(adapters=[ns['LoRAAdapter']])
extract('comfy/lora.py',['load_lora'])
suffixes=[('.lora_up.weight','.lora_down.weight'),('_lora.up.weight','_lora.down.weight'),('.lora_B.weight','.lora_A.weight'),('.lora.up.weight','.lora.down.weight'),('.lora_B','.lora_A'),('.lora_linear_layer.up.weight','.lora_linear_layer.down.weight'),('.lora_B.default.weight','.lora_A.default.weight')]
cases=[]
def pack(t):return {'dtype':{torch.float32:'F32',torch.float16:'F16',torch.bfloat16:'BF16',torch.float64:'F64'}[t.dtype],'shape':list(t.shape),'values':t.flatten().tolist()}
def capture(name,tensors,aliases):
    patches=ns['load_lora'](tensors,{prefix:'weight' for prefix in aliases},log_missing=False)
    adapter=patches['weight'];weight=torch.tensor([1.,2.,3.,4.,5.,6.]).reshape(2,3)
    result=adapter.calculate_weight(weight.clone(),'weight',.5,1,None,lambda x:x)
    cases.append({'name':name,'tensors':{k:pack(v) for k,v in tensors.items()},'aliases':aliases,'targetShape':[2,3],
        'weight':pack(weight),'output':pack(result),'claimedKeys':sorted(adapter.loaded_keys),
        'unclaimedKeys':sorted(set(tensors)-adapter.loaded_keys)})
for i,(up,down) in enumerate(suffixes):
    dtype=[torch.float32,torch.float16,torch.bfloat16][i%3]
    capture('suffix-'+str(i),{'layer'+up:torch.tensor([.25,.5],dtype=dtype).reshape(2,1),'layer'+down:torch.tensor([1.,2.,3.],dtype=dtype).reshape(1,3),'layer.alpha':torch.tensor(2.,dtype=torch.float64)},['layer'])
t={'layer.lora_up.weight':torch.ones(2,1),'layer.lora_down.weight':torch.ones(1,3),
   'layer.lora_B.weight':torch.full((2,1),4.),'layer.lora_A.weight':torch.full((1,3),4.)}
capture('regular-over-B',t,['layer'])
t={'first.lora_up.weight':torch.ones(2,1),'first.lora_down.weight':torch.ones(1,3),
   'second.lora_B.weight':torch.full((2,1),4.),'second.lora_A.weight':torch.full((1,3),.5)}
capture('later-alias-wins',t,['first','second'])
with open(a.output,'x',encoding='utf-8',newline='\n') as f:
    json.dump({'sourceCommit':commit,'sourceHashes':hashes,'torch':torch.__version__,'absoluteTolerance':1e-6,'relativeTolerance':1e-6,
        'scope':'Actual source load_lora using the LoRAAdapter provider for LoRA-only input files. Other providers are outside this corpus. Alias order supplied explicitly; architecture mapping not qualified.',
        'cases':cases},f,indent=2,allow_nan=False);f.write('\n')
