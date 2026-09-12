"""Actual frozen load_lora/calculate_weight for mixed additive and LoRA files, CPU-only lab."""
import argparse,ast,hashlib,json,logging,subprocess,types
from typing import Callable,Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert torch.__version__=='2.10.0+cpu' and torch.version.cuda is None
torch.set_num_threads(1);torch.set_num_interop_threads(1)
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={}
ns=dict(torch=torch,nn=nn,F=F,logging=logging,Optional=Optional,Callable=Callable)
ns['comfy']=types.SimpleNamespace(model_management=types.SimpleNamespace(cast_to_device=lambda t,device,dtype:t.to(device=device,dtype=dtype)))
def extract(path,names):
 raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
 nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
extract('comfy/weight_adapter/base.py',['WeightAdapterBase','weight_decompose','pad_tensor_to_shape'])
extract('comfy/weight_adapter/lora.py',['LoRAAdapter'])
ns['weight_adapter']=types.SimpleNamespace(adapters=[ns['LoRAAdapter']],WeightAdapterBase=ns['WeightAdapterBase'])
extract('comfy/lora.py',['load_lora','calculate_weight'])
def pack(t):return {'shape':list(t.shape),'values':t.flatten().tolist()}
cases=[]
def capture(name,tensors,aliases,shape,strength):
 patches=ns['load_lora'](tensors,{prefix:'layer.weight' for prefix in aliases},log_missing=False)
 weights={key:torch.ones(shape if key.endswith('.weight') else [shape[0]]) for key in patches}
 outputs={key:ns['calculate_weight']([(strength,patch,1,None,None)],weights[key].clone(),key) for key,patch in patches.items()}
 cases.append({'name':name,'tensors':{k:pack(v) for k,v in tensors.items()},'aliases':aliases,'targetShape':shape,'strength':strength,'weights':{k:pack(v) for k,v in weights.items()},'outputs':{k:pack(v) for k,v in outputs.items()}})
base={'layer.lora_up.weight':torch.ones(2,1),'layer.lora_down.weight':torch.ones(1,3)}
capture('lora-plus-bias',{**base,'layer.diff_b':torch.tensor([.25,-.5])},['layer'],[2,3],.5)
capture('explicit-diff-overrides-lora',{**base,'layer.diff':torch.arange(6,dtype=torch.float32).reshape(2,3),'layer.diff_b':torch.tensor([2.,4.])},['layer'],[2,3],-.5)
capture('norm-then-explicit',{'layer.w_norm':torch.tensor([8.,8.]),'layer.b_norm':torch.tensor([9.,9.]),'layer.diff':torch.tensor([1.,2.]),'layer.diff_b':torch.tensor([3.,4.])},['layer'],[2],.5)
capture('later-alias-overrides-weight-only',{'first.diff':torch.ones(2),'first.diff_b':torch.tensor([2.,4.]),'second.diff':torch.tensor([3.,6.])},['first','second'],[2],.5)
capture('bias-only-zero-strength',{'layer.diff_b':torch.tensor([2.,4.])},['layer'],[2,3],0.)
with open(a.output,'x',encoding='utf-8',newline='\n') as f:
 json.dump({'sourceCommit':commit,'sourceHashes':hashes,'torch':torch.__version__,'scope':'Frozen load_lora and calculate_weight with LoRA provider and explicit additive tensors. Exactly representable values; no weights or .NET outputs. Other algorithms, set/padding and training-node qualification excluded.','cases':cases},f,indent=2,allow_nan=False);f.write('\n')
