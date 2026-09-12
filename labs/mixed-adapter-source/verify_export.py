"""Independent restricted safetensors decoding and frozen source acceptance of a mixed .NET export."""
import argparse,ast,hashlib,json,logging,math,pathlib,struct,subprocess,types
from typing import Callable,Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--adapter',required=True);p.add_argument('--report',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert torch.__version__=='2.10.0+cpu' and torch.version.cuda is None
report=json.loads(pathlib.Path(a.report).read_text());e=report['export'];raw=pathlib.Path(a.adapter).read_bytes()
assert report['status']=='ok' and e['reloadPredictionExact'] and hashlib.sha256(raw).hexdigest()==e['adapterSha256']
length=struct.unpack('<Q',raw[:8])[0];assert length%8==0 and length<=16*1024*1024
header=json.loads(raw[8:8+length]);body=raw[8+length:];cursor=0;tensors={};mapping={}
for name,info in header.items():
 assert info['dtype']=='F32';start,end=info['data_offsets'];assert start==cursor and end-start==math.prod(info['shape'])*4 and end<=len(body);cursor=end
 payload=body[start:end];assert hashlib.sha256(payload).hexdigest()==e['parameterHashes'][name]
 value=torch.frombuffer(bytearray(payload),dtype=torch.float32).clone().reshape(info['shape']);assert value.isfinite().all();tensors[name]=value
 suffix=next(s for s in ['.lora_up.weight','.lora_down.weight','.alpha','.diff_b','.diff'] if name.endswith(s))
 stem=name[:-len(suffix)];mapping[stem]=stem.removeprefix('diffusion_model.')+'.weight'
assert cursor==len(body) and len(tensors)==1250 and set(tensors)==set(e['parameterHashes'])
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={};ns=dict(torch=torch,nn=nn,F=F,logging=logging,Callable=Callable,Optional=Optional)
ns['comfy']=types.SimpleNamespace(model_management=types.SimpleNamespace(cast_to_device=lambda t,device,dtype:t.to(device=device,dtype=dtype)))
def extract(path,names):
 raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
 nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
extract('comfy/weight_adapter/base.py',['WeightAdapterBase','weight_decompose','pad_tensor_to_shape'])
extract('comfy/weight_adapter/lora.py',['LoRAAdapter'])
ns['weight_adapter']=types.SimpleNamespace(adapters=[ns['LoRAAdapter']],WeightAdapterBase=ns['WeightAdapterBase'])
extract('comfy/lora.py',['load_lora','calculate_weight'])
loaded=ns['load_lora'](tensors,mapping,log_missing=False);assert len(loaded)==e['targets']==686
counts={'lora':0,'weightDifferences':0,'biasDifferences':0}
for name,patch in loaded.items():
 if isinstance(patch,ns['LoRAAdapter']):counts['lora']+=1
 else:
  assert patch[0]=='diff';counts['biasDifferences' if name.endswith('.bias') else 'weightDifferences']+=1
  delta=patch[1][0];actual=ns['calculate_weight']([(1,patch,1,None,None)],torch.zeros_like(delta),name)
  assert torch.equal(actual,delta)
assert counts=={'lora':282,'weightDifferences':109,'biasDifferences':295}
with open(a.output,'x',encoding='utf-8',newline='\n') as f:
 json.dump({'status':'ok','sourceCommit':commit,'sourceHashes':hashes,'adapterSha256':e['adapterSha256'],'all1250PayloadHashesMatch':True,'sourceAcceptedTargets':len(loaded),'kinds':counts,'sourceAdditiveApplicationsExact':True,'scope':'Restricted independent F32 byte decoder, actual frozen mixed loader and additive application. Does not claim safetensors-package, pretrained source-forward/gradient or full training-node qualification.'},f,indent=2);f.write('\n')
print('All 1250 parameter hashes match; frozen source accepts 686 mixed targets.')
