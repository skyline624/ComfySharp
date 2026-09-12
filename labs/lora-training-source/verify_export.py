"""Independent F32/F64 byte decoding and frozen load_lora acceptance of a .NET export.
This deliberately limited laboratory decoder does not claim to be the safetensors package.
"""
import argparse,ast,hashlib,json,logging,math,pathlib,struct,subprocess,types
from typing import Callable,Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--adapter',required=True);p.add_argument('--report',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert torch.__version__=='2.10.0+cpu'
report=json.loads(pathlib.Path(a.report).read_text(encoding='utf-8'));data=pathlib.Path(a.adapter).read_bytes()
assert report['status']=='ok' and hashlib.sha256(data).hexdigest()==report['adapterSha256']
length=struct.unpack('<Q',data[:8])[0];assert length%8==0 and length<=16*1024*1024
header=json.loads(data[8:8+length]);body=data[8+length:];tensors={};cursor=0
for name,info in header.items():
    assert info['dtype'] in ['F32','F64'];start,end=info['data_offsets'];width=4 if info['dtype']=='F32' else 8
    assert start==cursor and end-start==math.prod(info['shape'])*width and end<=len(body);cursor=end
    tensor=torch.frombuffer(bytearray(body[start:end]),dtype=torch.float32 if width==4 else torch.float64).clone().reshape(info['shape']);assert tensor.isfinite().all()
    tensors[name]=tensor
assert cursor==len(body)
for target,expected in report['updatedParameterHashes'].items():
    weight,factor=target.rsplit('/',1);key='diffusion_model.'+weight.removesuffix('.weight')+'.lora_'+factor+'.weight'
    assert hashlib.sha256(tensors[key].numpy().tobytes()).hexdigest()==expected,key
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={};ns=dict(torch=torch,nn=nn,F=F,logging=logging,Callable=Callable,Optional=Optional)
ns['comfy']=types.SimpleNamespace(model_management=types.SimpleNamespace(cast_to_device=lambda t,device,dtype:t.to(device=device,dtype=dtype)))
def extract(path,names):
    raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
    definitions=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];assert len(definitions)==len(names)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),path,'exec'),ns)
extract('comfy/weight_adapter/base.py',['WeightAdapterBase','weight_decompose','pad_tensor_to_shape'])
extract('comfy/weight_adapter/lora.py',['LoRAAdapter'])
ns['weight_adapter']=types.SimpleNamespace(adapters=[ns['LoRAAdapter']]);extract('comfy/lora.py',['load_lora'])
targets={'input_blocks.0.0.weight':[320,4,3,3],'out.2.weight':[4,320,3,3]}
mapping={'diffusion_model.'+name.removesuffix('.weight'):name for name in targets}
loaded=ns['load_lora'](tensors,mapping,log_missing=False);assert set(loaded)==set(targets)
summaries={}
for name,adapter in loaded.items():
    result=adapter.calculate_weight(torch.zeros(targets[name]),name,1,1,None,lambda x:x)
    assert list(result.shape)==targets[name] and result.isfinite().all() and result.abs().sum()>0
    summaries[name]={'shape':list(result.shape),'deltaSha256':hashlib.sha256(result.numpy().tobytes()).hexdigest()}
with open(a.output,'x',encoding='utf-8',newline='\n') as f:json.dump({'status':'ok','sourceCommit':commit,'sourceHashes':hashes,'adapterSha256':report['adapterSha256'],'factorHashesMatch':True,'sourceAcceptsBothTargets':True,'sourceApplications':summaries,'scope':'Independent restricted byte decoding plus actual frozen LoRA-only source loading/application on zero base weights. No safetensors-package, pretrained numerical or training-node qualification.'},f,indent=2);f.write('\n')
print('Export payload hashes match .NET parameters; frozen load_lora accepts both targets.')
