"""Frozen SD1/SD2 LoKr factory, full parameter/RNG reference in a separate CPU lab."""
import argparse,ast,base64,gzip,hashlib,json,logging,pathlib,subprocess,sys,tempfile,types
from typing import Callable,Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert torch.__version__=='2.10.0+cpu' and torch.version.cuda is None
torch.set_num_threads(1);torch.set_num_interop_threads(1)
root=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'labs/sd-source'))
from unet import source_model_type,source_configuration
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={};evidence=[]
ns=dict(torch=torch,nn=nn,F=F,logging=logging,Callable=Callable,Optional=Optional)
def extract(path,names):
 raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
 nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
extract('comfy/weight_adapter/base.py',['WeightAdapterBase','WeightAdapterTrainBase','factorization'])
extract('comfy/weight_adapter/lokr.py',['LokrDiff','LoKrAdapter'])
extract('comfy_extras/nodes_train.py',['BiasDiff','_create_weight_adapter','_create_bias_adapter','_setup_lora_adapters'])
ns['adapters']=[ns['LoKrAdapter']];ns['adapter_maps']={'LoKr':ns['LoKrAdapter']}
with tempfile.TemporaryDirectory(prefix='comfysharp-lokr-source-') as directory:
 snapshot=pathlib.Path(directory)
 for relative in ['comfy/ldm/modules/diffusionmodules/util.py','comfy/ldm/modules/attention.py','comfy/ldm/modules/diffusionmodules/openaimodel.py']:
  dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(subprocess.check_output(['git','-C',a.source,'show',commit+':'+relative]))
 model_type,operations=source_model_type(snapshot.resolve(),evidence)
cases=[];payloads={}
def tensor_record(value):
 raw=value.detach().cpu().contiguous().numpy().tobytes();digest=hashlib.sha256(raw).hexdigest()
 encoded=base64.b64encode(raw).decode('ascii')
 if digest in payloads:assert payloads[digest]==encoded
 else:payloads[digest]=encoded
 return dict(shape=list(value.shape),dtype=str(value.dtype).removeprefix('torch.'),sha256=digest)
for linear in [False,True]:
 for rank in [2,7]:
  model=model_type(**source_configuration(linear),dtype=torch.float32,device='meta',operations=operations)
  model.to_empty(device='cpu');model.requires_grad_(False)
  for m in model.modules():
   if isinstance(m,(nn.Linear,nn.Conv2d,nn.GroupNorm,nn.LayerNorm)):m.weight_function=[]
  wrappers={};mp=types.SimpleNamespace(model=model,add_weight_wrapper=lambda name,adapter:wrappers.__setitem__(name,adapter))
  torch.manual_seed(317);parameters,owners=ns['_setup_lora_adapters'](mp,{},'LoKr',torch.float32,rank)
  cases.append(dict(linear=linear,rank=rank,seed=317,parameterCount=len(parameters),parameterBytes=sum(v.numel()*v.element_size() for v in parameters.values()),
   randomStateSha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
   targets={name:{k:tensor_record(v) for k,v in adapter.named_parameters()} for name,adapter in wrappers.items()}))
factorizations=[dict(dimension=d,factor=f,result=ns['factorization'](d,f)) for d in [1,7,127,128,250,360,512,1024] for f in [-1,0,1,2,4,7,8,16,128]]
data=dict(sourceCommit=commit,sourceHashes=hashes,sourceDeclarations=evidence,absoluteTolerance=3e-5,relativeTolerance=3e-5,cases=cases,factorizations=factorizations,payloadEncoding='base64-f32-little-endian',payloads=payloads,
 scope='Independent frozen LoKr/BiasDiff factories for all reduced SD1/SD2 targets at ranks 2 and 7; parameter/RNG evidence only, not pretrained workflow or platform qualification.')
raw=(json.dumps(data,indent=2,allow_nan=False)+'\n').encode()
with open(a.output,'xb') as f:f.write(gzip.compress(raw,mtime=0))
print(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),compressedSha256=hashlib.sha256(pathlib.Path(a.output).read_bytes()).hexdigest(),bytes=len(raw),cases=[{k:v for k,v in c.items() if k!='targets'} for c in cases])))
