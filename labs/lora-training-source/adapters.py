"""Frozen all-target ordinary LoRA initialization and mixed adapter gradients; separate CPU lab."""
import argparse,ast,hashlib,json,logging,pathlib,subprocess,sys,tempfile,types
from typing import Callable,Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--order-output',required=True);a=p.parse_args()
assert torch.__version__=='2.10.0+cpu';torch.set_num_threads(1);torch.set_num_interop_threads(1)
root=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'labs/sd-source'))
from unet import source_model_type,source_configuration
from common import fill_parameters,synthetic_tensor,tensor_record
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={};evidence=[]
ns=dict(torch=torch,nn=nn,F=F,logging=logging,Optional=Optional,Callable=Callable)
def extract(path,names):
 raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
 nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
extract('comfy/weight_adapter/base.py',['WeightAdapterBase','WeightAdapterTrainBase','tucker_weight_from_conv'])
extract('comfy/weight_adapter/lora.py',['LoRAAdapter','LoraDiff'])
extract('comfy_extras/nodes_train.py',['BiasDiff','_create_weight_adapter','_create_bias_adapter','_setup_lora_adapters'])
ns['adapters']=[ns['LoRAAdapter']];ns['adapter_maps']={'lora':ns['LoRAAdapter']}
with tempfile.TemporaryDirectory(prefix='comfysharp-adapter-source-') as directory:
 snapshot=pathlib.Path(directory)
 for relative in ['comfy/ldm/modules/diffusionmodules/util.py','comfy/ldm/modules/attention.py','comfy/ldm/modules/diffusionmodules/openaimodel.py']:
  dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(subprocess.check_output(['git','-C',a.source,'show',commit+':'+relative]))
 model_type,operations=source_model_type(snapshot.resolve(),evidence)
def digest(t):return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()
cases=[];order=None
for linear in [False,True]:
 model=model_type(**source_configuration(linear),dtype=torch.float32,device='meta',operations=operations)
 model.to_empty(device='cpu');model.requires_grad_(False);model.eval();fill_parameters(model);model.forward=model._forward
 # The lab's native Linear/Conv/Norm operation adapter marks the same plain-module weight-wrapper capability.
 for m in model.modules():
  if isinstance(m,(nn.Linear,nn.Conv2d,nn.GroupNorm,nn.LayerNorm)):m.weight_function=[]
 wrappers={}
 mp=types.SimpleNamespace(model=model,add_weight_wrapper=lambda name,adapter:wrappers.__setitem__(name,adapter))
 torch.manual_seed(317)
 parameters,owners=ns['_setup_lora_adapters'](mp,{},'lora',torch.float32,2)
 initial=[]
 for name,adapter in wrappers.items():
  params=dict(adapter.named_parameters());initial.append({'target':name,'shape':list(dict(model.named_parameters())[name].shape),'kind':'lora' if isinstance(adapter,ns['LoraDiff']) else 'difference','parameters':[{'name':key,'shape':list(t.shape),'sha256':digest(t),'requiresGrad':t.requires_grad} for key,t in params.items()]})
 assert len(initial)==686
 names=[r['target'] for r in initial]
 if order is None:order=names
 else:assert order==names
 state=digest(torch.get_rng_state());steps=[]
 latent=synthetic_tensor('adapters/latent',[1,4,8,8]);context=synthetic_tensor('adapters/context',[1,3,16]);times=torch.tensor([17.25]);target=synthetic_tensor('adapters/target',[1,4,8,8])
 optimizer=torch.optim.SGD(parameters.values(),lr=.01)
 observed=['input_blocks.0.0.weight','out.0.weight','out.0.bias','out.2.weight','out.2.bias']
 original=dict(model.named_parameters())
 for step in range(2):
  optimizer.zero_grad(set_to_none=True)
  patched={name:adapter(original[name]) for name,adapter in wrappers.items()}
  output=torch.func.functional_call(model,patched,(latent,),{'timesteps':times,'context':context,'transformer_options':{}},strict=False)
  loss=F.mse_loss(output,target);loss.backward()
  grads={name:{key:tensor_record(t.grad) for key,t in wrappers[name].named_parameters()} for name in observed}
  assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in parameters.values())
  assert all(t.grad is None for t in model.parameters())
  optimizer.step()
  steps.append({'output':tensor_record(output),'loss':loss.item(),'gradients':grads,'updated':{name:{key:tensor_record(t) for key,t in wrappers[name].named_parameters()} for name in observed}})
 cases.append({'linearProjection':linear,'seed':317,'rank':2,'initial':initial,'randomStateSha256':state,'parameterCount':len(parameters),'latent':tensor_record(latent),'context':tensor_record(context),'times':tensor_record(times),'target':tensor_record(target),'steps':steps})
data={'sourceCommit':commit,'sourceHashes':hashes,'sourceDeclarations':evidence,'torch':torch.__version__,'absoluteTolerance':3e-5,'relativeTolerance':3e-5,'scope':'Frozen all-target ordinary LoRA setup and two SGD updates through reduced full plain SD1/SD2 U-Nets. Native operation adapter supplies weight_function capability flags; wrapper dispatch is replaced by functional_call. Includes trainable alpha and discarded constructor RNG. CPU Float32 only; no complete node, existing adapters, alternative algorithms or pretrained training qualification.','cases':cases}
for path,value in [(a.output,data),(a.order_output,order)]:
 with open(path,'x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
print('Collected 686 ordered targets per topology, trainable alpha, full RNG state and two mixed-adapter SGD steps.')
