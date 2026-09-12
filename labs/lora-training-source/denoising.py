"""Execute frozen TrainSampler.fwd_bwd and full reduced U-Nets for LoRA objective references."""
import argparse,ast,hashlib,json,logging,math,pathlib,subprocess,sys,tempfile,types,warnings
from typing import Callable,Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert torch.__version__==('2.10.0' if sys.platform=='darwin' else '2.10.0+cpu') and torch.version.cuda is None
torch.set_num_threads(1);torch.set_num_interop_threads(1)
root=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'labs/sd-source'))
from unet import source_model_type,source_configuration
from common import synthetic_tensor,tensor_record,fill_parameters
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={};evidence=[]
ns={'torch':torch,'nn':nn,'F':F,'np':np,'math':math,'logging':logging,'Optional':Optional,'Callable':Callable,'comfy':types.SimpleNamespace(samplers=types.SimpleNamespace(Sampler=object))}
def extract(path,names):
 raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
 nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
extract('comfy/ldm/modules/diffusionmodules/util.py',['make_beta_schedule'])
extract('comfy/model_sampling.py',['reshape_sigma','EPS','V_PREDICTION','ModelSamplingDiscrete'])
extract('comfy/weight_adapter/base.py',['WeightAdapterTrainBase','tucker_weight_from_conv'])
extract('comfy/weight_adapter/lora.py',['LoraDiff'])
extract('comfy_extras/nodes_train.py',['make_batch_extra_option_dict','TrainSampler','_create_loss_function'])
with tempfile.TemporaryDirectory(prefix='comfysharp-denoising-source-') as directory:
 snapshot=pathlib.Path(directory)
 for path in ['comfy/ldm/modules/diffusionmodules/util.py','comfy/ldm/modules/attention.py','comfy/ldm/modules/diffusionmodules/openaimodel.py']:
  dest=snapshot/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]))
 model_type,operations=source_model_type(snapshot.resolve(),evidence)
cases=[]
for linear in [False,True]:
 for sigma_values in [[],[.65],[.1,3.0]]:
  model=model_type(**source_configuration(linear),dtype=torch.float32,device='meta',operations=operations)
  model.to_empty(device='cpu');model.requires_grad_(False);model.eval();fill_parameters(model);model.forward=model._forward
  sampling=type('Sampling',(ns['ModelSamplingDiscrete'],ns['V_PREDICTION' if linear else 'EPS']),{})()
  parameters=dict(model.named_parameters());adapters={};initial={}
  for name in ['input_blocks.0.0.weight','out.2.weight']:
   shape=parameters[name].shape;columns=parameters[name].numel()//shape[0]
   up=synthetic_tensor(name+'/up',[shape[0],2])*.03125;down=synthetic_tensor(name+'/down',[2,columns])*.03125
   adapters[name]=ns['LoraDiff']((up,down,2.5,None,None,None));initial[name]={'up':tensor_record(up),'down':tensor_record(down),'alpha':2.5}
  latent=synthetic_tensor('objective/latent',[2,4,8,8])*.18215;noise=synthetic_tensor('objective/noise',[2,4,8,8]);context=synthetic_tensor('objective/context',[2,3,16])
  sigma=torch.tensor(sigma_values if sigma_values else 0.,dtype=torch.float32)
  class Wrapper:
   def __init__(self):self.inner_model=types.SimpleNamespace(model_sampling=sampling);self.conds={};self.captures={}
   def __call__(self,xt,sigmas,**kwargs):
    self.xt=xt;self.sigmas=sigmas
    scaled=sampling.calculate_input(sigmas,xt);times=sampling.timestep(sigmas).float().reshape(-1)
    patched={name:adapter(parameters[name]) for name,adapter in adapters.items()}
    raw=torch.func.functional_call(model,patched,(scaled,),{'timesteps':times,'context':context,'transformer_options':{}},strict=False)
    denoised=sampling.calculate_denoised(sigmas,raw,xt)
    self.captures={k:tensor_record(v) for k,v in [('noisy',xt),('scaled',scaled),('times',times),('raw',raw),('denoised',denoised)]}
    return denoised
  wrapper=Wrapper();sampler=ns['TrainSampler'](ns['_create_loss_function']('MSE'),None,grad_acc=2,training_dtype=torch.float32)
  # CPU autocast disables itself for Float32: no mixed-precision emulation in this corpus.
  with warnings.catch_warnings():
   warnings.filterwarnings('ignore',message='In CPU autocast.*')
   loss=sampler.fwd_bwd(wrapper,sigma,noise,latent,[{},{}],[0,1],{},2,bwd=True)
  gradients={name:{'up':tensor_record(adapter.lora_up.weight.grad),'down':tensor_record(adapter.lora_down.weight.grad)} for name,adapter in adapters.items()}
  assert all(v.grad is None for v in model.parameters())
  cases.append({'linearProjection':linear,'predictionKind':'velocity' if linear else 'epsilon','initial':initial,'latent':tensor_record(latent),'noise':tensor_record(noise),'context':tensor_record(context),'sigma':tensor_record(sigma),'loss':loss.item(),'gradAccumulation':2,'captures':wrapper.captures,'gradients':gradients,'noisyGradient':tensor_record(wrapper.xt.grad),'sigmaGradient':tensor_record(wrapper.sigmas.grad)})
helpers={str(q.relative_to(root)).replace('\\','/'):hashlib.sha256(q.read_bytes()).hexdigest() for q in [root/'labs/sd-source/unet.py',root/'labs/sd-source/common.py']}
data={'sourceCommit':commit,'sourceHashes':hashes,'sourceDeclarations':evidence,'helperHashes':helpers,'torch':torch.__version__,'absoluteTolerance':3e-5,'relativeTolerance':3e-5,'scope':'Actual frozen TrainSampler.fwd_bwd, EPS/V prediction and reduced full U-Net with ordinary LoRA. Plain conditioning wrapper replaces external guider dispatch. Float32 CPU, fixed supplied diffusion latents/noise/sigmas; no dataset sampling, mixed precision, pretrained training or complete node qualification.','cases':cases}
with open(a.output,'x',encoding='utf-8',newline='\n') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
print('Collected six denoising loss/gradient cases: SD1 EPS, SD2 V, zero/broadcast/per-image sigma.')
