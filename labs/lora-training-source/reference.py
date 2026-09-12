"""Frozen full-topology U-Net LoRA gradients. Separate source-only CPU laboratory."""
import argparse,ast,hashlib,json,logging,pathlib,subprocess,sys,tempfile,types
from typing import Callable,Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert torch.__version__=='2.10.0+cpu';torch.set_num_threads(1);torch.set_num_interop_threads(1)
repo=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo/'labs/sd-source'))
from unet import source_model_type,source_configuration
from common import fill_parameters,synthetic_tensor,tensor_record
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={};evidence=[]
ns=dict(torch=torch,nn=nn,F=F,logging=logging,Optional=Optional,Callable=Callable)
def extract(path,names):
    raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
    selected=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];assert len(selected)==len(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),path,'exec'),ns)
extract('comfy/weight_adapter/base.py',['WeightAdapterTrainBase','tucker_weight_from_conv'])
extract('comfy/weight_adapter/lora.py',['LoraDiff'])
# Read exact Git blobs rather than a Windows checkout whose line endings may differ.
with tempfile.TemporaryDirectory(prefix='comfysharp-training-source-') as directory:
    snapshot=pathlib.Path(directory)
    for relative in ['comfy/ldm/modules/diffusionmodules/util.py','comfy/ldm/modules/attention.py','comfy/ldm/modules/diffusionmodules/openaimodel.py']:
        destination=snapshot/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(subprocess.check_output(['git','-C',a.source,'show',commit+':'+relative]))
    model_type,operations=source_model_type(snapshot.resolve(),evidence)
cases=[]
for linear in [False,True]:
    model=model_type(**source_configuration(linear),dtype=torch.float32,device='meta',operations=operations)
    model.to_empty(device='cpu');model.requires_grad_(False);model.eval();fill_parameters(model)
    # Bypass only external wrapper dispatch, exactly as the previous source lab does.
    model.forward=model._forward
    parameters=dict(model.named_parameters());adapters={};initial={}
    for name in ['input_blocks.0.0.weight','out.2.weight']:
        shape=parameters[name].shape;columns=parameters[name].numel()//shape[0]
        up=synthetic_tensor(name+'/up',[shape[0],2])*0.03125
        down=synthetic_tensor(name+'/down',[2,columns])*0.03125
        adapters[name]=ns['LoraDiff']((up,down,2.5,None,None,None))
        initial[name]={'up':tensor_record(up),'down':tensor_record(down),'alpha':2.5}
    latent=synthetic_tensor('training/latent',[1,4,8,8]);context=synthetic_tensor('training/context',[1,3,16]);times=torch.tensor([17.25])
    target=synthetic_tensor('training/target',[1,4,8,8])
    optimizer=torch.optim.SGD([v for adapter in adapters.values() for v in [adapter.lora_up.weight,adapter.lora_down.weight]],lr=.01)
    steps=[]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        patched={name:adapter(parameters[name]) for name,adapter in adapters.items()}
        output=torch.func.functional_call(model,patched,(latent,),{'timesteps':times,'context':context,'transformer_options':{}},strict=False)
        loss=F.mse_loss(output,target);loss.backward()
        gradients={name:{'up':tensor_record(adapter.lora_up.weight.grad),'down':tensor_record(adapter.lora_down.weight.grad)} for name,adapter in adapters.items()}
        assert all(v.grad is None for v in model.parameters())
        optimizer.step()
        steps.append({'output':tensor_record(output),'loss':loss.item(),'gradients':gradients,'updated':{name:{'up':tensor_record(adapter.lora_up.weight),'down':tensor_record(adapter.lora_down.weight)} for name,adapter in adapters.items()}})
    cases.append({'linearProjection':linear,'latent':tensor_record(latent),'context':tensor_record(context),'timesteps':tensor_record(times),'target':tensor_record(target),'initial':initial,'learningRate':.01,'steps':steps})
helpers={str(p.relative_to(repo)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in [repo/'labs/sd-source/unet.py',repo/'labs/sd-source/common.py']}
with open(a.output,'x',encoding='utf-8',newline='\n') as f:
    json.dump({'sourceCommit':commit,'sourceHashes':hashes,'sourceDeclarations':evidence,'helperHashes':helpers,'torch':torch.__version__,'absoluteTolerance':3e-5,'relativeTolerance':3e-5,
      'scope':'Actual frozen U-Net _forward and LoraDiff, basic attention with native PyTorch operations, full plain topology at reduced widths. Differentiable functional weight replacement; first and final convolution targets, frozen base parameters. Two SGD steps; no checkpointing, bypass, offload, pretrained model or complete TrainLoraNode qualification.', 'cases':cases},f,indent=2,allow_nan=False);f.write('\n')
print('Collected two full-topology source cases, two SGD steps each; base weights frozen.')
