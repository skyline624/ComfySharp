"""Frozen SD adapter resume factory oracle; isolated CPU laboratory only."""
import argparse, ast, hashlib, json, logging, pathlib, subprocess, sys, tempfile, types
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--output',required=True); a=p.parse_args()
assert torch.__version__=='2.10.0+cpu'
torch.set_num_threads(1); torch.set_num_interop_threads(1)
root=pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0,str(root/'labs/sd-source'))
from unet import source_model_type, source_configuration
from common import tensor_record
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a'; hashes={}; evidence=[]
ns=dict(torch=torch,nn=nn,F=F,logging=logging,Callable=Callable,Optional=Optional)
def extract(path,names):
    raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]); hashes[path]=hashlib.sha256(raw).hexdigest()
    selected=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(selected)==len(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),path,'exec'),ns)
extract('comfy/weight_adapter/base.py',['WeightAdapterBase','WeightAdapterTrainBase','tucker_weight_from_conv'])
extract('comfy/weight_adapter/lora.py',['LoRAAdapter','LoraDiff'])
extract('comfy_extras/nodes_train.py',['BiasDiff','_create_weight_adapter','_create_bias_adapter','_setup_lora_adapters'])
ns['adapters']=[ns['LoRAAdapter']]; ns['adapter_maps']={'LoRA':ns['LoRAAdapter']}
with tempfile.TemporaryDirectory(prefix='comfysharp-resume-source-') as directory:
    snapshot=pathlib.Path(directory)
    for relative in ['comfy/ldm/modules/diffusionmodules/util.py','comfy/ldm/modules/attention.py','comfy/ldm/modules/diffusionmodules/openaimodel.py']:
        path=snapshot/relative; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(subprocess.check_output(['git','-C',a.source,'show',commit+':'+relative]))
    model_type,operations=source_model_type(snapshot.resolve(),evidence)
suffixes=[('.lora_up.weight','.lora_down.weight'),('_lora.up.weight','_lora.down.weight'),('.lora_B.weight','.lora_A.weight'),
          ('.lora.up.weight','.lora.down.weight'),('.lora_B','.lora_A'),('.lora_linear_layer.up.weight','.lora_linear_layer.down.weight'),('.lora_B.default.weight','.lora_A.default.weight')]
prefixes=['time_embed.0','time_embed.2','input_blocks.0.0','input_blocks.1.1.transformer_blocks.0.attn1.to_q',
          'input_blocks.1.1.proj_in','input_blocks.3.0.op','out.2']
def digest(value): return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()
cases=[]
for linear,dtype in [(False,torch.float32),(False,torch.float16),(False,torch.bfloat16),(True,torch.float32)]:
    model=nn.Module(); model.diffusion_model=model_type(**source_configuration(linear),dtype=torch.float32,device='meta',operations=operations)
    model.to_empty(device='cpu'); model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module,(nn.Linear,nn.Conv2d,nn.GroupNorm,nn.LayerNorm)): module.weight_function=[]
    weights=dict(model.named_parameters()); existing={}
    for i,(prefix,(u,d)) in enumerate(zip(prefixes,suffixes)):
        key='diffusion_model.'+prefix; weight=weights[key+'.weight']; rank=1+i%3
        def values(shape,shift):
            return ((torch.arange(shape[0]*shape[1],dtype=torch.float32).reshape(shape)+shift)%23-11).div(71).to(dtype)
        existing[key+u]=values([weight.shape[0],rank],i)
        existing[key+d]=values([rank,weight.numel()//weight.shape[0]],i+5)
        existing[key+'.alpha']=torch.tensor(9.) # Not read by the frozen training factory.
        if i%2==0: existing[key+'.weight.alpha']=torch.tensor(1.75+i)
        # LoraDiff ignores the inference DoRA/reshape metadata in this two-factor path.
        if i==0:
            existing[key+'.weight.dora_scale']=torch.ones([weight.shape[0],1])
            existing[key+'.reshape_weight']=torch.tensor(list(weight.shape))
            existing[key+'.lora_B.weight']=values([weight.shape[0],rank],17)
            existing[key+'.lora_A.weight']=values([rank,weight.numel()//weight.shape[0]],19)
    existing['diffusion_model.out.0.diff']=torch.full([32],3.)
    existing['diffusion_model.out.2.diff_b']=torch.full([4],4.)
    wrappers={}; mp=types.SimpleNamespace(model=model,add_weight_wrapper=lambda name,adapter:wrappers.__setitem__(name,adapter))
    torch.manual_seed(317)
    parameters,owners=ns['_setup_lora_adapters'](mp,existing,'LoRA',torch.float32,2)
    all_hash=hashlib.sha256(); count=0; size=0
    for name,adapter in wrappers.items():
        for index,value in enumerate(adapter.parameters()):
            all_hash.update((name.removeprefix('diffusion_model.')+'/'+str(index)+'\0').encode())
            all_hash.update(value.detach().contiguous().numpy().tobytes()); count+=1; size+=value.numel()*value.element_size()
    cases.append(dict(linearProjection=linear,dtype=str(dtype),seed=317,freshRank=2,
                      existing={k:tensor_record(v.float()) | dict(dtype=str(v.dtype)) for k,v in existing.items()},
                      allParameterSha256=all_hash.hexdigest(),randomStateSha256=digest(torch.get_rng_state()),
                      parameterCount=count,parameterBytes=size,targets=len(wrappers),
                      resumed={prefix+'.weight':{k:tensor_record(v) for k,v in wrappers['diffusion_model.'+prefix+'.weight'].named_parameters()} for prefix in prefixes}))
with open(a.output,'x',encoding='utf-8',newline='\n') as destination:
    json.dump(dict(sourceCommit=commit,sourceHashes=hashes,sourceDeclarations=evidence,cases=cases,
                   scope='Exact frozen LoRA resume factory initialization on reduced SD1/SD2. Seven spellings, rank override, F32/F16/BF16 source factors, source alpha/reset rules and RNG. Other adapter loaders are outside this oracle.'),destination,indent=2,allow_nan=False)
    destination.write('\n')
print('Collected four resume factory cases with all-parameter and RNG hashes.')
