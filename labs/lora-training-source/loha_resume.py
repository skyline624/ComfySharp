"""Frozen mixed LoRA/LoHa resume factory, complete parameter values and RNG; separate CPU lab."""
import argparse,ast,gzip,hashlib,json,logging,pathlib,subprocess,sys,tempfile,types
from typing import Callable,Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert torch.__version__==('2.10.0' if sys.platform=='darwin' else '2.10.0+cpu') and torch.version.cuda is None
torch.set_num_threads(1);torch.set_num_interop_threads(1)
root=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'labs/sd-source'))
from unet import source_model_type,source_configuration
from common import tensor_record
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={};evidence=[]
ns=dict(torch=torch,nn=nn,F=F,logging=logging,Callable=Callable,Optional=Optional)
def extract(path,names):
    raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
    nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
extract('comfy/weight_adapter/base.py',['WeightAdapterBase','WeightAdapterTrainBase','tucker_weight_from_conv'])
extract('comfy/weight_adapter/lora.py',['LoRAAdapter','LoraDiff'])
extract('comfy/weight_adapter/loha.py',['HadaWeight','HadaWeightTucker','LohaDiff','LoHaAdapter'])
extract('comfy_extras/nodes_train.py',['BiasDiff','_create_weight_adapter','_create_bias_adapter','_setup_lora_adapters'])
ns['adapters']=[ns['LoRAAdapter'],ns['LoHaAdapter']];ns['adapter_maps']={'LoRA':ns['LoRAAdapter'],'LoHa':ns['LoHaAdapter']}
with tempfile.TemporaryDirectory(prefix='comfysharp-loha-resume-') as directory:
    snapshot=pathlib.Path(directory)
    for relative in ['comfy/ldm/modules/diffusionmodules/util.py','comfy/ldm/modules/attention.py','comfy/ldm/modules/diffusionmodules/openaimodel.py']:
        path=snapshot/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(subprocess.check_output(['git','-C',a.source,'show',commit+':'+relative]))
    model_type,operations=source_model_type(snapshot.resolve(),evidence)
prefixes=['time_embed.0','time_embed.2','input_blocks.0.0','input_blocks.1.1.transformer_blocks.0.attn1.to_q','input_blocks.3.0.op','out.2']
cases=[]
for linear,dtype,algorithm in [(False,torch.float32,'LoHa'),(False,torch.float16,'LoHa'),(True,torch.bfloat16,'LoHa'),(True,torch.float64,'LoRA')]:
    model=nn.Module();model.diffusion_model=model_type(**source_configuration(linear),dtype=torch.float32,device='meta',operations=operations)
    model.to_empty(device='cpu');model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m,(nn.Linear,nn.Conv2d,nn.GroupNorm,nn.LayerNorm)):m.weight_function=[]
    weights=dict(model.named_parameters());existing={}
    def values(shape,shift):
        count=1
        for n in shape:count*=n
        return (((torch.arange(count,dtype=torch.float32)+shift)%23-11)/71).reshape(shape).to(dtype)
    for i,prefix in enumerate(prefixes):
        key='diffusion_model.'+prefix;weight=weights[key+'.weight'];rows=weight.shape[0];columns=weight.numel()//rows
        tucker=i in [2,4,5];rank=1+i%2
        if tucker:
            existing[key+'.hada_w1_a']=values([rank,rows],i);existing[key+'.hada_w1_b']=values([2,weight.shape[1]],i+3)
            existing[key+'.hada_w2_a']=values([rank,rows],i+5);existing[key+'.hada_w2_b']=values([3,weight.shape[1]],i+7)
            existing[key+'.hada_t1']=values([rank,2,*weight.shape[2:]],i+9);existing[key+'.hada_t2']=values([rank,3,*weight.shape[2:]],i+11)
        else:
            existing[key+'.hada_w1_a']=values([rows,rank],i);existing[key+'.hada_w1_b']=values([rank,columns],i+3)
            existing[key+'.hada_w2_a']=values([rows,rank+1],i+5);existing[key+'.hada_w2_b']=values([rank+1,columns],i+7)
        existing[key+'.alpha']=torch.tensor(17.) # Export alpha is intentionally ignored by resume.
        if i%2==0:existing[key+'.weight.alpha']=torch.tensor(1.75+i)
        existing[key+'.weight.dora_scale']=torch.ones([rows,1]) # LohaDiff ignores inference DoRA.
        if i==0:
            existing[key+'.lora_up.weight']=values([rows,1],17);existing[key+'.lora_down.weight']=values([1,columns],19)
    existing['diffusion_model.out.0.diff']=torch.full([32],3.)
    existing['diffusion_model.out.2.diff_b']=torch.full([4],4.)
    wrappers={};mp=types.SimpleNamespace(model=model,add_weight_wrapper=lambda name,adapter:wrappers.__setitem__(name,adapter))
    torch.manual_seed(317);parameters,owners=ns['_setup_lora_adapters'](mp,existing,algorithm,torch.float32,2)
    cases.append(dict(linearProjection=linear,dtype=str(dtype),algorithm=algorithm,seed=317,freshRank=2,
        existing={k:tensor_record(v.float())|dict(dtype=str(v.dtype)) for k,v in existing.items()},
        parameterBytes=sum(p.numel()*p.element_size() for p in parameters.values()),parameterCount=len(parameters),
        randomStateSha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        targets={name.removeprefix('diffusion_model.'):{k:tensor_record(v) for k,v in adapter.named_parameters()} for name,adapter in wrappers.items()},
        resumed=[prefix+'.weight' for prefix in prefixes]))
data=dict(sourceCommit=commit,sourceHashes=hashes,sourceDeclarations=evidence,absoluteTolerance=3e-5,relativeTolerance=3e-5,cases=cases,
    scope='Complete reduced SD1/SD2 factory with mixed resumed LoRA/LoHa and fresh LoRA/LoHa, F32/F16/BF16/F64 source factors, Tucker, alpha/reset/provider-priority and RNG. Not pretrained gradient or platform qualification.')
raw=(json.dumps(data,indent=2,allow_nan=False)+'\n').encode()
with open(a.output,'xb') as f:f.write(gzip.compress(raw,mtime=0))
print(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),compressedSha256=hashlib.sha256(pathlib.Path(a.output).read_bytes()).hexdigest(),bytes=len(raw),cases=len(cases))))
