"""Frozen mixed LoRA/LoHa/LoKr resume factory and RNG; separate CPU source lab."""
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
extract('comfy/weight_adapter/base.py',['WeightAdapterBase','WeightAdapterTrainBase','tucker_weight_from_conv','factorization'])
extract('comfy/weight_adapter/lora.py',['LoRAAdapter','LoraDiff'])
extract('comfy/weight_adapter/loha.py',['HadaWeight','HadaWeightTucker','LohaDiff','LoHaAdapter'])
extract('comfy/weight_adapter/lokr.py',['LokrDiff','LoKrAdapter'])
extract('comfy_extras/nodes_train.py',['BiasDiff','_create_weight_adapter','_create_bias_adapter','_setup_lora_adapters'])
ns['adapters']=[ns['LoRAAdapter'],ns['LoHaAdapter'],ns['LoKrAdapter']]
ns['adapter_maps']={k:ns[v] for k,v in [('LoRA','LoRAAdapter'),('LoHa','LoHaAdapter'),('LoKr','LoKrAdapter')]}
with tempfile.TemporaryDirectory(prefix='comfysharp-lokr-resume-') as directory:
    snapshot=pathlib.Path(directory)
    for relative in ['comfy/ldm/modules/diffusionmodules/util.py','comfy/ldm/modules/attention.py','comfy/ldm/modules/diffusionmodules/openaimodel.py']:
        path=snapshot/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(subprocess.check_output(['git','-C',a.source,'show',commit+':'+relative]))
    model_type,operations=source_model_type(snapshot.resolve(),evidence)
payloads={}
def record(value):
    raw=value.detach().float().cpu().contiguous().numpy().tobytes();digest=hashlib.sha256(raw).hexdigest()
    encoded=base64.b64encode(raw).decode('ascii')
    if digest in payloads:assert payloads[digest]==encoded
    else:payloads[digest]=encoded
    return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=digest)
prefixes=['time_embed.0','time_embed.2','input_blocks.0.0','input_blocks.1.1.transformer_blocks.0.attn1.to_q','input_blocks.3.0.op','out.2']
cases=[]
for linear,dtype,algorithm in [(False,torch.float32,'LoKr'),(True,torch.float16,'LoKr'),(False,torch.bfloat16,'LoHa'),(True,torch.float64,'LoRA')]:
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
        ol,im=2,2;ok,inn=rows//ol,weight.shape[1]//im;kernel=list(weight.shape[2:])
        factors={'lokr_w1':values([ol,im],i),'lokr_w2':values([ok,inn]+kernel,i+3)}
        if i==2: # Direct factors win, but inactive decomposed parameters remain registered.
            factors.update(lokr_w1_a=values([ol,3],5),lokr_w1_b=values([3,im],7),
                lokr_w2_a=values([1,ok],9),lokr_w2_b=values([1,inn],11),lokr_t2=values([1,1]+kernel,13))
        if i==3: # Unequal ranks on both rebuilt sides; both alpha factors apply in training.
            factors={'lokr_w1_a':values([ol,3],5),'lokr_w1_b':values([3,im],7),'lokr_w2_a':values([ok,1],9),'lokr_w2_b':values([1,inn],11)}
        if i==4: # Working single-channel Tucker side.
            factors={'lokr_w1':values([rows,weight.shape[1]],5),'lokr_w2_a':values([2,1],7),'lokr_w2_b':values([3,1],9),'lokr_t2':values([2,3]+kernel,11)}
        if i==5: # LoKrAdapter loads these, but LokrDiff does not register or use them.
            factors.update(lokr_w1_b=values([7,8],1),lokr_t2=values([1,1,2,2],3))
        existing.update({key+'.'+k:v for k,v in factors.items()})
        if i<2:
            existing.update({key+'.hada_w1_a':values([rows,1],1),key+'.hada_w1_b':values([1,columns],2),
                key+'.hada_w2_a':values([rows,2],3),key+'.hada_w2_b':values([2,columns],4)})
        if i==0:
            existing[key+'.lora_up.weight']=values([rows,1],17);existing[key+'.lora_down.weight']=values([1,columns],19)
        existing[key+'.alpha']=torch.tensor(17.)
        if i%2==0:existing[key+'.weight.alpha']=torch.tensor(1.75+i)
        existing[key+'.weight.dora_scale']=torch.ones([rows,1])
    existing['diffusion_model.out.0.diff']=torch.full([32],3.)
    existing['diffusion_model.out.2.diff_b']=torch.full([4],4.)
    wrappers={};mp=types.SimpleNamespace(model=model,add_weight_wrapper=lambda name,adapter:wrappers.__setitem__(name,adapter))
    torch.manual_seed(317);parameters,owners=ns['_setup_lora_adapters'](mp,existing,algorithm,torch.float32,2)
    cases.append(dict(linearProjection=linear,dtype=str(dtype),algorithm=algorithm,seed=317,freshRank=2,
        existing={k:record(v) for k,v in existing.items()},parameterBytes=sum(p.numel()*p.element_size() for p in parameters.values()),parameterCount=len(parameters),
        randomStateSha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        targets={name.removeprefix('diffusion_model.'):{k:record(v) for k,v in adapter.named_parameters()} for name,adapter in wrappers.items()},
        resumed=[prefix+'.weight' for prefix in prefixes]))
data=dict(sourceCommit=commit,sourceHashes=hashes,sourceDeclarations=evidence,absoluteTolerance=3e-5,relativeTolerance=3e-5,cases=cases,
    payloadEncoding='base64-f32-little-endian',payloads=payloads,
    scope='Complete reduced SD factories with mixed resumed LoRA/LoHa/LoKr and fresh LoRA/LoHa/LoKr, source F32/F16/BF16/F64 factors, Tucker, inactive and orphan factors, alpha/reset/priority and RNG. No pretrained workflow or platform qualification.')
raw=(json.dumps(data,indent=2,allow_nan=False)+'\n').encode()
with open(a.output,'xb') as f:f.write(gzip.compress(raw,mtime=0))
print(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),compressedSha256=hashlib.sha256(pathlib.Path(a.output).read_bytes()).hexdigest(),bytes=len(raw),compressedBytes=pathlib.Path(a.output).stat().st_size,cases=[{k:v for k,v in c.items() if k not in ['targets','existing']} for c in cases])))
