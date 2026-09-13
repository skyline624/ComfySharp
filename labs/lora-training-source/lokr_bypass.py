"""Frozen LoKr inference/training grouped bypass outputs, gradients and SGD updates."""
import argparse,ast,base64,gzip,hashlib,json,logging,pathlib,subprocess,types
from typing import Callable,Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert torch.__version__=='2.10.0+cpu' and torch.version.cuda is None
torch.set_num_threads(1);torch.set_num_interop_threads(1)
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={};payloads={}
ns=dict(torch=torch,nn=nn,F=F,logging=logging,Callable=Callable,Optional=Optional)
def extract(path,names):
    raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
    nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
extract('comfy/weight_adapter/base.py',['WeightAdapterBase','WeightAdapterTrainBase'])
extract('comfy/weight_adapter/lokr.py',['LoKrAdapter','LokrDiff'])
def values(shape,shift):
    count=1
    for n in shape:count*=n
    return (((torch.arange(count,dtype=torch.float32)+shift)%19-9)/31).reshape(shape)
def record(value):
    if value is None:return None
    raw=value.detach().float().cpu().contiguous().numpy().tobytes();digest=hashlib.sha256(raw).hexdigest();encoded=base64.b64encode(raw).decode('ascii')
    if digest in payloads:assert payloads[digest]==encoded
    else:payloads[digest]=encoded
    return dict(shape=list(value.shape),sha256=digest)
cases=[]
for dims in range(4):
    kinds=['direct','first','second','both','inactive'] if dims==0 else ['direct','first','spatial-second','spatial-both','matrix-second','matrix-direct']
    if dims==2:kinds+=['tucker','tucker-rect']
    for kind in kinds:
        out2=2 if kind=='tucker' else 3;inn=2;kernel=[1 if kind=='matrix-direct' else 2]*dims
        for strength,alpha,dtype in [(1.,None,torch.float32),(-.4,2.25,torch.float16),(.75,1.25,torch.bfloat16),(0.,1.75,torch.float32),(1.,0.,torch.float32)]:
            factors=dict(lokr_w1=values([2,2],0),lokr_w2=values([out2,inn]+([] if kind=='matrix-direct' else kernel),3))
            if kind in ['first','both','spatial-both','inactive']:
                factors.update(lokr_w1_a=values([2,3],5),lokr_w1_b=values([3,2],7))
                if kind!='inactive':del factors['lokr_w1']
            if kind in ['second','both','spatial-second','spatial-both','matrix-second','inactive']:
                factors.update(lokr_w2_a=values([out2,1],9),lokr_w2_b=values([1,inn]+(kernel if kind.startswith('spatial') else []),11))
                if kind!='inactive':del factors['lokr_w2']
            if kind.startswith('tucker'):
                del factors['lokr_w2'];factors.update(lokr_w2_a=values([2,out2],9),lokr_w2_b=values([3,inn],11),lokr_t2=values([2,3]+kernel,13))
            factors={k:v.to(dtype) for k,v in factors.items()}
            # Source inference h ignores DoRA; retain it to ensure the adapter snapshot does too.
            factors['dora_scale']=values([2*out2,1]+[1]*dims,5).abs()+.5
            input_shape=[2,3,4] if dims==0 else [1,4]+[4,5,3][:dims]
            x0=values(input_shape,1);weight=values([2*out2,4]+kernel,17)
            stride,padding=(1,0) if dims==0 else (2,1)
            op=F.linear if dims==0 else [F.conv1d,F.conv2d,F.conv3d][dims-1]
            kwargs={} if dims==0 else dict(stride=stride,padding=padding)
            def configure(adapter,multiplier):
                adapter.multiplier=multiplier;adapter.is_conv=dims>0;adapter.conv_dim=dims;adapter.kw_dict=kwargs
            inference={};x=x0.clone().requires_grad_();base=op(x,weight,**kwargs)
            adapter=ns['LoKrAdapter'].load('layer',{'layer.'+k:v for k,v in factors.items()},alpha,factors['dora_scale']);configure(adapter,strength)
            try:
                result=base+adapter.h(x,base);result.square().mean().backward()
                inference=dict(output=record(result),inputGradient=record(x.grad),error=None)
            except Exception as error:inference=dict(error=dict(type=type(error).__name__,message=str(error)))
            # Parameter wrappers may share F32 factor storage; capture inputs before SGD mutates it.
            recorded_factors={k:record(v) for k,v in factors.items()}
            training={};x=x0.clone().requires_grad_();base=op(x,weight,**kwargs)
            try:
                order=['lokr_w1','lokr_w2',None,'lokr_w1_a','lokr_w1_b','lokr_w2_a','lokr_w2_b','lokr_t2','dora_scale']
                v=[(1. if alpha is None else alpha) if k is None else factors[k].float() if k in factors else None for k in order]
                train=ns['LokrDiff'](tuple(v)).requires_grad_(True);configure(train,1.)
                optimizer=torch.optim.SGD(train.parameters(),lr=.003);result=base+train.h(x,base);loss=result.square().mean();loss.backward()
                training=dict(initial={k:record(v) for k,v in train.named_parameters()},output=record(result),inputGradient=record(x.grad),
                    gradients={k:record(v.grad) for k,v in train.named_parameters()},loss=loss.item(),error=None)
                optimizer.step();training['updated']={k:record(v) for k,v in train.named_parameters()}
            except Exception as error:training=dict(error=dict(type=type(error).__name__,message=str(error)))
            cases.append(dict(dims=dims,kind=kind,strength=strength,alpha=alpha,dtype=str(dtype),kernel=kernel if dims else None,stride=stride,padding=padding,
                factors=recorded_factors,input=record(x0),weight=record(weight),inference=inference,training=training))
data=dict(sourceCommit=commit,sourceHashes=hashes,absoluteTolerance=3e-5,relativeTolerance=3e-5,payloadEncoding='base64-f32-little-endian',payloads=payloads,cases=cases,
    scope='Frozen LoKr h linear and Conv1d/2d/3d source operators, F32 activations, F32/F16/BF16 factors cast to F32, native input/parameter gradients and one SGD update. Explicit source constructor/operator errors. No full model workflow, mixed activation precision or platform qualification.')
raw=(json.dumps(data,indent=2,allow_nan=False)+'\n').encode()
with open(a.output,'xb') as f:f.write(gzip.compress(raw,mtime=0))
print(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),compressedSha256=hashlib.sha256(pathlib.Path(a.output).read_bytes()).hexdigest(),cases=len(cases),
    inferenceSuccess=sum(c['inference']['error'] is None for c in cases),trainingSuccess=sum(c['training']['error'] is None for c in cases),bytes=len(raw),compressedBytes=pathlib.Path(a.output).stat().st_size)))
