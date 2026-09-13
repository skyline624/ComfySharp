"""Frozen LoRA/LoHa h and addition across linear and Conv1d/2d/3d module geometries."""
import argparse,ast,base64,gzip,hashlib,json,logging,pathlib,subprocess
from typing import Optional,Callable
import torch
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);args=p.parse_args()
assert torch.__version__=='2.10.0+cpu';torch.set_num_threads(1);torch.set_num_interop_threads(1)
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={};payloads={};rows=[]
ns=dict(torch=torch,nn=torch.nn,F=F,Optional=Optional,Callable=Callable,logging=logging,_warn_loha_bypass_inefficient=lambda:None)
for path,names in [('comfy/weight_adapter/base.py',['WeightAdapterBase']),('comfy/weight_adapter/lora.py',['LoRAAdapter']),('comfy/weight_adapter/loha.py',['LoHaAdapter'])]:
    raw=subprocess.check_output(['git','-C',args.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
    nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
def values(shape,offset=0):
    count=1
    for n in shape:count*=n
    return ((torch.arange(count,dtype=torch.float32)+offset)%19-9).reshape(shape)/31
def record(t):
    raw=t.detach().float().contiguous().numpy().tobytes();sha=hashlib.sha256(raw).hexdigest();payloads[sha]=base64.b64encode(raw).decode('ascii')
    return dict(shape=list(t.shape),sha256=sha)
def case(kind,name,target,factors,spatial=None):
    dims=len(target)-2;x=values([1,3,target[1]] if dims==0 else [1,target[1]]+spatial,4);weight=values(target,9)
    op=F.linear if dims==0 else [F.conv1d,F.conv2d,F.conv3d][dims-1];base=op(x,weight)
    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        data={k:values(shape,i+1).to(dtype) for i,(k,shape) in enumerate(factors.items())};dora=values([7],8)
        adapter=ns[kind+'Adapter'].load('layer',{'layer.'+k:v for k,v in data.items()},1.25,dora)
        adapter.is_conv=dims>0;adapter.conv_dim=dims;adapter.kernel_size=target[2:];adapter.in_channels=target[1];adapter.kw_dict=dict(stride=1,padding=0);adapter.multiplier=-.4
        output=None;error=None
        try:
            with torch.no_grad():output=adapter.g(base+adapter.h(x,base))
        except Exception as failure:error=dict(type=type(failure).__name__,message=str(failure))
        rows.append(dict(kind=kind,name=name,dtype=str(dtype),target=target,input=record(x),base=record(base),factors={k:record(v) for k,v in data.items()},
            dora=record(dora),alpha=1.25,strength=-.4,error=error,output=None if output is None else record(output)))
def lora(up,down,mid=None):
    result={'lora_up.weight':up,'lora_down.weight':down}
    if mid is not None:result['lora_mid.weight']=mid
    return result
def loha(out,columns,core=None):
    if core is None:return dict(hada_w1_a=[out,2],hada_w1_b=[2,columns],hada_w2_a=[out,2],hada_w2_b=[2,columns])
    return dict(hada_w1_a=[2,out],hada_w1_b=[2,columns],hada_w2_a=[2,out],hada_w2_b=[2,columns],hada_t1=[2,2]+core,hada_t2=[2,2]+core)
case('LoRA','linear',[3,2],lora([3,2],[2,2]))
case('LoRA','linear-mid',[3,2],lora([3,4],[2,2],[4,2]))
for dims in [1,2,3]:
    case('LoRA','flat-conv-'+str(dims),[3,2]+[3]*dims,lora([3,2],[2,2*3**dims]),[4]*dims)
    case('LoRA','mid-conv-'+str(dims),[3,2]+[3]*dims,lora([3,2],[2,2]+[1]*dims,[2,2]+[3]*dims),[4]*dims)
    case('LoRA','sequential-spatial-'+str(dims),[3,2]+[3]*dims,lora([3,2]+[2]*dims,[2,2]+[2]*dims),[4]*dims)
for kind in ['LoRA','LoHa']:
    factory=lambda out,columns:lora([out,2],[2,columns]) if kind=='LoRA' else loha(out,columns)
    case(kind,'broadcast-output',[3,2],factory(1,2))
    case(kind,'broadcast-base',[1,2],factory(3,2))
case('LoRA','broadcast-spatial',[3,2,1],lora([3,2],[2,2,3]),[3])
case('LoRA','broadcast-base-spatial',[3,2,3],lora([3,2],[2,2,1]),[3])
case('LoRA','error-rank',[3,2],lora([3,3],[2,2]))
case('LoRA','error-flatten',[3,2,3],lora([3,2],[2,5]),[4])
case('LoRA','error-kernel-dim',[3,2,3],lora([3,2],[2,2,3,3]),[4])
case('LoRA','error-spatial',[3,2,2],lora([3,2],[2,2,3]),[4])
case('LoHa','linear',[3,2],loha(3,2))
for dims in [1,2,3]:case('LoHa','flat-conv-'+str(dims),[3,2]+[3]*dims,loha(3,2*3**dims),[4]*dims)
case('LoHa','tucker',[3,2,3,3],loha(3,2,[3,3]),[4,4])
case('LoHa','broadcast-spatial',[3,2,1,1],loha(3,2,[3,3]),[3,3])
case('LoHa','broadcast-base-spatial',[3,2,3,3],loha(3,2,[1,1]),[3,3])
case('LoHa','error-flatten',[3,2,3],loha(3,5),[4])
case('LoHa','error-core-conv1',[3,2,3],loha(3,2,[3,3]),[4])
case('LoHa','error-core-conv3',[3,2,3,3,3],loha(3,2,[3,3]),[4,4,4])
case('LoHa','error-spatial',[3,2,2,2],loha(3,2,[3,3]),[4,4])
result=dict(backendCommit=commit,sourceHashes=hashes,absoluteTolerance=3e-5,relativeTolerance=3e-5,cases=rows,payloads=payloads)
raw=json.dumps(result,separators=(',',':')).encode();packed=gzip.compress(raw,mtime=0)
with pathlib.Path(args.output).open('xb') as output:output.write(packed)
print(json.dumps(dict(compressedSha256=hashlib.sha256(packed).hexdigest(),rawSha256=hashlib.sha256(raw).hexdigest(),compressedBytes=len(packed),rawBytes=len(raw),cases=len(rows),success=sum(r['error'] is None for r in rows),errors=[dict(kind=r['kind'],name=r['name'],error=r['error']) for r in rows if r['error'] and r['dtype']=='torch.float32'])))
