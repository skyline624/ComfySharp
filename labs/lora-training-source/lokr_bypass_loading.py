"""Frozen LoKr h operator chains and native addition for bypass-only file geometries."""
import argparse,ast,base64,gzip,hashlib,json,logging,pathlib,subprocess
from typing import Optional,Callable
import torch
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);args=p.parse_args()
assert torch.__version__=='2.10.0+cpu';torch.set_num_threads(1);torch.set_num_interop_threads(1)
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={};payloads={};rows=[]
ns=dict(torch=torch,nn=torch.nn,F=F,Optional=Optional,Callable=Callable,logging=logging)
for path,names in [('comfy/weight_adapter/base.py',['WeightAdapterBase','WeightAdapterTrainBase']),('comfy/weight_adapter/lokr.py',['LoKrAdapter'])]:
    raw=subprocess.check_output(['git','-C',args.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
    nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
raw=subprocess.check_output(['git','-C',args.source,'show',commit+':comfy/weight_adapter/bypass.py']);hashes['comfy/weight_adapter/bypass.py']=hashlib.sha256(raw).hexdigest()
def values(shape,offset=0):
    count=1
    for n in shape:count*=n
    return ((torch.arange(count,dtype=torch.float32)+offset)%19-9).reshape(shape)/31
def record(t):
    raw=t.detach().contiguous().numpy().tobytes();sha=hashlib.sha256(raw).hexdigest();payloads[sha]=base64.b64encode(raw).decode('ascii')
    return dict(shape=list(t.shape),sha256=sha)
def case(name,target,factors,spatial=None,stride=1,padding=0):
    dims=len(target)-2;weight=values(target,9);x=values([1,3,target[1]] if dims==0 else [1,target[1]]+spatial,4)
    data={k:values(v,i+1) for i,(k,v) in enumerate(factors.items())};dora=values([7],8)
    op=F.linear if dims==0 else [F.conv1d,F.conv2d,F.conv3d][dims-1]
    base=op(x,weight) if dims==0 else op(x,weight,stride=stride,padding=padding)
    adapter=ns['LoKrAdapter'].load('layer',{'layer.'+k:v for k,v in data.items()},1.25,dora)
    adapter.is_conv=dims>0;adapter.conv_dim=dims;adapter.kw_dict=dict(stride=stride,padding=padding);adapter.multiplier=.75
    error=None;output=None
    try:
        with torch.no_grad():output=adapter.g(base+adapter.h(x,base))
    except Exception as failure:error=dict(type=type(failure).__name__,message=str(failure))
    rows.append(dict(name=name,target=target,input=record(x),base=record(base),factors={k:record(v) for k,v in data.items()},
        dora=record(dora),alpha=1.25,strength=.75,stride=stride,padding=padding,error=error,output=None if output is None else record(output)))
case('linear',[6,4],dict(lokr_w1=[2,2],lokr_w2=[3,2]))
case('linear-broadcast-delta',[6,4],dict(lokr_w1=[1,1],lokr_w2=[1,4]))
case('linear-broadcast-base',[1,4],dict(lokr_w1=[2,2],lokr_w2=[3,2]))
for dims in [1,2,3]:
    case('spatial-b-'+str(dims),[6,4]+[3]*dims,dict(lokr_w1=[2,2],lokr_w2_a=[3,1],lokr_w2_b=[1,2]+[3]*dims),[4]*dims)
    case('tucker-'+str(dims),[6,4]+[3]*dims,dict(lokr_w1=[2,2],lokr_w2_a=[3,4],lokr_w2_b=[1,2],lokr_t2=[4,1]+[3]*dims),[4]*dims)
case('matrix-tucker-core',[6,4,1],dict(lokr_w1=[2,2],lokr_w2_a=[3,4],lokr_w2_b=[1,2],lokr_t2=[4,1]),[4])
case('sequential-spatial',[6,4,3,3],dict(lokr_w1=[2,2],lokr_w2_a=[3,1,2,2],lokr_w2_b=[1,2,2,2]),[4,4])
case('conv-broadcast-channels',[6,4,3],dict(lokr_w1=[1,2],lokr_w2=[1,2,3]),[4])
case('conv-broadcast-delta-spatial',[6,4,1,1],dict(lokr_w1=[2,2],lokr_w2=[3,2,3,3]),[3,3])
case('conv-broadcast-base-spatial',[6,4,3,3],dict(lokr_w1=[2,2],lokr_w2=[3,2,1,1]),[3,3])
case('linear-ignores-core',[6,4],dict(lokr_w1=[2,2],lokr_w2_a=[3,1],lokr_w2_b=[1,2],lokr_t2=[7]))
case('error-linear-rank',[6,4],dict(lokr_w1=[2,2],lokr_w2_a=[3,2],lokr_w2_b=[1,2]))
case('error-conv-direct-matrix',[6,4,1],dict(lokr_w1=[2,2],lokr_w2=[3,2]),[4])
case('error-conv-decomposed-matrix',[6,4,1,1],dict(lokr_w1=[2,2],lokr_w2_a=[3,1],lokr_w2_b=[1,2]),[4,4])
case('error-tucker-channels',[6,4,3,3],dict(lokr_w1=[2,2],lokr_w2_a=[3,4],lokr_w2_b=[1,2],lokr_t2=[4,2,3,3]),[4,4])
case('error-input-channels',[6,5],dict(lokr_w1=[2,2],lokr_w2=[3,2]))
case('error-output-channels',[6,4],dict(lokr_w1=[1,2],lokr_w2=[2,2]))
case('error-output-spatial',[6,4,2,2],dict(lokr_w1=[2,2],lokr_w2=[3,2,3,3]),[4,4])
data=dict(backendCommit=commit,sourceHashes=hashes,absoluteTolerance=3e-5,relativeTolerance=3e-5,cases=rows,payloads=payloads)
raw=json.dumps(data,separators=(',',':')).encode();packed=gzip.compress(raw,mtime=0)
with pathlib.Path(args.output).open('xb') as output:output.write(packed)
print(json.dumps(dict(compressedSha256=hashlib.sha256(packed).hexdigest(),rawSha256=hashlib.sha256(raw).hexdigest(),compressedBytes=len(packed),rawBytes=len(raw),cases=[dict(name=c['name'],error=c['error'],shape=None if c['output'] is None else c['output']['shape']) for c in rows])))
