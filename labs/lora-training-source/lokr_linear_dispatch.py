"""Frozen LoKr linear bypass: leaf metadata selects native mm or bmm under no_grad."""
import argparse, ast, base64, gzip, hashlib, json, logging, pathlib, subprocess
from typing import Callable, Optional
import torch
import torch.nn.functional as F

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);args=p.parse_args()
assert torch.__version__=='2.10.0+cpu' and torch.version.cuda is None
torch.set_num_threads(1);torch.set_num_interop_threads(1)
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={};payloads={}
ns=dict(torch=torch,nn=torch.nn,Optional=Optional,Callable=Callable,F=F,logging=logging)
for path,names in [('comfy/weight_adapter/base.py',['WeightAdapterBase','WeightAdapterTrainBase','factorization']),('comfy/weight_adapter/lokr.py',['LokrDiff','LoKrAdapter'])]:
    raw=subprocess.check_output(['git','-C',args.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
    nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
def record(value):
    raw=value.detach().contiguous().numpy().tobytes();sha=hashlib.sha256(raw).hexdigest()
    payloads[sha]=base64.b64encode(raw).decode('ascii')
    return dict(shape=list(value.shape),sha256=sha)
rows=[]
for out_width,in_width,length in [(320,320,64),(320,768,3),(2560,320,3),(320,1280,3),(1280,1280,3)]:
    torch.manual_seed(317);owner=ns['LoKrAdapter'].create_train(torch.empty(out_width,in_width),rank=7).requires_grad_(True)
    with torch.no_grad():owner.lokr_w1.copy_(torch.randn_like(owner.lokr_w1)*.03)
    owner.is_conv=False;owner.multiplier=1.
    x=torch.randn(1,length,in_width);base=torch.zeros(1,length,out_width)
    with torch.no_grad():trained=owner.h(x,base)
    frozen=ns['LoKrAdapter'].load('layer',{'layer.'+k:v.detach().clone() for k,v in owner.named_parameters()},1.,None)
    frozen.is_conv=False;frozen.multiplier=1.
    with torch.no_grad():loaded=frozen.h(x,base)
    # Trace only the cross-group linear that consumes the transposed ND view.
    assert hasattr(owner,'lokr_w1') and hasattr(owner,'lokr_w2')
    with torch.no_grad():hidden=F.linear(x.reshape(*x.shape[:-1],owner.lokr_w1.shape[1],-1),owner.lokr_w2).transpose(-1,-2)
    profiles=[];cross_results=[]
    for enabled in [True,False]:
        owner.requires_grad_(enabled)
        with torch.no_grad(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            cross=F.linear(hidden,owner.lokr_w1)
        profiles.append(dict(requiresGrad=enabled,operators={e.key:e.count for e in prof.key_averages() if e.key in ['aten::mm','aten::bmm']}))
        cross_results.append(record(cross))
    with torch.no_grad():disabled=owner.h(x,base)
    assert torch.equal(disabled,loaded)
    assert profiles[0]['operators']=={'aten::mm':1} and profiles[1]['operators']=={'aten::bmm':1}, profiles
    rows.append(dict(input=record(x),first=record(owner.lokr_w1),second=record(owner.lokr_w2),
        training=record(trained),inference=record(loaded),frozenOwners=record(disabled),cross=cross_results,
        hiddenShape=list(hidden.shape),hiddenStride=list(hidden.stride()),profiles=profiles,
        maxAbsoluteDifference=(trained-loaded).abs().max().item()))
result=dict(backendCommit=commit,sourceHashes=hashes,torchVersion=torch.__version__,absoluteTolerance=3e-5,relativeTolerance=3e-5,cases=rows,payloads=payloads)
raw=json.dumps(result,separators=(',',':')).encode('utf-8');compressed=gzip.compress(raw,mtime=0)
with pathlib.Path(args.output).open('xb') as output:output.write(compressed)
print(json.dumps(dict(compressedSha256=hashlib.sha256(compressed).hexdigest(),rawSha256=hashlib.sha256(raw).hexdigest(),compressedBytes=len(compressed),rawBytes=len(raw),cases=[dict(shape=c['input']['shape'],profiles=c['profiles'],maxAbsoluteDifference=c['maxAbsoluteDifference']) for c in rows])))
