"""Complete frozen LoHa SD factory oracle, separate CPU laboratory."""
import argparse,ast,gzip,hashlib,json,logging,pathlib,subprocess,sys,tempfile,types
from typing import Callable,Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert torch.__version__=='2.10.0+cpu' and torch.version.cuda is None
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
extract('comfy/weight_adapter/base.py',['WeightAdapterBase','WeightAdapterTrainBase'])
extract('comfy/weight_adapter/loha.py',['HadaWeight','HadaWeightTucker','LohaDiff','LoHaAdapter'])
extract('comfy_extras/nodes_train.py',['BiasDiff','_create_weight_adapter','_create_bias_adapter','_setup_lora_adapters'])
ns['adapters']=[ns['LoHaAdapter']];ns['adapter_maps']={'LoHa':ns['LoHaAdapter']}
with tempfile.TemporaryDirectory(prefix='comfysharp-loha-source-') as directory:
    snapshot=pathlib.Path(directory)
    for relative in ['comfy/ldm/modules/diffusionmodules/util.py','comfy/ldm/modules/attention.py','comfy/ldm/modules/diffusionmodules/openaimodel.py']:
        dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(subprocess.check_output(['git','-C',a.source,'show',commit+':'+relative]))
    model_type,operations=source_model_type(snapshot.resolve(),evidence)
# SD1/SD2 projections have the same flattened dimensions; both C# configurations are checked.
model=model_type(**source_configuration(False),dtype=torch.float32,device='meta',operations=operations)
model.to_empty(device='cpu');model.requires_grad_(False)
for m in model.modules():
    if isinstance(m,(nn.Linear,nn.Conv2d,nn.GroupNorm,nn.LayerNorm)):m.weight_function=[]
wrappers={};mp=types.SimpleNamespace(model=model,add_weight_wrapper=lambda name,adapter:wrappers.__setitem__(name,adapter))
torch.manual_seed(317);parameters,owners=ns['_setup_lora_adapters'](mp,{},'LoHa',torch.float32,2)
data=dict(sourceCommit=commit,sourceHashes=hashes,sourceDeclarations=evidence,seed=317,rank=2,absoluteTolerance=3e-5,relativeTolerance=3e-5,
          parameterCount=len(parameters),parameterBytes=sum(v.numel()*v.element_size() for v in parameters.values()),
          randomStateSha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
          targets={name:{k:tensor_record(v) for k,v in adapter.named_parameters()} for name,adapter in wrappers.items()},
          scope='Independent frozen LoHa/BiasDiff factory for all reduced SD targets. Float32 parameter values and CPU RNG state; not pretrained gradients, resume, mixed precision or platform qualification.')
raw=(json.dumps(data,indent=2,allow_nan=False)+'\n').encode()
with open(a.output,'xb') as f:f.write(gzip.compress(raw,mtime=0))
print(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),compressedSha256=hashlib.sha256(pathlib.Path(a.output).read_bytes()).hexdigest(),bytes=len(raw),targets=len(wrappers),parameterCount=len(parameters),parameterBytes=data['parameterBytes'],rng=data['randomStateSha256'])))
