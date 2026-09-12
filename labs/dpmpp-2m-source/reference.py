"""Development-only collector of the frozen DPM++ 2M function."""
import argparse,ast,hashlib,json,subprocess
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
args=parser.parse_args();assert torch.__version__=='2.10.0+cpu'
torch.set_num_threads(1);torch.set_num_interop_threads(1)
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';path='comfy/k_diffusion/sampling.py'
raw=subprocess.check_output(['git','-C',args.source,'show',commit+':'+path])
definition=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='sample_dpmpp_2m')
namespace={'torch':torch,'trange':lambda n,**kw:range(n)}
exec(compile(ast.Module(body=[definition],type_ignores=[]),path,'exec'),namespace)
cases=[]
for schedule in ([1,0],[2,.75,.125,0],[.02,.005,.001,0],[1,1,0],[2,1,1,0],[1,1,.5,0]):
    initial=torch.tensor([.125,-.75,2.5,-3.125,0,.25,-.5,1.25],dtype=torch.float32).reshape(2,4,1,1)
    calls=[]
    def model(x,sigma):
        calls.append({'x':x.flatten().tolist() if bool(x.isfinite().all()) else None,'sigma':sigma.tolist()})
        return x*.25+sigma.reshape(2,1,1,1)*.125
    output=namespace['sample_dpmpp_2m'](model,initial,torch.tensor(schedule,dtype=torch.float32),disable=True)
    finite=bool(output.isfinite().all())
    cases.append({'sigmas':schedule,'initial':initial.flatten().tolist(),'calls':calls,'finite':finite,'output':output.flatten().tolist() if finite else None})
with open(args.output,'x',encoding='utf-8',newline='\n') as output:
    json.dump({'sourceCommit':commit,'sourceHashes':{path:hashlib.sha256(raw).hexdigest()},'torch':torch.__version__,
        'absoluteTolerance':1e-6,'relativeTolerance':1e-6,'scope':'DPM++ 2M ODE trajectory and call inputs with an analytical denoiser, not pretrained numerical qualification. Undefined trajectories must be diagnosed.','cases':cases},output,indent=2,allow_nan=False)
    output.write('\n')
