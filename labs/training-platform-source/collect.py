"""Execute only prospectively pinned upstream training collectors; never consult .NET outputs."""
import argparse,hashlib,json,os,pathlib,platform,subprocess,sys

ROOT=pathlib.Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'labs/training-platform-source/protocol.json'
COMPONENTS={'batches':('batches.py',6),'denoising':('denoising.py',6),'adapters':('adapters.py',2)}
def sha(data):return hashlib.sha256(data).hexdigest()
def git(source,*args):return subprocess.check_output(['git','-C',str(source),*args])
def target():
 key=(platform.system(),platform.machine().lower())
 mapping={('Windows','amd64'):'win-x64',('Linux','x86_64'):'linux-x64',('Darwin','arm64'):'osx-arm64'}
 if key not in mapping:raise RuntimeError('Unsupported source collection platform: '+str(key))
 return mapping[key]
def preflight(source):
 protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
 if platform.python_version()!=protocol['runtime']['python']:raise RuntimeError('Incorrect laboratory Python version.')
 for relative,expected in protocol['inputs'].items():
  if sha((ROOT/relative).read_bytes())!=expected:raise RuntimeError('Collector input hash differs: '+relative)
 for relative,expected in protocol['sources'].items():
  if sha(git(source,'show',protocol['backendCommit']+':'+relative))!=expected:raise RuntimeError('Frozen source hash differs: '+relative)
 return protocol
def collect(source,output):
 protocol=preflight(source);rid=target()
 # Native imports happen only after the code, source and dependency-lock preflight.
 import torch,numpy,einops
 expected='2.10.0' if rid=='osx-arm64' else '2.10.0+cpu'
 if torch.__version__!=expected or torch.version.cuda is not None or numpy.__version__!='2.2.6' or einops.__version__!='0.8.1':
  raise RuntimeError('Installed source environment differs from the CPU protocol.')
 output.mkdir(parents=True,exist_ok=False)
 records={}
 for component,(script,count) in COMPONENTS.items():
  destination=output/(component+'.json')
  command=[sys.executable,'-I','-B',str(ROOT/'labs/lora-training-source'/script),'--source',str(source),'--output',str(destination)]
  if component=='adapters':command+=['--order-output',str(output/'unet-training-order.json')]
  subprocess.run(command,check=True)
  data=json.loads(destination.read_text(encoding='utf-8'))
  if data['sourceCommit']!=protocol['backendCommit'] or len(data['cases'])!=count:raise RuntimeError('Collector identity/count differs: '+component)
  for name,digest in data['sourceHashes'].items():
   if protocol['sources'].get(name)!=digest:raise RuntimeError('Unpinned executed source: '+name)
  if component!='batches' and (data['absoluteTolerance']!=3e-5 or data['relativeTolerance']!=3e-5):raise RuntimeError('A numerical profile changed.')
  if component=='batches' and any(len(c['batches'])!=4 for c in data['cases']):raise RuntimeError('A batch recipe changed.')
  if component=='adapters' and any(len(c['initial'])!=686 or c['parameterCount']!=1250 or len(c['steps'])!=2 for c in data['cases']):raise RuntimeError('An all-target adapter case changed.')
  records[component]={'file':destination.name,'sha256':sha(destination.read_bytes()),'bytes':destination.stat().st_size,'cases':count}
 manifest={'schema':1,'profile':protocol['profile'],'target':rid,'backendCommit':protocol['backendCommit'],'collectorCommit':git(ROOT,'rev-parse','HEAD').decode().strip(),
  'protocolSha256':sha(PROTOCOL.read_bytes()),'inputs':protocol['inputs'],'sources':protocol['sources'],'artifacts':records,
  'orderSha256':sha((output/'unet-training-order.json').read_bytes()),
  'runtime':{'python':platform.python_version(),'system':platform.system(),'machine':platform.machine(),'torch':torch.__version__,'numpy':numpy.__version__,'einops':einops.__version__,'device':'cpu','dtype':'float32','collectorThreads':1,'collectorInteropThreads':1,'torchBuild':torch.__config__.show()},
  'runId':os.environ.get('GITHUB_RUN_ID'),'runAttempt':os.environ.get('GITHUB_RUN_ATTEMPT'),
  'scope':'Frozen original source collection on the recorded CPU platform. No .NET outputs consulted or tracked fixtures modified; not application/model/platform acceptance by itself.'}
 with (output/'manifest.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(manifest,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':'collected','target':rid,'components':list(records)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
 if a.validate_only:preflight(a.source);print('Pinned training source preflight passed.')
 elif a.output is None:p.error('--output is required for collection')
 else:collect(a.source,a.output)
