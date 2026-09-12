"""Observe the unchanged adapter source collector. Outputs are diagnostic, never new acceptance fixtures."""
import argparse,hashlib,importlib.util,json,pathlib,runpy,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('training_preflight',ROOT/'labs/training-platform-source/collect.py');preflight=importlib.util.module_from_spec(spec);spec.loader.exec_module(preflight)
preflight.preflight(a.source)
protocol=json.loads((ROOT/'labs/training-trace/protocol.json').read_text())
for name,h in protocol['inputs'].items():
 if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('Diagnostic input changed: '+name)
a.output.mkdir(parents=True,exist_ok=False)
collector=ROOT/'labs/lora-training-source/adapters.py'
subprocess.run([sys.executable,'-I','-B',str(collector),'--source',str(a.source),'--output',str(a.output/'baseline.json'),'--order-output',str(a.output/'baseline-order.json')],check=True)
import torch
assert torch.__version__=='2.10.0+cpu' and torch.version.cuda is None
original=torch.func.functional_call;calls=0;records={}
boundaries={'time_embed':'timeEmbedding','input_blocks.2.1':'down0','input_blocks.5.1':'down1','input_blocks.8.1':'down2','input_blocks.11.0':'down3','middle_block.2':'middle','output_blocks.2.1':'up0','output_blocks.5.2':'up1','output_blocks.8.2':'up2'}
def snap(t):
 t=t.detach().cpu().contiguous()
 return {'shape':list(t.shape),'stride':list(t.stride()),'sha256':hashlib.sha256(t.numpy().tobytes()).hexdigest(),'values':t.flatten().tolist()}
def observed(*args,**kwargs):
 global calls
 case,step=divmod(calls,2);calls+=1;model=args[0];row=records.setdefault(case,{})
 if step==0:
  row['baseWeights']={name:{'shape':list(t.shape),'stride':list(t.stride()),'sha256':hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()} for name,t in model.named_parameters()}
  row['initial/latent']=snap(args[2][0]);row['initial/times']=snap(args[3]['timesteps']);row['initial/context']=snap(args[3]['context'])
 handles=[]
 for name,module in model.named_modules():
  if name in boundaries:
   def hook(module,inputs,output,label=boundaries[name]):row['step-'+str(step)+'/'+label]=snap(output)
   handles.append(module.register_forward_hook(hook))
 def input_hook(name,label,index=0):
  def hook(module,inputs):row[f'step-{step}/{label}']=snap(inputs[index])
  handles.append(model.get_submodule(name).register_forward_pre_hook(hook))
 def output_hook(name,label):
  def hook(module,inputs,output):row[f'step-{step}/{label}']=snap(output)
  handles.append(model.get_submodule(name).register_forward_hook(hook))
 input_hook('input_blocks.9.0.op','input_blocks.9.0.op.input');output_hook('input_blocks.9.0.op','input_blocks.9.0.op.output')
 for index in [10,11]:
  prefix=f'input_blocks.{index}.0';input_hook(prefix,prefix+'.input');input_hook(prefix,prefix+'.embedding',1)
  for child in ['in_layers.0','in_layers.1','in_layers.2','emb_layers.0','emb_layers.1','out_layers.0','out_layers.1','out_layers.3']:
   input_hook(prefix+'.'+child,prefix+'.'+child+'.input');output_hook(prefix+'.'+child,prefix+'.'+child+'.output')
  output_hook(prefix,prefix+'.output')
 try:
  result=original(*args,**kwargs);row['step-'+str(step)+'/output']=snap(result);return result
 finally:
  for handle in handles:handle.remove()
torch.func.functional_call=observed
sys.argv=[str(collector),'--source',str(a.source),'--output',str(a.output/'observed.json'),'--order-output',str(a.output/'observed-order.json')]
try:runpy.run_path(str(collector),run_name='__main__')
finally:torch.func.functional_call=original
assert calls==4
assert (a.output/'baseline.json').read_bytes()==(a.output/'observed.json').read_bytes(),'Observation changed source results.'
assert (a.output/'baseline-order.json').read_bytes()==(a.output/'observed-order.json').read_bytes()
data=json.loads((a.output/'observed.json').read_text())
for case,row in records.items():
 for step,entry in enumerate(data['cases'][case]['steps']):
  for target,parameters in entry['gradients'].items():
   for name,value in parameters.items():row[f'step-{step}/gradient/{target}/{name}']=value
 with (a.output/f'case-{case}.json').open('x',encoding='utf-8',newline='\n') as f:
  json.dump({'target':preflight.target(),'threads':torch.get_num_threads(),'interopThreads':torch.get_num_interop_threads(),'cpuCapability':torch.backends.cpu.get_cpu_capability(),'records':row},f);f.write('\n')
with (a.output/'manifest.json').open('x',encoding='utf-8',newline='\n') as f:
 json.dump({'sourceUnchangedByObservation':True,'torchBuild':torch.__config__.show(),'protocolSha256':hashlib.sha256((ROOT/'labs/training-trace/protocol.json').read_bytes()).hexdigest(),'sourceHashes':data['sourceHashes'],'baselineSha256':hashlib.sha256((a.output/'baseline.json').read_bytes()).hexdigest()},f,indent=2);f.write('\n')
