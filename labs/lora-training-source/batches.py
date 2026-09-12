"""Frozen dataset preparation and training batch/RNG sequencing, without model evaluation."""
import argparse,ast,hashlib,json,logging,math,pathlib,subprocess,types
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert torch.__version__=='2.10.0+cpu';torch.set_num_threads(1)
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={}
comfy=types.SimpleNamespace(samplers=types.SimpleNamespace(Sampler=object),sample=types.SimpleNamespace())
extras=types.SimpleNamespace(nodes_custom_sampler=types.SimpleNamespace())
ns={'torch':torch,'np':np,'math':math,'logging':logging,'comfy':comfy,'comfy_extras':extras}
def extract(path,names):
 raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
 selected=[n for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];assert len(selected)==len(names)
 exec(compile(ast.Module(body=selected,type_ignores=[]),path,'exec'),ns)
extract('comfy/sample.py',['prepare_noise_inner','prepare_noise']);comfy.sample.prepare_noise=ns['prepare_noise']
extract('comfy_extras/nodes_custom_sampler.py',['Noise_RandomNoise']);extras.nodes_custom_sampler.Noise_RandomNoise=ns['Noise_RandomNoise']
extract('comfy/ldm/modules/diffusionmodules/util.py',['make_beta_schedule'])
extract('comfy/model_sampling.py',['ModelSamplingDiscrete'])
extract('comfy/latent_formats.py',['LatentFormat','SD15'])
extract('comfy_extras/nodes_train.py',['TrainSampler','_process_latents_standard_mode','_process_latents_bucket_mode','_prepare_latents_and_count','_run_training_loop'])
def record(t):return {'shape':list(t.shape),'values':t.detach().flatten().tolist()}
def state_hash():return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
cases=[]
recipes=[('standard',[(4,4,2,3)],False,2,17,False),('concatenated',[(2,4,2,2),(1,4,2,2)],False,2,18446744073709546615,False),('oversized-batch',[(2,4,2,3)],False,9,23,False),('multi-resolution',[(2,4,2,3),(1,4,3,3)],False,2,41,False),('buckets',[(3,4,2,3),(1,4,3,3)],True,2,67,False),('zero-latents',[(2,4,2,2)],False,1,97,True)]
for name,shapes,bucket_mode,batch_size,seed,zero in recipes:
 inputs=[torch.zeros(shape) if zero else (torch.arange(math.prod(shape),dtype=torch.float32).reshape(shape)+i*100)/256 for i,shape in enumerate(shapes)]
 processed=ns['_process_latents_bucket_mode' if bucket_mode else '_process_latents_standard_mode']([{'samples':t} for t in inputs])
 processed,num_images,multi=ns['_prepare_latents_and_count'](processed,torch.float32,bucket_mode)
 sampler=ns['TrainSampler'](None,None,batch_size=batch_size,grad_acc=2,seed=seed,real_dataset=processed if multi else None,bucket_latents=processed if bucket_mode else None,training_dtype=torch.float32)
 model_wrap=types.SimpleNamespace(inner_model=types.SimpleNamespace(model_sampling=ns['ModelSamplingDiscrete']()))
 captured=[];current=[]
 def fwd_bwd(self,model_wrap,batch_sigmas,batch_noise,batch_latent,cond,indices,extra_args,dataset_size,bwd=True):
  current.append({'indices':indices,'latent':record(batch_latent),'noise':record(batch_noise),'sigma':record(batch_sigmas)})
  return torch.tensor(0.,requires_grad=True)
 sampler.fwd_bwd=types.MethodType(fwd_bwd,sampler)
 pbar=types.SimpleNamespace(set_postfix=lambda *_:None)
 class Guider:
  def sample(self,initial_noise,latent_image,train_sampler,sigmas,seed):
   initial_state=state_hash();initial_record=record(initial_noise)
   # Same boundary as CFGGuider.inner_sample; bucket/multires methods read their separate original dataset.
   if torch.count_nonzero(latent_image)>0:latent_image=ns['SD15']().process_in(latent_image)
   for i in range(4):
    current.clear();noisegen=ns['Noise_RandomNoise'](seed+i*1000)
    if bucket_mode:sampler._train_step_bucket_mode(model_wrap,[],{},noisegen,latent_image,pbar)
    elif multi:sampler._train_step_multires_mode(model_wrap,[],{},noisegen,latent_image,num_images,pbar)
    else:sampler._train_step_standard_mode(model_wrap,[],{},noisegen,latent_image,num_images,pbar)
    captured.append({'step':i,'noiseSeed':seed+i*1000,'groups':list(current),'lossWeightPerGroup':1/len(current),'stateSha256':state_hash()})
   self.initial_state=initial_state;self.initial_noise=initial_record
 guider=Guider();ns['_run_training_loop'](guider,sampler,processed,num_images,seed,bucket_mode,multi)
 cases.append({'name':name,'bucketMode':bucket_mode,'mode':'Buckets' if bucket_mode else 'MultiResolution' if multi else 'Standard','batchSize':batch_size,'seed':seed,'count':num_images,'inputs':[record(t) for t in inputs],'latentScale':.18215,'initialNoise':guider.initial_noise,'initialStateSha256':guider.initial_state,'batches':captured})
with open(a.output,'x',encoding='utf-8',newline='\n') as f:json.dump({'sourceCommit':commit,'sourceHashes':hashes,'torch':torch.__version__,'scope':'Actual frozen dataset helpers, run-loop initial noise and three training batch methods; model evaluation replaced by a recording zero loss. Plain SD15 guider latent conversion follows the original boundary. Float32 CPU only; no model/workflow or RNG parity with other native builds claimed.','cases':cases},f,indent=2,allow_nan=False);f.write('\n')
print('Collected six dataset modes/edge cases, four batches each, exact noise/state/selection records.')
