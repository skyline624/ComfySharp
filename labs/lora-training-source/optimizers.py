"""Source-only optimizer/loss/accumulation corpus, not a model-family qualification."""
import argparse,ast,hashlib,json,pathlib,subprocess,torch
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert torch.__version__=='2.10.0+cpu';torch.set_num_threads(1)
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a'
raw=subprocess.check_output(['git','-C',a.source,'show',commit+':comfy_extras/nodes_train.py'])
names=['_create_optimizer','_create_loss_function']
selected=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(selected)==2
ns={'torch':torch};exec(compile(ast.Module(body=selected,type_ignores=[]),'nodes_train.py','exec'),ns)
def record(t):return {'shape':list(t.shape),'values':t.detach().flatten().tolist()}
initial_up=torch.tensor([[.25,-.5],[.125,.75]])
initial_down=torch.tensor([[.5,-.25,.125],[-.75,.25,.5]])
base=torch.tensor([[.75,-.25,.5],[-.125,.5,.25]])
cases=[]
for opt in ['Adam','AdamW','SGD','RMSprop']:
 for loss_name in ['MSE','L1','Huber','SmoothL1']:
  up=torch.nn.Parameter(initial_up.clone());down=torch.nn.Parameter(initial_down.clone());unused=torch.nn.Parameter(torch.tensor([.875]))
  optimizer=ns['_create_optimizer'](opt,[up,down,unused],.0125);criterion=ns['_create_loss_function'](loss_name)
  batches=[]
  for micro in range(6):
   inputs=torch.tensor([[.25+micro*.125,-.5,1.0],[1.5,-.25+micro*.0625,.75]])
   target=torch.tensor([[2.5,-.75],[-1.5,.25+micro*.25]])
   train_up=micro not in [2,3]  # previously active parameter becomes unused for the middle update
   prediction=inputs@(base+((up if train_up else up.detach())@down)*1.25).T
   loss=criterion(prediction.float(),target.float());(loss/2).backward()
   if (micro+1)%2==0:optimizer.step();optimizer.zero_grad()
   batches.append({'input':record(inputs),'target':record(target),'trainUp':train_up,'loss':loss.item(),'updated':(micro+1)%2==0,'up':record(up),'down':record(down),'unused':unused.item()})
  cases.append({'optimizer':opt,'loss':loss_name,'batches':batches})
boundaries=[]
for name in ['MSE','L1','Huber','SmoothL1']:
 predicted=torch.tensor([-2.,-1.,-.25,0.,.25,1.,2.],requires_grad=True);target=torch.zeros_like(predicted)
 loss=ns['_create_loss_function'](name)(predicted,target);loss.backward()
 boundaries.append({'loss':name,'value':loss.item(),'gradient':record(predicted.grad)})
data={'sourceCommit':commit,'sourceSha256':hashlib.sha256(raw).hexdigest(),'torch':torch.__version__,'absoluteTolerance':2e-6,'relativeTolerance':2e-6,
 'scope':'Frozen source optimizer/loss factories, default options, three updates with two accumulated microbatches each. Analytic LoRA linear layer, unused leaf, and loss boundary gradients. No dataset or pretrained-model qualification.',
 'learningRate':.0125,'accumulationSteps':2,'initialUp':record(initial_up),'initialDown':record(initial_down),'base':record(base),'cases':cases,'boundaries':boundaries}
with open(a.output,'x',encoding='utf-8',newline='\n') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
print('Collected 16 optimizer/loss combinations, 96 microbatches, 48 optimizer updates, four boundary gradients.')
