"""Run frozen alias functions on checkpoint header metadata; never read tensor payloads."""
import argparse,ast,hashlib,json,pathlib,struct,subprocess,types
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';hashes={};ns={}
def extract(path,names):
    raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path]);hashes[path]=hashlib.sha256(raw).hexdigest()
    selected=[]
    for n in ast.parse(raw).body:
        if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets):selected.append(n)
    assert len(selected)==len(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),path,'exec'),ns)
extract('comfy/utils.py',['UNET_MAP_ATTENTIONS','TRANSFORMER_BLOCKS','UNET_MAP_RESNET','UNET_MAP_BASIC','unet_to_diffusers'])
class UnrelatedModelTypes:
    def __getattr__(self,name):return type(name,(),{})
ns['comfy']=types.SimpleNamespace(utils=types.SimpleNamespace(unet_to_diffusers=ns['unet_to_diffusers']),model_base=UnrelatedModelTypes())
extract('comfy/lora.py',['LORA_CLIP_MAP','model_lora_keys_unet','model_lora_keys_clip'])
with open(a.checkpoint,'rb') as f:
    length=struct.unpack('<Q',f.read(8))[0];assert length<16*1024*1024
    header=json.loads(f.read(length))
config={'num_res_blocks':[2]*4,'channel_mult':[1,2,4,4],'transformer_depth':[1]*6+[0]*2,'transformer_depth_output':[1]*9+[0]*3,'transformer_depth_middle':1}
class Model:
    def __init__(self,shapes):self.shapes=shapes;self.model_config=types.SimpleNamespace(unet_config=config)
    def state_dict(self):return self.shapes
def capture(name,shapes,prefix,fn):
    mapping=fn(Model(shapes),{});groups={}
    for alias,target in mapping.items():
        if target in shapes and len(shapes[target])>=2:
            groups.setdefault(target[len(prefix):],[]).append(alias)
    return {'name':name,'shapes':{k[len(prefix):]:v for k,v in shapes.items() if len(v)>=2},'aliasesByTarget':dict(sorted(groups.items()))}
unet={'diffusion_model.'+k[len('model.diffusion_model.'):]:v['shape'] for k,v in header.items() if k.startswith('model.diffusion_model.') and k.endswith('.weight')}
clip={'clip_l.transformer.'+k[len('cond_stage_model.transformer.'):]:v['shape'] for k,v in header.items() if k.startswith('cond_stage_model.transformer.') and k.endswith('.weight')}
assert len(unet)>300 and len(clip)>90
cases=[capture('sd15-checkpoint',unet,'diffusion_model.',ns['model_lora_keys_unet']),capture('clip-l-checkpoint',clip,'clip_l.transformer.',ns['model_lora_keys_clip'])]
# Explicit metadata diagnostic: 33 layers exercises source's 32-layer legacy-alias limit.
for wrapper in ['clip_l','clip_g']:
    shapes={wrapper+'.transformer.text_model.embeddings.token_embedding.weight':[49408,4],wrapper+'.transformer.text_model.embeddings.position_embedding.weight':[77,4]}
    for i in range(33):
        for part in ns['LORA_CLIP_MAP']:shapes[f'{wrapper}.transformer.text_model.encoder.layers.{i}.{part}.weight']=[8,4] if part=='mlp.fc1' else [4,8] if part=='mlp.fc2' else [4,4]
    shapes[wrapper+'.transformer.text_projection.weight']=[4,4]
    cases.append(capture(wrapper+'-33-layer-metadata',shapes,wrapper+'.transformer.',ns['model_lora_keys_clip']))
output={'sourceCommit':commit,'sourceHashes':hashes,'scope':'Exact source alias functions with metadata-only model facade. Plain SD config; standalone CLIP. Existing rank>=2 weights only; source mappings to absent/rank-one weights are excluded. Per-target priority preserved; global unrelated-target iteration order is not a contract. Checkpoint payload not read.','cases':cases}
with open(a.output,'x',encoding='utf-8',newline='\n') as f:json.dump(output,f,ensure_ascii=False,separators=(',',':'));f.write('\n')
print(json.dumps({c['name']:{'targets':len(c['aliasesByTarget']),'aliases':sum(map(len,c['aliasesByTarget'].values()))} for c in cases}))
