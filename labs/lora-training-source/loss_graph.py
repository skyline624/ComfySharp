"""Run frozen LossGraphNode with drawing/preview spies; collect source layout contracts."""
import argparse,ast,hashlib,json,pathlib,subprocess,types
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';path='comfy_extras/nodes_train.py'
raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path])
calls=[]
class Drawing:
    def __init__(self,image):pass
    def line(self,points,**kwargs):calls.append(dict(kind='line',points=points,**kwargs))
    def text(self,point,text,**kwargs):calls.append(dict(kind='text',point=point,text=text,fill=kwargs['fill']))
ns=dict(Image=types.SimpleNamespace(new=lambda mode,size,color:np.full((size[1],size[0],3),255,dtype=np.uint8)),
    ImageDraw=types.SimpleNamespace(Draw=Drawing),ImageFont=types.SimpleNamespace(truetype=lambda *args:None),np=np,torch=torch,
    io=types.SimpleNamespace(ComfyNode=object,NodeOutput=lambda **kw:kw),ui=types.SimpleNamespace(PreviewImage=lambda image,**kw:image))
node=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='LossGraphNode')
exec(compile(ast.Module(body=[node],type_ignores=[]),path,'exec'),ns)
cases=[]
for values in [[1.,.8,.2,.4],[-.5,-2.,-1.], [1.125,2.675,3.125], [0.,1.], [], [1.], [2.,2.], [1.001,2.675]]:
    calls.clear();entry=dict(losses=values)
    try:
        result=ns['LossGraphNode'].execute({'loss':values},'ignored-prefix')['ui']
        entry.update(shape=list(result.shape),calls=list(calls))
    except (ValueError,ZeroDivisionError) as error:entry['error']=type(error).__name__
    cases.append(entry)
data=dict(sourceCommit=commit,sourceFile=path,sourceSha256=hashlib.sha256(raw).hexdigest(),cases=cases,
    scope='Actual frozen LossGraphNode layout and text; drawing/font/PreviewImage are spies. Pixel rendering, persistence and metadata tested separately. No Pillow or cross-font pixel parity claimed.')
with open(a.output,'x',encoding='utf-8',newline='\n') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
print(hashlib.sha256(pathlib.Path(a.output).read_bytes()).hexdigest())
