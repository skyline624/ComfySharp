"""Isolated frozen filename counter oracle; filesystem and tensor loading are stubbed."""
import argparse, ast, hashlib, json, subprocess, types
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
commit='1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a';path='comfy_extras/nodes_train.py'
raw=subprocess.check_output(['git','-C',a.source,'show',commit+':'+path])
function=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='_load_existing_lora')
ns=dict(folder_paths=types.SimpleNamespace(get_full_path_or_raise=lambda folder,name:name),
        comfy=types.SimpleNamespace(utils=types.SimpleNamespace(load_torch_file=lambda path:{})))
exec(compile(ast.Module(body=[function],type_ignores=[]),path,'exec'),ns)
cases=[]
for name in ['[None]','adapter_2_steps_00001_.safetensors','folder/name_+12_steps_x','x_-3_steps_y','x_ 42 _steps_y',
             'x_１２_steps_y','x_١٢_steps_y','x_92233720368547758080_steps_y','x_1_000_steps_y','x_steps_1_steps_y','model.safetensors','x__steps_y','x_2.0_steps_y']:
    try: cases.append(dict(name=name,steps=str(ns['_load_existing_lora'](name)[1])))
    except ValueError: cases.append(dict(name=name,error='ValueError'))
with open(a.output,'x',encoding='utf-8',newline='\n') as f:
    json.dump(dict(sourceCommit=commit,sourceSha256=hashlib.sha256(raw).hexdigest(),cases=cases),f,indent=2,ensure_ascii=False);f.write('\n')
print('Collected thirteen source filename-counter cases.')
