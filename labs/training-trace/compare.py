"""Describe the first source/.NET divergence; never change expected tensors or acceptance thresholds."""
import argparse,json,pathlib,struct,math
def f32(value):return struct.unpack('<f',struct.pack('<f',value))[0]
p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--actual',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--require-complete',action='store_true');p.add_argument('--require-tolerance',action='store_true');a=p.parse_args()
reports=[]
for case in range(2):
 expected=json.loads((a.source/f'case-{case}.json').read_text());actual=json.loads((a.actual/f'case-{case}.json').read_text())
 source=expected['records'];dotnet=actual['records'];weights=[]
 assert set(source['baseWeights'])==set(dotnet['baseWeights'])
 for name,s in source['baseWeights'].items():
  d=dotnet['baseWeights'][name]
  if s['shape']!=d['shape'] or s['sha256']!=d['sha256']:weights.append(name)
 rows=[]
 for key,s in source.items():
  if key=='baseWeights' or key not in dotnet:continue
  d=dotnet[key];assert s['shape']==d['shape'];assert len(s['values'])==len(d['values'])
  reference=list(map(f32,s['values']));observed=list(map(f32,d['values']))
  # System.Text.Json emits shortest round-trippable Single decimals; restore Single bits
  # before comparing, including the test's Float32 subtraction, rather than comparing JSON text precision.
  errors=[abs(f32(x-y)) for x,y in zip(reference,observed)]
  outside=[i for i,(x,y,e) in enumerate(zip(reference,observed,errors)) if not math.isfinite(x) or not math.isfinite(y) or e>3e-5+(0 if key.endswith('/loss') else 3e-5*abs(x))]
  unequal=[i for i,e in enumerate(errors) if e!=0]
  rows.append({'name':key,'exact':s['sha256']==d['sha256'],'maxAbs':max(errors,default=0),'outsideOriginalTolerance':len(outside),'firstDifferent':unequal[0] if unequal else None,'firstOutside':outside[0] if outside else None})
 reports.append({'case':case,'sourceTarget':expected['target'],'actualTarget':actual['target'],'sourceThreads':[expected['threads'],expected['interopThreads']],'actualThreads':[actual['threads'],actual['interopThreads']],'baseWeightDifferences':weights,'missingCaptures':sorted(set(source)-set(dotnet)),'completed':actual.get('completed',False),'comparisons':rows,'firstNonExact':next((r['name'] for r in rows if not r['exact']),None)})
with a.output.open('x',encoding='utf-8',newline='\n') as f:json.dump({'scope':'Diagnostic only. Original test verdict and 3e-5 absolute/relative assertions remain authoritative. Missing later traces mean execution stopped early.','cases':reports},f,indent=2);f.write('\n')
print(json.dumps([{'case':r['case'],'baseWeightDifferences':len(r['baseWeightDifferences']),'firstNonExact':r['firstNonExact'],'outside':[c['name'] for c in r['comparisons'] if c['outsideOriginalTolerance']]} for r in reports]))
if a.require_complete and any(r['missingCaptures'] or not r['completed'] for r in reports):raise SystemExit('Incomplete training evidence.')
if a.require_tolerance and any(r['baseWeightDifferences'] or any(c['outsideOriginalTolerance'] for c in r['comparisons']) for r in reports):raise SystemExit('Same-host source tolerance failed.')
