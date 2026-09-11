"""Strict stdlib validation and comparison of the fourteen reduced suite cases."""
import hashlib
import math
import struct
from allocation import validate as validate_allocation

BOUNDARIES=('timeEmbedding','down0','down1','down2','down3','middle','up0','up1','up2','output')
IDS=tuple(m+'/'+c for m in ('sd15-reduced','sd2-reduced') for c in
          ('square','odd-rectangle','batch-distinct-time','batch-shared-time-odd'))+tuple(
          'guidance-'+length+'/'+policy for length in ('2-3','3-3','2-10') for policy in ('separate','concatenateCompatible'))


def require(value,message):
    if not value:raise ValueError(message)


def values(record):
    require(record['dtype']=='float32','Expected Float32.')
    require(math.prod(record['shape'])==len(record['values']),'Wrong element count.')
    require(len(record['stride'])==len(record['shape']) and isinstance(record['aligned64'],bool),'Missing layout evidence.')
    payload=struct.pack('<'+'f'*len(record['values']),*record['values'])
    require(hashlib.sha256(payload).hexdigest()==record['sha256'],'Tensor payload SHA differs.')
    result=struct.unpack('<'+'f'*len(record['values']),payload)
    require(all(math.isfinite(x) for x in result),'Nonfinite value outside this finite U-Net/CFG corpus.')
    return result


def parameter_map(records):
    result={p['name']:(p['shape'],p['sha256']) for p in records}
    require(len(result)==len(records)==686,'Parameter names missing or duplicated.')
    return result


def verify(doc,source=False):
    validate_allocation(doc)
    require(doc['id'] in IDS,'Unexpected case.')
    kind='unet' if doc['id'].startswith(('sd15-','sd2-')) else 'guidance'
    require(doc['kind']==kind,'Case kind differs.')
    if not source:require(doc['synthetic'] and doc['inputsUnchanged'],'Invalid synthetic/input integrity evidence.')
    parameters=parameter_map(doc['parameters'])
    require(parameters==parameter_map(doc['parametersAfter']),'Parameters changed across forwards.')
    require(set(parameters)==set(doc['parameterLayouts']),'Parameter layouts incomplete.')
    require(doc['parameterLayouts']==doc['parameterLayoutsAfter'],'Parameter layouts changed.')
    require(all(v['storageOffset']==0 and v['addressModulo64']==0 and v['aligned64'] is True and v['dtype']=='float32' for v in doc['parameterLayouts'].values()),'Parameter storage evidence differs.')
    for name,(shape,_) in parameters.items():
        layout=doc['parameterLayouts'][name]
        require(layout['shape']==shape and len(layout['stride'])==len(shape) and isinstance(layout['aligned64'],bool),'Invalid parameter layout.')
    require(set(doc['inputs'])==({'latent','timesteps','context'} if kind=='unet' else {'latent','sigma','positive','negative'}),'Inputs incomplete.')
    require(set(doc['tensors'])==(set(BOUNDARIES) if kind=='unet' else {'output'}),'Capture labels incomplete.')
    for value in (*doc['inputs'].values(),*doc['tensors'].values()):values(value)
    if not source:
        require(doc['inputHashesAfter']=={n:v['sha256'] for n,v in doc['inputs'].items()},'Input hashes changed.')
    observer=doc['observer']
    sequence=['off','on','off'] if kind=='unet' else ['off','off','off']
    require(observer['sequence']==sequence,'Wrong observation sequence.')
    require(observer['bitIdentical'] and observer['outputHashes']==[doc['tensors']['output']['sha256']]*3,'Repeats/observer neutrality failed.')


def compare(actual,expected):
    require(actual['id']==expected['id'] and actual['kind']==expected['kind'],'Case mismatch.')
    require(actual['config']==expected['config'] and actual['options']==expected['options'],'Operation configuration/policy differs.')
    require(parameter_map(actual['parameters'])==parameter_map(expected['parameters']),'Parameter identities differ.')
    require({n:(v['shape'],v['sha256']) for n,v in actual['inputs'].items()}==
            {n:(v['shape'],v['sha256']) for n,v in expected['inputs'].items()},'Input identities differ.')
    def record(a,b):
        require(a['shape']==b['shape'],'Tensor shapes differ.')
        av,bv=values(a),values(b)
        errors=[abs(x-y) for x,y in zip(av,bv)]
        outside=[i for i,e in enumerate(errors) if e>3e-5+3e-5*abs(bv[i])]
        return {'byteIdentical':a['sha256']==b['sha256'],'maximumAbsoluteError':max(errors,default=0),
            'outsideBound':len(outside),'firstOutside':outside[0] if outside else None,
            'strideEqual':a['stride']==b['stride'],'actualStride':a['stride'],'expectedStride':b['stride'],
            'actualAligned64':a['aligned64'],'expectedAligned64':b['aligned64'],
            'actualAddressModulo64':a['addressModulo64'],'expectedAddressModulo64':b['addressModulo64'],
            'actualStorageOffset':a['storageOffset'],'expectedStorageOffset':b['storageOffset']}
    order=BOUNDARIES if actual['kind']=='unet' else ('output',)
    tensors={n:record(actual['tensors'][n],expected['tensors'][n]) for n in order}
    return {'tensors':tensors,'inputs':{n:record(actual['inputs'][n],expected['inputs'][n]) for n in actual['inputs']},
        'parameterLayoutsEqual':actual['parameterLayouts']==expected['parameterLayouts'],
        'firstDataDifference':next((n for n,r in tensors.items() if not r['byteIdentical']),None),
        'causality':'not_assessed; exact actual primitive inputs and parameters are required'}


def exact_control(comparison):
    return comparison['parameterLayoutsEqual'] and all(r['byteIdentical'] and r['strideEqual'] and
        r['actualAligned64']==r['expectedAligned64'] and r['actualAddressModulo64']==r['expectedAddressModulo64'] and r['actualStorageOffset']==r['expectedStorageOffset'] for category in ('inputs','tensors') for r in comparison[category].values())


def input_layout_control(comparison):
    """Outputs may differ; package attribution still requires equal actual inputs/layouts."""
    return comparison['parameterLayoutsEqual'] and all(r['byteIdentical'] and r['strideEqual'] and
        r['actualAligned64']==r['expectedAligned64'] and r['actualAddressModulo64']==r['expectedAddressModulo64'] and r['actualStorageOffset']==r['expectedStorageOffset'] for r in comparison['inputs'].values())


def check_corpus_inputs(doc,unet,guidance):
    """Only committed input/configuration records are used, never expected output values."""
    if doc['kind']=='unet':
        case=next(c for c in unet['cases'] if c['id']==doc['id'])
        model=unet['models'][case['model']]
        names=('latent','timesteps','context');config=model['config'];parameters=model['parameters']
        options={}
    else:
        identifier,policy=doc['id'].split('/')
        case=next(c for c in guidance['cases'] if c['id']==identifier)
        names=('latent','sigma','positive','negative');config=guidance['config'];parameters=guidance['parameters']
        options={'policy':policy,'scale':case['scale'],'predictionKind':guidance['predictionKind'],
                 'concatEligible':case['concatEligible'],'commonLength':case['commonLength']}
    require(doc['config']==config and doc['options']==options,'Frozen operation inputs differ.')
    require(parameter_map(doc['parameters'])==parameter_map(parameters),'Frozen parameter inputs differ.')
    require({n:(doc['inputs'][n]['shape'],doc['inputs'][n]['sha256']) for n in names}==
            {n:(case[n]['shape'],case[n]['sha256']) for n in names},'Frozen tensor inputs differ.')
