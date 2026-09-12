"""Laboratory-only comparison; validates manifests and payloads before reporting discrepancies.
Does not update .NET fixtures, tolerance profiles or a family qualification status.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

def payload(root, row):
    name=row['File']
    assert name==Path(name).name and name.endswith('.f32')
    assert row['Dtype']=='float32-le' and 0<row['Bytes']<=16*1024*1024
    raw=(root/name).read_bytes()
    assert len(raw)==row['Bytes'] and hashlib.sha256(raw).hexdigest()==row['Sha256']
    assert len(raw)==int(np.prod(row['Shape']))*4
    value=np.frombuffer(raw,dtype='<f4').astype(np.float64)
    assert np.isfinite(value).all()
    return value

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--product',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    protocol_path=Path(__file__).with_name('protocol.json')
    protocol=json.loads(protocol_path.read_text())
    source=json.loads((args.source/'result.json').read_text())
    product=json.loads((args.product/'result.json').read_text())
    assert source['status']==product['status']=='ok'
    assert source['profile']==protocol['profile']
    source_hashes={name.replace('\\','/'):value for name,value in source['laboratoryHashes'].items()}
    assert len(source_hashes)==len(source['laboratoryHashes'])
    assert source_hashes['labs/sd15-pretrained-source/protocol.json']==hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    assert source['modelSha256']==product['settings']['modelSha256']==protocol['modelSha256']
    assert source['case']==protocol['case']
    for key in ('prompt','negative','seed','width','height','steps','cfg','threads'):
        assert product['settings'][key]==protocol['case'][key]
    assert product['settings']['backend']=='cpu' and product['settings']['dtype']=='Float32'
    assert product['settings']['scheduler']=='karras' and product['settings']['sampler']=='euler'
    references={row['File']:row for row in source['traces']}
    actual={row['File']:row for row in product['traces']}
    assert len(source['traces'])==len(product['traces'])==len(references)==len(actual)==8 and references.keys()==actual.keys()
    rows=[]
    atol=protocol['comparison']['absoluteTolerance']; rtol=protocol['comparison']['relativeTolerance']
    for name,reference in references.items():
        assert reference['Shape']==actual[name]['Shape']
        expected=payload(args.source,reference)
        observed=payload(args.product/'traces',actual[name])
        absolute=np.abs(observed-expected)
        exact_required=name.removesuffix('.f32') in protocol['comparison']['inputsMustBeExact']
        threshold=atol+rtol*np.abs(expected)
        failed=int(np.count_nonzero(absolute>threshold))
        exact=reference['Sha256']==actual[name]['Sha256']
        rows.append({'name':name,'shape':reference['Shape'],'elements':len(expected),'bitExact':exact,
            'exactRequired':exact_required,'outsideTolerance':failed,'maxAbsoluteError':float(absolute.max()),
            'meanAbsoluteError':float(absolute.mean()),'rmse':float(np.sqrt(np.mean(absolute*absolute))),
            'passed':exact if exact_required else failed==0,'referenceSha256':reference['Sha256'],'productSha256':actual[name]['Sha256']})
    report={'profile':protocol['profile'],'protocolSha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'sourceReportSha256':hashlib.sha256((args.source/'result.json').read_bytes()).hexdigest(),
        'productReportSha256':hashlib.sha256((args.product/'result.json').read_bytes()).hexdigest(),
        'modelSha256':protocol['modelSha256'],'absoluteTolerance':atol,'relativeTolerance':rtol,
        'passed':all(row['passed'] for row in rows),'familyQualified':False,'comparisons':rows}
    with args.output.open('x',encoding='utf-8') as file: json.dump(report,file,indent=2)
    print(json.dumps(report,indent=2))
    return 0 if report['passed'] else 1

if __name__=='__main__': sys.exit(main())
