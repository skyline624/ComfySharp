"""Comparator controls using tiny fixtures, not evidence of a model family."""
import contextlib, hashlib, importlib.util, io, json, struct, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pretrained_compare',HERE/'compare.py')
compare=importlib.util.module_from_spec(spec); spec.loader.exec_module(compare)

class ComparatorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.source=self.root/'source'; self.source.mkdir()
        self.product=self.root/'product'; (self.product/'traces').mkdir(parents=True)
        protocol=json.loads((HERE/'protocol.json').read_text())
        self.reference={'status':'ok','profile':protocol['profile'],'modelSha256':protocol['modelSha256'],
            'case':protocol['case'],'laboratoryHashes':{'labs/sd15-pretrained-source/protocol.json':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()},'traces':[]}
        self.actual={'status':'ok','settings':dict(protocol['case'],modelSha256=protocol['modelSha256'],backend='cpu',dtype='Float32',scheduler='karras',sampler='euler'),'traces':[]}
        self.names=['positive-hidden','negative-hidden','noise','sigmas','initial','diffusion','raw-vae','image']
        for name in self.names:
            self.reference['traces'].append(self.write(self.source,name,1.0))
            self.actual['traces'].append(self.write(self.product/'traces',name,1.0))
    def tearDown(self): self.temp.cleanup()
    def write(self,root,name,value):
        raw=struct.pack('<f',value); filename=name+'.f32'; (root/filename).write_bytes(raw)
        return {'File':filename,'Shape':[1],'Dtype':'float32-le','Bytes':4,'Sha256':hashlib.sha256(raw).hexdigest()}
    def run_comparison(self):
        (self.source/'result.json').write_text(json.dumps(self.reference))
        (self.product/'result.json').write_text(json.dumps(self.actual))
        args=['compare','--source',str(self.source),'--product',str(self.product),'--output',str(self.root/'report.json')]
        with patch.object(sys,'argv',args),contextlib.redirect_stdout(io.StringIO()): return compare.main()
    def test_identical_payloads_pass(self): self.assertEqual(0,self.run_comparison())
    def test_windows_manifest_separators_are_supported(self):
        self.reference['laboratoryHashes']={name.replace('/','\\'):value for name,value in self.reference['laboratoryHashes'].items()}
        self.assertEqual(0,self.run_comparison())
    def test_activation_within_tolerance_passes(self):
        self.actual['traces'][0]=self.write(self.product/'traces','positive-hidden',1.00001)
        self.assertEqual(0,self.run_comparison())
    def test_exact_input_rejects_even_small_difference(self):
        self.actual['traces'][2]=self.write(self.product/'traces','noise',1.00001)
        self.assertEqual(1,self.run_comparison())
    def test_latent_error_fails(self):
        self.actual['traces'][5]=self.write(self.product/'traces','diffusion',1.01)
        self.assertEqual(1,self.run_comparison())
    def test_hash_mismatch_is_rejected(self):
        (self.product/'traces/image.f32').write_bytes(struct.pack('<f',0))
        with self.assertRaises(AssertionError): self.run_comparison()
    def test_duplicate_capture_is_rejected(self):
        self.actual['traces'].append(self.actual['traces'][0])
        with self.assertRaises(AssertionError): self.run_comparison()

if __name__=='__main__': unittest.main()
