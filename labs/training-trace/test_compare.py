"""Failure gates for the separate laboratory comparator; no native dependency."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


class ComparisonGates(unittest.TestCase):
    def invoke(self, mutate):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, actual = root/'source', root/'actual'
            source.mkdir(); actual.mkdir()
            value = {'shape': [], 'values': [0.0], 'sha256': hashlib.sha256(struct.pack('<f', 0)).hexdigest()}
            reference = {'target': 'linux-x64', 'threads': 1, 'interopThreads': 1,
                'records': {'baseWeights': {}, 'step-0/loss': value, 'step-1/output': value}}
            observed = copy.deepcopy(reference)
            observed['completed'] = True
            mutate(observed)
            for case in range(2):
                (source/f'case-{case}.json').write_text(json.dumps(reference))
                (actual/f'case-{case}.json').write_text(json.dumps(observed))
            return subprocess.run([sys.executable, '-I', '-S', '-B', str(Path(__file__).with_name('compare.py')),
                '--source', str(source), '--actual', str(actual), '--output', str(root/'report.json'),
                '--require-complete', '--require-tolerance'], capture_output=True, text=True)

    def test_complete_equal_evidence_passes(self):
        result = self.invoke(lambda _: None)
        self.assertEqual(0, result.returncode, result.stderr)

    def test_missing_second_step_fails(self):
        result = self.invoke(lambda d: d['records'].pop('step-1/output'))
        self.assertNotEqual(0, result.returncode)
        self.assertIn('Incomplete training evidence', result.stderr)

    def test_incomplete_run_fails_even_with_matching_records(self):
        self.assertNotEqual(0, self.invoke(lambda d: d.update(completed=False)).returncode)

    def test_loss_outside_original_absolute_bound_fails(self):
        result = self.invoke(lambda d: d['records']['step-0/loss'].update(values=[0.000031]))
        self.assertNotEqual(0, result.returncode)
        self.assertIn('Same-host source tolerance failed', result.stderr)

    def test_nonfinite_observation_fails(self):
        result = self.invoke(lambda d: d['records']['step-1/output'].update(values=[float('nan')]))
        self.assertNotEqual(0, result.returncode)


if __name__ == '__main__':
    unittest.main()
