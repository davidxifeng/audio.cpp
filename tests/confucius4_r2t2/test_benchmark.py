import copy
import unittest
from benchmark import compare, percentile


def report():
    return dict(schema_version=1, config=dict(iterations=1),
                provenance=dict(audio_sha256='audio', host='host'),
                runs=[dict(wall_ms=100., rtf=.1, finish_ms=2., text='hello', language='English',
                           finish_delta='', chunks=[dict(audio_end_ms=320., wall_ms=98., delta='hello')])])


class ComparisonTests(unittest.TestCase):
    def test_speedup(self):
        a = report()
        b = copy.deepcopy(a)
        b['runs'][0]['wall_ms'] = 50.
        self.assertEqual(compare(a, b)['speedup'], 2.)

    def test_percentile(self):
        self.assertEqual(percentile([10., 0.], .95), 9.5)
        self.assertEqual(percentile([4.], .95), 4.)

    def test_reject_mismatches(self):
        for field in ('config', 'audio', 'host', 'text', 'delta', 'timing', 'unstable'):
            with self.subTest(field=field):
                a, b = report(), report()
                if field == 'config': b['config']['chunk_ms'] = 160
                if field == 'audio': b['provenance']['audio_sha256'] = 'other'
                if field == 'host': b['provenance']['host'] = 'other'
                if field == 'text': b['runs'][0]['text'] = 'hi'
                if field == 'delta': b['runs'][0]['chunks'][0]['delta'] = 'hi'
                if field == 'timing': b['runs'][0]['wall_ms'] = float('nan')
                if field == 'unstable':
                    b['config']['iterations'] = 2
                    b['runs'].append(copy.deepcopy(b['runs'][0]))
                    b['runs'][1]['text'] = 'hi'
                with self.assertRaises(ValueError): compare(a, b)


if __name__ == '__main__':
    unittest.main()
