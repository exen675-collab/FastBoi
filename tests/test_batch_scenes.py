import json
import tempfile
import unittest
from pathlib import Path

import batch_scenes


class BatchScenesTests(unittest.TestCase):
    def test_normalize_and_dry_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            refs = root / 'refs'
            refs.mkdir()
            (refs / 'a.png').write_bytes(b'x')
            payload = {
                'version': 1,
                'defaults': {'size': next(iter(batch_scenes.SIZES)), 'seed': -1},
                'scenes': [
                    {
                        'id': 's01',
                        'prompt': '  rain on neon street  ',
                        'duration_seconds': 12,
                        'reference_image_count': 1,
                        'reference_images': [{'index': 1, 'path': 'refs/a.png'}],
                    },
                    {
                        'prompt': 'second scene',
                        'seed': 7,
                    },
                ],
            }
            path = root / 'scenes.json'
            path.write_text(json.dumps(payload), encoding='utf-8')

            normalized = batch_scenes.normalize_document(
                batch_scenes.load_json(path), path.parent
            )
            self.assertEqual(len(normalized['scenes']), 2)
            self.assertEqual(normalized['scenes'][0]['prompt'], 'rain on neon street')
            self.assertEqual(normalized['scenes'][0]['reference_image_count'], 1)
            self.assertTrue(normalized['scenes'][0]['reference_images'][0]['path_exists'])
            self.assertFalse(normalized['scenes'][0]['engine_applies']['reference_images'])
            self.assertFalse(normalized['scenes'][0]['engine_applies']['duration_seconds'])
            self.assertEqual(normalized['scenes'][1]['id'], 'scene_002')
            self.assertEqual(normalized['scenes'][1]['seed'], 7)

            rc = batch_scenes.main([str(path), '--dry-run', '--run-id', 'testrun'])
            self.assertEqual(rc, 0)

    def test_rejects_bad_ref_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = {
                'version': 1,
                'scenes': [{
                    'id': 'bad',
                    'prompt': 'hello',
                    'size': next(iter(batch_scenes.SIZES)),
                    'reference_image_count': 2,
                    'reference_images': [{'index': 1, 'url': 'https://example.com/a.png'}],
                }],
            }
            path = root / 'bad.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaises(ValueError):
                batch_scenes.normalize_document(batch_scenes.load_json(path), path.parent)

    def test_rejects_wrong_version(self):
        with self.assertRaises(ValueError):
            batch_scenes.normalize_document({'version': 2, 'scenes': [{'prompt': 'x'}]}, Path('.'))


if __name__ == '__main__':
    unittest.main()
