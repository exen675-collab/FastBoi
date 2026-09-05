import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import backend


class BackendTests(unittest.TestCase):
    def test_validation(self):
        size = next(iter(backend.SIZES))
        for prompt, seed in [('', 1), ('ok', -2), ('ok', 1.5), ('ok', 2**32)]:
            with self.assertRaises(ValueError):
                backend.validate_request(prompt, size, seed)
        self.assertEqual(backend.validate_request(' hello ', size, 42)[::2], ('hello', 42))
        self.assertGreaterEqual(backend.validate_request('hello', size, -1)[2], 0)

    def test_reuse_and_outputs(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(backend, 'ROOT', Path(temp)):
            Path(temp, 'config.json').write_text('{}')
            engine = backend.Backend()
            engine.args = types.SimpleNamespace()
            def generate(request):
                request.write_bytes(b'fake-test-only')
                return types.SimpleNamespace(video_path=str(request))
            engine.generator = types.SimpleNamespace(generate=generate)
            engine.recipe = types.SimpleNamespace(build_request=lambda args, path, seed: path,
                                                  _actual_output_path=lambda result, path: Path(result.video_path))
            first = engine.generate('hello', next(iter(backend.SIZES)), 42)
            second = engine.generate('hello', next(iter(backend.SIZES)), 42)
            self.assertNotEqual(first[0], second[0])
            self.assertTrue(Path(first[0]).exists())
            self.assertEqual(json.loads(Path(first[1]).read_text())['seed'], 42)

    def test_failure_resets_engine(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(backend, 'ROOT', Path(temp)):
            engine = backend.Backend()
            engine.args = types.SimpleNamespace()
            closed = []
            def fail(request):
                raise RuntimeError('CUDA OOM')
            engine.generator = types.SimpleNamespace(generate=fail, shutdown=lambda: closed.append(True))
            engine.recipe = types.SimpleNamespace(build_request=lambda *args: None)
            with self.assertRaises(RuntimeError):
                engine.generate('hello', next(iter(backend.SIZES)), 42)
            self.assertIsNone(engine.generator)
            self.assertEqual(closed, [True])


if __name__ == '__main__':
    unittest.main()
