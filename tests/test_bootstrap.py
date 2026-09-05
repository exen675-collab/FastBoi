import unittest
from unittest.mock import patch

import bootstrap


class PreflightTests(unittest.TestCase):
    @patch('bootstrap.platform.system', return_value='Linux')
    @patch('bootstrap.platform.machine', return_value='x86_64')
    def test_small_gpu_stops_before_install(self, *_):
        with patch('bootstrap.subprocess.check_output', return_value='RTX 4060, 8188, 591.86\n'), \
                self.assertRaisesRegex(RuntimeError, 'Za mało VRAM'):
            bootstrap.hardware_check()
        with patch('bootstrap.subprocess.check_output', return_value='RTX 4060, 8188, 591.86\n'):
            # Opt-in bypass: warns instead of raising (still expected to OOM later).
            self.assertEqual(bootstrap.hardware_check(allow_low_vram=True), 'cu130')

    @patch('bootstrap.platform.system', return_value='Linux')
    @patch('bootstrap.platform.machine', return_value='x86_64')
    def test_cuda_selection(self, *_):
        for driver, expected in [('580.10', 'cu130'), ('570.10', 'cu126')]:
            with patch('bootstrap.subprocess.check_output', return_value=f'GPU, 80000, {driver}\n'):
                self.assertEqual(bootstrap.hardware_check(), expected)


if __name__ == '__main__':
    unittest.main()
