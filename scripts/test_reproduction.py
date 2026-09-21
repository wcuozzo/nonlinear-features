"""Small integration checks for notebook execution and saved-weight plotting.

The notebook check runs its actual cells with two training steps and one seed per
configuration. It checks execution, not convergence or scientific results.
"""
from pathlib import Path
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT / 'scripts'))
import core
import export_shallow_figures as figures


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_verbose_training(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            result = core.run_experiment(n=5, m=2, l=1, n_steps=2,
                batch_size=8, tied_weights=True, device='cpu', verbose=True)
        self.assertIn('Results: hidden_state_linearity=', output.getvalue())
        self.assertTrue(np.isfinite(result['eval_mse']))

    def test_pentagon_notebook_cells(self):
        path = ROOT / 'results/core/tied_untied/toy_models_sanity_check.ipynb'
        notebook = json.loads(path.read_text())
        scope = {}
        calls = []

        def short_training(**kwargs):
            calls.append(kwargs['S'])
            kwargs.update(n_steps=2, n_seeds=1, batch_size=8,
                          device='cpu', verbose=False)
            return core.run_experiment_multi_seed(**kwargs)

        previous = Path.cwd()
        try:
            os.chdir(ROOT)
            with contextlib.redirect_stdout(io.StringIO()):
                for index, cell in enumerate(notebook['cells']):
                    if cell['cell_type'] != 'code':
                        continue
                    exec(compile(''.join(cell['source']), f'{path}:cell{index}', 'exec'), scope)
                    # Replace only the expensive training budget after imports.
                    scope['run_experiment_multi_seed'] = short_training
            self.assertEqual(calls, [.95, .5, .75, .95])
            self.assertEqual(len(scope['toy_geometry']['norms']), 5)
            self.assertEqual(len(scope['geometry']['angles']), 10)
            self.assertTrue(np.isfinite(scope['toy_geometry']['min_angle']))
        finally:
            plt.close('all')
            os.chdir(previous)

    def test_saved_weight_export(self):
        writes, reads = figures.load_vectors()
        saved = json.loads((ROOT / 'figures/untied_read_write_example.json').read_text())
        # Check orientation against the recorded reconstruction operator: a
        # write/read transpose error would reverse the asymmetric interference.
        np.testing.assert_allclose(reads.T @ writes,
            saved['models']['trained_untied']['operator'], rtol=5e-6, atol=5e-6)
        self.assertLess(writes[:, 0] @ reads[:, 2], -100)
        self.assertLess(abs(writes[:, 2] @ reads[:, 0]), .01)
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, str(ROOT / 'scripts/export_shallow_figures.py'),
                            '--output-dir', directory], cwd=directory,
                           check=True, capture_output=True, text=True)
            expected = {'fig_untied_read_write.png', 'fig_untied_motif_directions.png',
                        'fig_untied_motif_magnitudes.png', 'fig_untied_motif_asymmetry.png',
                        'fig_pentagon_n5.png'}
            self.assertEqual({p.name for p in Path(directory).iterdir()}, expected)
            for name in expected:
                with Image.open(Path(directory) / name) as image:
                    self.assertGreater(min(image.size), 400)
                    image.verify()


if __name__ == '__main__':
    unittest.main()
