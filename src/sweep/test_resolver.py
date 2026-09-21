"""Resolver control-flow and actual warm-start pool regression checks."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

import precise
import resolve_monotonicity_violations as resolver
from check_results import check_warm_start_coverage
import test_launcher


class ResolverTests(unittest.TestCase):
    def frame(self, deep_loss):
        return pd.DataFrame([
            dict(n=16, m=2, l=1, S=.9, mse_full=.01, mse_full_ci95=0),
            dict(n=16, m=2, l=2, S=.9, mse_full=deep_loss, mse_full_ci95=0)])

    def test_final_round_verdict(self):
        for final_loss, expected in [(.005, 0), (.02, 2)]:
            with self.subTest(final_loss=final_loss), tempfile.TemporaryDirectory() as directory, \
                    patch.object(precise, 'compile_precise_csv',
                                 side_effect=[self.frame(.02), self.frame(final_loss)]) as evaluate, \
                    patch.object(resolver, 'run_one_round', return_value=0) as train, \
                    contextlib.redirect_stdout(io.StringIO()):
                status = resolver.main(['--store-dir', directory, '--max-rounds', '1'])
                self.assertEqual(status, expected)
                self.assertEqual(evaluate.call_count, 2)
                train.assert_called_once()

    def test_auto_resume_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as exit_result:
                resolver.main(['--store-dir', directory, '--resume', '--run-id', 'test', '--freeze-sources'])
            self.assertEqual(exit_result.exception.code, 2)
            self.assertIn('--resume requires explicit --configs', error.getvalue())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_explicit_resume_skips_completed_config(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(resolver, 'run_one_round', return_value=0) as train, \
                contextlib.redirect_stdout(io.StringIO()):
            seeds = Path(directory) / 'seeds'
            seeds.mkdir()
            (seeds / 'n16_m2_l2_S0.9.json').write_text(json.dumps(dict(seeds=[
                dict(run_id='test', warm_start_arm='diverse_pool')])))
            status = resolver.main(['--store-dir', directory, '--resume', '--run-id', 'test',
                                    '--configs', '16,2,2,0.9', '16,2,3,0.9'])
            self.assertEqual(status, 0)
            self.assertEqual(train.call_args.args[0], [(16, 2, 3, .9, 30)])

    def test_smoke_pool_passes_warm_start_audit(self):
        calls = test_launcher.LauncherTests().launch(SMOKE='1')
        args = calls['resolve_monotonicity_violations.py']
        count = int(args[args.index('--K') + 1])
        root = Path(__file__).resolve().parents[2]
        models = root / 'data_and_models/results_canonical_sweep_2026-08-18_seed42/models'
        # Build actual pools from saved models. Only source ranking/evaluation
        # is stubbed; loading, allocation, and embedding use the real code.
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(resolver, 'device', torch.device('cpu')), \
                patch.object(resolver, 'eval_mse', return_value=.01):
            seeds = Path(directory) / 'seeds'
            seeds.mkdir()
            for m in [2, 4, 8]:
                for depth in [1, 2, 3, 4]:
                    pool, _, labels, _, _ = resolver.build_diverse_pool(
                        16, m, depth, .9, count, models_dir=str(models), verbose=False)
                    self.assertEqual(len(pool), count)
                    (seeds / f'n16_m{m}_l{depth}_S0.9.json').write_text(json.dumps(dict(seeds=[
                        dict(run_id='smoke', warm_start_arm='diverse_pool',
                             warm_start_source=label, mse_full=.01) for label in labels])))
            failures, _ = check_warm_start_coverage(directory)
            self.assertEqual(failures, [])


if __name__ == '__main__':
    unittest.main()
