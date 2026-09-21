"""Regression checks for all-run sanity and phase-specific warm-start coverage.

Run: python -m unittest discover -s src/sweep -p 'test_check_results.py'
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

from check_results import (
    check_new_seed_sanity,
    check_warm_start_coverage,
    check_zero_noise_determinism,
    main,
    require_seed_logs,
    check_saved_model_fidelity,
)


def seed(run_id='canonical_test', arm='large_noise', source='large_noise(l=1,noise=0.0)',
         **extra):
    return dict(run_id=run_id, warm_start_arm=arm, warm_start_source=source,
                **{'mse_full': 0.01, 'init_mse': 0.01, 'warm_start_noise': 0.0,
                   'hidden_state_linearity': 0.5, **extra})


class SanityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Path(self.temp.name)
        (self.store / 'seeds').mkdir()

    def write(self, seeds, l=2, m=4):
        path = self.store / 'seeds' / f'n16_m{m}_l{l}_S0.9.json'
        path.write_text(json.dumps(dict(config=dict(n=16, m=m, l=l, S=0.9), seeds=seeds)))

    def run_audit(self, store=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch('sys.argv', ['check_results.py', '--store-dir',
                               str(store or self.store), '--skip-model-check']), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main()
        return code, stdout.getvalue(), stderr.getvalue()

    def snapshot(self):
        return {p.relative_to(self.store).as_posix(): p.read_bytes() if p.is_file() else None
                for p in self.store.rglob('*')}

    def test_missing_store_is_not_created(self):
        missing = self.store / 'absent'
        code, _, error = self.run_audit(missing)
        self.assertEqual(code, 2)
        self.assertIn('python scripts/verify_results.py', error)
        self.assertFalse(missing.exists())

    def test_snapshot_without_logs_is_unchanged(self):
        (self.store / 'seeds').rmdir()
        (self.store / 'compiled').mkdir()
        (self.store / 'compiled/sweep_results.csv').write_text('n,m,l,S,mse_full\n16,4,1,0.9,0.01\n')
        before = self.snapshot()
        code, _, error = self.run_audit()
        self.assertEqual(code, 2)
        self.assertIn('No per-config seed logs', error)
        self.assertEqual(self.snapshot(), before)

    def test_partial_logs_do_not_truncate_either_summary(self):
        self.write([seed()], l=1)
        (self.store / 'compiled').mkdir()
        for filename in ['sweep_results.csv', 'sweep_results_precise.csv']:
            with self.subTest(filename=filename):
                path = self.store / 'compiled' / filename
                path.write_text('n,m,l,S,mse_full\n16,4,1,0.9,0.01\n16,4,2,0.9,0.005\n')
                before = self.snapshot()
                code, _, error = self.run_audit()
                self.assertEqual(code, 2)
                self.assertIn('Missing seed logs for 1 configurations', error)
                self.assertEqual(self.snapshot(), before)
                path.unlink()

    def test_empty_or_invalid_logs_are_rejected_without_writes(self):
        path = self.store / 'seeds/n16_m4_l1_S0.9.json'
        for text in ['{"config": {}, "seeds": []}', '{invalid json', 'null']:
            with self.subTest(text=text):
                path.write_text(text)
                before = self.snapshot()
                code, _, error = self.run_audit()
                self.assertEqual(code, 2)
                self.assertIn('Cannot audit training store', error)
                self.assertEqual(self.snapshot(), before)

    def test_complete_training_store_still_audits(self):
        self.write([seed(steps_used=2, converged=True)], l=1)
        require_seed_logs(self.store)
        code, output, error = self.run_audit()
        self.assertEqual(code, 0, output + error)
        self.assertIn('SANITY CHECK: CLEAN', output)
        self.assertTrue((self.store / 'compiled/sweep_results.csv').is_file())

    def test_missing_checkpoint_fails_audit(self):
        self.write([seed(steps_used=2, converged=True)], l=1)
        output = io.StringIO()
        with patch('sys.argv', ['check_results.py', '--store-dir', str(self.store)]), \
                contextlib.redirect_stdout(output):
            code = main()
        self.assertEqual(code, 1)
        self.assertIn('Missing 1 expected checkpoint', output.getvalue())
        self.assertNotIn('SANITY CHECK: CLEAN', output.getvalue())

    def test_checkpoint_presence_checked_outside_evaluation_sample(self):
        frame = pd.DataFrame([dict(n=16, m=4, l=1, S=.9, mse_full=.01)])
        with self.assertRaisesRegex(FileNotFoundError, 'model_n16_m4_l1_S0.9.pt'):
            check_saved_model_fidelity(self.store, frame, sample_size=0)

    def test_sanity_checks_every_run_name_and_missing_run_id(self):
        for run_id in ['canonical_seed7', 'smoke_test', 'custom', 'canonical_resolve_r1', None]:
            with self.subTest(run_id=run_id):
                bad = seed(run_id)
                bad.update(mse_full=0.02, hidden_state_linearity=2.0,
                           loss_curve=[[i, 0.01 * (i + 1) ** 4] for i in range(20)])
                if run_id is None:
                    del bad['run_id']
                self.write([bad])
                issues = check_new_seed_sanity(self.store)
                for kind in ['weird_gain', 'floor_broken', 'convergence_unstable']:
                    self.assertEqual(len(issues[kind]), 1, kind)

    def test_missing_optional_telemetry_is_supported(self):
        self.write([dict(mse_full=0.01)])
        self.assertFalse(any(check_new_seed_sanity(self.store).values()))

    def test_handoffs_stay_within_run_and_adjacent_depths(self):
        self.write([seed('earlier')], l=1)
        deeper = seed('later')
        deeper['init_mse'] = 0.1
        self.write([deeper])
        self.assertFalse(check_new_seed_sanity(self.store)['handoff_broken'])
        self.write([seed('later')], l=1)
        self.assertEqual(len(check_new_seed_sanity(self.store)['handoff_broken']), 1)
        (self.store / 'seeds' / 'n16_m4_l2_S0.9.json').unlink()
        self.write([deeper], l=3)
        self.assertFalse(check_new_seed_sanity(self.store)['handoff_broken'])

    def test_random_initializations_do_not_define_a_handoff(self):
        self.write([seed()], l=1)
        cold = seed(arm='explore_random', source='random_init')
        cold['init_mse'] = 0.1
        self.write([cold])
        self.assertFalse(check_new_seed_sanity(self.store)['handoff_broken'])

    def test_resolver_frozen_sources_do_not_define_a_chain_handoff(self):
        self.write([seed('resolve', 'diverse_pool', 'shallow(l=1)+noise=0.0')])
        self.write([seed('resolve', 'diverse_pool', 'shallow(l=2)+noise=0.0', init_mse=0.1),
                    seed('resolve', 'near_warm_start', 'near_warm_start(l=2,noise=1.0e-07)',
                         init_mse=0.1)], l=3)
        self.assertFalse(check_new_seed_sanity(self.store)['handoff_broken'])

    def test_zero_noise_checks_all_runs_but_only_matching_siblings(self):
        first = seed()
        worse = seed()
        worse['mse_full'] = 0.02
        self.write([first, worse])
        self.assertEqual(len(check_zero_noise_determinism(self.store)), 1)
        for change in [dict(run_id='other'), dict(warm_start_arm='other'),
                       dict(warm_start_source='random_init'),
                       dict(warm_start_source='large_noise(l=2,noise=0.0)')]:
            with self.subTest(change=change):
                self.write([first, {**worse, **change}])
                self.assertFalse(check_zero_noise_determinism(self.store))

    def test_chain_needs_shallow_but_not_same_depth_neighbors(self):
        self.write([seed()])
        self.assertFalse(check_warm_start_coverage(self.store)[0])
        self.write([seed(arm='explore_random', source='random_init')])
        self.assertTrue(check_warm_start_coverage(self.store)[0])

    def test_resolver_needs_narrower_neighbor_not_s_neighbor(self):
        pool = [seed(arm='diverse_pool', source='shallow(l=1)+noise=0.0'),
                seed(arm='diverse_pool', source='S_neighbor(S=0.85)+noise=0.0')]
        self.write(pool)
        failures, _ = check_warm_start_coverage(self.store)
        self.assertEqual(len(failures), 1)
        self.assertIn('narrower-m', failures[0][1])
        pool.append(seed(arm='diverse_pool', source='neighbor(m=2)_seed0'))
        self.write(pool)
        self.assertFalse(check_warm_start_coverage(self.store)[0])

    def test_resolver_coverage_cannot_be_borrowed_from_another_run(self):
        pool = [seed('good', 'diverse_pool', 'shallow(l=1)+noise=0.0'),
                seed('good', 'diverse_pool', 'neighbor(m=2)_seed0'),
                seed('bad', 'diverse_pool', 'S_neighbor(S=0.85)+noise=0.0')]
        self.write(pool)
        failures, _ = check_warm_start_coverage(self.store)
        self.assertEqual(len(failures), 2)
        self.assertTrue(all("'bad'" in message for _, message in failures))

    def test_boundary_configs_do_not_require_unavailable_sources(self):
        self.write([seed(arm='diverse_pool', source='shallow(l=1)+noise=0.0')], m=2)
        self.write([seed(arm='diverse_pool', source='stored+noise=0.001')], l=1)
        self.assertFalse(check_warm_start_coverage(self.store)[0])


if __name__ == '__main__':
    unittest.main()
