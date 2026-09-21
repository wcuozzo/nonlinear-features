"""
Additive results store for sweep experiments.

Key property: more compute is ALWAYS additive. Running more seeds for a config
can only improve results, never make them worse. The store accumulates all seed
results across runs and always derives the "best" from the full pool.

Usage:
    store = ResultsStore('data_and_models/results_db')
    store.add_seeds(n=16, m=2, l=2, S=0.9, seed_results=[...], run_meta={...})
    summary = store.compile()          # Best-seed summary for all configs
    seeds = store.get_seeds(16, 2, 2, 0.9)  # All seeds for one config
"""

import os
import json
import time
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path


class ResultsStore:
    """Append-only store for per-seed experiment results.

    Storage layout:
        {store_dir}/
            seeds/                  # Per-config seed results
                n16_m2_l2_S0.9.json
                n128_m64_l4_S0.95.json
                ...
            models/                 # Best model weights per config
                model_n16_m2_l2_S0.9.pt
                ...
            compiled/               # Derived summaries (regenerated)
                sweep_results.csv   # Best-seed summary
                all_seeds.csv       # All seeds flat
            runs.jsonl              # Run metadata log
    """

    def __init__(self, store_dir='data_and_models/results_db'):
        self.store_dir = Path(store_dir)
        self.seeds_dir = self.store_dir / 'seeds'
        self.models_dir = self.store_dir / 'models'
        self.compiled_dir = self.store_dir / 'compiled'

        for d in [self.seeds_dir, self.models_dir, self.compiled_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _config_key(self, n, m, l, S):
        return f'n{n}_m{m}_l{l}_S{S}'

    def _config_path(self, n, m, l, S):
        return self.seeds_dir / f'{self._config_key(n, m, l, S)}.json'

    def _load_config(self, n, m, l, S):
        path = self._config_path(n, m, l, S)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {'config': {'n': n, 'm': m, 'l': l, 'S': S}, 'seeds': []}

    @staticmethod
    def _json_default(obj):
        """Handle numpy types in JSON serialization."""
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

    def _save_config(self, n, m, l, S, data):
        path = self._config_path(n, m, l, S)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=self._json_default)

    def add_seeds(self, n, m, l, S, seed_results, run_id=None,
                  model_state_dict=None, training_meta=None):
        """Add seed results for a config. Always additive.

        Args:
            n, m, l, S: config parameters
            seed_results: list of dicts, one per seed, each containing:
                - seed_value: int, the random seed used
                - mse_full: float
                - hidden_state_linearity: float — R^2 of best affine x->z fit
                  (legacy key 'linearity_score' accepted)
                - converged: bool
                - steps_used: int
                Optional:
                - loss_curve: list of (step, loss) tuples
                - gain_curve: list of (step, gain) tuples
            run_id: str, identifier for this run (auto-generated if None)
            model_state_dict: if provided, save the best model
            training_meta: dict of training hyperparameters
        """
        if run_id is None:
            run_id = f'run_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'

        data = self._load_config(n, m, l, S)

        # Track existing seeds to avoid exact duplicates
        existing_seeds = {(s['seed_value'], s.get('run_id', ''))
                         for s in data['seeds']}

        timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')
        added = 0
        for sr in seed_results:
            key = (sr['seed_value'], run_id)
            if key in existing_seeds:
                continue  # Skip exact duplicates

            entry = {
                'seed_value': sr['seed_value'],
                'mse_full': float(sr['mse_full']),
                'hidden_state_linearity': float(sr.get('hidden_state_linearity',
                                                        sr.get('linearity_score', float('nan')))),
                'converged': bool(sr['converged']),
                'steps_used': int(sr['steps_used']),
                'run_id': run_id,
                'timestamp': timestamp,
            }
            # Preserve ALL extra fields from caller (init_mse, warm_start_source,
            # warm_start_noise, etc.) so we don't silently drop provenance.
            for key, val in sr.items():
                if key not in entry:
                    entry[key] = val
            if training_meta:
                entry['training_meta'] = training_meta

            data['seeds'].append(entry)
            added += 1

        self._save_config(n, m, l, S, data)

        # Save model if provided and it's the new best
        if model_state_dict is not None:
            import torch
            model_path = self.models_dir / f'model_{self._config_key(n, m, l, S)}.pt'

            # Check if new model is better than existing
            save = True
            if model_path.exists():
                best = self._get_best_seed(data)
                # Only save if this run produced the current best
                new_best_mse = min(sr['mse_full'] for sr in seed_results)
                if new_best_mse > best['mse_full'] + 1e-8:
                    save = False

            if save:
                torch.save(model_state_dict, model_path)

        # Log run
        self._log_run(run_id, n, m, l, S, added, training_meta)

        return added

    def _log_run(self, run_id, n, m, l, S, n_seeds_added, training_meta):
        log_path = self.store_dir / 'runs.jsonl'
        entry = {
            'run_id': run_id,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'config': {'n': n, 'm': m, 'l': l, 'S': S},
            'n_seeds_added': n_seeds_added,
        }
        if training_meta:
            entry['training_meta'] = training_meta
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry, default=self._json_default) + '\n')

    def _get_best_seed(self, data):
        """Get the best seed from a config's data: PURE MINIMUM MSE.

        The scaling law's object is the loss, so final selection is min mse_full,
        full stop. nonlinear_gain / linearity_score are REPORTED metrics of the
        selected seed, never selection criteria: earlier versions filtered or
        tie-broke on gain ("prefer the nonlinear basin"), which (a) could crown a
        worse-MSE model and (b) made fleet nonlinearity statistics circular —
        seeds were selected FOR high gain, then the selected fleet was reported
        as nonlinear. Basin exploration is the warm-start pools' job (shallow
        embed / perturbed / neighbor-m / random), not the selector's.
        """
        if not data['seeds']:
            return None
        return min(data['seeds'], key=lambda s: s['mse_full'])

    def get_seeds(self, n, m, l, S):
        """Get all seed results for a config."""
        data = self._load_config(n, m, l, S)
        return data['seeds']

    def get_all_configs(self):
        """List all configs that have results."""
        configs = []
        for path in self.seeds_dir.glob('*.json'):
            with open(path) as f:
                data = json.load(f)
            configs.append(data['config'])
        return configs

    def compile(self):
        """Compile best-seed summary across all configs.

        Returns DataFrame compatible with existing sweep_results.csv format.
        Also saves to compiled/sweep_results.csv.
        """
        rows = []
        for path in sorted(self.seeds_dir.glob('*.json')):
            with open(path) as f:
                data = json.load(f)

            seeds = data['seeds']
            if not seeds:
                continue

            cfg = data['config']
            n, m, l, S = cfg['n'], cfg['m'], cfg['l'], cfg['S']

            # Best seed selection
            best = self._get_best_seed(data)

            # Hidden-state linearity (R^2 of affine x->z fit; legacy key 'linearity_score')
            def _hl(s):
                return s.get('hidden_state_linearity', s.get('linearity_score', float('nan')))
            hls = [_hl(s) for s in seeds]

            row = {
                'n': n, 'm': m, 'l': l, 'S': S,
                'hidden_state_linearity': _hl(best),
                'mse_full': best['mse_full'],
                'hidden_linearity_mean': float(np.mean(hls)),
                'hidden_linearity_std': float(np.std(hls)),
                'steps_used': best['steps_used'],
                'steps_mean': float(np.mean([s['steps_used'] for s in seeds])),
                'converged': best['converged'],
                'n_converged': sum(1 for s in seeds if s['converged']),
                'n_seeds': len(seeds),
                'n_runs': len(set(s.get('run_id', '') for s in seeds)),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values(['n', 'm', 'l', 'S']).reset_index(drop=True)
            # nonlinear_gain: CONFIG-LEVEL depth gain vs the forced-linear counterpart —
            # (mse[l=1] - mse[l]) / mse[l=1] at the same (n, m, S). Exactly 0 at l=1;
            # NaN if the l=1 sibling is missing. ('hidden_state_linearity' is the separate
            # z-space diagnostic: R^2 of the best affine x->z fit.)
            l1 = df[df['l'] == 1].set_index(['n', 'm', 'S'])['mse_full']
            key = list(zip(df['n'], df['m'], df['S']))
            mse_l1 = pd.Series([l1.get(k, np.nan) for k in key], index=df.index)
            df['nonlinear_gain'] = (mse_l1 - df['mse_full']) / mse_l1

        # Save compiled CSV
        csv_path = self.compiled_dir / 'sweep_results.csv'
        df.to_csv(csv_path, index=False)

        return df

    def import_csv(self, csv_path, run_id=None):
        """Import results from an existing sweep_results.csv.

        This creates synthetic per-seed entries from the summary statistics.
        Use import_seed_level() for actual per-seed data.
        """
        if run_id is None:
            run_id = f'import_{Path(csv_path).stem}_{time.strftime("%Y%m%d_%H%M%S")}'

        df = pd.read_csv(csv_path)
        imported = 0

        for _, row in df.iterrows():
            n, m, l, S = int(row['n']), int(row['m']), int(row['l']), row['S']

            # Create a single seed entry from the best-seed result
            seed_results = [{
                'seed_value': 0,  # Unknown original seed
                'mse_full': float(row['mse_full']),
                'mse_linear': float(row['mse_linear']),
                'hidden_state_linearity': float(row.get('hidden_state_linearity',
                                                        row.get('linearity_score', float('nan')))),
                'converged': bool(row.get('converged', False)),
                'steps_used': int(row.get('steps_used', 0)),
            }]

            added = self.add_seeds(n, m, l, S, seed_results, run_id=run_id)
            imported += added

        return imported

    def stats(self):
        """Print store statistics."""
        configs = list(self.seeds_dir.glob('*.json'))
        total_seeds = 0
        total_runs = set()

        for path in configs:
            with open(path) as f:
                data = json.load(f)
            total_seeds += len(data['seeds'])
            for s in data['seeds']:
                total_runs.add(s.get('run_id', 'unknown'))

        return {
            'n_configs': len(configs),
            'n_seeds': total_seeds,
            'n_runs': len(total_runs),
            'avg_seeds_per_config': total_seeds / max(len(configs), 1),
        }

    def needs_more_seeds(self, n, m, l, S, min_seeds=30, max_cv=0.3):
        """Check if a config needs more seeds.

        Returns (needs_more, reason) tuple.
        """
        seeds = self.get_seeds(n, m, l, S)
        if len(seeds) < min_seeds:
            return True, f'only {len(seeds)} seeds (need {min_seeds})'

        gains = [s['nonlinear_gain'] for s in seeds]
        mean_g = np.mean(gains)
        std_g = np.std(gains)
        cv = std_g / (abs(mean_g) + 1e-6)

        if cv > max_cv and mean_g > 0.01:
            return True, f'CV={cv:.2f} > {max_cv}'

        conv_rate = sum(1 for s in seeds if s['converged']) / len(seeds)
        if conv_rate < 0.5:
            return True, f'conv_rate={conv_rate:.0%} < 50%'

        return False, 'sufficient'

    def get_configs_needing_work(self, all_configs=None, min_seeds=30, max_cv=0.3):
        """Get list of configs that need more compute."""
        if all_configs is None:
            all_configs = self.get_all_configs()

        needs_work = []
        for cfg in all_configs:
            needs, reason = self.needs_more_seeds(
                cfg['n'], cfg['m'], cfg['l'], cfg['S'],
                min_seeds=min_seeds, max_cv=max_cv)
            if needs:
                needs_work.append({**cfg, 'reason': reason})

        return needs_work
