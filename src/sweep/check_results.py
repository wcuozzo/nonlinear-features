"""
Comprehensive sanity check after a violation-fix sweep.

Goes beyond "0 violations":
  A. Monotonicity (depth + bottleneck + input_dim + sparsity)
  B. All-run seed/MSE plausibility:
       - hidden_state_linearity in [-1.05, 1.05] (R^2 of affine x->z fit)
       - final_mse <= init_mse * 1.05 (floor enforcement worked)
       - init_mse of stage l ~ best final_mse of stage l-1 (handoff sanity)
  C. Saved-model fidelity:
       - re-loading the .pt file gives MSE matching reported mse_full
  D. Loss-curve stability:
       - flag a strongly ascending trend in the last quarter
  E. Cross-seed determinism for zero-noise seeds:
       - two seeds with noise=0 from same source should give similar final MSE
       - if they diverge wildly, something stochastic is broken
  G. Warm-start coverage:
       - deeper configs need shallow starts; only resolver pools need narrower-m starts

Exit code 0 if all green, 1 if any anomaly, 2 if seed logs are missing or invalid.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'lib'))
from results_store import ResultsStore
from train_sweep import find_violations, group_violations_by_nms
from warm_start import eval_mse, load_best_model

device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────

def _load_seeds(store_dir, n, m, l, S):
    path = Path(store_dir) / 'seeds' / f'n{n}_m{m}_l{l}_S{S}.json'
    if not path.exists():
        return []
    return json.load(open(path))['seeds']


def require_seed_logs(store_dir):
    """Check audit inputs before ResultsStore can create or overwrite files."""
    store_dir = Path(store_dir)
    paths = sorted((store_dir / 'seeds').glob('*.json'))
    if not paths:
        raise ValueError(f'No per-config seed logs found in {store_dir / "seeds"}.')
    dimensions = ['n', 'm', 'l', 'S']
    logged = set()
    for path in paths:
        data = json.loads(path.read_text())
        if not isinstance(data.get('seeds'), list) or not data['seeds']:
            raise ValueError(f'No seed results recorded in {path}.')
        logged.add(tuple(data['config'][key] for key in dimensions))
    # A partially exported store must not replace a complete summary with a
    # smaller one just because a few seed files happen to be present.
    for name in ['sweep_results.csv', 'sweep_results_precise.csv']:
        path = store_dir / 'compiled' / name
        if path.is_file():
            frame = pd.read_csv(path, usecols=dimensions)
            recorded = set(frame[dimensions].itertuples(index=False, name=None))
            missing = recorded - logged
            if missing:
                raise ValueError(f'Missing seed logs for {len(missing)} configurations in {path}.')


# ────────────────────────────────────────────────────────────────────────
# A. Cross-axis monotonicity
# ────────────────────────────────────────────────────────────────────────

from monotonicity import check_monotonicity_all_axes  # shared with resolve_violations

# ────────────────────────────────────────────────────────────────────────
# B. Seed sanity across all runs
# ────────────────────────────────────────────────────────────────────────

def check_new_seed_sanity(store_dir):
    """Scan every recorded seed, regardless of run name (legacy API name).

    Checks needing optional telemetry run only where that telemetry exists.
    Handoffs compare adjacent depths within the same chain run. Resolver pools
    use frozen pre-round sources, so their results do not define a handoff.
    """
    seeds_dir = Path(store_dir) / 'seeds'

    weird_gain = []
    floor_broken = []
    handoff_broken = []
    convergence_unstable = []

    # Do not compare unrelated training runs in an accumulated store.
    by_group = defaultdict(lambda: defaultdict(list))
    for path in sorted(seeds_dir.glob('*.json')):
        data = json.loads(path.read_text())
        cfg = data['config']
        n, m, l, S = cfg['n'], cfg['m'], cfg['l'], cfg['S']
        for sd in data['seeds']:
            by_group[(n, m, S, sd.get('run_id'))][l].append(sd)

    for (n, m, S, _run_id), l_to_seeds in by_group.items():
        for l, seeds in l_to_seeds.items():
            for sd in seeds:
                hl = sd.get('hidden_state_linearity', sd.get('linearity_score'))
                if hl is not None and np.isfinite(hl) and not (-1.05 <= hl <= 1.05):
                    weird_gain.append((n, m, l, S, hl))

                init = sd.get('init_mse')
                if init is not None and sd['mse_full'] > init * 1.05:
                    floor_broken.append((n, m, l, S, init, sd['mse_full']))

                # Robust convergence check: training loss is noisy per-batch,
                # so use the LINEAR-FIT slope of the last quarter as the signal.
                # If slope > +tolerance, training was still ascending (bad).
                # Tolerance is scaled by the median loss to be relative.
                lc = sd.get('loss_curve', [])
                if len(lc) >= 20:
                    last_q = lc[-max(5, len(lc) // 4):]
                    steps = np.array([s for s, _ in last_q], dtype=float)
                    losses = np.array([v for _, v in last_q], dtype=float)
                    if len(steps) >= 3 and steps.max() > steps.min():
                        slope, intercept = np.polyfit(steps, losses, 1)
                        median_loss = float(np.median(losses))
                        # Slope per step; normalize by (median_loss / total_steps)
                        relative = slope * (steps.max() - steps.min()) / max(median_loss, 1e-12)
                        # Threshold scales with absolute loss: at noise-floor MSE
                        # (~1e-5), batch-to-batch variation is naturally larger.
                        # Use 30% for absolute losses above 0.001, 100% for below.
                        thresh = 0.30 if float(np.median(losses)) > 0.001 else 1.0
                        if relative > thresh:
                            convergence_unstable.append(
                                (n, m, l, S, float(np.min(losses)),
                                 float(losses[-1]), float(relative)))

        # Handoff check: stage l's best init_mse should be ~ stage l-1's best final_mse
        for l, seeds in l_to_seeds.items():
            if l - 1 not in l_to_seeds:
                continue
            curr_best_init = min(
                (sd['init_mse'] for sd in seeds
                 if sd.get('init_mse') is not None
                 and _has_shallow(sd.get('warm_start_source'))
                 and (sd.get('warm_start_arm') == 'large_noise'
                      or sd.get('warm_start_source_l') == l - 1)),
                default=np.inf)
            prev_best_final = min(sd['mse_full'] for sd in l_to_seeds[l - 1])
            if curr_best_init < np.inf:
                # Init at stage l should be close to final at stage l-1
                # (the warm-start source is the previous best)
                # Allow up to 5% slack from eval-sample noise
                if curr_best_init > prev_best_final * 1.05 + 1e-6:
                    handoff_broken.append((n, m, S, l - 1, l,
                                           prev_best_final, curr_best_init))

    return dict(
        weird_gain=weird_gain,
        floor_broken=floor_broken,
        handoff_broken=handoff_broken,
        convergence_unstable=convergence_unstable,
    )


# ────────────────────────────────────────────────────────────────────────
# C. Saved-model fidelity
# ────────────────────────────────────────────────────────────────────────

def check_saved_model_fidelity(store_dir, df, n_samples=200000, sample_size=30,
                                eval_seed=42, use_precise_eval=True):
    """Reload each saved .pt and verify its MSE matches the CSV.

    If `use_precise_eval` and df is the precise CSV, we re-eval at the same
    (n_samples=200000, seed=42) used by lib/precise.compile_precise_csv — matches should
    then be within float precision (~1e-7). Any mismatch flags a real bug
    (e.g. .pt file corrupted, recompile out-of-date). CAVEAT: run on the SAME
    device type that generated the CSV — torch.manual_seed draws differ across
    cpu/mps/cuda RNGs, so cross-device checks show ~0.1% spurious mismatches.
    """
    from core import Autoencoder, evaluate_mse
    import core as _core
    _core.device = device

    # Check every expected checkpoint, even when MSE evaluation is sampled.
    missing = []
    for row in df.itertuples():
        name = f'model_n{int(row.n)}_m{int(row.m)}_l{int(row.l)}_S{float(row.S)}.pt'
        if not (Path(store_dir) / 'models' / name).is_file():
            missing.append(name)
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} expected checkpoint(s): ' + ', '.join(missing))

    np.random.seed(0)
    idxs = np.random.choice(len(df), size=min(sample_size, len(df)), replace=False)
    mismatches = []
    for i in idxs:
        row = df.iloc[i]
        n, m, l, S = int(row.n), int(row.m), int(row.l), row.S
        model_path = Path(store_dir) / 'models' / f'model_n{n}_m{m}_l{l}_S{S}.pt'
        model = Autoencoder(n, m, l, tied_weights=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        if use_precise_eval:
            mse = evaluate_mse(model, S, n_samples=n_samples, seed=eval_seed)['eval_mse']
            tol = 1e-3  # tight enough to catch real mismatches, loose enough
                        # for sub-display-precision float32 numerical drift
        else:
            mse = eval_mse(model, S, n_samples=n_samples, device=device)
            tol = 0.03
        reported = row['mse_full']
        if mse > reported * (1 + tol) + 1e-9 or reported > mse * (1 + tol) + 1e-9:
            mismatches.append((n, m, l, S, reported, mse))
    return mismatches


# ────────────────────────────────────────────────────────────────────────
# E. Zero-noise determinism
# ────────────────────────────────────────────────────────────────────────

def check_zero_noise_determinism(store_dir):
    """Within each run/source/arm, zero-noise siblings should have similar final MSE.

    Different seed_value but same noise=0 from same source = different starting
    perturbation paths (the embed should be identical), so they should converge
    to similar minima. Diverging suggests stochasticity beyond the noise schedule.
    """
    by_group_l = defaultdict(list)
    for path in sorted(Path(store_dir, 'seeds').glob('*.json')):
        data = json.loads(path.read_text())
        cfg = data['config']
        key = (cfg['n'], cfg['m'], cfg['l'], cfg['S'])
        for sd in data['seeds']:
            source = sd.get('warm_start_source')
            if sd.get('warm_start_noise') == 0.0 and _has_shallow(source):
                sibling_key = (key, sd.get('run_id'), source,
                               sd.get('warm_start_source_l'), sd.get('warm_start_arm'))
                by_group_l[sibling_key].append(sd)

    diverging = []
    for (key, *_source), seeds in by_group_l.items():
        if len(seeds) < 2:
            continue
        mses = [sd['mse_full'] for sd in seeds]
        if max(mses) > min(mses) * 1.5 and max(mses) - min(mses) > 1e-5:
            diverging.append((key, min(mses), max(mses)))
    return diverging


# ────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────

def _has_shallow(src):
    return bool(src) and ('large_noise' in src or 'near_warm_start' in src or 'shallow' in src)


def _has_neighbor(src):
    return bool(src) and src.startswith('neighbor(m=')


def check_warm_start_coverage(store_dir):
    """For every config with seeds, verify the warm-start sources it SHOULD have
    were actually present in its pool. Returns (failures, warnings)."""
    import glob, re
    failures, warnings = [], []
    for p in sorted(glob.glob(str(Path(store_dir) / 'seeds' / 'n*.json'))):
        mo = re.search(r'n(\d+)_m(\d+)_l(\d+)_S([\d.]+)\.json$', p)
        if not mo:
            continue
        n, m, l, S = int(mo[1]), int(mo[2]), int(mo[3]), mo[4]
        seeds = json.loads(Path(p).read_text()).get('seeds', [])
        if not seeds:
            failures.append(((n, m, l, S), 'no seeds (config not trained)'))
            continue
        srcs = [s.get('warm_start_source') for s in seeds]
        # HARD: every l>=2 config must have gotten a shallow embed (the chain's uniform warm
        # start). l=1 is the chain base (trained from random in-chain) — no source expected.
        if l >= 2 and not any(_has_shallow(s) for s in srcs):
            failures.append(((n, m, l, S), f'l={l}: NO shallow warm-start seed (cold-started)'))
        # The first-pass chain has no guaranteed same-depth neighbors yet.
        # Only a later diverse resolver pool promises a narrower-m source.
        # Check each pool separately so another run cannot mask missing coverage.
        resolver_pools = defaultdict(list)
        for seed in seeds:
            if seed.get('warm_start_arm') == 'diverse_pool':
                resolver_pools[seed.get('run_id')].append(seed.get('warm_start_source'))
        for run_id, pool_srcs in resolver_pools.items():
            if l >= 2 and not any(_has_shallow(s) for s in pool_srcs):
                failures.append(((n, m, l, S), f'resolver run {run_id!r}: NO shallow warm-start seed'))
            if l >= 2 and m > 2 and not any(_has_neighbor(s) for s in pool_srcs):
                failures.append(((n, m, l, S), f'resolver run {run_id!r}: NO narrower-m neighbor seed'))
        # SOFT: did a cold RANDOM init produce the best minimum? (unusual for l>=2)
        best = min(seeds, key=lambda s: s.get('mse_full', float('inf')))
        bsrc = best.get('warm_start_source') or ''
        if l >= 2 and 'random' in bsrc:
            warnings.append(((n, m, l, S), f'best minimum came from a random (cold) init: {bsrc!r}'))
    return failures, warnings


def report_section(title, items, formatter=None, max_show=10):
    if not items:
        print(f'  ✓ {title}: 0')
        return False
    print(f'  ✗ {title}: {len(items)}')
    for x in items[:max_show]:
        if formatter:
            print(f'      {formatter(x)}')
        else:
            print(f'      {x}')
    if len(items) > max_show:
        print(f'      ... and {len(items)-max_show} more')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store-dir', default='data_and_models/results_db')
    parser.add_argument('--skip-model-check', action='store_true')
    parser.add_argument('--model-sample', type=int, default=30)
    parser.add_argument('--device', default=None,
                        help='Force eval device (cpu/mps/cuda:0). MUST match the device '
                             'that generated the precise CSV — RNG draws differ across '
                             'device types (see check_saved_model_fidelity caveat).')
    parser.add_argument('--precise', action='store_true',
                        help='Use sweep_results_precise.csv for monotonicity (recommended)')
    args = parser.parse_args()
    try:
        require_seed_logs(args.store_dir)
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
        print(f'Cannot audit training store: {error}', file=sys.stderr)
        print('This audit requires per-config seed logs before recompiling training summaries.\n'
              'To check the bundled results without seed logs, run from the repository root:\n'
              '  python scripts/verify_results.py', file=sys.stderr)
        return 2
    if args.device:
        global device
        device = torch.device(args.device)
        import core as _core
        _core.device = device

    print(f'\n{"="*72}\nCOMPREHENSIVE SANITY CHECK: {args.store_dir}\n{"="*72}')
    any_issue = False

    # Recompile
    store = ResultsStore(args.store_dir)
    df = store.compile()

    # For monotonicity, use precise CSV if available + requested
    precise_path = Path(args.store_dir) / 'compiled' / 'sweep_results_precise.csv'
    if args.precise and precise_path.exists():
        df_for_mono = pd.read_csv(precise_path)
        print(f'Configs in store: {len(df)};  using PRECISE CSV ({len(df_for_mono)} models) for monotonicity')
    else:
        df_for_mono = df
        print(f'Configs in store: {len(df)};  using NOISY CSV for monotonicity')

    df_idx = df_for_mono.set_index(['n', 'm', 'l', 'S'])

    # ── A. Monotonicity across all axes (CI-aware: improvement / tie / violation) ──
    print('\n[A] Cross-axis monotonicity:')
    from monotonicity import check_monotonicity_with_summary
    mono, mono_summary = check_monotonicity_with_summary(df_idx)
    for axis, cnt in mono_summary.items():
        total = sum(cnt.values()) or 1
        print(f'    {axis:11s}: {cnt["improvement"]:4d} significant improvements, '
              f'{cnt["tie"]:4d} ties (within eval CI), {cnt["violation"]:3d} violations '
              f'({100*cnt["improvement"]/total:.0f}% / {100*cnt["tie"]/total:.0f}% / '
              f'{100*cnt["violation"]/total:.0f}%)')
    for axis, viols in mono.items():
        any_issue |= report_section(
            f'{axis} violations', viols,
            formatter=lambda v: f'better={v[0]} ({v[2]:.5f}) worse={v[1]} ({v[3]:.5f})  ratio={v[3]/v[2]:.2f}x',
            max_show=5,
        )

    # ── B. Seed sanity ────────────────────────────────────────────────
    print('\n[B] Seed sanity (all runs):')
    nss = check_new_seed_sanity(args.store_dir)
    any_issue |= report_section(
        'out-of-range nonlinear_gain', nss['weird_gain'],
        formatter=lambda v: f'n={v[0]} m={v[1]} l={v[2]} S={v[3]} gain={v[4]:.3f}')
    any_issue |= report_section(
        'floor broken (final > 1.05 * init)', nss['floor_broken'],
        formatter=lambda v: f'n={v[0]} m={v[1]} l={v[2]} S={v[3]} init={v[4]:.5f} final={v[5]:.5f}')
    any_issue |= report_section(
        'broken handoff (stage l init > 1.05 * stage l-1 final)', nss['handoff_broken'],
        formatter=lambda v: f'n={v[0]} m={v[1]} S={v[2]} l={v[3]}->{v[4]} '
                            f'prev_final={v[5]:.5f} curr_init={v[6]:.5f}')
    any_issue |= report_section(
        'convergence unstable (last quarter ascending >30%; >100% near noise floor)', nss['convergence_unstable'],
        formatter=lambda v: f'n={v[0]} m={v[1]} l={v[2]} S={v[3]} '
                            f'min={v[4]:.5f} last={v[5]:.5f}  rel_slope={v[6]:+.2f}')

    # ── C. Saved-model fidelity ───────────────────────────────────────
    if not args.skip_model_check:
        print('\n[C] Saved-model fidelity (sampled):')
        # If precise CSV available, use it (consistent eval, tight tolerance)
        check_df = df_for_mono if args.precise and 'mse_full' in df_for_mono.columns else df
        try:
            mismatches = check_saved_model_fidelity(args.store_dir, check_df, sample_size=args.model_sample)
        except FileNotFoundError as error:
            print(f'  Saved-model check failed: {error}')
            return 1
        any_issue |= report_section(
            f'saved .pt vs CSV mismatch (sample {args.model_sample})', mismatches,
            formatter=lambda v: f'n={v[0]} m={v[1]} l={v[2]} S={v[3]} '
                                f'csv={v[4]:.5f} live={v[5]:.5f}  ratio={v[5]/max(v[4], 1e-12):.2f}x')

    # ── E. Zero-noise determinism ─────────────────────────────────────
    print('\n[E] Zero-noise seed determinism:')
    diverging = check_zero_noise_determinism(args.store_dir)
    any_issue |= report_section(
        'zero-noise siblings diverging (max/min > 1.5)', diverging,
        formatter=lambda v: f'n={v[0][0]} m={v[0][1]} l={v[0][2]} S={v[0][3]} '
                            f'min={v[1]:.5f} max={v[2]:.5f}')

    print('\n[G] Warm-start coverage (l>=2 needs shallow; resolver l>=2, m>2 also needs narrower-m):')
    ws_fail, ws_warn = check_warm_start_coverage(args.store_dir)
    any_issue |= report_section(
        'configs MISSING an expected warm start (cold-started -> biased minimum)', ws_fail,
        formatter=lambda v: f'n={v[0][0]} m={v[0][1]} l={v[0][2]} S={v[0][3]}: {v[1]}')
    report_section(  # soft: surfaced but does NOT fail the run
        'best-seed-from-random (investigate; not a hard fail)', ws_warn,
        formatter=lambda v: f'n={v[0][0]} m={v[0][1]} l={v[0][2]} S={v[0][3]}: {v[1]}')

    print()
    if any_issue:
        print('SANITY CHECK: ISSUES FOUND')
        return 1
    print('SANITY CHECK: CLEAN ✓')
    return 0


if __name__ == '__main__':
    sys.exit(main())
