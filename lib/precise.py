"""
Canonical-CSV compiler: re-evaluate every saved .pt on the canonical fixed draw
(core.evaluate_mse: seed 42, n=200k, chunked, paired) and write
compiled/sweep_results_precise.csv with mse_full + mse_full_ci95 +
nonlinear_gain (depth delta vs the trained l=1 sibling).

Single-sourced here; consumers:
  - resolve_monotonicity_violations (each round of its internal loop)
  - precise_eval.py (thin CLI for standalone refreshes: wrap_up, merge_store)
"""
import torch
from pathlib import Path
import pandas as pd

import core
from core import Autoencoder, evaluate_mse

device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')

try:
    from tqdm import tqdm
except ImportError:                                     # pragma: no cover
    def tqdm(x, **k): return x


def compile_precise_csv(store_dir='data_and_models/results_db', n_samples=200000, seed=42):
    core.device = device
    models_dir = Path(store_dir) / 'models'

    rows = []
    files = sorted(models_dir.glob('model_*.pt'))
    for path in tqdm(files, desc='Re-eval'):
        stem = path.stem  # model_n128_m2_l4_S0.95
        parts = stem.split('_')
        # parts: ['model','n128','m2','l4','S0.95']
        n = int(parts[1][1:])
        m = int(parts[2][1:])
        l = int(parts[3][1:])
        S = float(parts[4][1:])

        model = Autoencoder(n, m, l, tied_weights=False).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        ev = evaluate_mse(model, S, n_samples=n_samples, seed=seed)
        rows.append(dict(n=n, m=m, l=l, S=S, mse_full=ev['eval_mse'],
                         mse_full_ci95=ev['eval_mse_ci95']))

    df = pd.DataFrame(rows).sort_values(['n', 'm', 'l', 'S']).reset_index(drop=True)

    # Compute two clean metrics measuring how much depth helps over the linear
    # baseline (the actual trained l=1 model at same n, m, S — NOT a linear
    # approximation of a deeper model, which would unfairly include decoder
    # mismatch in the linear baseline).
    #   nonlinear_mse_decrease = mse_l1 - mse_l                (absolute drop)
    #   pct_nonlinear_mse_decrease = (mse_l1 - mse_l) / mse_l1 (fractional drop)
    # By construction both are 0 at l=1.
    l1_lookup = df[df.l == 1].set_index(['n', 'm', 'S'])['mse_full'].to_dict()
    def absolute_decrease(row):
        key = (int(row.n), int(row.m), float(row.S))
        if key not in l1_lookup:
            return float('nan')
        return l1_lookup[key] - row.mse_full
    def pct_decrease(row):
        key = (int(row.n), int(row.m), float(row.S))
        if key not in l1_lookup:
            return float('nan')
        mse_l1 = l1_lookup[key]
        if mse_l1 <= 0:
            return float('nan')
        return (mse_l1 - row.mse_full) / mse_l1
    df['nonlinear_mse_decrease'] = df.apply(absolute_decrease, axis=1)
    df['pct_nonlinear_mse_decrease'] = df.apply(pct_decrease, axis=1)
    # Canonical name: nonlinear_gain == pct_nonlinear_mse_decrease — the fractional
    # loss improvement over the trained l=1 sibling at the same (n, m, S).
    df['nonlinear_gain'] = df['pct_nonlinear_mse_decrease']

    out = Path(store_dir) / 'compiled' / 'sweep_results_precise.csv'
    df.to_csv(out, index=False)
    print(f'Wrote {out} ({len(df)} configs)')
    return df


def find_violations_precise(df):
    """Same shape as find_violations but on the precise CSV."""
    df_idx = df.set_index(['n', 'm', 'l', 'S'])
    n_vals = sorted(df.n.unique())
    m_vals = sorted(df.m.unique())
    S_vals = sorted(df.S.unique())

    violations = {}
    for _, row in df.iterrows():
        n, m, l, S = int(row.n), int(row.m), int(row.l), row.S
        mse = row.mse_full

        for l2 in range(int(l) + 1, 5):
            k = (n, m, l2, S)
            if k in df_idx.index:
                mse2 = float(df_idx.loc[k, 'mse_full'])
                if mse2 > mse * 1.001:
                    if k not in violations or mse < violations[k]['mse_target']:
                        violations[k] = dict(type='depth',
                                             mse_target=mse, mse_current=mse2,
                                             gap=mse2 / mse,
                                             shallow_l=int(l))
        # Other axes (bottleneck, input_dim, sparsity) likewise:
        mi = m_vals.index(m)
        for mi2 in range(mi + 1, len(m_vals)):
            m2 = m_vals[mi2]
            if m2 >= n:
                continue
            k = (n, m2, l, S)
            if k in df_idx.index:
                mse2 = float(df_idx.loc[k, 'mse_full'])
                if mse2 > mse * 1.001:
                    if k not in violations or mse < violations[k]['mse_target']:
                        violations[k] = dict(type='bottleneck',
                                             mse_target=mse, mse_current=mse2,
                                             gap=mse2 / mse,
                                             shallow_l=int(l))
        ni = n_vals.index(n)
        for ni2 in range(0, ni):
            n2 = n_vals[ni2]
            if m >= n2:
                continue
            k = (n2, m, l, S)
            if k in df_idx.index:
                mse2 = float(df_idx.loc[k, 'mse_full'])
                if mse2 > mse * 1.001:
                    if k not in violations or mse < violations[k]['mse_target']:
                        violations[k] = dict(type='input_dim',
                                             mse_target=mse, mse_current=mse2,
                                             gap=mse2 / mse,
                                             shallow_l=int(l))
        si = S_vals.index(S)
        for si2 in range(si + 1, len(S_vals)):
            S2 = S_vals[si2]
            k = (n, m, l, S2)
            if k in df_idx.index:
                mse2 = float(df_idx.loc[k, 'mse_full'])
                if mse2 > mse * 1.001:
                    if k not in violations or mse < violations[k]['mse_target']:
                        violations[k] = dict(type='sparsity',
                                             mse_target=mse, mse_current=mse2,
                                             gap=mse2 / mse,
                                             shallow_l=int(l))

    return pd.DataFrame(
        [dict(n=k[0], m=k[1], l=k[2], S=k[3], **v) for k, v in violations.items()]
    ).sort_values('gap', ascending=False).reset_index(drop=True) if violations else pd.DataFrame()

