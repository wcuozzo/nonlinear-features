"""
Cross-axis monotonicity: expected directions, CI-aware classification, violation scan.

Expected (candidate = deeper / wider-m / sparser / smaller-n should have <= MSE
than its bound config):
  depth      l up   => MSE down   (PROVABLE floor: identity-embed nests shallower in
                                   deeper — guarantees deeper can MATCH, never beat)
  bottleneck m up   => MSE down   (PROVABLE: wider bottleneck can zero extra dims)
  sparsity   S up   => MSE down   (easier problem; same weights transfer across S)
  input_dim  n down => MSE down   (easier at fixed m; NO nesting proof — empirical)

Every ordered comparison is classified three ways, using the eval CI
(mse_full_ci95 in the precise CSV; RSS-combined — conservative for the paired
eval, whose errors are positively correlated) plus the legacy x1.001 floor:
  improvement — candidate significantly better  (gap >  margin)
  tie         — statistically indistinguishable (|gap| <= margin)
  violation   — candidate significantly WORSE   (gap < -margin)
Violations fail the gate and get resolved. Ties are legitimate wherever the
true gain is ~0 (e.g. depth in the linear regime, where only MATCHING is
provable) — and the tie/improvement split across the grid is itself signal.
Legacy CSVs without the CI column degrade to the pure x1.001 tolerance.

Shared by check_results (the gate) and resolve_monotonicity_violations (the
fixer), so detection logic cannot drift between them.
"""


def violating_configs(mono_results):
    """Flatten {axis: [(bound_key, candidate_key, mse_bound, mse_cand), ...]}
    into {candidate_key: set(axes)} — the configs to resolve, tagged by axis."""
    out = {}
    for axis, viols in mono_results.items():
        for _bound, cand, _mb, _mc in viols:
            out.setdefault(cand, set()).add(axis)
    return out


def check_monotonicity_all_axes(df_idx, axes=('depth', 'bottleneck', 'input_dim', 'sparsity')):
    """Back-compatible wrapper: just the CI-confirmed violations dict."""
    violations, _ = check_monotonicity_with_summary(df_idx, axes)
    return violations


def check_monotonicity_with_summary(df_idx, axes=('depth', 'bottleneck', 'input_dim', 'sparsity'),
                                    rel_tol=0.001):
    """Scan every ordered comparable pair along each axis.

    Returns (violations, summary):
      violations: {axis: [(bound_key, candidate_key, mse_bound, mse_cand), ...]}
      summary:    {axis: {'improvement': n, 'tie': n, 'violation': n}}
    """
    n_vals = sorted(df_idx.index.get_level_values('n').unique())
    m_vals = sorted(df_idx.index.get_level_values('m').unique())
    S_vals = sorted(df_idx.index.get_level_values('S').unique())
    has_ci = 'mse_full_ci95' in df_idx.columns

    violations = {a: [] for a in axes}
    summary = {a: {'improvement': 0, 'tie': 0, 'violation': 0} for a in axes}

    def scan(axis, bound_key, candidate_key):
        """bound_key's MSE is the floor the candidate (expected-better config)
        must not significantly exceed."""
        mse_bound = float(df_idx.loc[bound_key, 'mse_full'])
        mse_cand = float(df_idx.loc[candidate_key, 'mse_full'])
        ci = 0.0
        if has_ci:
            ci = (float(df_idx.loc[bound_key, 'mse_full_ci95']) ** 2
                  + float(df_idx.loc[candidate_key, 'mse_full_ci95']) ** 2) ** 0.5
        margin = ci + rel_tol * mse_bound
        gap = mse_bound - mse_cand                    # > 0: candidate better
        if gap > margin:
            summary[axis]['improvement'] += 1
        elif gap < -margin:
            summary[axis]['violation'] += 1
            violations[axis].append((bound_key, candidate_key, mse_bound, mse_cand))
        else:
            summary[axis]['tie'] += 1

    for key in df_idx.index:
        n, m, l, S = key
        if 'depth' in axes:                            # candidate: deeper
            for l2 in range(int(l) + 1, 5):
                k2 = (n, m, l2, S)
                if k2 in df_idx.index:
                    scan('depth', key, k2)
        if 'bottleneck' in axes:                       # candidate: wider m
            mi = m_vals.index(m)
            for mi2 in range(mi + 1, len(m_vals)):
                m2 = m_vals[mi2]
                if m2 >= n:
                    continue
                k2 = (n, m2, l, S)
                if k2 in df_idx.index:
                    scan('bottleneck', key, k2)
        if 'input_dim' in axes:                        # candidate: smaller n
            ni = n_vals.index(n)
            for ni2 in range(0, ni):
                n2 = n_vals[ni2]
                if m >= n2:
                    continue
                k2 = (n2, m, l, S)
                if k2 in df_idx.index:
                    scan('input_dim', key, k2)
        if 'sparsity' in axes:                         # candidate: sparser
            si = S_vals.index(S)
            for si2 in range(si + 1, len(S_vals)):
                S2 = S_vals[si2]
                k2 = (n, m, l, S2)
                if k2 in df_idx.index:
                    scan('sparsity', key, k2)

    return violations, summary
