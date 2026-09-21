"""Multi-start searches for untied models with unit-norm constraints.

Run with a local Python containing numpy and torch. No external services or GPU
are required. All vectors stored here are columns of 2 x 5 matrices: E writes
input features, D reads output features, and prediction = relu(x @ E.T @ D + b).
The unit-encoder condition is analogous to unit dictionary atoms in an SAE.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
N, M, S = 5, 2, 0.9


def sample(count, generator):
    return (torch.rand(count, N, generator=generator) > S).float() * torch.rand(
        count, N, generator=generator
    )


def unit(v):
    return v / v.norm(dim=-2, keepdim=True).clamp_min(1e-12)


def directions(angle):
    return torch.stack((angle.cos(), angle.sin()), dim=-2)


def angle(v):
    return torch.atan2(v[..., 1, :], v[..., 0, :])


def rotate(v, offset):
    return directions(angle(v) + offset) * v.norm(dim=-2, keepdim=True)


def saved_models():
    data = json.loads((ROOT / 'figures/untied_read_write_example.json').read_text())
    out = {}
    for kind in ('tied', 'untied'):
        a = data['models']['trained_' + kind]
        out[kind] = tuple(torch.tensor(v, dtype=torch.float32) for v in (
            a['encoder'], np.array(a['decoder']).T.tolist(), a['bias']))
    return out


def make_starts(mode, per_family=6):
    saved = saved_models()
    families = ('tied_warm', 'random_tilt', 'random_independent',
                'pentagon_tilt', 'free_preserve_diagonal', 'free_raw')
    starts = []
    for family in families:
        for k in range(per_family):
            seed = 1700 + 100 * families.index(family) + k
            g = torch.Generator().manual_seed(seed)
            perturb = (0.0, 0.03, 0.15, 0.3, 0.6, 1.0)[k % 6]
            length = (1.05, 1.2, 1.5, 2.0, 4.0, 8.0)[k % 6]
            if family == 'tied_warm':
                E, D, b = (v.clone() for v in saved['tied'])
                E = rotate(E, perturb * torch.randn(N, generator=g))
                D = rotate(D, perturb * torch.randn(N, generator=g))
            elif family.startswith('free_'):
                E, D, b = (v.clone() for v in saved['untied'])
                E = rotate(E, perturb * torch.randn(N, generator=g))
                D = rotate(D, perturb * torch.randn(N, generator=g))
            else:
                theta = 2 * math.pi * torch.rand(N, generator=g)
                if family == 'pentagon_tilt':
                    theta = 2 * math.pi * torch.arange(N) / N
                    theta = theta + 0.15 * torch.randn(N, generator=g)
                E = directions(theta)
                if family == 'random_independent':
                    D = length * directions(2 * math.pi * torch.rand(N, generator=g))
                    b = torch.full((N,), 0.05)
                else:
                    signs = 2 * (torch.rand(N, generator=g) > 0.5).float() - 1
                    D = length * directions(theta + signs * math.acos(1 / length))
                    b = torch.full((N,), -0.03)
                if mode == 'unit_decoder':
                    E, D = D, E
            # Rescaling the partner keeps each diagonal E_i dot D_i unchanged;
            # it does NOT preserve off-diagonal entries or the overall function.
            if family in ('tied_warm', 'free_preserve_diagonal'):
                if mode == 'unit_encoder':
                    D = D * E.norm(dim=0, keepdim=True)
                    E = unit(E)
                elif mode == 'unit_decoder':
                    E = E * D.norm(dim=0, keepdim=True)
                    D = unit(D)
            starts.append(dict(id=f'{family}_{k}', family=family, seed=seed,
                               initial_free_length=length if 'random' in family or 'pentagon' in family else None,
                               E=E, D=D, b=b))
    return starts


def make_controls(mode):
    saved = saved_models()
    out = []
    for k in range(12):
        g = torch.Generator().manual_seed(3000 + k)
        if k < 4:
            E, D, b = (v.clone() for v in saved['tied'])
            E += 0.1 * k * torch.randn(E.shape, generator=g)
            D += 0.1 * k * torch.randn(D.shape, generator=g)
            family = 'tied_warm'
        elif mode == 'free' and k < 8:
            E, D, b = (v.clone() for v in saved['untied'])
            E = rotate(E, 0.03 * (k - 4) * torch.randn(N, generator=g))
            D = rotate(D, 0.03 * (k - 4) * torch.randn(N, generator=g))
            family = 'free_warm'
        else:
            E = 0.4 * torch.randn(M, N, generator=g)
            D = E.clone() if mode == 'tied' else 0.4 * torch.randn(M, N, generator=g)
            b = torch.zeros(N)
            family = 'random'
        out.append(dict(id=f'{family}_{k}', family=family, seed=3000 + k,
                        E=E, D=D, b=b))
    return out


class Models(torch.nn.Module):
    def __init__(self, starts, mode, parameterization):
        super().__init__()
        self.mode, self.parameterization = mode, parameterization
        E = torch.stack([a['E'] for a in starts])
        D = torch.stack([a['D'] for a in starts])
        self.e = torch.nn.Parameter(angle(E) if mode == 'unit_encoder' and parameterization == 'angle' else E)
        if mode != 'tied':
            self.d = torch.nn.Parameter(angle(D) if mode == 'unit_decoder' and parameterization == 'angle' else D)
        self.b = torch.nn.Parameter(torch.stack([a['b'] for a in starts]))

    def weights(self):
        E = self.e
        if self.mode == 'unit_encoder':
            E = directions(E) if self.parameterization == 'angle' else unit(E)
        D = E if self.mode == 'tied' else self.d
        if self.mode == 'unit_decoder':
            D = directions(D) if self.parameterization == 'angle' else unit(D)
        return E, D, self.b

    def forward(self, x):
        E, D, b = self.weights()
        return torch.relu((x @ E.transpose(1, 2)) @ D + b[:, None, :])


@torch.no_grad()
def evaluate(E, D, b, x, chunk=4096):
    totals = torch.zeros(E.shape[0], dtype=torch.float64)
    for start in range(0, len(x), chunk):
        a = x[start:start + chunk]
        y = torch.relu((a @ E.transpose(1, 2)) @ D + b[:, None, :])
        totals += (y - a).square().mean(dim=2).double().sum(dim=1)
    return totals / len(x)


def save_json(path, payload):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def serialize(starts, mode, parameterization, E, D, b, losses, best_steps):
    out = []
    for i, a in enumerate(starts):
        meta = {k: v for k, v in a.items() if k not in ('E', 'D', 'b')}
        matrix = E[i].T @ D[i]
        norms_e, norms_d = E[i].norm(dim=0), D[i].norm(dim=0)
        cos = (E[i] * D[i]).sum(dim=0) / (norms_e * norms_d).clamp_min(1e-12)
        off = matrix[~torch.eye(N, dtype=torch.bool)]
        out.append(dict(**meta, mode=mode, parameterization=parameterization,
                        validation_mse=float(losses[i]), best_step=int(best_steps[i]),
                        E=E[i].tolist(), D=D[i].tolist(), b=b[i].tolist(),
                        own_responses=matrix.diag().tolist(),
                        write_norms=norms_e.tolist(), read_norms=norms_d.tolist(),
                        angles_deg=torch.rad2deg(torch.acos(cos.clamp(-1, 1))).tolist(),
                        asymmetry=float((matrix - matrix.T).norm() / (2 * matrix.norm().clamp_min(1e-12))),
                        offdiag_min=float(off.min()), offdiag_max=float(off.max())))
    return out


def train(starts, mode, parameterization, steps, batch, seed, path, validation,
          lr=0.003, check_every=5000, model_factory=None):
    model = (model_factory or Models)(starts, mode, parameterization)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        E, D, b = (v.detach().clone() for v in model.weights())
    best_loss = evaluate(E, D, b, validation)
    best_E, best_D, best_b = E.clone(), D.clone(), b.clone()
    best_steps = torch.zeros(len(starts), dtype=torch.int64)
    trace = []
    started = time.monotonic()
    for step in range(1, steps + 1):
        # Constant high rate for the first half, then cosine cooldown to 1%.
        progress = max(0, (step / steps - 0.5) / 0.5)
        rate = lr * (0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2)
        optimizer.param_groups[0]['lr'] = rate
        x = sample(batch, generator)
        losses = (model(x) - x).square().mean(dim=(1, 2))
        optimizer.zero_grad(set_to_none=True)
        # Sum independent model losses: each model has its ordinary MSE gradient.
        losses.sum().backward()
        optimizer.step()
        if step % check_every == 0 or step == steps:
            with torch.no_grad():
                E, D, b = (v.detach().clone() for v in model.weights())
            score = evaluate(E, D, b, validation)
            if not torch.isfinite(score).all():
                raise RuntimeError(f'Nonfinite loss in {mode}/{parameterization}')
            improved = score < best_loss
            best_loss[improved] = score[improved]
            best_E[improved], best_D[improved], best_b[improved] = E[improved], D[improved], b[improved]
            best_steps[improved] = step
            entry = dict(step=step, lr=rate, elapsed_seconds=time.monotonic()-started,
                         validation_mse=score.tolist(), best_validation_mse=best_loss.tolist())
            trace.append(entry)
            payload = dict(mode=mode, parameterization=parameterization, steps=steps,
                           completed_steps=step, batch_size=batch, training_seed=seed,
                           learning_rate=lr, schedule='flat half then cosine to 1%',
                           weight_decay=0, bias='learned output bias, no encoder bias',
                           trace=trace,
                           runs=serialize(starts, mode, parameterization, best_E, best_D,
                                          best_b, best_loss, best_steps))
            save_json(path, payload)
            print(f'{path.stem}: {step}/{steps}; val best={best_loss.min():.8f}, '
                  f'median={best_loss.median():.8f}; {entry["elapsed_seconds"]:.1f}s', flush=True)
    return payload


def from_record(a):
    meta = {k: a[k] for k in ('id', 'family', 'seed')}
    return dict(**meta, **{k: torch.tensor(a[k]) for k in ('E', 'D', 'b')})


def self_check():
    starts = make_starts('unit_encoder')[:2]
    model = Models(starts, 'unit_encoder', 'angle')
    g = torch.Generator().manual_seed(123)
    x = sample(64, g)
    y = model(x)
    E, D, b = model.weights()
    torch.testing.assert_close(E.norm(dim=1), torch.ones(2, N))
    reference = torch.stack([torch.relu((x @ E[i].T) @ D[i] + b[i]) for i in range(2)])
    torch.testing.assert_close(y, reference)
    batch_grad = torch.autograd.grad((y-x).square().mean(dim=(1, 2)).sum(), model.e)[0]
    separate = Models(starts[:1], 'unit_encoder', 'angle')
    grad = torch.autograd.grad((separate(x)-x).square().mean(), separate.e)[0]
    torch.testing.assert_close(batch_grad[:1], grad)
    for mode in ('unit_encoder', 'unit_decoder'):
        starts = make_starts(mode)
        raw = saved_models()['untied']
        a = next(z for z in starts if z['id']=='free_preserve_diagonal_0')
        torch.testing.assert_close((a['E']*a['D']).sum(dim=0), (raw[0]*raw[1]).sum(dim=0))
    print('Forward, independent-gradient, unit-norm, and warm-start checks passed.', flush=True)


def report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = {p.stem: json.loads(p.read_text()) for p in sorted(output.glob('explore_*.json'))}
    groups.update({p.stem: json.loads(p.read_text()) for p in sorted(output.glob('refine_*.json'))})
    for mode in ('tied', 'free', 'unit_encoder', 'unit_decoder'):
        if f'refine_{mode}' not in groups:
            raise RuntimeError(f'Finish refinement for {mode} before opening the test set')
    if any(group['completed_steps'] != group['steps'] for group in groups.values()):
        raise RuntimeError('Finish training and validation selection before opening the test set')
    winners = {}
    for mode in ('tied', 'free', 'unit_encoder', 'unit_decoder'):
        candidates = [(name, a) for name, group in groups.items() for a in group['runs'] if a['mode'] == mode]
        name, a = min(candidates, key=lambda t: t[1]['validation_mse'])
        winners[mode] = dict(a, source_group=name)
    for name, (E, D, b) in saved_models().items():
        winners['original_' + name] = dict(E=E.tolist(), D=D.tolist(), b=b.tolist())
    names = list(winners)
    weights = [torch.stack([torch.tensor(winners[name][key]) for name in names]) for key in ('E', 'D', 'b')]
    E, D, b = weights
    count, seed = 1000000, 914999
    x = sample(count, torch.Generator().manual_seed(seed))
    errors = np.empty((len(names), count), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, count, 8192):
            a = x[start:start+8192]
            y = torch.relu((a @ E.transpose(1, 2)) @ D + b[:, None, :])
            errors[:, start:start+len(a)] = (y-a).square().mean(dim=2).double().numpy()
    base = errors[0]
    base_mean = base.mean()
    active = (x > 0).sum(dim=1).numpy()
    for i, name in enumerate(names):
        err = errors[i]
        ratio = err.mean()/base_mean
        gain = 100*(1-ratio)
        # Delta-method uncertainty for the paired ratio of test-sample means.
        gain_se = 100*np.std(err-ratio*base, ddof=1)/math.sqrt(count)/base_mean
        winners[name].update(test_mse=float(err.mean()),
                             test_mse_se=float(err.std(ddof=1)/math.sqrt(count)),
                             improvement_percent=float(gain),
                             improvement_ci95_percent=[float(gain-1.96*gain_se), float(gain+1.96*gain_se)],
                             mse_by_active_count={str(k):float(err[active==k].mean()) for k in range(6) if (active==k).any()})
    summary = dict(n=N, m=M, sparsity=S,
                   objective='MSE averaged over examples and all five features; output ReLU; learned output bias; no encoder bias or weight decay',
                   validation_seed=914001, validation_samples=131072,
                   test_seed=seed, test_samples=count,
                   selection='Select checkpoints and one winner per constraint by validation MSE only; evaluate winners on one shared independent test draw',
                   uncertainty='Paired test-sampling 95% intervals, not uncertainty over training seeds or global optimality',
                   training_randomness='One fresh-data RNG stream per group, shared across its initializations; different groups/refinement use different streams',
                   source_pretraining='Saved tied and free warm starts have 30000 and 60000 preceding steps respectively; new random starts do not',
                   global_optimality_established=False,
                   counts={name:len(group['runs']) for name, group in groups.items()},
                   winners=winners)
    save_json(output/'summary.json', summary)
    labels = {'tied':'Tied', 'free':'Untied, unconstrained',
              'unit_encoder':'Unit writes (SAE decoder analogue)',
              'unit_decoder':'Unit reads'}
    lines = ['# One-sided unit-norm follow-up', '',
             'Five input features, two hidden dimensions, sparsity 0.9; output ReLU and learned output bias.',
             'MSE averages over examples and all five features.', '',
             '## Previous experiments', '',
             'The original notebook started from one trained tied solution, ran encoder-only normalization twice',
             '(once for its table and once for its plot), and ran one encoder-normalized warm start from a free solution.',
             'The displayed table reported about +1% improvement for the tied start and -32% for the free warm start.',
             'It also tested both-sided normalization and free models with/without weight decay.',
             'These were not a multi-start search; losses used separate fresh evaluation draws.', '',
             '## New search', '']
    for name, group in groups.items():
        lines.append(f'- `{name}`: {len(group["runs"])} runs, {group["steps"]:,} steps, batch {group["batch_size"]:,}, learning rate {group["learning_rate"]}.')
    lines += ['', 'One-sided starts cover tied warm starts, random directions with read/write lengths 1.05–8,',
              'independent random reads/writes, perturbed pentagons, and free-model warm starts with and without',
              'partner rescaling to preserve diagonal responses (off-diagonal responses still change).',
              'Both angle parameters and forward-normalized Cartesian parameters are tested.',
              'Adam uses a constant learning rate for half the run and cosine cooldown to 1% thereafter.',
              'All models within a group receive the same fresh training batches; different groups use different streams.',
              'Refinement includes the four lowest-validation-loss candidates and the best candidate from every initial family.',
              'Source tied/free warm starts have 30,000/60,000 preceding training steps; this is an optimization search, not a comparison of training efficiency.', '',
              '## Held-out results', '',
              'Checkpoint and model selection use 131,072 validation examples; these results use a separate shared draw of 1,000,000 examples.',
              'Improvement is relative to the validation-selected tied control, evaluated on those same test examples.', '',
              '| Constraint | Test MSE | Improvement vs. tied | Paired 95% interval |',
              '| --- | ---: | ---: | ---: |']
    for name in labels:
        a = winners[name]
        lo, hi = a['improvement_ci95_percent']
        precision = 4 if name.startswith('unit_') else 2
        interval = 'reference' if name == 'tied' else f'[{lo:+.{precision}f}%, {hi:+.{precision}f}%]'
        lines.append(f'| {labels[name]} | {a["test_mse"]:.6e} | {a["improvement_percent"]:+.{precision}f}% | {interval} |')
    lines += ['', 'Intervals quantify test-sampling error, not variation across training procedures or uncertainty about the global optimum.',
              'No global-optimality claim follows from these numerical searches.', '',
              'The SAE analogy concerns which vectors are normalized: toy encoder/write columns correspond to SAE decoder/dictionary columns.',
              'The objective here is toy feature-reconstruction MSE, not SAE activation-reconstruction loss plus an activation sparsity penalty.', '',
              '## Geometry of selected solutions', '',
              '| Constraint | Mean read/write angle | Free-side norm range | Mean self-response | Asymmetry |',
              '| --- | ---: | ---: | ---: | ---: |']
    for mode in ('unit_encoder', 'unit_decoder'):
        a = winners[mode]
        norms = a['read_norms'] if mode == 'unit_encoder' else a['write_norms']
        lines.append(f'| {labels[mode]} | {np.mean(a["angles_deg"]):.2f}° | {min(norms):.3f}–{max(norms):.3f} | {np.mean(a["own_responses"]):.3f} | {a["asymmetry"]:.4f} |')
    lines += ['', '## Interpretation', '',
              'These experiments test whether other initial configurations or parameterizations recover a one-sided advantage.',
              'The best one-sided solutions return to nearly aligned reads and writes; the larger free-side norms also accommodate self-responses above one and negative output biases.',
              'For example, unit writes with aligned reads of length 1.21 implement the same reconstruction operator as a tied model with all vectors of length $\\sqrt{1.21}$.',
              'Thus a free-side norm above one need not imply a directional benefit from untying.', '',
              'The poor free-model warm start in the original notebook was not evidence that optimization effects had been ruled out:',
              'normalizing a side changes the function, and the new warm starts that preserve diagonal responses can recover the tied-level solution.',
              'Preserving diagonal responses does not preserve the entire function or remove all optimization difficulties.', '',
              'One-sided normalization does not mathematically require alignment, and a broader search could still find better solutions.',
              'Even with both sides at unit norm, alignment follows only if the own-feature response is exactly one; unit norms alone do not force symmetry.', '']
    lines += ['', '## Files and reproduction', '',
              '- [Full result and selected weights](summary.json).',
              '- `explore_*.json` and `refine_*.json`: every run, validation trace, selected checkpoint weights, and exact training settings.',
              '- [Search plot](search_results.png): all exploratory validation results; held-out test results are in the table above.',
              '- [Experiment script](../one_sided_norm_experiments.py).', '',
              '```sh',
              'python results/core/tied_untied/one_sided_norm_experiments.py --phase explore',
              'python results/core/tied_untied/one_sided_norm_experiments.py --phase refine --steps 60000 --batch 4096',
              'MPLCONFIGDIR=/tmp/one-sided-norm-mpl python results/core/tied_untied/one_sided_norm_experiments.py --phase report',
              '```', '']
    (output/'REPORT.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    families = ('tied_warm', 'random_tilt', 'random_independent', 'pentagon_tilt', 'free_preserve_diagonal', 'free_raw')
    ticklabels = ('Tied\nwarm', 'Random\nangled', 'Random\nindependent', 'Pentagon\nangled', 'Free\ndiagonal kept', 'Free\nraw')
    for ax, mode in zip(axes, ('unit_encoder', 'unit_decoder')):
        for param, marker, color in [('angle','o','#226c9b'),('cartesian','x','#ca7530')]:
            runs = groups[f'explore_{mode}_{param}']['runs']
            for f, family in enumerate(families):
                subset = [a for a in runs if a['family']==family]
                shift = -0.08 if param=='angle' else 0.08
                jitter = np.linspace(-.1,.1,len(subset))
                ax.scatter(f+shift+jitter,[a['validation_mse'] for a in subset],
                           marker=marker,color=color,s=30,alpha=.8,label=param if f==0 else None)
        ax.axhline(winners['tied']['validation_mse'],color='#555555',lw=1,label='best tied')
        ax.axhline(winners['free']['validation_mse'],color='#51906b',lw=1,label='best free')
        ax.set_xticks(range(len(families)),ticklabels,fontsize=8)
        ax.set_title(labels[mode],fontsize=11)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Validation MSE (best checkpoint per run)')
    axes[1].legend(fontsize=8,frameon=False,loc='upper right')
    fig.tight_layout()
    fig.savefig(output/'search_results.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    for name in labels:
        a = winners[name]
        print(f'{labels[name]}: MSE={a["test_mse"]:.8f}, gain={a["improvement_percent"]:+.3f}%, CI={a["improvement_ci95_percent"]}',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('explore', 'refine', 'report', 'self-check'), default='explore')
    parser.add_argument('--output', type=Path, default=HERE/'one_sided_norm_results')
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--modes', nargs='+', choices=('tied', 'free', 'unit_encoder', 'unit_decoder'))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    self_check()
    if args.phase == 'self-check':
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == 'report':
        report(args.output)
        return
    val = sample(131072, torch.Generator().manual_seed(914001))
    if args.phase == 'explore':
        groups = [('tied', 'cartesian', make_controls('tied')),
                  ('free', 'cartesian', make_controls('free'))]
        for mode in ('unit_encoder', 'unit_decoder'):
            starts = make_starts(mode)
            groups.append((mode, 'angle', starts))
            groups.append((mode, 'cartesian', [a for i, a in enumerate(starts) if i % 3 == 0]))
        for i, (mode, param, starts) in enumerate(groups):
            if args.modes and mode not in args.modes:
                continue
            path = args.output/f'explore_{mode}_{param}.json'
            if path.exists() and json.loads(path.read_text())['completed_steps'] == args.steps:
                print(f'Skipping completed {path.name}', flush=True)
                continue
            train(starts, mode, param, args.steps, args.batch, 914100+i, path, val)
    else:
        for i, mode in enumerate(('tied', 'free', 'unit_encoder', 'unit_decoder')):
            if args.modes and mode not in args.modes:
                continue
            candidates = []
            for path in sorted(args.output.glob(f'explore_{mode}_*.json')):
                candidates.extend(json.loads(path.read_text())['runs'])
            candidates.sort(key=lambda a: a['validation_mse'])
            if not candidates:
                raise RuntimeError(f'Missing exploration for {mode}')
            chosen = candidates[:4]
            for family in dict.fromkeys(a['family'] for a in candidates):
                candidate = next(a for a in candidates if a['family'] == family)
                if candidate not in chosen:
                    chosen.append(candidate)
            starts = []
            for rank, a in enumerate(chosen):
                b = from_record(a)
                b['id'] = f'{a["parameterization"]}_{b["id"]}_refine_{rank}'
                b['source_parameterization'] = a['parameterization']
                starts.append(b)
            param = 'angle' if mode.startswith('unit_') else 'cartesian'
            path = args.output/f'refine_{mode}.json'
            if path.exists() and json.loads(path.read_text())['completed_steps'] == args.steps:
                print(f'Skipping completed {path.name}', flush=True)
                continue
            train(starts, mode, param, args.steps, args.batch, 914200+i, path, val, lr=0.001)


if __name__ == '__main__':
    main()
