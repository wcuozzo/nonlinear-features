"""Regenerate the README's representation figures from ONE selected sweep.

Run from any directory:
    python results/core/nonlinear_representations/canonical_analysis.py

The adjacent canonical_source.json selects the store. This deliberately never
falls back to results_db or combines runs. results/REPRODUCING.md documents the
training procedure, source limitations, and how to replace the selected run.
"""
from pathlib import Path
import colorsys
import copy
import hashlib
import json
import sys

import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = json.loads(Path(__file__).with_name('canonical_source.json').read_text())
STYLE_PATH = Path(__file__).with_name('geometry_style.json')
GEOMETRY_STYLE = json.loads(STYLE_PATH.read_text())
STORE = ROOT / SOURCE['store']
CSV = STORE / 'compiled/sweep_results_precise.csv'
sys.path.insert(0, str(ROOT / 'lib'))
from core import Autoencoder
from contour_labels import label_magnitude, add_uphill_arrows


def load_frame():
    df = pd.read_csv(CSV)
    keys = ['n', 'm', 'l', 'S']
    if df.duplicated(keys).any() or not (df.mse_full > 0).all():
        raise ValueError('Expected unique configurations with positive MSE')
    if not (df.m < df.n).all() or not set(df.l) == {1, 2, 3, 4}:
        raise ValueError('Expected compressed configurations at depths 1–4')
    df['r'] = df.m / ((1-df.S)*df.n)
    df['y'] = df.mse_full / variance(df.S)
    return df


def variance(S):
    return (1-S)/3 - (1-S)**2/4


def model_path(n, m, l, S):
    return STORE / f'models/model_n{n}_m{m}_l{l}_S{S}.pt'


def load_model(n, m, l, S):
    model = Autoencoder(n, m, l, tied_weights=False).cpu()
    model.load_state_dict(torch.load(model_path(n, m, l, S), map_location='cpu', weights_only=True))
    return model.eval()


def fit_depth(d):
    sparsities = sorted(d.S.unique())
    indices = d.S.map({S: i for i, S in enumerate(sparsities)}).to_numpy()
    rates = d.r.to_numpy()
    target = np.log(d.mse_full.to_numpy())
    def predict(_, A, *cs):
        exponent = np.asarray(cs)[indices] * rates
        return np.log(variance(d.S.to_numpy())) - np.log1p(A*np.expm1(np.clip(exponent, None, 500)))
    params, _ = curve_fit(predict, rates, target, p0=[0.3]+[1.]*len(sparsities),
                          bounds=(1e-5, 1e3), maxfev=400000)
    prediction = predict(rates, *params)
    return dict(A=float(params[0]), c={str(S):float(c) for S,c in zip(sparsities,params[1:])},
                log_mse_r2=float(1-np.sum((target-prediction)**2)/np.sum((target-target.mean())**2)),
                configs=len(d))


def scaling_figure(df, fits):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True, layout='constrained')
    colors = plt.cm.viridis(np.linspace(.12, .85, df.S.nunique()))
    for l, ax in zip(range(1,5), axes):
        fit = fits[str(l)]
        d = df[df.l==l]
        xmax = 0
        for S,color in zip(sorted(d.S.unique()),colors):
            sub = d[d.S==S]
            E = sub.r*fit['c'][str(S)]
            xmax = max(xmax, E.max())
            ax.scatter(E,sub.y,s=22,color=color,label=f'S = {S}',alpha=.8)
        E=np.linspace(0,xmax*1.03,400)
        ax.plot(E,1/(1+fit['A']*np.expm1(E)),color='#262626',label='logistic fit')
        ax.set(yscale='log',xlabel='Effective rate c(S) · r',
               title=f"l = {l}  |  A = {fit['A']:.2f}\nlog-MSE R² = {fit['log_mse_r2']:.3f}")
        ax.grid(alpha=.15)
    axes[0].set_ylabel('MSE / source variance')
    axes[0].legend(fontsize=8)
    return fig


def sparsity_figure(fits, compression=1/8):
    """Separate sparsity's effects on rate, fitted exponent, and MSE scale."""
    sparsities = sorted(float(s) for s in fits['1']['c'])
    S = np.asarray(sparsities)
    rate = compression / (1-S)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained',
                             gridspec_kw={'width_ratios': [1, 1.5, 1]})
    fig.suptitle(f'Sparsity and the scaling law at fixed m/n = {compression:g}', fontsize=15)
    axes[0].plot(S, rate, 'o-', color='#555555', lw=2, ms=7)
    axes[0].set(title='Nominal rate increases', ylabel=r'$r = (m/n)/(1-S)$', ylim=(0, 2.9))
    axes[0].text(.05, .94, f'{rate[-1]/rate[0]:.2f}× from S = 0.85 to 0.95',
                 transform=axes[0].transAxes, va='top', fontsize=10)

    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    for (depth, fit), color, marker in zip(sorted(fits.items(), key=lambda item: int(item[0])),
                                          colors, ['o', 's', '^', 'D']):
        slopes = np.asarray([fit['c'][str(s)] for s in sparsities])
        exponent = slopes * rate
        axes[1].plot(S, exponent, marker=marker, color=color, lw=1.8, ms=7,
                     label=rf'$l={depth}$: {exponent[-1]/exponent[0]:.2f}×')
    axes[1].set(title='Fitted exponent increases more slowly',
                ylabel=r'$c_{l,S}\,r$', ylim=(0, 1.85))
    axes[1].legend(loc='lower right', ncol=2, fontsize=9, title='Change from S = 0.85 to 0.95',
                   title_fontsize=9, frameon=False)

    baseline = variance(S)
    axes[2].plot(S, baseline, 'o-', color='#555555', lw=2, ms=7)
    axes[2].set(title='Baseline MSE decreases', ylabel=r'$V(S)$', ylim=(0, .052))
    axes[2].text(.05, .94, f'{baseline[-1]/baseline[0]:.2f}× from S = 0.85 to 0.95',
                 transform=axes[2].transAxes, va='top', fontsize=10)
    for ax in axes:
        ax.set(xlabel=r'Sparsity $S$', xticks=S, xlim=(.843, .957))
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    return fig


def heatmap_figure(df):
    ns,ms,ss=sorted(df.n.unique()),sorted(df.m.unique()),sorted(df.S.unique())
    fig,axes=plt.subplots(len(ns),len(ms),figsize=(16,10.8),layout='constrained')
    fig.suptitle('MSE(l, S) heatmaps at fixed (n, m)',fontsize=14)
    norm=LogNorm(df.mse_full.min(),df.mse_full.max())
    cmap=plt.get_cmap('viridis')
    for i,n in enumerate(ns):
        for j,m in enumerate(ms):
            ax=axes[i,j]
            d=df[(df.n==n)&(df.m==m)]
            if d.empty:
                ax.axis('off');continue
            table=d.pivot(index='l',columns='S',values='mse_full').reindex(index=[1,2,3,4],columns=ss)
            im=ax.imshow(table,cmap=cmap,norm=norm,aspect='auto',origin='lower')
            ax.set_title(f'n = {n}, m = {m}',fontsize=10)
            ax.set_xticks(range(len(ss)),labels=[str(x) for x in ss],fontsize=8)
            ax.set_yticks(range(4),labels=[1,2,3,4],fontsize=8)
            if j==0:ax.set_ylabel('Depth l')
            if i==len(ns)-1:ax.set_xlabel('Sparsity S')
            for row in range(4):
                for col in range(len(ss)):
                    value=table.iloc[row,col]
                    # Select text against the rendered cell color, not a loss cutoff.
                    rgb=np.asarray(cmap(norm(value))[:3])
                    linear_rgb=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
                    luminance=linear_rgb @ np.array([.2126,.7152,.0722])
                    white_contrast=1.05/(luminance+.05)
                    black_contrast=(luminance+.05)/.05
                    color='white' if white_contrast>black_contrast else 'black'
                    ax.text(col,row,f'{value:.1e}',ha='center',va='center',fontsize=9,
                            color=color)
    fig.colorbar(im,ax=axes,label='MSE (log scale)',shrink=.85,pad=.015)
    return fig


def feature_paths(model):
    ts = torch.linspace(0, 1, 401)
    x = torch.eye(model.n)[:, None, :] * ts[None, :, None]
    with torch.no_grad():
        return model.encode(x.reshape(-1, model.n)).reshape(model.n, len(ts), 2).numpy()


def feature_speeds(model, points=1201):
    # Double precision keeps differencing from amplifying float32 rounding noise.
    encoder = copy.deepcopy(model).double()
    ts = torch.linspace(0, 1, points, dtype=torch.float64)
    x = torch.eye(model.n, dtype=torch.float64)[:, None, :] * ts[None, :, None]
    with torch.no_grad():
        paths = encoder.encode(x.reshape(-1, model.n)).reshape(model.n, points, 2).numpy()
    velocity = np.diff(paths, axis=1) / np.diff(ts.numpy())[None, :, None]
    return ts.numpy(), np.linalg.norm(velocity, axis=2)


def geometry(model, points=700, extent_multiplier=1.):
    paths = feature_paths(model)
    center = paths[0, 0]
    # A few long trajectories can be hundreds of times larger than the rest.
    # Show the central geometry explicitly, without rescaling individual features.
    radii = np.max(np.abs(paths - center), axis=(1, 2))
    half_width = max(float(np.quantile(radii, .8)) * 1.25, 1e-6) * extent_multiplier
    bounds = [center[0]-half_width, center[0]+half_width,
              center[1]-half_width, center[1]+half_width]
    xs = np.linspace(bounds[0], bounds[1], points)
    ys = np.linspace(bounds[2], bounds[3], points)
    xx, yy = np.meshgrid(xs, ys)
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    labels, magnitudes, l1_magnitudes, maxima = [], [], [], []
    with torch.no_grad():
        for chunk in grid.split(32768):
            output = model.decode(chunk)
            labels.append(output.argmax(1).numpy())
            magnitudes.append(torch.log1p(torch.linalg.vector_norm(output, dim=1)).numpy())
            l1_magnitudes.append(torch.log1p(output.abs().sum(dim=1)).numpy())
            maxima.append(output.max(1).values.numpy())
    return dict(paths=paths, center=center, bounds=bounds, xs=xs, ys=ys,
                labels=np.concatenate(labels).reshape(points, points),
                magnitude=np.concatenate(magnitudes).reshape(points, points),
                magnitude_l1=np.concatenate(l1_magnitudes).reshape(points, points),
                maxima=np.concatenate(maxima).reshape(points, points),
                half_width=half_width)


def palette(n):
    colors = np.array([colorsys.hsv_to_rgb(((i*37) % n)/n,
                         .7 + .05*((i//5) % 3 - 1),
                         .88 + .04*((i//8) % 3 - 1)) for i in range(n)])
    # Freeze the approved B1 mapping across every corresponding n=64 panel.
    if n == GEOMETRY_STYLE['palette_n']:
        colors = colors[GEOMETRY_STYLE['palette_indices']]
    return colors


def draw_regions(ax, data, n, contour_style=None):
    masked = np.ma.masked_where(data['maxima'] <= 0, data['labels'])
    cmap = ListedColormap(palette(n))
    cmap.set_bad('#eeeeee')
    ax.imshow(masked, origin='lower', extent=data['bounds'], cmap=cmap,
              vmin=0, vmax=n-1, interpolation='nearest', aspect='equal')
    use_l1 = contour_style is not None and contour_style.get('norm') == 'l1'
    magnitude = data['magnitude_l1'] if use_l1 else data['magnitude']
    if contour_style is None:
        levels = np.linspace(magnitude.min(), magnitude.max(), 20)[1:-1]
        color, linewidth, alpha = '#222222', .5, .65
    else:
        if 'contour_interval' in contour_style:
            step = float(contour_style['contour_interval'])
            if not np.isfinite(step) or step <= 0:
                raise ValueError('Contour interval must be finite and positive')
            raw_min, raw_max = np.expm1([magnitude.min(), magnitude.max()])
            raw_levels = np.arange(np.floor(raw_min/step)+1, np.ceil(raw_max/step))*step
            levels = np.log1p(raw_levels)
        else:
            levels = np.linspace(magnitude.min(), magnitude.max(), contour_style['contour_count']+3)[2:-1]
        color = contour_style['contour_color']
        if contour_style.get('higher_is_darker'):
            grays = np.linspace(*contour_style['gray_range'], len(levels))
            color = np.repeat(grays[:, None], 3, axis=1)
            data['contour_tones'] = [dict(raw_value=float(np.expm1(v)),gray=float(g))
                                     for v,g in zip(levels,grays)]
        linewidth, alpha = contour_style['contour_linewidth'], contour_style['contour_alpha']
    if len(levels) and magnitude.min() < magnitude.max():
        contours = ax.contour(data['xs'], data['ys'], magnitude, levels=levels,
                              colors=color, linewidths=linewidth, alpha=alpha)
        if contour_style is not None and contour_style.get('labels'):
            labels = label_magnitude(ax, contours, data['bounds'], region_labels=data['labels'],
                                     feature_colors=palette(n), **contour_style['labels'])
            data['contour_labels'] = labels
            if contour_style.get('direction_marks'):
                data['direction_marks'] = add_uphill_arrows(ax, contours, magnitude,
                    data['xs'], data['ys'], labels, **contour_style['direction_marks'])
    ax.set_xlim(data['bounds'][:2])
    ax.set_ylim(data['bounds'][2:])
    ax.set_box_aspect(1)
    ax.tick_params(labelsize=9)


def geometry_figures():
    # The original overview emphasized the nonlinear depths; the expanded gallery
    # retains l=1 as well. Spatial rows share exactly the same viewport per column.
    fig, axes = plt.subplots(3, 3, figsize=(12, 12.8))
    fig.subplots_adjust(left=.09, right=.98, bottom=.055, top=.89, wspace=.35, hspace=.36)
    fig.suptitle('m = 2 visualizations  (n = 64, S = 0.95)', fontsize=16, y=.98)
    used = []
    all_speeds = []
    for j, l in enumerate([2, 3, 4]):
        model = load_model(64, 2, l, .95)
        used.append(model_path(64, 2, l, .95))
        data = geometry(model)
        ax = axes[0, j]
        for path, color in zip(data['paths'], palette(64)):
            ax.plot(path[:, 0], path[:, 1], color=color, lw=1.25, alpha=.95)
            ax.plot(*path[-1], 'o', color=color, markersize=4.2,
                    markeredgecolor='#444444', markeredgewidth=.35)
        ax.plot(*data['center'], '+', color='#222222', markersize=6, mew=1.2)
        ax.set_xlim(data['bounds'][:2])
        ax.set_ylim(data['bounds'][2:])
        ax.set_aspect('equal')
        ax.set_box_aspect(1)
        ax.set_title(f'l = {l}', fontsize=13, pad=8)
        ax.set_xlabel('z₁', fontsize=11)
        ax.set_ylabel('z₂', fontsize=11)
        ax.grid(alpha=.2, linewidth=.5)
        ax.tick_params(labelsize=9)
        ts, speeds = feature_speeds(model)
        all_speeds.append(speeds)
        speed_ax = axes[1, j]
        for speed, color in zip(speeds, palette(64)):
            speed_ax.stairs(speed, ts, baseline=None, color=color, lw=.9, alpha=.85)
        speed_ax.set_yscale('log')
        speed_ax.set_xlim(0, 1)
        speed_ax.set_xlabel('Input value t', fontsize=11)
        speed_ax.set_ylabel(r'$\|dz(t e_i)/dt\|_2$ (log scale)', fontsize=10)
        speed_ax.set_box_aspect(1)
        speed_ax.grid(alpha=.2, linewidth=.5)
        speed_ax.tick_params(labelsize=9)
        draw_regions(axes[2, j], data, 64)
        axes[2, j].set_xlabel('z₁', fontsize=11)
        axes[2, j].set_ylabel('z₂', fontsize=11)
        if l == 4:
            hero_style = GEOMETRY_STYLE['hero']
            hero_data = geometry(model, points=hero_style['grid_points'],
                                 extent_multiplier=hero_style['extent_multiplier'])
            hero, hero_ax = plt.subplots(figsize=(7, 7), layout='constrained')
            draw_regions(hero_ax, hero_data, 64, contour_style=hero_style)
            hero_ax.set(xlabel='z₁', ylabel='z₂')
            hero_ax.set_title('n = 64, m = 2, L = 4, S = 0.95', fontsize=11)
            hero_ax.set_xlabel('z₁', fontsize=9)
            hero._geometry_annotations = dict(labels=hero_data.get('contour_labels',[]),
                                              direction_marks=hero_data.get('direction_marks',[]),
                                              contour_tones=hero_data.get('contour_tones',[]))
    speeds = np.concatenate(all_speeds)
    if not np.all(np.isfinite(speeds)) or np.any(speeds <= 0):
        raise ValueError('Logarithmic speed panels require finite, positive speeds; do not hide zero-speed intervals.')
    speed_limits = (10.**np.floor(np.log10(speeds.min())),
                    10.**np.ceil(np.log10(speeds.max())))
    for ax in axes[1]:
        ax.set_ylim(speed_limits)
    # Row headings describe the view; axis labels describe the plotted coordinates.
    fig.canvas.draw()
    for row, title in enumerate(['Feature Trajectories', 'Feature Speeds', 'Decoder Regions']):
        bounds = axes[row, 1].get_position()
        # Keep headings close to their own plots, with the larger gap above them.
        offset = .035 if row == 0 else 6 / (72 * fig.get_figheight())
        fig.text(bounds.x0 + bounds.width / 2, bounds.y1 + offset, title,
                 ha='center', va='bottom', fontsize=14, fontweight='normal')
    return fig, hero, used


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    df=load_frame()
    # Every table row must have a checkpoint compatible with the untied architecture.
    for row in df.itertuples():load_model(row.n,row.m,row.l,row.S)
    fits={str(l):fit_depth(df[df.l==l]) for l in range(1,5)}
    out=ROOT/'figures'
    geometry_plot,hero,used=geometry_figures()
    plots={'fig_canonical_scaling.png':scaling_figure(df,fits),
           'fig_scaling_sparsity.png':sparsity_figure(fits),
           'fig_mse_heatmap.png':heatmap_figure(df),
           'fig_m2_visualizations.png':geometry_plot,
           'fig_hero_argmax_regions.png':hero}
    for name,fig in plots.items():
        fig.savefig(out/name,dpi=160);plt.close(fig)
    manifest=STORE/'manifest.json'
    report=dict(source=SOURCE,source_manifest=json.loads(manifest.read_text()) if manifest.exists() else None,
                configs=len(df),fits=fits,
                geometry=dict(depths=[2,3,4], center="z(0)", extent_quantile=.8, margin=1.25,
                              contour="log1p(L2 norm of decoder output)", grid_points=700,
                              speed=dict(quantity="L2 norm of dz(t e_i)/dt", samples=1201,
                                         method="float64 finite differences", scale="log; shared across depths",
                                         normalization="none"),
                              hero_contour=f"log1p({GEOMETRY_STYLE['hero']['norm'].upper()} norm); raw norm labels near the perimeter",
                              rendering_style=GEOMETRY_STYLE,
                              hero_annotations=hero._geometry_annotations),
                inputs={str(p.relative_to(ROOT)):digest(p) for p in [CSV,STYLE_PATH,*used]},
                outputs={name:digest(out/name) for name in plots})
    (out/'canonical_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'source':SOURCE,'configs':len(df),'fits':fits},indent=2))


if __name__=='__main__':main()
