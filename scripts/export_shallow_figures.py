"""Recreate the README's five shallow-model diagrams from saved weights.

Run from the repository root:
    python scripts/export_shallow_figures.py
Use --output-dir to preview the figures elsewhere. No training is performed.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
COLORS = plt.get_cmap('tab10').colors
INK = '#19232e'
GRAY = '#d7dce3'

# Annotation positions reproduce the label arrangement of the README diagrams.
LABELS = {
    ('E', 0): (1.20, -.54), ('E', 1): (1.08, .82),
    ('E', 2): (-.19, 1.29), ('E', 3): (.24, -1.30),
    ('E', 4): (-1.12, -.82), ('D', 0): (1.25, .20),
    ('D', 1): (.44, 1.29), ('D', 2): (-1.16, .79),
    ('D', 3): (-.45, -1.34), ('D', 4): (-1.30, -.15),
}


def load_vectors():
    saved = json.loads((ROOT / 'figures/untied_read_write_example.json').read_text())
    model = saved['models']['trained_untied']
    # Each column is a vector in the two-dimensional bottleneck.
    return np.asarray(model['encoder']), np.asarray(model['decoder']).T


def vector_panel(ax, writes, reads, active=None, compact=False):
    ax.set_aspect('equal')
    ax.set(xlim=(-1.62, 1.62), ylim=(-1.58, 1.58),
           xticks=[-1, 0, 1], yticks=[-1, 0, 1], xlabel='$z_1$', ylabel='$z_2$')
    ax.axhline(0, color='#eceef2', lw=1, zorder=0)
    ax.axvline(0, color='#eceef2', lw=1, zorder=0)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=GRAY, lw=1.2))
    for spine in ax.spines.values():
        spine.set_color('#cdd3dc')
    ax.tick_params(color='#adb7c6')
    for kind, vectors in [('E', writes), ('D', reads)]:
        for i, vector in enumerate(vectors.T):
            selected = active is None or (kind, i) in active
            norm = np.linalg.norm(vector)
            direction = vector / norm
            ax.plot([0, direction[0]], [0, direction[1]],
                    color=COLORS[i] if selected else GRAY,
                    ls='-' if kind == 'E' else '--',
                    lw=2.8 if selected else 1.5, alpha=1 if selected else .7,
                    zorder=3 if selected else 1)
            ax.plot(*direction, 'o' if kind == 'E' else 's',
                    color=COLORS[i] if selected else GRAY,
                    ms=7 if selected else 4, zorder=4 if selected else 2)
            if selected:
                position = LABELS[kind, i]
                if compact and (kind, i) == ('E', 4):
                    position = (-1.10, -.82)
                ax.annotate(f'${kind}_{i}$\n{norm:.4g}', xy=direction,
                            xytext=position, ha='center', va='center', color=INK,
                            fontsize=13, linespacing=1.2,
                            arrowprops=dict(arrowstyle='-', color='#c1c9d4', lw=.8))


def pair_angle(a, b):
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def motif(writes, reads, active, sections):
    fig = plt.figure(figsize=(12.8, 5.2))
    ax = fig.add_axes((.07, .15, .34, .82))
    vector_panel(ax, writes, reads, active, compact=True)
    text = fig.add_axes((.53, .03, .45, .93))
    text.axis('off')
    for heading, y, rows in sections:
        text.text(0, y, heading, fontsize=14, color=INK, va='top')
        for label, value, row_y, detail in rows:
            text.text(0, row_y, label, fontsize=15, va='top')
            text.text(.98, row_y, f'{value:+.4g}'.replace('-', '−'),
                      fontsize=15, va='top', ha='right')
            if detail:
                text.text(0, row_y - .08, detail, fontsize=12,
                          color='#4d596b', va='top')
    return fig


def shallow_figures():
    writes, reads = load_vectors()
    E, D = writes.T, reads.T
    fig, ax = plt.subplots(figsize=(7, 7.2))
    vector_panel(ax, writes, reads)
    ax.set_title(r'$l=1,\ n=5,\ m=2,\ S=0.9$', pad=20)
    figures = {'fig_untied_read_write.png': fig}
    figures['fig_untied_motif_directions.png'] = motif(writes, reads,
        {('E', 0), ('E', 1), ('E', 3), ('D', 1), ('D', 3)}, [
            ('Nearly perpendicular to other reads', .93, [
                ('$E_0 \\cdot D_1$', E[0] @ D[1], .81, None),
                ('$E_0 \\cdot D_3$', E[0] @ D[3], .66, None)]),
            ('Still overlaps other writes', .44, [
                ('$E_0 \\cdot E_1$', E[0] @ E[1], .32, None),
                ('$E_0 \\cdot E_3$', E[0] @ E[3], .17, None)]),
        ])
    figures['fig_untied_motif_magnitudes.png'] = motif(writes, reads,
        {('E', 3), ('E', 4), ('D', 3), ('D', 4)}, [
            ('Small vectors, nearly aligned', .95, [
                ('$E_4 \\cdot D_3$', E[4] @ D[3], .83,
                 f'Angle: {pair_angle(E[4], D[3]):.1f}°')]),
            ('Large vectors, nearly perpendicular', .59, [
                ('$E_3 \\cdot D_4$', E[3] @ D[4], .47,
                 f'Angle: {pair_angle(E[3], D[4]):.4f}°')]),
            ('Own-feature responses stay near one', .24, [
                ('$E_3 \\cdot D_3$', E[3] @ D[3], .12, None),
                ('$E_4 \\cdot D_4$', E[4] @ D[4], -.02, None)]),
        ])
    figures['fig_untied_motif_asymmetry.png'] = motif(writes, reads,
        {('E', 0), ('E', 2), ('D', 0), ('D', 2)}, [
            ('Feature 0 strongly suppresses read 2', .93, [
                ('$E_0 \\cdot D_2$', E[0] @ D[2], .81, None)]),
            ('Feature 2 barely affects read 0', .59, [
                ('$E_2 \\cdot D_0$', E[2] @ D[0], .47, None)]),
            ('Own-feature responses stay near one', .24, [
                ('$E_0 \\cdot D_0$', E[0] @ D[0], .12, None),
                ('$E_2 \\cdot D_2$', E[2] @ D[2], -.02, None)]),
        ])
    return figures


def pentagon_figure():
    checkpoint = ROOT / 'results/exploratory/seed_models/pentagon_n5_m2_l1_S0.95.pt'
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)['encoder.weight'].numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set(aspect='equal', xlim=(-1.6, 1.6), ylim=(-1.6, 1.6),
           xlabel='$z_1$', ylabel='$z_2$',
           title='n=5, m=2, l=1, S=0.95 — encoder columns')
    ax.grid(alpha=.12, lw=.5)
    ax.axhline(0, color='#cccccc', lw=.5)
    ax.axvline(0, color='#cccccc', lw=.5)
    order = np.argsort(np.arctan2(weights[1], weights[0]))
    polygon = weights[:, np.r_[order, order[0]]]
    ax.plot(*polygon, '--', color='#cccccc', lw=.8)
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, 5))
    for i, vector in enumerate(weights.T):
        ax.annotate('', xy=vector, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color=colors[i], lw=2))
        ax.scatter(*vector, color=colors[i], s=55, zorder=3)
        ax.text(*(vector * 1.20), f'$f_{i}$', color=colors[i],
                ha='center', va='center', fontsize=12)
    return fig


def export(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.size': 12, 'axes.titlesize': 14,
                         'axes.labelsize': 13, 'savefig.facecolor': 'white'}):
        figures = shallow_figures()
    with plt.rc_context({'font.size': 9, 'axes.titlesize': 10}):
        figures['fig_pentagon_n5.png'] = pentagon_figure()
    for name, fig in figures.items():
        fig.savefig(output_dir / name, dpi=180, bbox_inches='tight', pad_inches=.10)
        plt.close(fig)
    return list(figures)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'figures')
    args = parser.parse_args()
    for filename in export(args.output_dir):
        print(args.output_dir / filename)
