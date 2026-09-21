"""Render the README architecture diagram; run this file from any directory."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[3]
    fig, ax = plt.subplots(figsize=(16, 6.5))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.03, top=0.98)
    ax.set(xlim=(0, 14), ylim=(0, 6.5))
    ax.axis("off")

    blue, orange = "#245589", "#b64f2e"
    xs = [0.65, 2.5, 4.75, 7, 9.25, 11.5, 13.35]
    wide = np.linspace(1.65, 4.3, 6)
    narrow = np.array([2.5, 3.45])
    ys = [wide, wide, wide, narrow, wide, wide, wide]
    colors = ["#a15cb2", "#5095d1", "#5095d1", "#d86d49",
              "#5095d1", "#5095d1", "#5cb461"]

    for i in range(6):
        projection = i == 2
        for y1 in ys[i]:
            for y2 in ys[i + 1]:
                ax.plot([xs[i], xs[i + 1]], [y1, y2],
                        color=orange if projection else "#91979d",
                        alpha=0.48 if projection else 0.32,
                        lw=1.05 if projection else 0.8,
                        ls="--" if i in (1, 4) else "-", zorder=1)
    for x, yy, color in zip(xs, ys, colors):
        ax.scatter(np.full(len(yy), x), yy, s=340, facecolor=color,
                   edgecolor="#30363b", linewidth=1.2, zorder=3)
    for x in (3.625, 10.375):
        ax.text(x, 2.975, "…", ha="center", va="center", fontsize=28,
                color="#70777d", backgroundcolor="white", zorder=4)

    def bracket(x1, x2, y, label):
        ax.plot([x1, x1, x2, x2], [y - 0.16, y, y, y - 0.16],
                color=blue, lw=1.5)
        ax.text((x1 + x2) / 2, y + 0.23, label, ha="center",
                va="bottom", fontsize=17, weight="bold", color=blue)

    bracket(xs[0], xs[3], 5.93, r"Encoder: $l$ Linear Layers")
    bracket(xs[3] + 0.25, xs[-1], 5.93, r"Decoder: $l$ Linear Layers")
    ax.text(2.6, 5.43, r"$l-1$ Hidden Layers", ha="center", fontsize=14,
            color=blue, weight="bold")
    ax.text(2.6, 5.05, "Linear + Bias + ReLU", ha="center", fontsize=13,
            color="#454e57")

    ax.text(6.15, 5.43, "Final Encoder Layer", ha="center", fontsize=14,
            color=orange, weight="bold")
    ax.text(6.15, 5.05, r"Linear ($n \to m$)", ha="center", fontsize=13,
            color=orange)
    ax.text(6.15, 4.69, "No Bias · No ReLU", ha="center", fontsize=13,
            color=orange, weight="bold")
    ax.annotate("", xy=(6.15, 3.88), xytext=(6.15, 4.51),
                arrowprops=dict(arrowstyle="->", color=orange, lw=1.3))

    ax.text(10.65, 5.43, "Linear + Bias + ReLU", ha="center", fontsize=14,
            color=blue, weight="bold")
    ax.text(10.65, 5.05, "At Every Layer, Including the Output", ha="center",
            fontsize=12.5, color="#454e57")

    labels = ["Input\n" + r"$x \in \mathbb{R}^{n}$",
              "Hidden\n" + r"$(n)$", "Hidden\n" + r"$(n)$",
              "Bottleneck\n" + r"$z \in \mathbb{R}^{m}$",
              "Hidden\n" + r"$(n)$", "Hidden\n" + r"$(n)$",
              "Output\n" + r"$\hat{x} \in \mathbb{R}^{n}$"]
    for x, label in zip(xs, labels):
        ax.text(x, 0.98, label, ha="center", va="top", fontsize=14,
                linespacing=1.35, color="#22272c")
    ax.text(7, 0.12,
            r"At $l=1$, the hidden layers are omitted: "
            r"$x\;\longrightarrow\;W_{\mathrm{enc}}x=z"
            r"\;\longrightarrow\;\mathrm{ReLU}(W_{\mathrm{dec}}z+b)=\hat{x}$.",
            ha="center", fontsize=12, color="#555e67")
    fig.savefig(root / "figures/fig_architecture.png", dpi=160,
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
