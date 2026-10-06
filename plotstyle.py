"""Common figure style: palette and panel labels placed below each panel."""

import matplotlib.pyplot as plt

STYLE = {   # method: (colour, marker, line style)
    "PGS (ours)": ("#2a78d6", "o", "-"),
    "DRL-Q":      ("#eb6834", "s", "--"),
    "SAS":        ("#1baf7a", "^", "--"),
    "LL":         ("#eda100", "D", "--"),
    "JSQ-speed":  ("#e87ba4", "v", ":"),
    "RR":         ("#008300", "P", ":"),
    "Random":     ("#4a3aa7", "X", ":"),
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 2, "lines.markersize": 5, "font.family": "DejaVu Sans"})


def panel_labels(fig, axes, pad=0.02):
    """Writes (a), (b), ... centred below each panel, on one common baseline."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    h = fig.bbox.height
    y = min(ax.get_tightbbox(r).y0 for ax in axes) / h - pad
    for i, ax in enumerate(axes):
        p = ax.get_position()
        fig.text((p.x0 + p.x1) / 2, y, f"({chr(97 + i)})", ha="center", va="top", fontsize=10)


def save(fig, path_without_ext):
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_without_ext}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
