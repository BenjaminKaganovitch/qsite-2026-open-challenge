"""Shared matplotlib style: phase colours (validated categorical slots 1-3 + neutral), overlays."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import annni_core as ac

PHASE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#b9b8b3"]  # ferro, antiphase, para, unpolarised
PHASE_CMAP = ListedColormap(PHASE_COLORS)
PHASE_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.titlesize": 9.5})


def overlays(ax, lw=1.2, legend=True, pe=True):
    k = np.linspace(0, 1, 400)
    ax.plot(k, ac.h_ising(k), color=INK, lw=lw, ls="-", label="Ising (2nd-order est.)")
    ax.plot(k, ac.h_pt_lower(k), color=INK, lw=lw, ls="--", label="lower PT fit")
    ax.plot(k, ac.h_kt_upper(k), color=INK, lw=lw, ls=":", label="upper KT fit")
    if pe:
        kk = np.linspace(0.13, 0.5, 200)
        ax.plot(kk, ac.h_peschel_emery(kk), color=INK2, lw=0.8, ls="-.", label="Peschel-Emery disorder line")
    ax.set_xlim(0, 1); ax.set_ylim(0, 2)
    if legend:
        ax.legend(loc="upper left", fontsize=6.5, frameon=False)


def phase_map(ax, K, H, lab, amb=None, title="", mod=None):
    dk, dh = K[1] - K[0], H[1] - H[0]
    ax.imshow(lab.T, origin="lower", extent=[K[0] - dk / 2, K[-1] + dk / 2, H[0] - dh / 2, H[-1] + dh / 2],
              cmap=PHASE_CMAP, norm=PHASE_NORM, aspect="auto", interpolation="nearest")
    KK, HH = np.meshgrid(K, H, indexing="ij")
    if amb is not None and amb.any():
        ax.scatter(KK[amb], HH[amb], s=4, marker="x", color=INK, lw=0.5, label="ambiguous (threshold ±0.1)")
    if mod is not None and mod.any():
        ax.scatter(KK[mod], HH[mod], s=2.5, marker="o", color="white", lw=0, label="q* ∉ {0, π/2}")
    ax.set_xlabel("κ"); ax.set_ylabel("h"); ax.set_title(title)


def phase_legend(fig, loc="lower center", ncol=4, extra=(), y=0.0):
    from matplotlib.patches import Patch
    names = ["ferro", "antiphase", "paramagnet", "unpolarised (order & m_x low)"]
    h = [Patch(color=c, label=n) for c, n in zip(PHASE_COLORS, names)] + list(extra)
    fig.legend(handles=h, loc=loc, ncol=ncol, fontsize=7, frameon=False, bbox_to_anchor=(0.5, y))
