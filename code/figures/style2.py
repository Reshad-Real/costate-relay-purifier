"""Figure style for the two-column Elsevier layout. Figures are drawn at their printed size
(single column 3.45 in, double column 7.1 in) so that 8-pt text stays 8 pt in the PDF."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SINGLE = 3.45
DOUBLE = 7.10
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

# entity -> (colour, marker): validated categorical palette; low-contrast hues carry markers/labels
ENT = {
    "MPC-FF": ("#2a78d6", "o"),
    "Relay": ("#eb6834", "s"),
    "Costate-FB": ("#1baf7a", "D"),
    "MPC-FB": ("#eda100", "^"),
    "PI": ("#e87ba4", "v"),
    "AO": ("#008300", "P"),
    "MPC-oracle": ("#4a3aa7", "*"),
    "Timer+Relay-PM": ("#e34948", "X"),
}


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.2, "lines.markersize": 3.6,
        "savefig.dpi": 600, "figure.dpi": 150, "legend.frameon": False, "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.2, "legend.columnspacing": 1.0, "axes.titlepad": 3,
        "axes.labelpad": 2, "pdf.fonttype": 42,
    })


def panel(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            va="bottom", ha="right", color=INK)


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
