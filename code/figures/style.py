"""Shared figure style: fixed entity->colour/marker mapping (validated categorical palette,
light mode; low-contrast hues always carry a distinct marker and a direct label)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

# entity -> (colour, marker); colour follows the entity in every figure
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
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
        "lines.markersize": 5, "savefig.dpi": 300, "figure.dpi": 150, "legend.frameon": False,
    })


def panel(ax, letter):
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right", color=INK)


def save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
