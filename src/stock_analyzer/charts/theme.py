import matplotlib as mpl
from ..tokens import COLORS

def apply_theme() -> None:
    mpl.rcParams.update({
        "figure.facecolor": COLORS["bg"], "axes.facecolor": COLORS["bg"],
        "axes.edgecolor": COLORS["hairline"], "axes.grid": True,
        "grid.color": COLORS["hairline"], "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"], "font.size": 9,
    })
