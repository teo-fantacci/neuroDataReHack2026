"""Colour-vision-deficiency-safe plotting style (tuned for red-green / deuteranopia).

Choices, and why:

* **Categorical -> Okabe-Ito.** An 8-colour qualitative palette explicitly designed to
  stay distinguishable under red-green colour-blindness (Okabe & Ito 2008). Used for HMM
  states, region lines, etc. We deliberately avoid the default matplotlib cycle, whose
  green/red pair is a classic confusion.
* **Sequential -> cividis.** Perceptually uniform *and* optimised so that a deuteranope
  and a normal-vision viewer see nearly the same image (Nunez, Anderton & Renslow 2018).
  Preferred over viridis here because it was designed for red-green colour-vision deficiency.
* **Diverging -> RdBu_r.** The confusion axis is red-green; a red-**blue** diverging map is
  safe because blue is well separated from red for deuteranopes. Used for signed
  quantities like net transition flux (blue = reverse, red = forward).

Nothing here is greyscale-only.
"""

from __future__ import annotations

from pathlib import Path

# Okabe-Ito, minus black (kept for text/axes). Order chosen for maximal separation.
OKABE_ITO = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black (last resort)
]

SEQUENTIAL = "cividis"   # heatmaps of a single non-negative quantity
DIVERGING = "RdBu_r"     # signed quantities (e.g. net flux); red-blue is CVD-safe here

# Total width of the black outline stroked under every line/marker. Our lines are ~1.8-2.0
# wide, so ~3.0 leaves a ~0.5 pt dark edge each side: enough to make yellow readable on
# white, not so much that thin lines turn black.
OUTLINE_LW = 3.0


def stroke(linewidth: float = OUTLINE_LW, foreground: str = "black"):
    """Black outline for ONE line: ``ax.plot(..., path_effects=stroke())``.

    Why the outline exists
    ----------------------
    Okabe-Ito is built for **hue** separation under red-green deficiency, not for
    **luminance** contrast against the page. Its yellow (``#F0E442``) is nearly invisible as
    a thin line on white — a legibility failure for every viewer. A dark edge restores the
    contrast *without altering a single hue*, so the colour-blind-safe property is preserved
    exactly. It also rescues light-on-dark overlays (e.g. a white reaction-time curve drawn
    over a cividis raster).

    Why it is NOT a global rcParam
    ------------------------------
    matplotlib's global ``path.effects`` is tempting and **wrong here**: it strokes *text*
    too, thickening every glyph until titles, ticks and legends look bold and smeared
    (verified — it wrecked the figures). Outline lines per-artist; let text alone.
    """
    import matplotlib.patheffects as pe

    return [pe.Stroke(linewidth=linewidth, foreground=foreground), pe.Normal()]


def set_style() -> None:
    """Apply the palette and a few sensible rcParams. Call once per session/notebook.

    Markers get a thin black edge globally (safe: it only touches marker outlines).
    **Lines** must be outlined per-artist with ``path_effects=stroke()`` -- see the note on
    ``stroke`` for why this is not done globally.
    """
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "axes.prop_cycle": cycler(color=OKABE_ITO),
        "image.cmap": SEQUENTIAL,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        # Marker edges are safe to set globally and already buy most of the legibility.
        "lines.markeredgecolor": "black",
        "lines.markeredgewidth": 0.6,
        "scatter.edgecolors": "black",
    })


def state_colors(n_states: int) -> list[str]:
    """Distinct CVD-safe colours for discrete HMM states (cycles past 8 if needed)."""
    return [OKABE_ITO[i % len(OKABE_ITO)] for i in range(n_states)]


# Stable region -> colour map, so a region keeps its colour across every figure.
_REGION_COLORS = {
    "VISp": OKABE_ITO[4],   # blue    (visual / sensory)
    "VISpm": OKABE_ITO[1],  # sky blue (higher-order visual)
    "VISam": OKABE_ITO[2],  # green
    "LP": OKABE_ITO[6],     # purple  (association thalamus)
    "MOs": OKABE_ITO[0],    # orange  (premotor)
    "SCm": OKABE_ITO[5],    # vermillion (motor)
    "MRN": OKABE_ITO[3],    # yellow  (motor)
}


def region_color(region: str) -> str:
    """Consistent colour for a brain region across all figures."""
    return _REGION_COLORS.get(region, OKABE_ITO[7])


# ---------------------------------------------------------------------------
# Structured figure saving: results/figures/<phase>/<name>.png
# Keeps the results tree tidy as the project grows so plots never overwrite.
# ---------------------------------------------------------------------------


def figure_path(phase: str, name: str, root: Path | str | None = None, ext: str = "png") -> Path:
    """Return (and create) a structured path: results/figures/<phase>/<name>.<ext>."""
    base = Path(root) if root is not None else _repo_root()
    out = base / "results" / "figures" / phase
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{name}.{ext}"


def savefig(fig, phase: str, name: str, root: Path | str | None = None, ext: str = "png") -> Path:
    """Save ``fig`` under the structured results tree and return the path."""
    path = figure_path(phase, name, root=root, ext=ext)
    fig.savefig(path)
    return path


def _repo_root() -> Path:
    # src/viz/style.py -> repo root is two parents up from src/
    return Path(__file__).resolve().parents[2]
