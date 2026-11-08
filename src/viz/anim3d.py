"""Slow-rotating 3-D GIFs for slides.

The interactive plotly scenes in the dim-red notebooks are great on screen but do not embed in a
slide deck; a GIF does. So this draws a 3-D scene **once** and then sweeps the camera azimuth,
writing a GIF. The rotation carries **no scientific claim** — it only lets a 3-D trajectory be read
on a flat slide from more than one angle.

Colour-vision safe by construction (matches `src/viz/style.py`): sequential scalar colouring
(e.g. time) uses **cividis**; categorical series use **Okabe-Ito**; markers get a thin black edge.
``depthshade`` is turned OFF so a point keeps its colour as it rotates (depth-shading makes the
cloud flicker light/dark through the turn).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def rotating_3d_gif(
    panels: list[dict],
    out_path: str | Path,
    *,
    suptitle: str = "",
    colorbar_label: str | None = None,
    legend: list[tuple[str, str]] | None = None,
    n_frames: int = 90,
    elev: float = 18.0,
    azim0: float = -60.0,
    fps: int = 12,
    dpi: int = 95,
    figsize: tuple[float, float] | None = None,
    seq_cmap: str = "cividis",
    max_scatter: int = 600,
    seed: int = 0,
) -> Path:
    """Write a slow-rotating 3-D GIF with one or more panels that rotate in sync.

    Parameters
    ----------
    panels
        One dict per 3-D panel, laid out in a row. Each dict:

        * ``"series"`` : list of series dicts, each with
            - ``"xyz"``    : (N, 3) array of points.
            - ``"kind"``   : ``"scatter"`` | ``"line"`` | ``"linemarker"`` (default scatter).
            - ``"c"``      : (N,) scalar array -> mapped through ``seq_cmap``; OR a colour string.
            - ``"s"``      : marker size (scatter / linemarker).
            - ``"marker"`` : marker symbol (default ``"o"``).
            - ``"lw"``     : line width (line / linemarker).
            - ``"line_color"`` : connector colour for ``linemarker`` (default ``"0.8"``).
        * ``"axis_labels"`` : (x, y, z) strings.
        * ``"title"`` : panel title.
    colorbar_label
        If given and any series uses a scalar ``c``, draw one shared colourbar with this label.
        The scalar normalisation is shared across every panel so colours are comparable.
    legend
        Optional list of ``(label, marker)`` proxies drawn once (e.g. dPCA condition symbols).
    n_frames, fps
        ``n_frames`` covers a full 360 deg; ``fps`` sets playback speed. Default 90 frames @ 12 fps
        = one slow revolution in 7.5 s.

    Returns the written path.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import colormaps, colors
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3-D projection)

    from src.viz.style import OKABE_ITO

    rng = np.random.default_rng(seed)
    n = len(panels)
    figsize = figsize or (4.6 * n + 1.2, 4.9)
    fig = plt.figure(figsize=figsize)
    axes = [fig.add_subplot(1, n, j + 1, projection="3d") for j in range(n)]

    # one scalar normalisation across every panel, so the shared colourbar is honest
    scal = [np.asarray(s["c"]).ravel() for p in panels for s in p["series"]
            if isinstance(s.get("c"), np.ndarray)]
    norm = None
    if scal:
        allc = np.concatenate(scal)
        norm = colors.Normalize(vmin=float(allc.min()), vmax=float(allc.max()))
    cmap = colormaps[seq_cmap]

    for ax, panel in zip(axes, panels):
        for s in panel["series"]:
            xyz = np.asarray(s["xyz"], dtype=float)
            kind = s.get("kind", "scatter")
            c = s.get("c")
            is_scalar = isinstance(c, np.ndarray)

            if kind in ("line", "linemarker"):
                ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                        color=s.get("line_color", "0.8"), lw=s.get("lw", 2.0), zorder=1)

            if kind in ("scatter", "linemarker"):
                pts, cc = xyz, c
                if kind == "scatter" and xyz.shape[0] > max_scatter:
                    take = rng.choice(xyz.shape[0], max_scatter, replace=False)
                    pts = xyz[take]
                    cc = c[take] if is_scalar else c
                ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    c=(cc if is_scalar else None),
                    color=(None if is_scalar else (cc if cc is not None else OKABE_ITO[4])),
                    cmap=(cmap if is_scalar else None), norm=(norm if is_scalar else None),
                    s=s.get("s", 14), marker=s.get("marker", "o"),
                    edgecolors="black", linewidths=0.2, depthshade=False, zorder=3,
                )

        xl, yl, zl = panel.get("axis_labels", ("PC1", "PC2", "PC3"))
        ax.set_xlabel(xl, fontsize=8, labelpad=-6)
        ax.set_ylabel(yl, fontsize=8, labelpad=-6)
        ax.set_zlabel(zl, fontsize=8, labelpad=-6)
        ax.set_title(panel.get("title", ""), fontsize=10)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    if norm is not None and colorbar_label:
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02).set_label(colorbar_label)

    if legend:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker=m, color="0.5", lw=0, markerfacecolor="0.7",
                          markeredgecolor="black", markersize=7, label=lab)
                   for lab, m in legend]
        axes[-1].legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=0.99)

    def update(frame):
        for ax in axes:
            ax.view_init(elev=elev, azim=azim0 + 360.0 * frame / n_frames)
        return axes

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out_path
