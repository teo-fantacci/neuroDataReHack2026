"""Animations of the rSLDS regime SEQUENCE. Illustration, built on the certified result.

Scientific guardrail (read before using)
-----------------------------------------
The certified rSLDS result is the **sequence**: the population passes through a reliable,
position-dependent ordering of metastable regimes (recurrence test + order_consistency, both
backends, both alignments -- see docs/worklog/2026-07-15_rslds-first-real-test.md). The
**geometry** of each regime's flow (attractor vs line-attractor/integrator) is NOT certified:
the |lambda|-stability test showed |lambda| is >97% reproduced on time-shuffled data, i.e. it is
the smoother, not the dynamics.

So these animations show *when the population switches regime and in what order*. The latent
plane is drawn only so the switches have somewhere to happen; **its shape carries no claim**, and
the panels say so. The load-bearing content is the regime tape and the switch times.

Two builders
------------
* :func:`single_trial_gif`  -- one representative trial: a marker walks each area's latent path,
  flashing on every regime change, with a 3-lane regime tape building up underneath. Matches the
  "watch one decision happen" intuition; noisier, because a single trial is a single sample.
* :func:`occupancy_sweep_gif` -- the trial-AVERAGED view: a time cursor sweeps the regime-occupancy
  of all three areas at once, while a switch-time density builds up beneath it. Cleaner, and it is
  the *certified* quantity (occupancy and switch timing are averages over all trials, not one
  noisy path). This is the one to trust for "which area switches when".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.viz.style import OKABE_ITO, region_color, state_colors, stroke


# --------------------------------------------------------------------------- helpers


def _collapse(row: np.ndarray) -> list[int]:
    """Regime visit sequence with consecutive repeats removed."""
    out = [int(row[0])]
    for s in row[1:]:
        if int(s) != out[-1]:
            out.append(int(s))
    return out


def _switch_times(states_1d: np.ndarray, bc: np.ndarray) -> np.ndarray:
    """Times (s) at which the regime changes within one trial."""
    idx = np.flatnonzero(states_1d[1:] != states_1d[:-1]) + 1
    return bc[idx]


def modal_sequence(states: np.ndarray) -> tuple[int, ...]:
    """The most common regime visit-sequence across trials (repeats collapsed)."""
    from collections import Counter

    return Counter(tuple(_collapse(r)) for r in states).most_common(1)[0][0]


def pick_representative_trial(
    fits: dict,
    *,
    rt: np.ndarray | None = None,
    bin_centers: np.ndarray | None = None,
    max_switches: int = 6,
    richness_bonus: float = 0.4,
) -> int:
    """Index of a trial that follows the modal regime sequence in as many areas as possible.

    Scoring, per trial:

    * **+1** for each area whose collapsed sequence equals that area's modal sequence (the
      "representative" part);
    * **+``richness_bonus``** for each area that shows at least one *post-onset* switch (so the
      chosen trial actually illustrates switching in all three areas where possible, rather than a
      trial where a weak-sequence area like VISp happens to sit still);
    * **−0.5** if any area flickers (> ``max_switches`` changes) — flicker is not a clean sequence.

    Ties break toward the median-reaction-time trial (if ``rt`` given), i.e. a *typical* one. The
    richness bonus is small on purpose: it nudges toward an illustrative trial *among* those that
    already match the modal patterns; it does not manufacture a sequence where there is none.
    """
    regions = list(fits)
    n_trials = fits[regions[0]].states.shape[0]
    modal = {r: modal_sequence(fits[r].states) for r in regions}
    post = None if bin_centers is None else (np.asarray(bin_centers) >= 0.0)

    score = np.zeros(n_trials)
    for r in regions:
        st = fits[r].states
        for i in range(n_trials):
            if tuple(_collapse(st[i])) == modal[r]:
                score[i] += 1.0
            row = st[i]
            n_sw = int((row[1:] != row[:-1]).sum())
            if n_sw > max_switches:
                score[i] -= 0.5
            seg = row[post] if post is not None else row
            if len(seg) > 1 and (seg[1:] != seg[:-1]).any():
                score[i] += richness_bonus

    best = np.flatnonzero(score >= score.max() - 1e-9)
    if rt is not None and len(best) > 1:
        return int(best[np.argmin(np.abs(rt[best] - np.nanmedian(rt)))])
    return int(best[0])


def _latent_limits(fits, pad=0.1, trial_idx=None):
    """Axis limits per region for the latent panels.

    ``trial_idx=None`` uses the 1-99 percentile of **all** (trial, bin) points — the whole
    cloud. But most of that cloud's extent is *cross-trial* variability, not the within-trial
    motion, so a single trial fills only ~15-20% of it and looks compressed. For a single-trial
    animation, pass ``trial_idx`` to zoom to **that trial's** extent (+ padding) so the path
    actually fills the panel. The switches are where the science is; they should be visible.
    """
    lims = {}
    for r, res in fits.items():
        if trial_idx is None:
            p = res.x.reshape(-1, res.x.shape[-1])[:, :2]
            lo, hi = np.percentile(p, [1, 99], axis=0)
        else:
            p = res.x[trial_idx, :, :2]
            lo, hi = p.min(axis=0), p.max(axis=0)
        d = pad * np.maximum(hi - lo, 1e-6)
        lims[r] = (lo - d, hi + d)
    return lims


# --------------------------------------------------------------------------- single trial


def single_trial_gif(
    fits: dict,
    trial_idx: int,
    bin_centers: np.ndarray,
    out_path: str | Path,
    *,
    align_label: str = "stimulus onset",
    fps: int = 8,
    hold: int = 8,
    dpi: int = 95,
) -> Path:
    """Animate ONE trial: a marker walks each area's latent path, flashing on regime change,
    with a 3-lane regime tape building up underneath. Writes a GIF, returns its path.

    ``fits`` : ``{region: RSLDSResult}`` (all sharing the same trials/bins).
    ``trial_idx`` : which trial (see :func:`pick_representative_trial`).
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    regions = list(fits)
    bc = np.asarray(bin_centers)
    T = len(bc)
    # zoom each panel to THIS trial's extent (not the whole cloud), so the path fills the panel
    lims = _latent_limits(fits, pad=0.18, trial_idx=trial_idx)
    cols = {r: state_colors(fits[r].n_states) for r in regions}

    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, len(regions), height_ratios=[3.0, 1.7], hspace=0.42, wspace=0.28)
    ax_top = [fig.add_subplot(gs[0, j]) for j in range(len(regions))]
    ax_tape = fig.add_subplot(gs[1, :])

    def draw(t: int):
        t = min(t, T - 1)
        # ---- top: latent panels ----
        for a, r in zip(ax_top, regions):
            a.clear()
            res = fits[r]
            x = res.x[trial_idx]                       # (T, D)
            st = res.states[trial_idx]
            a.plot(x[:, 0], x[:, 1], color="0.86", lw=1.0, zorder=1)   # full path, faint
            a.plot(x[: t + 1, 0], x[: t + 1, 1], color="0.45", lw=1.6, zorder=2)  # walked
            # coloured breadcrumbs by regime
            a.scatter(x[: t + 1, 0], x[: t + 1, 1],
                      c=[cols[r][s] for s in st[: t + 1]], s=22, zorder=3,
                      edgecolors="white", linewidths=0.3)
            # current marker
            a.scatter([x[t, 0]], [x[t, 1]], s=170, color=cols[r][st[t]], zorder=5,
                      edgecolors="black", linewidths=1.0)
            # flash ring on a switch
            if t > 0 and st[t] != st[t - 1]:
                a.scatter([x[t, 0]], [x[t, 1]], s=520, facecolor="none",
                          edgecolor="black", linewidths=2.2, zorder=6)
            (lo, hi) = lims[r]
            a.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xticks=[], yticks=[])
            a.set_title(f"{r}   regime {st[t]}", color=region_color(r), fontsize=11)
            a.set_xlabel("latent PC1", fontsize=8)
            if r == regions[0]:
                a.set_ylabel("latent PC2", fontsize=8)
        ax_top[len(regions) // 2].annotate(
            "latent geometry is illustrative (a smoothing artefact) — watch the SWITCHES",
            xy=(0.5, 1.16), xycoords="axes fraction", ha="center", fontsize=8.5, color="0.4")

        # ---- bottom: regime tape, one lane per area ----
        ax_tape.clear()
        lane_h = 0.62
        for lane, r in enumerate(regions):
            y = len(regions) - 1 - lane            # VISp on top
            st = fits[r].states[trial_idx]
            dt = bc[1] - bc[0]
            for b in range(t + 1):
                ax_tape.add_patch(plt.Rectangle((bc[b] - dt / 2, y - lane_h / 2), dt, lane_h,
                                                facecolor=cols[r][st[b]], edgecolor="none"))
            # switch ticks so far
            for sw in _switch_times(st[: t + 1], bc[: t + 1]) if t > 0 else []:
                ax_tape.plot(sw, y + lane_h / 2 + 0.08, "v", color="black", markersize=7)
            ax_tape.text(bc[0] - 0.02, y, r, ha="right", va="center", fontsize=10,
                         color=region_color(r), fontweight="bold")
        ax_tape.axvline(0, color="0.15", ls="--", lw=1.3)
        ax_tape.axvline(bc[t], color=OKABE_ITO[5], lw=2.2, path_effects=stroke())   # cursor
        ax_tape.set(xlim=(bc[0] - 0.06, bc[-1] + 0.01),
                    ylim=(-0.6, len(regions) - 0.4), yticks=[])
        ax_tape.set_xlabel(f"time from {align_label} (s)")
        ax_tape.set_title("regime tape (▼ = a switch) — the certified content", fontsize=10)

        fig.suptitle(f"One trial (#{trial_idx}) — t = {bc[t]:+.2f} s from {align_label}",
                     fontsize=13, y=0.99)

    frames = list(range(T)) + [T - 1] * hold
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- occupancy sweep


def occupancy_sweep_gif(
    fits: dict,
    bin_centers: np.ndarray,
    out_path: str | Path,
    *,
    align_label: str = "stimulus onset",
    fps: int = 8,
    hold: int = 8,
    dpi: int = 95,
) -> Path:
    """Trial-AVERAGED sweep: a time cursor crosses the regime occupancy of all areas at once,
    while a switch-time density builds up beneath. Writes a GIF, returns its path.

    This is the trustworthy visualisation: occupancy P(regime | t) and the distribution of switch
    times are averages over ALL trials, so nothing here rests on one noisy single-trial path.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    regions = list(fits)
    bc = np.asarray(bin_centers)
    T = len(bc)
    cols = {r: state_colors(fits[r].n_states) for r in regions}

    # precompute occupancy + switch-time density per region
    occ = {r: np.stack([(fits[r].states == k).mean(axis=0)
                        for k in range(fits[r].n_states)]) for r in regions}
    sw_density = {}
    for r in regions:
        st = fits[r].states
        changes = np.zeros(T)
        for i in range(st.shape[0]):
            changes[np.flatnonzero(st[i, 1:] != st[i, :-1]) + 1] += 1
        sw_density[r] = changes / st.shape[0]        # mean switches per trial entering each bin

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(len(regions) + 1, 1, height_ratios=[2, 2, 2, 2.2], hspace=0.32)
    ax_occ = [fig.add_subplot(gs[j]) for j in range(len(regions))]
    ax_sw = fig.add_subplot(gs[len(regions)])

    def draw(t: int):
        t = min(t, T - 1)
        for a, r in zip(ax_occ, regions):
            a.clear()
            a.stackplot(bc, occ[r], colors=cols[r], alpha=0.92, edgecolor="white", lw=0.25)
            a.axvline(0, color="0.15", ls="--", lw=1.1)
            a.axvline(bc[t], color=OKABE_ITO[5], lw=2.4, path_effects=stroke())
            a.set(xlim=(bc[0], bc[-1]), ylim=(0, 1), xticks=[])
            a.set_ylabel(f"{r}\nP(regime|t)", fontsize=9, color=region_color(r))
        ax_occ[0].set_title("Regime occupancy (trial-averaged) with a sweeping time cursor",
                            fontsize=11)

        ax_sw.clear()
        for r in regions:
            ax_sw.plot(bc[: t + 1], sw_density[r][: t + 1], lw=2.2, color=region_color(r),
                       path_effects=stroke(), label=r)
        ax_sw.axvline(0, color="0.15", ls="--", lw=1.1)
        ax_sw.axvline(bc[t], color=OKABE_ITO[5], lw=2.4, path_effects=stroke())
        ax_sw.set(xlim=(bc[0], bc[-1]), ylim=(0, max(max(v.max() for v in sw_density.values()),
                                                     1e-3) * 1.15))
        ax_sw.set_xlabel(f"time from {align_label} (s)")
        ax_sw.set_ylabel("switches / trial\nentering this bin", fontsize=9)
        ax_sw.legend(frameon=False, fontsize=9, ncol=len(regions), loc="upper right")
        ax_sw.set_title("When do the areas switch? (switch-time density, all trials)", fontsize=10)

        fig.suptitle(f"Occupancy sweep — t = {bc[t]:+.2f} s from {align_label}",
                     fontsize=13, y=0.995)

    frames = list(range(T)) + [T - 1] * hold
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out_path
