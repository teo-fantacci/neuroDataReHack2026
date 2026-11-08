"""Pre-modelling data-summary figures for one {VISp, SCm, MRN} session.

Purpose
-------
Before any HMM / rSLDS / geometry, *look at the data*. This script produces the standard
sanity-and-structure figures a systems-neuro paper shows before modelling, all on the SAME
frozen preprocessing the models eat (``PreprocessConfig`` -- pre 0.3 s, post 0.8 s, 40 ms bins,
>= 1 Hz units), and all colour-vision-deficiency-safe (``src/viz/style.py``: Okabe-Ito
categorical, cividis heatmaps, no red-vs-green encoding anywhere).

Figures written to ``results/figures/preproc_summary/`` (align tag in the filename):

1. ``trial_timing``       -- where movement and reward fall INSIDE the analysis window
                             (answers "reaction time w.r.t. the window we selected").
2. ``rasters``            -- dot rasters of a few example units, trials sorted by RT, RT +
                             feedback overlaid (the raw thing a PSTH averages).
3. ``psth_over_trials``   -- PSTH = average over TRIALS: population per region (+/-SEM over
                             trials) and a few example single units.
4. ``psth_over_neurons``  -- the OTHER average: over NEURONS, on single example trials, so the
                             single-trial variability the trial-average hides is visible.
5. ``sequence_heatmap``   -- trial-averaged, peak-normalised, latency-sorted unit x time map
                             per region: the "is there a temporal sequence?" picture that
                             motivates the whole metastable-sequence hypothesis.
6. ``rate_distribution``  -- per-region firing-rate histogram with the 1 Hz cut marked
                             (justifies the unit threshold).
7. ``mean_variance``      -- per-unit count mean vs variance against the Poisson line y=x
                             (justifies the Poisson emission model / the Anscombe discussion).
8. ``behavior``           -- psychometric (choice vs signed contrast) + chronometric (RT vs
                             |contrast|): the behavioural sanity check.

Usage
-----
    python scripts/plot_preprocessing_summary.py                       # working session, stimOn
    python scripts/plot_preprocessing_summary.py --eid 90c61c38 --align firstMovement_times
    python scripts/plot_preprocessing_summary.py --all-figs --show-timing

Data is read from the ONE cache in ``data/ONE`` (downloads on first use for a fresh session).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.access import get_one                                    # noqa: E402
from src.data.build_matrices import (                                  # noqa: E402
    counts_to_rates, drop_silent_units, firing_rate_table,
    region_event_spikes, region_tensor_from_probes, select_active_units,
    smooth_over_time,
)
from src.data.preprocess import (                                      # noqa: E402
    PreprocessConfig, TRIPLET_EIDS, WORKING_EID, align_times,
    load_session, sigma_ms_to_bins, usable_mask,
)
from src.viz.style import (                                            # noqa: E402
    OKABE_ITO, SEQUENTIAL, region_color, savefig, set_style, stroke,
)

PHASE = "preproc_summary"
SMOOTH_MS = 60.0                       # display smoothing (project default, 2026-07-16)

# Okabe-Ito roles that are NEVER red-vs-green (safe for deutan/protan):
C_RT = OKABE_ITO[4]                    # blue      -- reaction time / movement
C_FB = OKABE_ITO[0]                    # orange    -- feedback / reward
C_BASE = OKABE_ITO[1]                  # sky blue  -- baseline window shade
C_POST = OKABE_ITO[0]                  # orange    -- post-stim window shade


# --------------------------------------------------------------------------- data


def _resolve_eid(eid: str) -> str:
    """Accept an 8-char prefix or a full eid."""
    if eid in TRIPLET_EIDS:
        return eid
    hits = [e for e in TRIPLET_EIDS if e.startswith(eid)]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(f"eid {eid!r} is not one of the 5 triplet sessions: {TRIPLET_EIDS}")


def load(eid: str, cfg: PreprocessConfig):
    """Load one session and return everything the figures need, in register with `mask`."""
    one = get_one(cache_dir=ROOT / "data" / "ONE")
    trials, mask, probe_loads = load_session(one, eid, cfg=cfg)
    ev = align_times(trials, mask, cfg.align_event)
    keep = usable_mask(trials, mask, cfg.align_event)          # subsets the masked trials
    t = trials[mask][keep]                                     # covariates in register with ev
    return probe_loads, ev, t


def _tensors(probe_loads, ev, cfg):
    """Per-region count tensors + smoothed rate tensors + bin centres (all aligned).

    Units are filtered exactly as ``preprocess_session`` does (drop silent, then >= min_rate_hz)
    so these figures show the SAME population the HMM / rSLDS see. The rate-distribution figure
    keeps all units on purpose -- it exists to justify this very cut.
    """
    counts, rates, bc = {}, {}, None
    sigma_bins = sigma_ms_to_bins(SMOOTH_MS, cfg.bin_size_s)
    for r in cfg.regions:
        c, bc = region_tensor_from_probes(
            probe_loads, r, ev, pre_s=cfg.pre_s, post_s=cfg.post_s, bin_size_s=cfg.bin_size_s)
        c, _ = drop_silent_units(c, min_total_spikes=1)
        c, _ = select_active_units(c, cfg.bin_size_s, min_rate_hz=cfg.min_rate_hz)
        counts[r] = c
        rates[r] = smooth_over_time(counts_to_rates(c, cfg.bin_size_s), sigma_bins)
    return counts, rates, bc


def _behaviour(t):
    """Signed contrast, choice, RT, feedback -- NaN-safe arrays over the usable trials."""
    cl = np.nan_to_num(t["contrastLeft"].to_numpy(dtype=float), nan=0.0)
    cr = np.nan_to_num(t["contrastRight"].to_numpy(dtype=float), nan=0.0)
    signed = cr - cl                                   # + = stimulus on the right
    choice = t["choice"].to_numpy(dtype=float)         # IBL: choice==-1 <=> mouse reported RIGHT
    rt = t["firstMovement_times"].to_numpy(dtype=float) - t["stimOn_times"].to_numpy(dtype=float)
    fb = t["feedback_times"].to_numpy(dtype=float) - t["stimOn_times"].to_numpy(dtype=float)
    return signed, choice, rt, fb


# --------------------------------------------------------------------------- figures


def fig_trial_timing(t, cfg):
    """RT and feedback distributions, drawn ON the analysis window. Figure 1."""
    import matplotlib.pyplot as plt

    _, _, rt, fb = _behaviour(t)
    quiesc = t["quiescencePeriod"].to_numpy(dtype=float)
    rt, fb, quiesc = rt[np.isfinite(rt)], fb[np.isfinite(fb)], quiesc[np.isfinite(quiesc)]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.5, 6.2), height_ratios=[2.2, 1])

    # --- top: RT / feedback histograms on the post-stimulus part of the window
    edges = np.linspace(0, cfg.post_s, 41)
    ax0.hist(rt[rt <= cfg.post_s], bins=edges, color=C_RT, alpha=0.7,
             label=f"reaction time (movement onset)  median {np.median(rt)*1000:.0f} ms")
    ax0.hist(fb[fb <= cfg.post_s], bins=edges, color=C_FB, alpha=0.55,
             label=f"feedback (reward/error)  median {np.median(fb)*1000:.0f} ms")
    for v, c in [(np.median(rt), C_RT), (np.median(fb), C_FB)]:
        ax0.axvline(v, color=c, lw=2, path_effects=stroke())
    ax0.axvspan(-cfg.pre_s, 0, color=C_BASE, alpha=0.15, zorder=0)
    ax0.axvline(0, color="black", lw=1.2)
    ax0.set_xlim(-cfg.pre_s, cfg.post_s)
    ax0.set_xlabel("time from stimulus onset (s)")
    ax0.set_ylabel("trials")
    fmoved = float(np.mean(rt < cfg.post_s))
    ffb = float(np.mean(fb < cfg.post_s))
    ax0.set_title(f"The [0, +{cfg.post_s:g}] s window is a MOVEMENT window:  "
                  f"moved on {fmoved:.0%} of trials, feedback on {ffb:.0%}  (n={rt.size})")
    ax0.legend(frameon=False, fontsize=8, loc="upper right")

    # --- bottom: enforced quiescence vs the chosen baseline pre_s
    ax1.hist(quiesc, bins=30, color=OKABE_ITO[2], alpha=0.7)
    ax1.axvline(cfg.pre_s, color="black", lw=2, path_effects=stroke(),
                label=f"pre_s = {cfg.pre_s:g} s baseline")
    fout = float(np.mean(quiesc < cfg.pre_s))
    ax1.set_xlabel("enforced quiescence period (s, wheel held still before stimulus)")
    ax1.set_ylabel("trials")
    ax1.set_title(f"Baseline sits inside the enforced-still period on {1-fout:.0%} of trials")
    ax1.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    return fig


def fig_rasters(probe_loads, ev, t, cfg, n_units_per_region=1):
    """Dot rasters of example units, trials sorted by RT, RT+feedback overlaid. Figure 2."""
    import matplotlib.pyplot as plt

    _, _, rt, fb = _behaviour(t)
    order = np.argsort(np.where(np.isfinite(rt), rt, np.inf))    # slow/NaN trials at the bottom
    rank = np.empty_like(order); rank[order] = np.arange(order.size)

    regions = list(cfg.regions)
    fig, axes = plt.subplots(1, len(regions), figsize=(4.4 * len(regions), 4.6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, r in zip(axes, regions):
        counts, _ = region_tensor_from_probes(
            probe_loads, r, ev, pre_s=cfg.pre_s, post_s=cfg.post_s, bin_size_s=cfg.bin_size_s)
        frt = firing_rate_table(counts, cfg.bin_size_s)
        # pick the unit with the highest peak PSTH (the clearest example)
        u = int(frt.sort_values("peak_rate_hz", ascending=False)["unit"].iloc[0])
        ti, ui, tr = region_event_spikes(
            probe_loads, r, ev, pre_s=cfg.pre_s, post_s=cfg.post_s)
        m = ui == u
        ax.scatter(tr[m], rank[ti[m]], s=1.4, marker="|", color=region_color(r),
                   linewidths=0.6, rasterized=True)
        # overlay RT and feedback per trial (sorted), thin lines
        ax.plot(rt[order], np.arange(order.size), color=C_RT, lw=1.4,
                path_effects=stroke(2.4), label="movement")
        ax.plot(fb[order], np.arange(order.size), color=C_FB, lw=1.4,
                path_effects=stroke(2.4), label="feedback")
        ax.axvline(0, color="black", lw=1)
        ax.set_xlim(-cfg.pre_s, cfg.post_s)
        ax.set_ylim(0, order.size)
        ax.set_title(f"{r}  unit {u}  ({frt.set_index('unit').loc[u, 'peak_rate_hz']:.0f} Hz peak)")
        ax.set_xlabel("time from stimulus onset (s)")
    axes[0].set_ylabel("trial (sorted by reaction time)")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Example single-unit rasters — the raw data a PSTH averages over", y=1.02)
    fig.tight_layout()
    return fig


def fig_psth_over_trials(rates, bc, cfg):
    """PSTH = mean over TRIALS. Population per region (+/-SEM), + example single units. Fig 3."""
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.3))

    # left: population PSTH per region (mean over that region's units of the per-trial rate)
    for r in cfg.regions:
        pop = rates[r].mean(axis=2)                    # (trials, bins) mean over units
        m = pop.mean(axis=0)
        sem = pop.std(axis=0) / np.sqrt(pop.shape[0])
        ax0.plot(bc, m, color=region_color(r), lw=2, path_effects=stroke(), label=r)
        ax0.fill_between(bc, m - sem, m + sem, color=region_color(r), alpha=0.2)
    ax0.axvline(0, color="black", lw=1)
    ax0.set_xlabel("time from stimulus onset (s)")
    ax0.set_ylabel("population firing rate (Hz)")
    ax0.set_title("Population PSTH  (mean over trials, +/-SEM over trials)")
    ax0.legend(frameon=False, fontsize=9)

    # right: a couple of example single units from SCm (usually the clearest), mean over trials
    r = "SCm" if "SCm" in cfg.regions else cfg.regions[0]
    unit_rates = rates[r]                               # (trials, bins, units)
    peak = unit_rates.mean(axis=0).max(axis=0)          # per-unit trial-averaged peak
    picks = np.argsort(peak)[::-1][:3]
    for k, u in enumerate(picks):
        m = unit_rates[:, :, u].mean(axis=0)
        sem = unit_rates[:, :, u].std(axis=0) / np.sqrt(unit_rates.shape[0])
        col = OKABE_ITO[[4, 0, 2][k]]
        ax1.plot(bc, m, color=col, lw=2, path_effects=stroke(), label=f"{r} unit {u}")
        ax1.fill_between(bc, m - sem, m + sem, color=col, alpha=0.2)
    ax1.axvline(0, color="black", lw=1)
    ax1.set_xlabel("time from stimulus onset (s)")
    ax1.set_ylabel("firing rate (Hz)")
    ax1.set_title("Example single-unit PSTHs  (mean over trials, +/-SEM)")
    ax1.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    return fig


def fig_psth_over_neurons(rates, bc, t, cfg, n_trials=6):
    """The OTHER average: mean over NEURONS, single trials. Shows single-trial variability. Fig 4."""
    import matplotlib.pyplot as plt

    _, _, rt, _ = _behaviour(t)
    # choose a spread of trials by RT (fast -> slow) so the panels differ meaningfully
    finite = np.flatnonzero(np.isfinite(rt))
    picks = finite[np.linspace(0, finite.size - 1, n_trials).round().astype(int)]

    fig, axes = plt.subplots(1, len(cfg.regions), figsize=(4.4 * len(cfg.regions), 4.2),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, cfg.regions):
        pop = rates[r].mean(axis=2)                     # (trials, bins) mean over neurons
        for k, tr in enumerate(picks):
            ax.plot(bc, pop[tr], color=OKABE_ITO[k % len(OKABE_ITO)], lw=1.3,
                    path_effects=stroke(2.2), alpha=0.9,
                    label=f"trial {tr} (RT {rt[tr]*1000:.0f} ms)")
        ax.plot(bc, pop.mean(axis=0), color="black", lw=2.2, ls="--", label="trial mean")
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"{r}  (mean over {rates[r].shape[2]} neurons)")
        ax.set_xlabel("time from stimulus onset (s)")
    axes[0].set_ylabel("population rate (Hz)")
    axes[-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Single-trial population rate (average over NEURONS) — the trial-average's raw input",
                 y=1.02)
    fig.tight_layout()
    return fig


def fig_sequence_heatmap(rates, bc, cfg):
    """Trial-averaged, peak-normalised, latency-sorted unit x time map per region. Figure 5."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(cfg.regions), figsize=(4.4 * len(cfg.regions), 4.4),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, cfg.regions):
        psth = rates[r].mean(axis=0).T                  # (units, bins), trial-averaged
        peak = psth.max(axis=1, keepdims=True)
        norm = psth / np.where(peak > 0, peak, 1.0)     # each unit peak-normalised to 1
        latency = np.argmax(norm, axis=1)               # bin of the peak
        order = np.argsort(latency)
        im = ax.imshow(norm[order], aspect="auto", cmap=SEQUENTIAL,
                       extent=[bc[0], bc[-1], norm.shape[0], 0], vmin=0, vmax=1)
        ax.axvline(0, color="white", lw=1.2)
        ax.set_title(f"{r}  ({norm.shape[0]} units, sorted by peak time)")
        ax.set_xlabel("time from stimulus onset (s)")
    axes[0].set_ylabel("unit (sorted by peak latency)")
    fig.colorbar(im, ax=axes[-1], fraction=0.046, label="peak-normalised rate")
    fig.suptitle("Sequential activation: is there a temporal sequence of activity? (motivates the HMM/rSLDS)",
                 y=1.02)
    fig.tight_layout()
    return fig


def fig_rate_distribution(probe_loads, ev, cfg):
    """Per-region firing-rate histogram with the 1 Hz cut marked. Figure 6."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bins = np.logspace(-2.5, 2, 40)
    for r in cfg.regions:
        counts, _ = region_tensor_from_probes(
            probe_loads, r, ev, pre_s=cfg.pre_s, post_s=cfg.post_s, bin_size_s=cfg.bin_size_s)
        window_s = counts.shape[1] * cfg.bin_size_s
        rate = counts.sum(axis=(0, 1)) / counts.shape[0] / window_s
        rate = rate[rate > 0]
        n_keep = int((rate >= cfg.min_rate_hz).sum())
        ax.hist(rate, bins=bins, histtype="step", lw=2.2, color=region_color(r),
                path_effects=stroke(3.2), label=f"{r}  ({n_keep}/{rate.size} kept >= {cfg.min_rate_hz:g} Hz)")
    ax.axvline(cfg.min_rate_hz, color="black", lw=1.6, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("mean in-window firing rate (Hz, log scale)")
    ax.set_ylabel("units")
    ax.set_title(f"Firing-rate distribution and the >= {cfg.min_rate_hz:g} Hz unit cut")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def fig_mean_variance(counts, cfg):
    """Per-unit spike-count mean vs variance against the Poisson line. Figure 7."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    lo, hi = 1e-3, 1.0
    for r in cfg.regions:
        per_trial = counts[r].sum(axis=1)               # (trials, units) spikes/trial
        m = per_trial.mean(axis=0)
        v = per_trial.var(axis=0)
        ax.scatter(m, v, s=18, color=region_color(r), alpha=0.7, label=r)
        lo, hi = min(lo, m.min() + 1e-3), max(hi, m.max())
    xs = np.array([lo, hi])
    ax.plot(xs, xs, color="black", lw=1.6, ls="--", label="Poisson (Var = Mean)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mean spike count per trial")
    ax.set_ylabel("variance of spike count per trial")
    ax.set_title("Mean vs variance: how Poisson are the counts?\n(motivates Poisson emissions; Anscombe cannot stabilise at these rates)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def fig_behavior(t, cfg):
    """Psychometric (choice vs signed contrast) + chronometric (RT vs |contrast|). Figure 8."""
    import matplotlib.pyplot as plt

    signed, choice, rt, _ = _behaviour(t)
    go = choice != 0                                    # drop no-go for the choice curve
    levels = np.unique(signed)

    p_right, rt_med, rt_lo, rt_hi = [], [], [], []
    for c in levels:
        sel = go & (signed == c)
        # documented mapping: choice == -1 <=> mouse reported the RIGHT stimulus
        p_right.append(float(np.mean(choice[sel] == -1)) if sel.any() else np.nan)
    for a in np.unique(np.abs(levels)):
        sel = (np.abs(signed) == a) & np.isfinite(rt)
        r = rt[sel]
        rt_med.append(np.median(r) if r.size else np.nan)
        rt_lo.append(np.percentile(r, 25) if r.size else np.nan)
        rt_hi.append(np.percentile(r, 75) if r.size else np.nan)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax0.plot(levels * 100, p_right, "-o", color=OKABE_ITO[4], lw=2, path_effects=stroke())
    ax0.axhline(0.5, color="black", lw=0.8, ls=":")
    ax0.axvline(0, color="black", lw=0.8, ls=":")
    ax0.set_xlabel("signed contrast (%)   (+ = stimulus on the right)")
    ax0.set_ylabel("P(mouse reports right)")
    ax0.set_ylim(-0.02, 1.02)
    ax0.set_title("Psychometric")

    abs_levels = np.unique(np.abs(levels)) * 100
    rt_med, rt_lo, rt_hi = map(np.asarray, (rt_med, rt_lo, rt_hi))
    ax1.plot(abs_levels, rt_med * 1000, "-o", color=OKABE_ITO[0], lw=2, path_effects=stroke())
    ax1.fill_between(abs_levels, rt_lo * 1000, rt_hi * 1000, color=OKABE_ITO[0], alpha=0.2)
    ax1.set_xlabel("|contrast| (%)")
    ax1.set_ylabel("reaction time (ms)  median, IQR")
    ax1.set_title("Chronometric")

    fig.suptitle(f"Behaviour  ({int(go.sum())} go trials)", y=1.02)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- driver


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eid", default=WORKING_EID, help="full eid or 8-char prefix")
    ap.add_argument("--align", default="stimOn_times",
                    choices=["stimOn_times", "firstMovement_times"])
    ap.add_argument("--show-timing", action="store_true")
    args = ap.parse_args()

    set_style()
    eid = _resolve_eid(args.eid)
    cfg = PreprocessConfig(align_event=args.align)      # pre 0.3 / post 0.8 / 40 ms / 1 Hz
    atag = args.align.replace("_times", "")
    print(f"session {eid[:8]}  align={args.align}  window[-{cfg.pre_s}, +{cfg.post_s}]s  "
          f"bin={cfg.bin_size_s*1000:.0f}ms  smooth={SMOOTH_MS:.0f}ms")

    t0 = time.time()
    probe_loads, ev, t = load(eid, cfg)
    counts, rates, bc = _tensors(probe_loads, ev, cfg)
    print(f"  loaded + binned in {time.time()-t0:.0f}s  "
          f"({{" + ", ".join(f'{r}:{counts[r].shape[2]}u' for r in cfg.regions) + f"}}, {len(ev)} trials)")

    jobs = [
        ("trial_timing",     lambda: fig_trial_timing(t, cfg)),
        ("rasters",          lambda: fig_rasters(probe_loads, ev, t, cfg)),
        ("psth_over_trials", lambda: fig_psth_over_trials(rates, bc, cfg)),
        ("psth_over_neurons", lambda: fig_psth_over_neurons(rates, bc, t, cfg)),
        ("sequence_heatmap", lambda: fig_sequence_heatmap(rates, bc, cfg)),
        ("rate_distribution", lambda: fig_rate_distribution(probe_loads, ev, cfg)),
        ("mean_variance",    lambda: fig_mean_variance(counts, cfg)),
        ("behavior",         lambda: fig_behavior(t, cfg)),
    ]
    for name, fn in jobs:
        ts = time.time()
        fig = fn()
        path = savefig(fig, PHASE, f"{eid[:8]}_{atag}_{name}")
        secs = time.time() - ts
        print(f"  [{secs:4.1f}s]  {path.relative_to(ROOT)}")
        if args.show_timing:
            import matplotlib.pyplot as plt
            plt.close(fig)


if __name__ == "__main__":
    main()
