"""Choose K by STATE STABILITY, at the MAXIMUM trial count the session offers.

    python scripts/run_state_stability.py

Why not held-out likelihood (short version; full argument in
docs/knowledge/ARI_and_state_stability.md):
a Poisson HMM assumes neurons are conditionally independent given the state, real
populations are correlated, and the model's only way to express correlation is to add
states. So held-out log-likelihood never turns over and its argmax just lands on whichever
end of the K range you searched — it gave K=2 (the bottom) when the scores were -inf, and
K=10 (the top) once that was fixed. Neither was a result.

This script asks the question our science actually makes: **are the states reproducible?**
Fit two HMMs on disjoint trial sets A and B, decode both on a third never-seen set C, and
measure whether the two independent fits carve C's timeline the same way (Adjusted Rand
Index: permutation-invariant, chance-corrected, comparable across K).

Two deliberate choices here, both aimed at making a LOW ARI trustworthy:

* **MAX_TRIALS = None -> all usable trials (843, not 400).** Each fit sees only ~1/3 of the
  trials, so a real-but-subtle state can fail to be recovered twice purely for want of data
  and masquerade as instability. Using every trial is the cheapest way to push that
  confound back.
* **N_RESTARTS > 1.** EM is non-convex; without restarts two fits can disagree from bad luck
  rather than different data, which biases the ARI (and hence K) DOWN. Each model is fit
  N_RESTARTS times and the best-likelihood solution is kept.

Memory: loading both probes' spike tables costs a few hundred MB and is the only expensive
part (the tensors are ~1.5 MB). They are cached to .npz, so re-runs are cheap and can
coexist with an rSLDS fit holding RAM.

Writes:  results/hmm/state_stability_maxtrials.json
         results/figures/phase2_cv/state_stability_maxtrials.png
"""

from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.build_matrices import (  # noqa: E402
    drop_silent_units,
    region_tensor_from_probes,
    select_active_units,
)
from src.dynamics.hmm import (  # noqa: E402
    cross_validate_states,
    select_k_by_elbow,
    stability_scan,
)
from src.viz.style import region_color, savefig, set_style, stroke  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

EID = "5c0c560e-9e1f-45e9-b66e-e4ee7855be84"
REGIONS = ["VISp", "SCm", "MRN"]
KS = [2, 3, 4, 5, 6, 7, 8, 9, 10]

PRE_S, POST_S, BIN_S = 0.5, 0.8, 0.02
MIN_RATE_HZ = 1.0
MAX_TRIALS = None          # <- ALL usable trials for this session (843)
SEED = 0

N_REPEATS = 5              # independent A/B/C splits per K
N_ITER = 30
N_RESTARTS = 2             # EM restarts per fit; >1 removes the local-minimum bias
RUN_CV = True              # also compute the likelihood curve + elbow, for comparison

CACHE = ROOT / "results" / "hmm" / f"tensors_{EID[:8]}_alltrials.npz"


def build_tensors() -> dict[str, np.ndarray]:
    """Load the session once, build per-region tensors, then free the spike tables."""
    from iblatlas.regions import BrainRegions

    from src.data.access import get_one, load_trials, load_units, sessions_table

    params = yaml.safe_load((ROOT / "config" / "params.yaml").read_text())
    one = get_one(cache_dir=str((ROOT / params["data"]["one_cache_dir"]).resolve()))

    pids = sessions_table(one).query("eid == @EID")["pid"].tolist()
    trials, mask = load_trials(one, EID)
    good = trials[mask].reset_index(drop=True)
    if MAX_TRIALS is not None and len(good) > MAX_TRIALS:
        rng = np.random.default_rng(SEED)
        good = good.iloc[np.sort(rng.choice(len(good), MAX_TRIALS, replace=False))].reset_index(drop=True)
    events = good["stimOn_times"].values
    print(f"trials: {len(good)} usable (mask.sum()={int(mask.sum())}, MAX_TRIALS={MAX_TRIALS})")

    br = BrainRegions()
    loads = []
    for pid in pids:
        spikes, clusters = load_units(one, pid)
        clusters = clusters.copy()
        clusters["Beryl"] = br.id2acronym(clusters["atlas_id"].values, mapping="Beryl")
        loads.append({"spikes": spikes, "clusters": clusters})

    out = {}
    for region in REGIONS:
        counts, _ = region_tensor_from_probes(
            loads, region, events, pre_s=PRE_S, post_s=POST_S, bin_size_s=BIN_S
        )
        n_all = counts.shape[2]
        counts, _ = drop_silent_units(counts, min_total_spikes=1)
        counts, _ = select_active_units(counts, BIN_S, min_rate_hz=MIN_RATE_HZ)
        out[region] = counts.astype(np.int32)
        print(f"  {region}: {n_all} good units -> {counts.shape[2]} kept (>= {MIN_RATE_HZ} Hz)"
              f"   tensor {counts.shape}")

    del loads          # ~16M spikes worth of arrays: drop before fitting
    gc.collect()
    return out


def main() -> int:
    set_style()

    if CACHE.exists():
        print(f"tensors <- {CACHE.relative_to(ROOT)}")
        z = np.load(CACHE)
        tensors = {r: z[r] for r in REGIONS}
        for r in REGIONS:
            print(f"  {r}: {tensors[r].shape}")
    else:
        print("building tensors (loads both probes — the only memory-heavy step)")
        tensors = build_tensors()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **tensors)
        print(f"cached -> {CACHE.relative_to(ROOT)}")

    n_trials = tensors[REGIONS[0]].shape[0]
    print(f"\nstability: K={KS}, {N_REPEATS} repeats, {N_RESTARTS} EM restarts, "
          f"{n_trials} trials (each fit sees ~{n_trials // 3})")

    results = {}
    for region in REGIONS:
        counts = tensors[region]
        print(f"\n=== {region}  ({counts.shape[2]} units, {counts.shape[0]} trials)")

        elbow_k = None
        if RUN_CV:
            t0 = time.time()
            cv = cross_validate_states(counts, KS, n_folds=3, n_iter=N_ITER, seed=SEED)
            elbow_k = select_k_by_elbow(cv)
            ll = cv["mean_test_ll"].to_numpy()
            argmax_k = int(cv["n_states"].to_numpy()[np.argmax(ll)])
            edge = "  [BOUNDARY - not a choice]" if argmax_k == max(KS) else ""
            print(f"  likelihood: elbow K={elbow_k}   argmax K={argmax_k}{edge}"
                  f"   [{time.time()-t0:.0f}s]")

        t0 = time.time()
        st = stability_scan(counts, KS, n_repeats=N_REPEATS, n_iter=N_ITER, seed=SEED,
                            n_restarts=N_RESTARTS, progress=True)
        ari = st["ari_mean"].to_numpy()
        peak_k = int(st["n_states"].to_numpy()[np.argmax(ari)])
        print(f"  stability : PEAK ARI at K={peak_k}  (ARI={ari.max():.3f})"
              f"   [{time.time()-t0:.0f}s]")

        results[region] = {
            "n_units": int(counts.shape[2]),
            "n_trials": int(counts.shape[0]),
            "K": list(KS),
            "ari": ari.tolist(),
            "ari_sd": st["ari_std"].to_numpy().tolist(),
            "matched_acc": st["matched_acc_mean"].to_numpy().tolist(),
            "stability_peak_k": peak_k,
            "elbow_k": elbow_k,
            "n_repeats": N_REPEATS,
            "n_restarts": N_RESTARTS,
            "min_rate_hz": MIN_RATE_HZ,
        }

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for region in REGIONS:
        r = results[region]
        c = region_color(region)
        ax[0].errorbar(r["K"], r["ari"], yerr=r["ari_sd"], marker="o", ms=5, capsize=3,
                       lw=1.8, color=c, label=f"{region} (peak K={r['stability_peak_k']})",
                       path_effects=stroke())
        ax[0].scatter([r["stability_peak_k"]], [max(r["ari"])], s=170, facecolors="none",
                      edgecolors=c, linewidths=2.2, zorder=5)
        ax[1].plot(r["K"], r["matched_acc"], marker="o", ms=5, lw=1.8, color=c,
                   label=region, path_effects=stroke())
    ax[0].axhline(0, color="black", lw=1, ls=":")
    ax[0].set(xlabel="number of HMM states K", ylabel="Adjusted Rand Index (0 = chance)",
              title=f"State stability, {n_trials} trials, {N_RESTARTS} EM restarts\n"
                    "(circled = peak = the honest K)")
    ax[0].set_xticks(KS)
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set(xlabel="number of HMM states K", ylabel="matched accuracy (Hungarian)",
              title="Matched agreement - NOT chance-corrected\n(falls with K for free; read with the ARI)")
    ax[1].set_xticks(KS)
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = savefig(fig, "phase2_cv", "state_stability_maxtrials", root=ROOT)

    out = ROOT / "results" / "hmm" / "state_stability_maxtrials.json"
    out.write_text(json.dumps(results, indent=1))

    print("\n" + "=" * 70)
    print("BEST K PER REGION")
    print("=" * 70)
    for region in REGIONS:
        r = results[region]
        print(f"  {region:5s}  stability peak K = {r['stability_peak_k']}"
              f"   (elbow K = {r['elbow_k']})   ARI = {max(r['ari']):.3f}")
    print(f"\nfigure -> {p}")
    print(f"json   -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
