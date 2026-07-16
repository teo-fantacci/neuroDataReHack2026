"""Survey the 5 replicate {VISp, SCm, MRN} sessions. Metadata + trials only (no spike downloads)."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd, yaml

from src.data.access import get_one, units_table, sessions_table, load_trials

REGIONS = ["VISp", "SCm", "MRN"]
MIN_UNITS = 20
MIN_RATE_HZ = 1.0   # our HMM/geometry active-unit cut

P = yaml.safe_load(Path("config/params.yaml").read_text())
one = get_one(str(Path(P["data"]["one_cache_dir"]).resolve()))
u = units_table(one)
sess = sessions_table(one)

# --- sessions holding all three regions with >= MIN_UNITS each -------------
per = (u[u.Beryl.isin(REGIONS)].groupby(["eid", "Beryl"], observed=True)
       .size().reset_index(name="n"))
per = per[per.n >= MIN_UNITS]
ok = [e for e, g in per.groupby("eid", observed=True) if set(g.Beryl) >= set(REGIONS)]
print(f"sessions with all of {REGIONS} at >= {MIN_UNITS} good units: {len(ok)}\n")

rows, cond_rows, probe_rows = [], [], []
for eid in ok:
    su = u[(u.eid == eid) & (u.Beryl.isin(REGIONS))]
    meta = sess[sess.eid == eid].iloc[0]

    trials, mask = load_trials(one, eid)
    good = trials[mask]
    n_tot, n_use = len(trials), int(mask.sum())

    # behaviour on usable trials
    cL = np.nan_to_num(good.contrastLeft.values, nan=0.0)
    cR = np.nan_to_num(good.contrastRight.values, nan=0.0)
    signed = cR - cL                       # >0 stim on right
    choice = good.choice.values            # IBL: -1 / +1 (0 = nogo, excluded by mask)
    perf = float((good.feedbackType.values == 1).mean())
    rt = (good.firstMovement_times.values - good.stimOn_times.values)
    rt = rt[np.isfinite(rt)]
    # easy-trial performance (|contrast| >= 0.5) = engagement proxy
    easy = np.abs(signed) >= 0.5
    perf_easy = float((good.feedbackType.values[easy] == 1).mean()) if easy.sum() else np.nan
    zero = np.abs(signed) < 1e-9

    # dPCA-style balanced design: stimulus side x choice, non-zero contrast only
    nz = ~zero
    cells = {}
    for s_lab, s_m in [("stimL", signed < 0), ("stimR", signed > 0)]:
        for c_lab, c_m in [("chL", choice == 1), ("chR", choice == -1)]:
            cells[f"{s_lab}/{c_lab}"] = int((nz & s_m & c_m).sum())
    kbal = min(cells.values())

    counts = su.groupby("Beryl", observed=True).size()
    act = su[su.firing_rate >= MIN_RATE_HZ].groupby("Beryl", observed=True).size()
    act = act.reindex(REGIONS).fillna(0).astype(int)
    counts = counts.reindex(REGIONS).fillna(0).astype(int)

    rows.append({
        "eid": eid, "subject": meta.subject, "lab": meta.lab, "date": str(meta.date),
        "n_probes_triplet": su.pid.nunique(),
        "trials_total": n_tot, "trials_usable": n_use,
        "perf": perf, "perf_easy": perf_easy,
        "rt_med_ms": float(np.median(rt) * 1000), "zero_contrast_frac": float(zero.mean()),
        **{f"{r}": int(counts[r]) for r in REGIONS},
        **{f"{r}_act": int(act[r]) for r in REGIONS},
        "total_units": int(counts.sum()), "total_act": int(act.sum()),
        "dpca_K": kbal,
    })
    cond_rows.append({"eid": eid[:8], **cells, "K_balanced": kbal})

    # which probe carries which region + per-region quality
    for (pid, reg), g in su.groupby(["pid", "Beryl"], observed=True):
        if len(g) == 0:
            continue
        probe_rows.append({
            "eid": eid[:8], "pid": pid[:8], "region": reg, "n": len(g),
            "n_act": int((g.firing_rate >= MIN_RATE_HZ).sum()),
            "fr_med": round(float(g.firing_rate.median()), 2),
            "presence_med": round(float(g.presence_ratio.median()), 3),
            "drift_med": round(float(g.drift.median()), 1),
            "contam_med": round(float(g.contamination.median()), 3),
        })

df = pd.DataFrame(rows).sort_values("trials_usable", ascending=False)
pd.set_option("display.width", 250)

print("=" * 130)
print("PER-SESSION OVERVIEW  (act = good units with session firing_rate >= 1 Hz, our HMM cut)")
print("=" * 130)
show = df[["eid", "subject", "lab", "n_probes_triplet", "trials_total", "trials_usable",
           "perf", "perf_easy", "rt_med_ms", "VISp", "SCm", "MRN", "total_units",
           "VISp_act", "SCm_act", "MRN_act", "total_act", "dpca_K"]]
print(show.to_string(index=False))

print("\n" + "=" * 130)
print("dPCA BALANCED DESIGN: usable trials per (stimulus side x choice) cell, non-zero contrast")
print("off-diagonal cells (stimL/chR, stimR/chL) are ERROR trials -> they cap K")
print("=" * 130)
print(pd.DataFrame(cond_rows).to_string(index=False))

print("\n" + "=" * 130)
print("PER-PROBE / PER-REGION YIELD AND QUALITY")
print("=" * 130)
pr = pd.DataFrame(probe_rows).sort_values(["eid", "region"])
print(pr.to_string(index=False))

print("\n" + "=" * 130)
print("POOLED TOTALS if we combine all 5 sessions")
print("=" * 130)
print(f"  usable trials : {int(df.trials_usable.sum())}   (working session alone: "
      f"{int(df[df.eid.str.startswith('5c0c560e')].trials_usable.iloc[0])})")
print(f"  dPCA K sum    : {int(df.dpca_K.sum())}          (working session alone: "
      f"{int(df[df.eid.str.startswith('5c0c560e')].dpca_K.iloc[0])})")
for r in REGIONS:
    tot, tot_act = int(df[r].sum()), int(df[r + "_act"].sum())
    print(f"  {r:5s}: {tot:3d} good units, {tot_act:3d} active (>= {MIN_RATE_HZ} Hz)"
          f"  [per-session range {df[r].min()}-{df[r].max()}]")

df.to_csv("results/triplet_sessions_survey.csv", index=False)
print("\nwrote results/triplet_sessions_survey.csv")
