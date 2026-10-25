"""
Bootstrap analysis - F1 2025
============================
Two questions answered with a bootstrap (resampling with replacement):

  A) Is the lap-time gap between two drivers SYSTEMATIC or just noise?
     -> paired lap-by-lap differences (same lap number = same track state,
        similar fuel load), then a 95 % confidence interval on the mean gap.

  B) How uncertain is the tyre degradation slope of each stint?
     -> resample the laps of a stint, refit the line 10 000 times,
        and get a 95 % confidence interval on the slope (ms/lap).

How to read the result:
  - If the 95 % CI of the gap does NOT contain 0 -> the gap is systematic.
  - If it contains 0 -> we cannot conclude anything.

Run:  python bootstrap_analysis.py
"""

import os
import warnings

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
CACHE_DIR = os.environ.get("F1_CACHE", "./f1_cache")  # set F1_CACHE to reuse an existing cache folder
OUT_DIR = "./outputs"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

YEAR = 2025
PAIRS = [("NOR", "PIA"), ("NOR", "LEC")]
N_BOOT = 10_000          # number of virtual resamples
CI = 95                  # confidence level (%)
SAME_COMPOUND = True     # compare only laps run on the same tyre compound
MIN_PAIRED_LAPS = 8      # minimum paired laps for a GP to be used
MIN_STINT_LAPS = 6       # minimum laps in a stint for the slope bootstrap
MAX_WET_SHARE = 0.20     # skip races with more than 20 % wet-tyre laps
SEED = 42

rng = np.random.default_rng(SEED)
lo_q, hi_q = (100 - CI) / 2, 100 - (100 - CI) / 2


# ============================================================
# HELPERS
# ============================================================
def is_dry_race(laps):
    """True if the race has few laps on INTERMEDIATE / WET tyres."""
    if "Compound" not in laps.columns:
        return True
    wet = laps["Compound"].isin(["INTERMEDIATE", "WET"]).mean()
    return wet <= MAX_WET_SHARE


def clean_laps(laps, driver):
    """Keep only 'clean' laps: green flag, no pit in/out, dry tyres, valid time."""
    d = laps.pick_drivers(driver).copy()
    d = d[d["LapTime"].notna() & d["PitInTime"].isna() & d["PitOutTime"].isna()]
    # TrackStatus '1' = green flag all lap (no yellow, SC, VSC, red)
    d = d[d["TrackStatus"].astype(str) == "1"]
    d = d[d["Compound"].isin(["SOFT", "MEDIUM", "HARD"])]
    d["LapTime_s"] = d["LapTime"].dt.total_seconds()
    # remove very slow laps (> 107 % of the driver's median)
    d = d[d["LapTime_s"] < 1.07 * d["LapTime_s"].median()]
    return d[["LapNumber", "LapTime_s", "Compound", "TyreLife", "Stint"]]


def paired_gaps(laps, d1, d2):
    """Lap-by-lap gap d1 - d2 (seconds). Negative = d1 faster."""
    a = clean_laps(laps, d1)
    b = clean_laps(laps, d2)
    keys = ["LapNumber", "Compound"] if SAME_COMPOUND else ["LapNumber"]
    m = a.merge(b, on=keys, suffixes=("_1", "_2"))
    return (m["LapTime_s_1"] - m["LapTime_s_2"]).to_numpy()


def bootstrap_mean(values):
    """Bootstrap distribution of the mean of `values`."""
    n = len(values)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    return values[idx].mean(axis=1)


def bootstrap_slope(x, y):
    """Bootstrap distribution of the regression slope of y on x."""
    n = len(x)
    slopes = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        xs, ys = x[idx], y[idx]
        if np.ptp(xs) == 0:          # degenerate resample, all same x
            slopes[i] = np.nan
            continue
        slopes[i] = np.polyfit(xs, ys, 1)[0]
    return slopes[~np.isnan(slopes)]


# ============================================================
# PREFLIGHT : load ONE race first, stop clearly if the data API fails
# ============================================================
def preflight():
    try:
        s = fastf1.get_session(YEAR, 1, "R")
        s.load(telemetry=False, weather=False, messages=False)
        n = len(s.laps)
        print(f"[preflight] OK, {n} laps loaded for round 1")
        return n > 0
    except Exception as e:
        print(f"[preflight] FAILED: {e}")
        return False


if not preflight():
    raise SystemExit(
        "\nFastF1 could not download data (see the WARNING lines above).\n"
        "Check: internet/VPN/firewall, `pip install -U fastf1`, wait a few minutes "
        "(rate limit), or point F1_CACHE to the folder of your earlier runs."
    )

# ============================================================
# MAIN LOOP : load every dry race once
# ============================================================
schedule = fastf1.get_event_schedule(YEAR, include_testing=False)

gap_rows = []                                   # per GP and pair
pooled = {f"{a}_{b}": [] for a, b in PAIRS}     # all paired laps of the season
slope_rows = []                                 # per stint

for _, event in schedule.iterrows():
    gp = event["EventName"]
    try:
        session = fastf1.get_session(YEAR, gp, "R")
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps
    except Exception as e:
        print(f"[skip] {gp}: {e}")
        continue

    if not is_dry_race(laps):
        print(f"[skip] {gp}: wet race")
        continue

    print(f"[ok]   {gp}")

    # ---- A) Paired lap-time gap ----
    for d1, d2 in PAIRS:
        label = f"{d1}_{d2}"
        try:
            gaps = paired_gaps(laps, d1, d2)
        except Exception as e:
            print(f"       {label}: {e}")
            continue
        if len(gaps) < MIN_PAIRED_LAPS:
            continue
        pooled[label].append(gaps)
        boot = bootstrap_mean(gaps)
        lo, hi = np.percentile(boot, [lo_q, hi_q])
        gap_rows.append(dict(
            gp=gp, pair=label, n_laps=len(gaps),
            mean_gap_s=round(gaps.mean(), 3),
            ci_low=round(lo, 3), ci_high=round(hi, 3),
            systematic=bool(lo > 0 or hi < 0),
        ))

    # ---- B) Bootstrap on the degradation slope ----
    for driver in sorted({d for pair in PAIRS for d in pair}):
        d = clean_laps(laps, driver)
        for stint, s in d.groupby("Stint"):
            if len(s) < MIN_STINT_LAPS:
                continue
            x = s["TyreLife"].to_numpy(float)
            y = s["LapTime_s"].to_numpy(float)
            slopes = bootstrap_slope(x, y) * 1000  # ms/lap
            if len(slopes) == 0:
                continue
            lo, hi = np.percentile(slopes, [lo_q, hi_q])
            slope_rows.append(dict(
                gp=gp, driver=driver, stint=int(stint),
                compound=s["Compound"].iloc[0], n_laps=len(s),
                slope_ms_per_lap=round(np.polyfit(x, y, 1)[0] * 1000, 1),
                ci_low=round(lo, 1), ci_high=round(hi, 1),
                degradation_significant=bool(lo > 0),
            ))

# ============================================================
# RESULTS
# ============================================================
df_gap = pd.DataFrame(gap_rows)
df_slope = pd.DataFrame(slope_rows)
df_gap.to_csv(f"{OUT_DIR}/bootstrap_gap_per_gp.csv", index=False)
df_slope.to_csv(f"{OUT_DIR}/bootstrap_degradation_slopes.csv", index=False)

print("\n" + "=" * 70)
print(f"A) LAP-TIME GAP (first driver minus second, {CI} % bootstrap CI)")
print("=" * 70)
print("Negative gap = first driver faster.\n")

for label, chunks in pooled.items():
    if not chunks:
        print(f"{label}: no data")
        continue
    gaps = np.concatenate(chunks)
    boot = bootstrap_mean(gaps)
    lo, hi = np.percentile(boot, [lo_q, hi_q])
    verdict = "SYSTEMATIC gap" if (lo > 0 or hi < 0) else "NOT conclusive (CI contains 0)"
    print(f"{label}: mean gap = {gaps.mean():+.3f} s over {len(gaps)} paired laps | "
          f"{CI} % CI [{lo:+.3f}, {hi:+.3f}] -> {verdict}")

    plt.figure(figsize=(8, 4))
    plt.hist(boot, bins=60, color="steelblue", alpha=0.8)
    plt.axvline(0, color="black", linestyle="--", label="no gap")
    plt.axvline(lo, color="red", linestyle=":", label=f"{CI} % CI")
    plt.axvline(hi, color="red", linestyle=":")
    plt.xlabel("Mean lap-time gap (s)  [first driver - second driver]")
    plt.ylabel("Number of resamples")
    plt.title(f"Bootstrap distribution of the gap {label.replace('_', ' vs ')} ({YEAR})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/bootstrap_gap_{label}.png", dpi=130)
    plt.close()

print("\nPer-GP table saved: outputs/bootstrap_gap_per_gp.csv")

print("\n" + "=" * 70)
print("B) TYRE DEGRADATION SLOPES WITH BOOTSTRAP CI")
print("=" * 70)
if df_slope.empty:
    print("No stint long enough.")
else:
    share = df_slope["degradation_significant"].mean() * 100
    print(f"{len(df_slope)} stints analysed; slope CI entirely > 0 in {share:.0f} % of them.")
    print("A CI that contains 0 means the slope is not distinguishable from noise.")
    print("Details: outputs/bootstrap_degradation_slopes.csv")

print("""
LIMITS 
- Consecutive laps are not independent (tyre wear, traffic, strategy), so a
  simple bootstrap can be too optimistic. Improvement: block bootstrap
  (resample whole stints instead of single laps).
- Pairing on lap number reduces the fuel effect (both cars lighten together)
  but does not remove strategy or traffic differences.
- Fuel effect (~0.03 s/lap) is not corrected in the slope: it hides part of
  the real tyre degradation.
- Bootstrap measures uncertainty, it does not fix a biased lap selection.
""")