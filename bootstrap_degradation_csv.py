"""
Bootstrap on tyre degradation, from your exported CSV (works OFFLINE)
=====================================================================
Input : outputs/all_degradation_results.csv  (made by tire_degradation.py)
        columns used: gp, pair, driver, stint, compound, n_laps, deg_ms_per_lap

Question: across the dry races, is NOR's degradation rate systematically
different from PIA's (same car), on the SAME tyre compound?

Method  : for each GP, take the first stint of each driver on the same compound,
          compute the difference (NOR - PIA) in ms/lap, then bootstrap the
          mean of these differences (resample the GPs with replacement).

Run:  python bootstrap_degradation_csv.py [path_to_csv]
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "outputs/all_degradation_results.csv"
D1, D2 = "NOR", "PIA"
MIN_LAPS = 8            # ignore very short stints (slope too uncertain)
N_BOOT, CI, SEED = 10_000, 95, 42
rng = np.random.default_rng(SEED)
lo_q, hi_q = (100 - CI) / 2, 100 - (100 - CI) / 2

df = pd.read_csv(CSV)
need = ["gp", "pair", "driver", "stint", "compound", "n_laps", "deg_ms_per_lap"]
miss = [c for c in need if c not in df.columns]
if miss:
    raise SystemExit(f"Missing columns: {miss}\nFound: {list(df.columns)}")

# NOR appears in two pairs: keep the teammate pair only, to avoid duplicates
df = df[df["pair"] == f"{D1}_vs_{D2}"]
df = df[df["n_laps"] >= MIN_LAPS]
first = df[df["stint"] == 1]

a = first[first["driver"] == D1][["gp", "compound", "deg_ms_per_lap"]]
b = first[first["driver"] == D2][["gp", "compound", "deg_ms_per_lap"]]
m = a.merge(b, on=["gp", "compound"], suffixes=(f"_{D1}", f"_{D2}"))
m["diff"] = m[f"deg_ms_per_lap_{D1}"] - m[f"deg_ms_per_lap_{D2}"]

print(f"{len(m)} races where both drivers have a first stint >= {MIN_LAPS} laps on the same compound\n")
if len(m) < 5:
    raise SystemExit("Not enough paired races for a meaningful bootstrap.")

x = m["diff"].to_numpy(float)
idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
boot = x[idx].mean(axis=1)
lo, hi = np.percentile(boot, [lo_q, hi_q])
verdict = "SYSTEMATIC difference" if (lo > 0 or hi < 0) else "not conclusive (CI contains 0)"
print(f"Mean degradation difference ({D1} - {D2}) = {x.mean():+.1f} ms/lap")
print(f"{CI} % bootstrap CI [{lo:+.1f}, {hi:+.1f}] ms/lap -> {verdict}")
print("Positive = NOR degrades faster than PIA.\n")
print(m.round(1).to_string(index=False))

plt.figure(figsize=(8, 4))
plt.hist(boot, bins=50, color="steelblue", alpha=0.8)
plt.axvline(0, color="black", linestyle="--", label="no difference")
plt.axvline(lo, color="red", linestyle=":", label=f"{CI} % CI")
plt.axvline(hi, color="red", linestyle=":")
plt.xlabel(f"Mean degradation difference {D1} - {D2} (ms/lap)")
plt.ylabel("Number of resamples")
plt.title("Bootstrap: tyre degradation NOR vs PIA (first stints, same compound)")
plt.legend()
plt.tight_layout()
plt.savefig("bootstrap_degradation_NOR_PIA.png", dpi=130)
print("\nSaved: bootstrap_degradation_NOR_PIA.png")
print("""
LIMITS 
- Each slope comes from a linear fit on few laps, so each point is itself uncertain.
- Fuel burn (~0.03 s/lap) is not corrected: it lowers every slope, for both drivers.
- Only dry races with a first stint >= 8 laps on the same compound are kept.
- Resampling GPs treats races as independent; circuits differ a lot in tyre wear.
""")