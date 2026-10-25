"""
Bootstrap from the season CSV (works OFFLINE, no FastF1 download needed)
=======================================================================
Input : season_2025_NOR_vs_PIA_teammates.csv  (made by telemetryAnalysis_NOR_vs_PIA_teammates.py)
Idea  : for each race we have one paired observation (same car, same weekend).
        We resample the ~24 races with replacement 10 000 times and ask:
        is the gap between teammates systematic or can it be luck?

Metrics (NOR minus PIA, race by race):
  - points            (positive = NOR scored more)
  - finishing position (negative = NOR finished ahead)
  - grid position      (negative = NOR qualified ahead)
  - share of races where NOR finished ahead of PIA

Run:  python bootstrap_from_csv.py [path_to_csv]
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "season_2025_NOR_vs_PIA_teammates.csv"
D1, D2 = "NOR", "PIA"
N_BOOT, CI, SEED = 10_000, 95, 42
rng = np.random.default_rng(SEED)
lo_q, hi_q = (100 - CI) / 2, 100 - (100 - CI) / 2

df = pd.read_csv(CSV)
need = [f"{d}_{c}" for d in (D1, D2) for c in ("points", "finish", "grid")]
missing = [c for c in need if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in {CSV}: {missing}\nColumns found: {list(df.columns)}")

for c in need:
    df[c] = pd.to_numeric(df[c], errors="coerce")

def boot_mean(x):
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    return x[idx].mean(axis=1)

metrics = {
    "points (NOR - PIA)": (df[f"{D1}_points"] - df[f"{D2}_points"]),
    "finish position (NOR - PIA)": (df[f"{D1}_finish"] - df[f"{D2}_finish"]),
    "grid position (NOR - PIA)": (df[f"{D1}_grid"] - df[f"{D2}_grid"]),
    "NOR ahead of PIA (share of races)": (df[f"{D1}_finish"] < df[f"{D2}_finish"]).astype(float).where(
        df[f"{D1}_finish"].notna() & df[f"{D2}_finish"].notna()),
}

print(f"{len(df)} races in the CSV\n")
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, (name, s) in zip(axes.ravel(), metrics.items()):
    x = s.dropna().to_numpy(float)
    if len(x) < 5:
        print(f"{name}: not enough races ({len(x)})")
        continue
    b = boot_mean(x)
    lo, hi = np.percentile(b, [lo_q, hi_q])
    ref = 0.5 if "share" in name else 0.0
    verdict = "SYSTEMATIC" if (lo > ref or hi < ref) else "not conclusive (CI contains the no-difference value)"
    print(f"{name}\n  mean = {x.mean():+.2f} over {len(x)} races | {CI} % CI [{lo:+.2f}, {hi:+.2f}] -> {verdict}\n")
    ax.hist(b, bins=50, color="steelblue", alpha=0.8)
    ax.axvline(ref, color="black", linestyle="--")
    ax.axvline(lo, color="red", linestyle=":")
    ax.axvline(hi, color="red", linestyle=":")
    ax.set_title(name, fontsize=10)
plt.tight_layout()
plt.savefig("bootstrap_season_NOR_PIA.png", dpi=130)
print("Saved: bootstrap_season_NOR_PIA.png")
print("""
LIMITS 
- Only ~24 races: wide intervals are normal and honest.
- Sprint points are not in the CSV; grid penalties and DNFs (mechanical,
  collisions) are counted like any other race, which adds noise.
- This tests the SEASON RESULT gap between teammates, not the lap-time gap.
""")