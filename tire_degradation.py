"""
F1 2025 — Tire Degradation & Pit Strategy: ALL DRY GPs
========================================================
Comparisons:
  - NOR vs PIA (teammates, same car — isolates driver skill)
  - NOR vs LEC (inter-team — car + driver combined)

For EVERY dry race in 2025:
  1. Detects wet races automatically and skips them
  2. Models tire degradation per stint (linear regression)
  3. Simulates 1-stop vs 2-stop strategy
  4. Exports one summary CSV with all GPs
"""

import fastf1
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from scipy import stats
import os, warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
CACHE_DIR = './f1_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs('./outputs', exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

YEAR        = 2025
PAIRS       = [('NOR', 'PIA'), ('NOR', 'LEC')]
COLORS      = {'NOR': 'orange', 'PIA': 'navy', 'LEC': '#dc2626'}
PIT_LOSS_S  = 22.0
MIN_DRY_PCT = 0.80   # race must be ≥80% dry laps to be included

# ============================================================
# HELPERS
# ============================================================

def is_dry_race(session_laps):
    """Returns True if ≥80% of laps have no intermediate/wet compound."""
    if 'Compound' not in session_laps.columns:
        return True
    wet = session_laps['Compound'].isin(['INTERMEDIATE', 'WET'])
    return (wet.sum() / max(len(session_laps), 1)) < (1 - MIN_DRY_PCT)


def get_stints(laps, driver):
    drv = laps.pick_drivers(driver).copy()
    drv['LapTime_s'] = drv['LapTime'].dt.total_seconds()
    drv = drv[
        drv['PitInTime'].isna() &
        drv['PitOutTime'].isna() &
        drv['LapTime_s'].notna() &
        (drv['LapTime_s'] > 0)
    ]
    med = drv['LapTime_s'].median()
    std = drv['LapTime_s'].std()
    drv = drv[abs(drv['LapTime_s'] - med) < 2 * std]
    stints = []
    for _, s in drv.groupby('Stint'):
        if len(s) >= 4:
            stints.append(s.sort_values('LapNumber').reset_index(drop=True))
    return stints


def fit_deg(stint_df):
    x = stint_df['TyreLife'].values.astype(float)
    y = stint_df['LapTime_s'].values.astype(float)
    slope, intercept, r, p, se = stats.linregress(x, y)
    return dict(slope_s=slope, slope_ms=slope*1000,
                intercept=intercept, r2=r**2, p=p, se=se, n=len(x))


def sim_strategy(stints_laps, deg_s, base_s, race_laps, pit_loss=PIT_LOSS_S):
    # adjust last stint so total = race_laps
    stints_laps = list(stints_laps)
    stints_laps[-1] = race_laps - sum(stints_laps[:-1])
    if stints_laps[-1] <= 0:
        return float('inf')
    total = sum(base_s + deg_s * age
                for sl in stints_laps for age in range(1, sl+1))
    total += (len(stints_laps) - 1) * pit_loss
    return total


# ============================================================
# MAIN LOOP
# ============================================================
schedule = fastf1.get_event_schedule(YEAR, include_testing=False)
all_deg_rows = []
all_strat_rows = []

for _, event in schedule.iterrows():
    gp = event['EventName']
    print(f"\n{'='*60}")
    print(f"  GP: {gp}")
    print(f"{'='*60}")

    try:
        session = fastf1.get_session(YEAR, gp, 'R')
        session.load()
        laps = session.laps.copy()
    except Exception as e:
        print(f"  Skipped (load error): {e}")
        continue

    if not is_dry_race(laps):
        print(f"  Skipped — wet/intermediate race detected")
        continue

    race_laps = int(laps['LapNumber'].max()) if 'LapNumber' in laps.columns else 50

    for d1, d2 in PAIRS:
        pair_label = f"{d1}_vs_{d2}"
        print(f"\n  --- {d1} vs {d2} ---")

        # ── Degradation ──────────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=False)
        compound_colors = {
            'SOFT':'#e63946','MEDIUM':'#f4a261',
            'HARD':'#adb5bd','INTER':'#52b788','WET':'#023e8a'
        }

        for ax, driver in zip(axes, [d1, d2]):
            stints = get_stints(laps, driver)
            ax.set_title(f'{driver} — Degradation (Silverstone replaced by {gp})',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Tyre Age (laps)')
            ax.set_ylabel('Lap Time (s)')

            for i, stint_df in enumerate(stints):
                compound = stint_df['Compound'].iloc[0] if 'Compound' in stint_df.columns else 'UNK'
                col = compound_colors.get(compound, 'gray')
                deg = fit_deg(stint_df)

                x = stint_df['TyreLife'].values.astype(float)
                y = stint_df['LapTime_s'].values.astype(float)
                xf = np.linspace(x.min(), x.max(), 100)
                yf = deg['slope_s'] * xf + deg['intercept']

                n = deg['n']
                t = stats.t.ppf(0.975, df=max(n-2,1))
                ci = t * deg['se'] * np.sqrt(1/n + (xf-x.mean())**2/max(((x-x.mean())**2).sum(),1e-9))

                ax.scatter(x, y, color=col, alpha=0.7, s=35,
                           label=f'Stint {i+1} {compound} ({n} laps)')
                ax.plot(xf, yf, color=col, lw=2)
                ax.fill_between(xf, yf-ci, yf+ci, color=col, alpha=0.13)
                ax.annotate(f'{deg["slope_ms"]:+.0f} ms/lap  R²={deg["r2"]:.2f}',
                            xy=(xf[-1], yf[-1]), fontsize=8, color=col,
                            xytext=(5,0), textcoords='offset points')

                all_deg_rows.append(dict(
                    gp=gp, pair=pair_label, driver=driver,
                    stint=i+1, compound=compound, n_laps=n,
                    deg_ms_per_lap=round(deg['slope_ms'],2),
                    r2=round(deg['r2'],3),
                    base_pace_s=round(deg['intercept'],3),
                ))
                print(f"    {driver} Stint {i+1} [{compound}] "
                      f"{deg['slope_ms']:+.1f} ms/lap | R²={deg['r2']:.2f} | {n} laps")

            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{d1} vs {d2} — Tire Degradation — {gp} {YEAR}',
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        fname = f'outputs/degradation_{pair_label}_{gp.replace(" ","_")}.png'
        plt.savefig(fname, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {fname}")

        # ── Strategy simulation ───────────────────────────────────
        for driver in [d1, d2]:
            rows = [r for r in all_deg_rows
                    if r['gp']==gp and r['pair']==pair_label and r['driver']==driver]
            if not rows:
                continue
            deg_s    = rows[0]['deg_ms_per_lap'] / 1000
            base_s   = rows[0]['base_pace_s']

            strategies = {
                '1-stop early':  [int(race_laps*0.38), int(race_laps*0.62)],
                '1-stop mid':    [int(race_laps*0.50), int(race_laps*0.50)],
                '1-stop late':   [int(race_laps*0.62), int(race_laps*0.38)],
                '2-stop equal':  [int(race_laps/3),    int(race_laps/3),    int(race_laps/3)],
                '2-stop early':  [int(race_laps*0.25), int(race_laps*0.38), int(race_laps*0.37)],
            }

            times = {k: sim_strategy(v, deg_s, base_s, race_laps)
                     for k, v in strategies.items()}
            best  = min(times, key=times.get)

            for strat, t in times.items():
                all_strat_rows.append(dict(
                    gp=gp, pair=pair_label, driver=driver,
                    strategy=strat, total_time_s=round(t,2),
                    is_optimal=(strat==best)
                ))

# ============================================================
# SUMMARY PLOTS
# ============================================================
df_deg   = pd.DataFrame(all_deg_rows)
df_strat = pd.DataFrame(all_strat_rows)

df_deg.to_csv('outputs/all_degradation_results.csv', index=False)
df_strat.to_csv('outputs/all_strategy_results.csv', index=False)

# -- Degradation heatmap: driver vs GP
for pair_label in df_deg['pair'].unique():
    sub = df_deg[df_deg['pair']==pair_label]
    # Average degradation per driver per GP (stint 1 only for consistency)
    pivot = sub[sub['stint']==1].pivot_table(
        index='driver', columns='gp', values='deg_ms_per_lap', aggfunc='mean')

    if pivot.empty:
        continue

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns)*1.2), 4))
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Degradation (ms/lap) — red=worse')
    ax.set_title(f'{pair_label.replace("_"," ")} — Stint 1 Degradation Rate Heatmap (All Dry GPs {YEAR})',
                 fontweight='bold')

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=8, color='black')

    plt.tight_layout()
    hname = f'outputs/heatmap_degradation_{pair_label}.png'
    plt.savefig(hname, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nSaved heatmap: {hname}")

print("\n" + "="*60)
print("ALL DONE")
print("="*60)
print("  outputs/all_degradation_results.csv")
print("  outputs/all_strategy_results.csv")
print("  outputs/degradation_*.png  (one per GP per pair)")
print("  outputs/heatmap_degradation_*.png  (season overview)")