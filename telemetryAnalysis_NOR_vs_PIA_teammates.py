"""
F1 2025 Season Analysis: Lando Norris (NOR) vs Oscar Piastri (PIA)
======================================================================
UNLIKE the NOR vs LEC script, these two drivers share the SAME CAR and
SAME TEAM (McLaren) all season. That means car performance, upgrades,
and (mostly) strategy are constants shared by both drivers — so any
points/pace gap between them is much closer to an isolated measure of
DRIVER performance, rather than a mix of car + driver like a cross-team
comparison would be.

This script:
1. Compares single-lap telemetry (speed, throttle, brake, G-forces) at Silverstone
2. Overlays REAL corner numbers from the official Silverstone track map
3. Detects braking zones and labels them with official corner numbers
4. Pulls the FULL 2025 season results and compares points, positions, DNFs
5. Adds a TEAMMATE-SPECIFIC analysis:
      - qualifying head-to-head (who out-qualified whom, race by race)
      - cumulative championship points over the season (trend, not just a total)
      - a concrete, data-driven explanation of WHERE the points gap comes from
6. Prints a methodology / limitations section at the end
"""

import fastf1
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIGURATION
# ============================================================
CACHE_DIR = './f1_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

driver1 = 'NOR'
driver2 = 'PIA'
YEAR = 2025

DRIVER_SURNAMES = {
    'NOR': 'Norris',
    'PIA': 'Piastri',
}

# ============================================================
# PART 1 — SINGLE-LAP TELEMETRY COMPARISON (Silverstone)
# ============================================================
print("="*70)
print("PART 1 — SILVERSTONE TELEMETRY COMPARISON (TEAMMATES)")
print("="*70)

session = fastf1.get_session(YEAR, 'Silverstone', 'R')
session.load()

lap1 = session.laps.pick_drivers(driver1).pick_fastest()
lap2 = session.laps.pick_drivers(driver2).pick_fastest()

tel1 = lap1.get_telemetry()
tel2 = lap2.get_telemetry()

# ---- Pull the REAL corner numbers from the official track map ----
circuit_info = session.get_circuit_info()
corners = circuit_info.corners  # columns: Number, Angle, Distance, X, Y, ...

def nearest_corner_number(distance, corners_df):
    """Match a distance-on-lap value to the closest official corner number."""
    idx = (corners_df['Distance'] - distance).abs().idxmin()
    return int(corners_df.loc[idx, 'Number'])

# ---- Plot 1: Speed / Throttle / Brake with corner numbers overlaid ----
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(tel1['Distance'], tel1['Speed'], label=driver1, color='orange')
axes[0].plot(tel2['Distance'], tel2['Speed'], label=driver2, color='navy')
axes[0].set_ylabel('Speed (km/h)')
axes[0].legend()
axes[0].set_title(f'{driver1} vs {driver2} — Teammate Telemetry Comparison (Silverstone {YEAR})')

axes[1].plot(tel1['Distance'], tel1['Throttle'], color='orange')
axes[1].plot(tel2['Distance'], tel2['Throttle'], color='navy')
axes[1].set_ylabel('Throttle (%)')

axes[2].plot(tel1['Distance'], tel1['Brake'], color='orange')
axes[2].plot(tel2['Distance'], tel2['Brake'], color='navy')
axes[2].set_ylabel('Braking')
axes[2].set_xlabel('Distance (m)')

y_top = axes[0].get_ylim()[1]
for _, corner in corners.iterrows():
    for ax in axes:
        ax.axvline(corner['Distance'], color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
    axes[0].text(corner['Distance'], y_top * 0.97, f"T{int(corner['Number'])}",
                 rotation=90, fontsize=7, ha='center', va='top', color='dimgray')

plt.tight_layout()
plt.savefig('comparison_telemetry_NOR_PIA_with_corners.png', dpi=150)
plt.show()  # <-- close this window to continue the script

# ---- Plot 2: Speed delta ----
common_dist = np.linspace(0, min(tel1['Distance'].max(), tel2['Distance'].max()), 1000)
speed1_interp = np.interp(common_dist, tel1['Distance'], tel1['Speed'])
speed2_interp = np.interp(common_dist, tel2['Distance'], tel2['Speed'])

plt.figure(figsize=(12, 4))
plt.plot(common_dist, speed1_interp - speed2_interp, color='darkgreen')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Distance (m)')
plt.ylabel(f'Speed delta {driver1} - {driver2} (km/h)')
plt.title('Where each driver gains or loses speed (same car)')
plt.savefig('delta_speed_NOR_PIA.png', dpi=150)
plt.show()

# ============================================================
# G-FORCES
# ============================================================
def add_gforce(tel):
    """Adds derived longitudinal and lateral G-force to the telemetry dataframe."""
    tel = tel.copy()
    speed_ms = tel['Speed'] / 3.6
    time_s = tel['Time'].dt.total_seconds()
    dt = time_s.diff()

    tel['LongG'] = speed_ms.diff() / dt / 9.81

    dx = tel['X'].diff()
    dy = tel['Y'].diff()
    ds = np.sqrt(dx**2 + dy**2)

    heading = np.arctan2(dy, dx)
    dheading = heading.diff()
    dheading = np.mod(dheading + np.pi, 2 * np.pi) - np.pi

    curvature = dheading / ds
    tel['LatG'] = (speed_ms**2 * curvature) / 9.81

    return tel

tel1 = add_gforce(tel1)
tel2 = add_gforce(tel2)

fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax[0].plot(tel1['Distance'], tel1['LongG'], color='orange', label=driver1)
ax[0].plot(tel2['Distance'], tel2['LongG'], color='navy', label=driver2)
ax[0].set_ylabel('Longitudinal G')
ax[0].axhline(0, color='gray', linewidth=0.5)
ax[0].legend()

ax[1].plot(tel1['Distance'], tel1['LatG'], color='orange')
ax[1].plot(tel2['Distance'], tel2['LatG'], color='navy')
ax[1].set_ylabel('Lateral G')
ax[1].set_xlabel('Distance (m)')

plt.tight_layout()
plt.savefig('gforces_comparison_NOR_PIA.png', dpi=150)
plt.show()

# ============================================================
# BRAKING ZONE ANALYSIS — labeled with REAL corner numbers
# ============================================================
def analyse_turns(tel, driver_name, corners_df):
    """Detects braking zones and labels each one with the nearest official corner number."""
    results = []
    braking = False
    brake_start = None

    for i in range(1, len(tel)):
        if tel['Brake'].iloc[i] == 1 and not braking:
            braking = True
            brake_start = tel['Distance'].iloc[i]
        elif tel['Brake'].iloc[i] == 0 and braking:
            braking = False
            brake_end = tel['Distance'].iloc[i]

            zone = tel[(tel['Distance'] >= brake_start) & (tel['Distance'] <= brake_end)]
            v_min = zone['Speed'].min()
            apex_dist = zone.loc[zone['Speed'].idxmin(), 'Distance']
            corner_number = nearest_corner_number(apex_dist, corners_df)

            results.append({
                'driver': driver_name,
                'corner': corner_number,
                'brake_start_m': round(brake_start, 1),
                'brake_end_m': round(brake_end, 1),
                'brake_length_m': round(brake_end - brake_start, 1),
                'apex_speed_kmh': round(v_min, 1),
                'apex_position_m': round(apex_dist, 1)
            })

    return pd.DataFrame(results).sort_values('corner').reset_index(drop=True)

turns1 = analyse_turns(tel1, driver1, corners)
turns2 = analyse_turns(tel2, driver2, corners)

print(f"\n=== Braking zones — {driver1} (with official corner numbers) ===")
print(turns1)
print(f"\n=== Braking zones — {driver2} (with official corner numbers) ===")
print(turns2)

turns1.to_csv(f'turns_{driver1}_silverstone_teammates.csv', index=False)
turns2.to_csv(f'turns_{driver2}_silverstone_teammates.csv', index=False)

# ============================================================
# PART 2 — FULL 2025 SEASON: RESULTS, WINNERS, POINTS COMPARISON
# ============================================================
print("\n" + "="*70)
print(f"PART 2 — FULL {YEAR} SEASON: {driver1} vs {driver2} (TEAMMATES)")
print("="*70)

schedule = fastf1.get_event_schedule(YEAR, include_testing=False)

season_data = []

for _, event in schedule.iterrows():
    gp_name = event['EventName']
    try:
        race = fastf1.get_session(YEAR, gp_name, 'R')
        race.load(telemetry=False, weather=False)

        results = race.results

        winner_row = results[results['Position'] == 1]
        winner = winner_row['Abbreviation'].values[0] if not winner_row.empty else 'N/A'

        d1_row = results[results['Abbreviation'] == driver1]
        d2_row = results[results['Abbreviation'] == driver2]

        def safe_val(row, col, default=None):
            return row[col].values[0] if not row.empty and pd.notna(row[col].values[0]) else default

        d1_pos = safe_val(d1_row, 'Position')
        d1_grid = safe_val(d1_row, 'GridPosition')
        d1_pts = safe_val(d1_row, 'Points', 0)
        d1_status = safe_val(d1_row, 'Status', 'N/A')

        d2_pos = safe_val(d2_row, 'Position')
        d2_grid = safe_val(d2_row, 'GridPosition')
        d2_pts = safe_val(d2_row, 'Points', 0)
        d2_status = safe_val(d2_row, 'Status', 'N/A')

        d1_gain = (d1_grid - d1_pos) if d1_grid is not None and d1_pos is not None else None
        d2_gain = (d2_grid - d2_pos) if d2_grid is not None and d2_pos is not None else None

        def count_incidents(race_session, driver_code):
            msgs = race_session.race_control_messages
            if msgs is None or msgs.empty or 'Message' not in msgs.columns:
                return 0
            surname = DRIVER_SURNAMES.get(driver_code, driver_code)
            mask = msgs['Message'].str.contains(surname, case=False, na=False)
            return int(mask.sum())

        d1_incidents = count_incidents(race, driver1)
        d2_incidents = count_incidents(race, driver2)

        # Who finished ahead of their teammate this race (raw head-to-head)
        teammate_winner = None
        if d1_pos is not None and d2_pos is not None:
            teammate_winner = driver1 if d1_pos < d2_pos else driver2

        season_data.append({
            'GP': gp_name,
            'Winner': winner,
            'Teammate_ahead': teammate_winner,
            f'{driver1}_grid': d1_grid,
            f'{driver1}_finish': d1_pos,
            f'{driver1}_positions_gained': d1_gain,
            f'{driver1}_points': d1_pts,
            f'{driver1}_status': d1_status,
            f'{driver1}_incidents_mentioned': d1_incidents,
            f'{driver2}_grid': d2_grid,
            f'{driver2}_finish': d2_pos,
            f'{driver2}_positions_gained': d2_gain,
            f'{driver2}_points': d2_pts,
            f'{driver2}_status': d2_status,
            f'{driver2}_incidents_mentioned': d2_incidents,
        })

        print(f"{gp_name:30s} winner={winner:4s} | {driver1} P{d1_pos} ({d1_pts} pts) | "
              f"{driver2} P{d2_pos} ({d2_pts} pts) | teammate ahead: {teammate_winner}")

    except Exception as e:
        print(f"Skipped {gp_name}: {e}")

df_season = pd.DataFrame(season_data)

# ---- Cumulative points over the season (the key teammate-trend chart) ----
df_season[f'{driver1}_cum_points'] = df_season[f'{driver1}_points'].cumsum()
df_season[f'{driver2}_cum_points'] = df_season[f'{driver2}_points'].cumsum()

df_season.to_csv(f'season_{YEAR}_{driver1}_vs_{driver2}_teammates.csv', index=False)

plt.figure(figsize=(13, 5))
plt.plot(df_season['GP'], df_season[f'{driver1}_cum_points'], marker='o', color='orange', label=driver1)
plt.plot(df_season['GP'], df_season[f'{driver2}_cum_points'], marker='o', color='navy', label=driver2)
plt.xticks(rotation=75, ha='right')
plt.ylabel('Cumulative championship points')
plt.title(f'{driver1} vs {driver2} — Cumulative Points Over the {YEAR} Season (Same Team, Same Car)')
plt.legend()
plt.tight_layout()
plt.savefig('cumulative_points_NOR_PIA.png', dpi=150)
plt.show()

# ============================================================
# PART 3 — TEAMMATE-SPECIFIC ANALYSIS (isolates driver skill)
# ============================================================
print("\n" + "="*70)
print("PART 3 — TEAMMATE HEAD-TO-HEAD: QUALIFYING & CONCRETE EXPLANATION")
print("="*70)

quali_ahead_count = {driver1: 0, driver2: 0}
races_with_quali_data = 0

for _, event in schedule.iterrows():
    gp_name = event['EventName']
    try:
        quali = fastf1.get_session(YEAR, gp_name, 'Q')
        quali.load(telemetry=False, weather=False)
        q_results = quali.results

        d1_q = q_results[q_results['Abbreviation'] == driver1]
        d2_q = q_results[q_results['Abbreviation'] == driver2]

        if not d1_q.empty and not d2_q.empty:
            d1_qpos = d1_q['Position'].values[0]
            d2_qpos = d2_q['Position'].values[0]
            if pd.notna(d1_qpos) and pd.notna(d2_qpos):
                races_with_quali_data += 1
                if d1_qpos < d2_qpos:
                    quali_ahead_count[driver1] += 1
                else:
                    quali_ahead_count[driver2] += 1
    except Exception as e:
        print(f"Qualifying skipped for {gp_name}: {e}")

print(f"\nQualifying head-to-head across {races_with_quali_data} sessions with data for both drivers:")
print(f"  {driver1} out-qualified {driver2}: {quali_ahead_count[driver1]} times")
print(f"  {driver2} out-qualified {driver1}: {quali_ahead_count[driver2]} times")

# ---- Race-day head-to-head (from Part 2 data) ----
race_ahead_count = df_season['Teammate_ahead'].value_counts().to_dict()
print(f"\nRace-day head-to-head (final classification):")
print(f"  {driver1} finished ahead of {driver2}: {race_ahead_count.get(driver1, 0)} times")
print(f"  {driver2} finished ahead of {driver1}: {race_ahead_count.get(driver2, 0)} times")

# ---- Season totals ----
total_d1 = df_season[f'{driver1}_points'].sum()
total_d2 = df_season[f'{driver2}_points'].sum()
dnf_d1 = (df_season[f'{driver1}_status'] != 'Finished').sum()
dnf_d2 = (df_season[f'{driver2}_status'] != 'Finished').sum()
avg_gain_d1 = df_season[f'{driver1}_positions_gained'].mean()
avg_gain_d2 = df_season[f'{driver2}_positions_gained'].mean()

leader = driver1 if total_d1 > total_d2 else driver2
gap = abs(total_d1 - total_d2)

print("\n" + "="*70)
print("CONCRETE EXPLANATION: WHERE THE POINTS GAP ACTUALLY CAME FROM")
print("="*70)
print(f"Season totals: {driver1} = {total_d1} pts | {driver2} = {total_d2} pts")
print(f"{leader} leads the head-to-head by {gap:.0f} points.\n")
print("Because both drivers share the exact same car all season, this gap should")
print("be explained almost entirely by the three factors below — check which ones")
print("actually favor the leader once the printed numbers are in front of you:\n")
print(f"  1. QUALIFYING  — {driver1} ahead {quali_ahead_count[driver1]}x vs {driver2} ahead {quali_ahead_count[driver2]}x")
print(f"  2. RACE EXECUTION — avg positions gained/race: {driver1} {avg_gain_d1:.2f} vs {driver2} {avg_gain_d2:.2f}")
print(f"  3. RELIABILITY — non-finishes: {driver1} {dnf_d1} vs {driver2} {dnf_d2}")
print("\nWhichever driver leads on 2 or 3 of these metrics is very likely the real")
print("source of the championship gap — not just 'being faster' in a vague sense.")

# ============================================================
# METHODOLOGY & LIMITATIONS
# ============================================================
print("\n" + "="*70)
print("METHODOLOGY & LIMITATIONS")
print("="*70)
print("""
1. Braking zones are detected from a binary Brake=1/0 signal. Very light lifts
   or partial trail-braking below the sensor's threshold may be missed, or
   split into extra short zones that don't represent a real distinct corner.

2. Corner numbers are matched to the NEAREST official corner marker distance
   from FastF1's circuit info. This is an approximation based on distance,
   not an exact corner-entry detection.

3. G-forces are DERIVED, not measured directly, and are sensitive to the
   resolution of the underlying position (X/Y) data.

4. "Positions gained" = Grid Position minus Finish Position. It does NOT
   account for grid penalties, safety car timing, or strategy-only pit stops.

5. "Race-control mentions" is a best-effort surname text search in official
   messages — not a verified penalty count.

6. UNLIKE the NOR vs LEC script, this comparison uses TEAMMATES on identical
   machinery, which removes car performance as a variable — but it does NOT
   remove team strategy as a variable. Pit stop order, tyre allocation
   priority, and team orders (if any were given) are shared-team factors
   that can still separate two drivers in identical cars, and are not
   directly visible in the results table used here.

7. Sprint race points are NOT included (only 'R' = main race sessions are
   queried), so season point totals will read slightly lower than the
   official standings on sprint weekends.

8. Qualifying head-to-head only counts sessions where BOTH drivers have a
   valid recorded qualifying position (e.g. excludes sessions where one
   driver was eliminated due to a red flag before setting a time, or DNS'd).
""")