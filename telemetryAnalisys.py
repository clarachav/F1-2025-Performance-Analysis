"""
F1 2025 Season Analysis: Lando Norris (NOR) vs Charles Leclerc (LEC)
======================================================================
This script:
1. Compares single-lap telemetry (speed, throttle, brake, G-forces) at Silverstone
2. Overlays REAL corner numbers from the official Silverstone track map onto every plot
3. Detects braking zones and labels each one with its official corner number
4. Pulls the FULL 2025 season results (every round, every winner) and compares
   NOR vs LEC points, finishing positions, positions gained/lost, and DNFs —
   to show CONCRETELY where the points gap actually came from
5. Prints a methodology / limitations section at the end (important for portfolio credibility)
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
driver2 = 'LEC'
YEAR = 2025

DRIVER_SURNAMES = {
    'NOR': 'Norris',
    'LEC': 'Leclerc',
}

# ============================================================
# PART 1 — SINGLE-LAP TELEMETRY COMPARISON (Silverstone)
# ============================================================
print("="*70)
print("PART 1 — SILVERSTONE TELEMETRY COMPARISON")
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

axes[0].plot(tel1['Distance'], tel1['Speed'], label=driver1, color='red')
axes[0].plot(tel2['Distance'], tel2['Speed'], label=driver2, color='blue')
axes[0].set_ylabel('Speed (km/h)')
axes[0].legend()
axes[0].set_title(f'{driver1} vs {driver2} — Telemetry Comparison (Silverstone {YEAR})')

axes[1].plot(tel1['Distance'], tel1['Throttle'], color='red')
axes[1].plot(tel2['Distance'], tel2['Throttle'], color='blue')
axes[1].set_ylabel('Throttle (%)')

axes[2].plot(tel1['Distance'], tel1['Brake'], color='red')
axes[2].plot(tel2['Distance'], tel2['Brake'], color='blue')
axes[2].set_ylabel('Braking')
axes[2].set_xlabel('Distance (m)')

# Draw a light vertical line + corner number label at each official corner
y_top = axes[0].get_ylim()[1]
for _, corner in corners.iterrows():
    for ax in axes:
        ax.axvline(corner['Distance'], color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
    axes[0].text(corner['Distance'], y_top * 0.97, f"T{int(corner['Number'])}",
                 rotation=90, fontsize=7, ha='center', va='top', color='dimgray')

plt.tight_layout()
plt.savefig('comparison_telemetry_with_corners.png', dpi=150)
plt.show()  # <-- close this window to continue the script

# ---- Plot 2: Speed delta ----
common_dist = np.linspace(0, min(tel1['Distance'].max(), tel2['Distance'].max()), 1000)
speed1_interp = np.interp(common_dist, tel1['Distance'], tel1['Speed'])
speed2_interp = np.interp(common_dist, tel2['Distance'], tel2['Speed'])

plt.figure(figsize=(12, 4))
plt.plot(common_dist, speed1_interp - speed2_interp)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Distance (m)')
plt.ylabel(f'Speed delta {driver1} - {driver2} (km/h)')
plt.title('Where each driver gains or loses speed')
plt.savefig('delta_speed.png', dpi=150)
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
ax[0].plot(tel1['Distance'], tel1['LongG'], color='red', label=driver1)
ax[0].plot(tel2['Distance'], tel2['LongG'], color='blue', label=driver2)
ax[0].set_ylabel('Longitudinal G')
ax[0].axhline(0, color='gray', linewidth=0.5)
ax[0].legend()

ax[1].plot(tel1['Distance'], tel1['LatG'], color='red')
ax[1].plot(tel2['Distance'], tel2['LatG'], color='blue')
ax[1].set_ylabel('Lateral G')
ax[1].set_xlabel('Distance (m)')

plt.tight_layout()
plt.savefig('gforces_comparison.png', dpi=150)
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

turns1.to_csv(f'turns_{driver1}_silverstone.csv', index=False)
turns2.to_csv(f'turns_{driver2}_silverstone.csv', index=False)

# ============================================================
# PART 2 — FULL 2025 SEASON: RESULTS, WINNERS, POINTS COMPARISON
# ============================================================
print("\n" + "="*70)
print(f"PART 2 — FULL {YEAR} SEASON: {driver1} vs {driver2}")
print("="*70)

schedule = fastf1.get_event_schedule(YEAR, include_testing=False)

season_data = []

for _, event in schedule.iterrows():
    gp_name = event['EventName']
    try:
        race = fastf1.get_session(YEAR, gp_name, 'R')
        race.load(telemetry=False, weather=False)  # faster: skip data we don't need here

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

        # Optional: rough incident/penalty count from race control messages (best-effort text match)
        def count_incidents(race_session, driver_code):
            msgs = race_session.race_control_messages
            if msgs is None or msgs.empty or 'Message' not in msgs.columns:
                return 0
            surname = DRIVER_SURNAMES.get(driver_code, driver_code)
            mask = msgs['Message'].str.contains(surname, case=False, na=False)
            return int(mask.sum())

        d1_incidents = count_incidents(race, driver1)
        d2_incidents = count_incidents(race, driver2)

        season_data.append({
            'GP': gp_name,
            'Winner': winner,
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

        print(f"{gp_name:30s} winner={winner:4s} | {driver1} P{d1_pos} ({d1_pts} pts) | {driver2} P{d2_pos} ({d2_pts} pts)")

    except Exception as e:
        print(f"Skipped {gp_name}: {e}")

df_season = pd.DataFrame(season_data)
df_season.to_csv(f'season_{YEAR}_{driver1}_vs_{driver2}.csv', index=False)

# ---- Championship totals & concrete comparison ----
total_d1 = df_season[f'{driver1}_points'].sum()
total_d2 = df_season[f'{driver2}_points'].sum()
dnf_d1 = (df_season[f'{driver1}_status'] != 'Finished').sum()
dnf_d2 = (df_season[f'{driver2}_status'] != 'Finished').sum()
avg_gain_d1 = df_season[f'{driver1}_positions_gained'].mean()
avg_gain_d2 = df_season[f'{driver2}_positions_gained'].mean()
incidents_d1 = df_season[f'{driver1}_incidents_mentioned'].sum()
incidents_d2 = df_season[f'{driver2}_incidents_mentioned'].sum()
wins_d1 = (df_season['Winner'] == driver1).sum()
wins_d2 = (df_season['Winner'] == driver2).sum()

print("\n" + "="*70)
print("SEASON TOTALS")
print("="*70)
print(f"{driver1}: {total_d1} pts | {wins_d1} wins | {dnf_d1} non-finishes | "
      f"avg positions gained/race: {avg_gain_d1:.2f} | race-control mentions: {incidents_d1}")
print(f"{driver2}: {total_d2} pts | {wins_d2} wins | {dnf_d2} non-finishes | "
      f"avg positions gained/race: {avg_gain_d2:.2f} | race-control mentions: {incidents_d2}")

leader = driver1 if total_d1 > total_d2 else driver2
gap = abs(total_d1 - total_d2)
print(f"\n{leader} leads the head-to-head by {gap:.0f} points this season.")
print("Use the table above (positions gained/lost, DNFs, race-control mentions) to identify")
print("WHERE the points gap came from: better qualifying, better race execution, fewer")
print("mistakes/incidents, or fewer mechanical retirements.")

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
   from FastF1's circuit info (session.get_circuit_info()). This is an
   approximation based on distance, not an exact corner-entry detection.

3. G-forces are DERIVED, not measured directly: longitudinal G comes from
   differentiating speed over time; lateral G is estimated from the curvature
   of the X/Y position trace. Both are sensitive to GPS/position data
   resolution and should be read as approximations.

4. "Positions gained" = Grid Position minus Finish Position. It does NOT
   account for grid penalties, safety car timing, or strategy-only pit stops,
   which can inflate or deflate this number independent of pure driving skill.

5. "Race-control mentions" is a best-effort text search for the driver's
   surname in official race control messages. It flags incidents, investigations
   or penalties WHERE THE DRIVER IS NAMED, but is not a verified penalty count
   and can miss messages that refer to a car number instead of a name.

6. This compares two different cars/teams (McLaren vs Ferrari in 2025), so
   the points and pace differences reflect a mix of DRIVER performance AND
   CAR performance — not driver skill in isolation. A fairer isolated
   comparison of pure driving would use teammates on identical machinery
   (e.g., NOR vs PIA at McLaren).

7. Sprint race points are NOT included in this script (only 'R' = main race
   sessions are queried). Any 2025 sprint weekends will show a slightly lower
   total points count than the official championship standings.
""")