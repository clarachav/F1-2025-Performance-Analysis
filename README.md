# F1 2025 Performance Analysis

## Overview
This project analyzes Formula 1 performance throughout the 2025 season using Python and FastF1 data. The primary objective is to explore race and driver performance through high-frequency telemetry, tyre degradation curves, pit-stop strategy simulations, and macroscopic championship data. 

The analysis focuses on two distinct comparison axes:
* **Lando Norris vs Charles Leclerc:** An inter-team comparison evaluating cross-car performance dynamics (McLaren vs Ferrari).
* **Lando Norris vs Oscar Piastri:** An intra-team teammate comparison sharing identical car packages and aerodynamic upgrades to isolate driver performance.

The project combines telemetry dynamic derivations, mathematical modeling, and statistical validation methods (such as Monte Carlo Bootstrap resampling) to investigate how driving inputs, tyre management, and strategic choices affect on-track results.

---

## Objectives
* **Analyze Telemetry Data:** Evaluate velocity profiles, throttle application, and braking inputs.
* **Kinematic Force Derivation:** Compute longitudinal and lateral G-forces from raw position and timing telemetry coordinates.
* **Model Tyre Degradation:** Calculate stint wear rates (in ms/lap) across dry race events using linear regression.
* **Simulate Race Strategies:** Reconstruct and optimize total race times for 1-stop and 2-stop scenarios.
* **Statistical Validation:** Apply Bootstrap resampling methods to verify if observed driver performance gaps are statistically significant trends or merely random race noise.

---

## Technologies
* **Python 3.11**
* **FastF1** — Formula 1 timing, telemetry, car data, and circuit map information.
* **Pandas** — Data manipulation, filtering, and structural aggregation.
* **NumPy** — Vectorized numerical computations and array interpolation.
* **SciPy** — Statistical analysis, linear regressions, and confidence intervals.
* **Matplotlib** — Data visualizations, subplots, and heatmaps.

---

## Main Analyses & Scripts Description

### 1. Telemetry & Kinematic Analysis
* `telemetryAnalisys.py`: Loads single-lap telemetry for cross-team comparisons (NOR vs LEC). Syncs speed, throttle, and braking data against official track coordinates.
* `telemetryAnalysis_NOR_vs_PIA_teammates.py`: Isolates intra-team behaviors using identical machinery. Derives a spatial speed delta profile to pinpoint exactly where on-track lap time is gained or lost.
* **G-Force Derivation:** Telemetry coordinates \((X, Y)\) and time vectors are processed to compute:
  * **Longitudinal G (\(G_{\text{Long}}\)):** First derivative of speed over time (\(\Delta v / \Delta t\)), highlighting deceleration thresholds (up to \(-5G\)).
  * **Lateral G (\(G_{\text{Lat}}\)):** Trigonometeric heading calculation (`arctan2`) used to compute local curvature (\(R\)) and centripetal acceleration (\(v^2/R\)).
* **Braking Zones Detection (`analyse_turns`):** Automatically maps braking impulses, computes braking distance, and associates the minimum speed apex with its official track corner number (e.g., T3, T6).

### 2. Tyre Degradation & Strategy Simulation
* `tire_degradation.py`: Automatically filters out wet/intermediate sessions. Filters clean laps using a robust \(2\sigma\) median threshold to eliminate traffic anomalies.
* **Degradation Modeling:** Computes the slope coefficient (\(\text{ms/lap}\)) of performance decay over stints. Draws a 95% Student-t confidence interval band around the fit line.
* **Strategy Simulation:** Simulates full-race timelines across 5 strategy variants (1-stop early/mid/late, 2-stop equal/early), factoring in a pit lane time loss penalty of 22 seconds (`PIT_LOSS_S = 22.0`). Generates global seasonal heatmaps of tyre wear.

### 3. Statistical Bootstrap Validation (Monte Carlo Resampling)
Because a racing season provides a small sample size (~24 events), classic means are highly sensitive to unprovoked DNFs or mechanical luck. This project implements non-parametric Bootstrap algorithms (10,000 simulations with replacement) to generate virtual populations:
* `bootstrap_analysis.py`: Analyzes the stability of lap-by-lap pace deltas and quantifies the mathematical uncertainty of individual stint degradation slopes.
* `bootstrap_degradation_csv.py`: Evaluates matched-stint degradation rate metrics over the entire championship. Evaluates whether one driver systematically degrades tyre compounds faster than their teammate in identical conditions.
* `bootstrap_from_csv.py`: Resamples seasonal outcomes (points gaps, finishing places, qualifying grid ratios). If the 95% Confidence Interval (CI) boundaries exclude the 0-difference baseline, the driver performance delta is flagged as **statistically systematic**.

---

## Key Takeaways & Conclusions
* **Telemetry & Dynamics:** Teammate comparisons isolated driver inputs, showing that braking consistency and throttle re-application profiles account for the majority of the lap time delta.
* **Tyre Degradation:** Modeled stint performance decay identified distinct degradation curves per compound, mapping out clear circuit-specific thermal management patterns (e.g., pointing out rear-tire overheat anomalies).
* **Statistical Validation:** Applied Bootstrap methods validated that the race-pace advantage observed for Lando Norris over a 24-race span is statistically significant and completely isolated from random variance.

---

## Project Structure
```text
F1-2025-Performance-Analysis/
│
├── f1_cache/                        # Local data cache directory (git-ignored)
├── outputs/                         # Generated CSV results, heatmaps, and plots
│
├── telemetryAnalisys.py             # Single-lap cross-team telemetry analysis
├── telemetryAnalysis_NOR_vs_PIA_teammates.py # Single-lap teammate telemetry & G-forces
├── tire_degradation.py              # Stint wear rates modeling and strategy simulation
│
├── bootstrap_analysis.py            # Telemetry gap and slope stability bootstrap
├── bootstrap_degradation_csv.py      # Whole-season paired stint degradation bootstrap
├── bootstrap_from_csv.py             # Macroscopic seasonal outcome bootstrap
│
├── *.csv                            # Aggregated analysis datasets
├── *.png                            # Main visualization exports
│
├── requirements.txt                 # Python project dependencies
├── .gitignore                       # System files and cache exclusions
└── README.md                        # Documentation
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/clarachav/F1-2025-Performance-Analysis.git
   cd F1-2025-Performance-Analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Python 3.11 is recommended for full library compatibility.*

---

## FastF1 Cache
FastF1 utilizes a local folder to store downloaded session datasets and telemetry streams. The local `f1_cache/` directory is intentionally excluded from version control via `.gitignore` to prevent committing heavy cache files. The folder will automatically initialize and populate locally upon script execution.

---

## Outputs
The repository outputs results directly into the workspace and the `outputs/` folder, including:
* Telemetry superposition comparison profiles.
* Spatial velocity delta indicators.
* Longitudinal and lateral derived G-force charts.
* Linear tyre degradation curves with Student-t confidence intervals.
* Macroscopic seasonal degradation rate heatmaps.
* Bootstrap Monte Carlo distribution histograms (95% CI boundaries).

---

## Author
**Clara Chavanon**  
EPF Engineering School — Aerospace Engineering Student  
Focus: Aerospace Dynamics, Metrology, Data Analysis & Motorsport Performance Engineering
