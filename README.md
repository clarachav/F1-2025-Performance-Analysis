F1 2025 Performance Analysis

Overview

This project analyzes Formula 1 performance throughout the 2025 season using Python and FastF1.

The objective is to explore race and driver performance through telemetry, tyre degradation, race strategy and championship data, with a focus on comparing:

Lando Norris vs Charles Leclerc — comparison between two different cars

Lando Norris vs Oscar Piastri — comparison between teammates in the same McLaren

The project combines data analysis, visualization and statistical methods to investigate how driving behaviour, car performance, tyre degradation and race strategy can affect on-track results.

Objectives

Analyze Formula 1 telemetry data

Compare speed, throttle and braking behaviour

Derive and analyze longitudinal and lateral G-forces

Study tyre degradation across races

Compare race strategies and pit-stop decisions

Analyze championship and race-performance data

Use statistical methods, including bootstrap analysis, to investigate performance differences

Technologies

Python 3.11

FastF1 — Formula 1 timing, telemetry and race data

Pandas — data manipulation and analysis

NumPy — numerical computation

Matplotlib — data visualization

SciPy — statistical analysis

Main Analyses

Telemetry analysis

Telemetry data is used to compare drivers through variables such as:

Speed

Throttle application

Braking

Longitudinal G-force

Lateral G-force

Corner-by-corner behaviour

Tyre degradation

The project analyzes tyre performance and degradation over race stints and compares degradation patterns between drivers.

Race strategy

Pit-stop and race-strategy data are analyzed to investigate differences in tyre choices, pit-stop timing and strategic decisions.

Championship and season analysis

The project also uses season-level data to compare driver performance and investigate the evolution of the 2025 championship.

Statistical analysis

Bootstrap methods are used to assess the robustness of observed performance differences across race data.

Project Structure

F1-2025-Performance-Analysis/
│
├── outputs/                         # Generated analysis results and plots
│
├── telemetryAnalisys.py             # Telemetry analysis
├── telemetryAnalysis_NOR_vs_PIA_teammates.py
├── tire_degradation.py              # Tyre degradation analysis
│
├── *.csv                            # Processed analysis data
├── *.png                            # Main visualizations
│
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Files excluded from version control
└── README.md

Installation

Clone the repository:

git clone https://github.com/clarachav/F1-2025-Performance-Analysis.git
cd F1-2025-Performance-Analysis

Install the required Python packages:

pip install -r requirements.txt

The project was developed with Python 3.11.

FastF1 Cache

FastF1 uses a local cache to store downloaded Formula 1 data and improve subsequent analyses.

The local f1_cache/ directory is intentionally not included in this repository because it contains large cached data files. The cache is generated locally when running the analyses.

Outputs

The repository contains the main generated results, including:

Telemetry comparison plots

Speed delta analysis

G-force comparisons

Tyre degradation curves

Pit-stop strategy analysis

Season and championship data

CSV files containing processed results

Author

Clara Chavanon

EPF Engineering School — Aerospace Engineering
