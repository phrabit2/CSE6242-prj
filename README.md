# ⚾ Change-Point-Based Detection and Visual Analysis of Baseball Performance Shifts

> **CSE 6242 — Data and Visual Analytics | Spring 2026 | Team 026**
>
> CI/CD: Auto-deploy to EC2 & S3 via GitHub Actions

**Team Members:** Hsiang Wen Hsiao(Ethan), I Lin Tsai(Irene), Qixiang Goh(Eric), Xueying Jin(Clara), Suho Lee(Suho)

---

## Overview

Traditional baseball metrics rely on season-long averages that act as **lagging indicators**, often concealing meaningful performance shifts until a player has already experienced extended decline. This project develops an **automated diagnostic dashboard** powered by **Change-Point Detection (CPD)** algorithms to pinpoint the exact onset of player slumps or breakouts using high-frequency MLB Statcast data.

Our system:
- **Detects** structural performance shifts across multivariate Statcast metrics (Exit Velocity, Launch Angle, Barrel Rate, etc.)
- **Diagnoses** root causes by distinguishing mechanical changes from psychological "clutch" performance dips
- **Visualizes** results through an interactive four-view dashboard with before/after snapshot comparison

## Key Features

| Feature | Description |
|---------|-------------|
| **Change-Point Timeline** | Interactive time series with detected breakpoints overlaid |
| **Before / After Snapshot** | Dual-state comparison of mechanical metrics pre/post shift |
| **Clutch vs. Core Breakdown** | Situational performance diagnostics |
| **Multi-Algorithm Support** | PELT, Binary Segmentation (CUSUM), Bayesian Online CPD |

## Tech Stack

- **Data Source:** [Baseball Savant (Statcast)](https://baseballsavant.mlb.com/) via `pybaseball`
- **CPD Engine:** `ruptures` (PELT, CUSUM, Bayesian)
- **Dashboard:** Plotly Dash + D3.js
- **Language:** Python 3.10+

## Project Structure

```
CSE6242_prj/
├── README.md
├── .gitignore
├── requirements.txt
│
├── docs/                          # Documentation & proposal
│   ├── team026proposal.pdf
│   └── references/
│
├── data/                          # Data directory (large files excluded from Git)
│   ├── raw/                       #   Raw Statcast data
│   ├── processed/                 #   Cleaned & aggregated data
│   └── README.md                  #   Data dictionary & fetch instructions
│
├── notebooks/                     # Jupyter notebooks for exploration & analysis
│
├── src/                           # Source code (Python package)
│   ├── data/
│   │   ├── fetch_statcast.py      #   Statcast data collection
│   │   └── preprocess.py          #   Data cleaning & feature engineering
│   ├── models/
│   │   ├── cpd.py                 #   CPD algorithms (PELT, BinSeg, Bayesian)
│   │   └── evaluate.py            #   Precision, Recall, F1 evaluation
│   └── visualization/
│       └── plots.py               #   Plotly-based visualization utilities
│
├── dashboard/                     # Plotly Dash web application
│   ├── app.py                     #   Main Dash app entry point
│   ├── layouts/                   #   Page layout definitions
│   ├── callbacks/                 #   Dash callback functions
│   ├── components/                #   Reusable UI components
│   └── assets/                    #   Static assets (CSS, JS/D3)
│
├── tests/                         # Unit tests
│   └── test_cpd.py                #   CPD algorithm tests
│
└── reports/                       # Generated figures & analysis outputs
    └── figures/
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/CSE6242_prj.git
cd CSE6242_prj
```

### 2. Set up the environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Fetch Statcast data

```bash
python src.data.fetch_statcast
```

### 4. Run the dashboard

```bash
cd dashboard
python app.py
```

Then open [http://localhost:8050](http://localhost:8050) in your browser.

### 5. Run tests

```bash
pytest tests/
```

## Project Timeline

| Phase | Period | Deliverables |
|-------|--------|-------------|
| **I — Literature Review** | 2/16 – 3/5 | Topic selection, literature survey |
| **II — Development** | 3/3 – 3/30 | Cleaned dataset, working CPD prototype, prototype dashboard |
| **III — Evaluation & Final** | 3/31 – 4/25 | Algorithm comparison (PELT vs CUSUM vs Bayesian), final four-view dashboard |

**Midterm Milestone (3/30):** Cleaned Statcast dataset + working CPD prototype + prototype dashboard

**Final Milestone (4/22):** Detection precision/recall evaluation + algorithm comparison + final dashboard

## References

- Truong, Oudre, & Vayatis (2020) — CPD computational framework
- Killick, Fearnhead, & Eckley (2012) — PELT algorithm
- Adams & MacKay (2007) — Bayesian Online CPD
- Taylor (2017) — Exit Velocity & Launch Angle predictive power
- Albert (2022) — Evolution of baseball metrics
- Lage et al. (2016) — StatCast Dashboard

See [docs/team026proposal.pdf](docs/team026proposal.pdf) for the full proposal and bibliography.

## License

This project is for academic purposes (Georgia Tech CSE 6242, Spring 2026).
