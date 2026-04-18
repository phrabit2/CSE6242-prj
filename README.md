# Performance Inflection Dashboard

> **CSE 6242 — Data and Visual Analytics | Spring 2026 | Team 026**

**Team Members:** Hsiang Wen Hsiao (Eric), I Lin Tsai (Irene), Qixiang Goh (Ethan), Xueying Jin (Clara), Suho Lee (Suho)

---

## Live Demo

**Access the deployed dashboard here:**

> **http://15.165.52.135:8501**

---

## Overview

Traditional baseball metrics rely on season-long averages that act as **lagging indicators**, often concealing meaningful performance shifts until a player has already experienced extended decline. This project develops an **automated diagnostic dashboard** powered by **Change-Point Detection (CPD)** algorithms to pinpoint the exact onset of player slumps or breakouts using high-frequency MLB Statcast data.

Our system analyzes **PA-level engineered features** across 420 qualified hitters from 2021–2025:

- **Detects** structural performance shifts using PELT (univariate) and ChangeForest (multivariate) algorithms
- **Diagnoses** root causes by distinguishing mechanical changes from psychological "clutch" performance dips
- **Visualizes** results through an interactive five-view dashboard with before/after snapshot comparison and a Smart Analyzer narrative engine

---

## The Four Performance Pillars

| Indicator | What It Measures |
|-----------|-----------------|
| **Hitting Decisions Score** | Plate discipline — quality of swing vs. take decisions |
| **Power Efficiency** | Raw power — converting swing effort to exit velocity |
| **wOBA Residual** | Luck vs. Skill — actual results vs. physics-based expectation |
| **Launch Angle Stability** | Swing consistency — repeatability of ball flight path |

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Welcome** | League-wide benchmark distributions & four-pillar overview |
| **Player Snapshot** | Season-level profile, percentile rankings, radar chart |
| **Peer Comparison** | Side-by-side radar, leaderboard, and density plots for up to 3 players |
| **Univariate Change Analyzer** | PELT-based single-metric CPD with interactive shift deep-dive |
| **Multivariate Change Analyzer** | ChangeForest RF-based multivariate CPD with feature importance |

---

## Project Structure

```
CSE6242_prj/
├── README.md
├── .gitignore
├── requirements.txt          # Minimal dependencies for final_dashboard.py
├── deploy.yml                # GitHub Actions CI/CD workflow (reference)
│
├── dashboard/                # Streamlit application (entry point + modules)
│   ├── final_dashboard.py    #   Entry point — run this with streamlit
│   ├── config.py             #   Constants, colour palette, indicator mappings
│   ├── styles.py             #   Global CSS injection
│   ├── data_loader.py        #   Data fetching & caching (Google Drive)
│   ├── cpd_engine.py         #   PELT & ChangeForest CPD algorithms
│   ├── ui_components.py      #   Shared UI helpers & deep-dive renderer
│   └── pages/
│       ├── welcome.py        #   Page 1: League overview
│       ├── snapshot.py       #   Page 2: Player snapshot
│       ├── peer_comparison.py#   Page 3: Peer comparison
│       ├── univariate.py     #   Page 4: Univariate change analyzer
│       └── multivariate.py   #   Page 5: Multivariate change analyzer
│
└── infra/                    # AWS infrastructure (Terraform)
    ├── main.tf
    ├── variables.tf
    ├── outputs.tf
    ├── userdata.sh
    ├── architecture_diagram.py
    └── cse6242_team26_architecture.png
```

---

## Installation & Local Run

### 1. Clone the repository

```bash
git clone https://github.com/phrabit2/CSE6242-prj.git
cd CSE6242-prj
git checkout final
```

### 2. Set up the environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
streamlit run dashboard/final_dashboard.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

> **Note:** On first run the app downloads `pa_master.csv` (~300 MB) from Google Drive to `/tmp/`. This takes 1–2 minutes. Subsequent runs use the cached file.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Data Source** | MLB Statcast via Google Drive (pre-processed PA-level CSV) |
| **CPD — Univariate** | `ruptures` (PELT, RBF kernel) |
| **CPD — Multivariate** | `changeforest` (Random Forest) |
| **Feature Importance** | `scikit-learn` RandomForestClassifier |
| **Dashboard** | `streamlit` |
| **Visualization** | `plotly`, `matplotlib`, `seaborn` |
| **Deployment** | AWS EC2 (ap-northeast-2) + GitHub Actions CI/CD |

---

## Deployment (CI/CD)

The `deploy.yml` file documents the GitHub Actions workflow that auto-deploys to EC2 on every push to `main`:

1. SSH into EC2 → `git pull` → `pip install` → restart `systemd` service
2. Sync data assets to S3 (`s3://team26-cpd-data-294342039761`)

The infra was provisioned with Terraform (see `infra/`).

---

## References

- Truong, Oudre, & Vayatis (2020) — CPD computational framework
- Killick, Fearnhead, & Eckley (2012) — PELT algorithm
- Londschien, Kovács, & Bühlmann (2023) — ChangeForest algorithm
- Adams & MacKay (2007) — Bayesian Online CPD
- Taylor (2017) — Exit Velocity & Launch Angle predictive power
- Lage et al. (2016) — StatCast Dashboard

---

## License

This project is for academic purposes (Georgia Tech CSE 6242, Spring 2026).
