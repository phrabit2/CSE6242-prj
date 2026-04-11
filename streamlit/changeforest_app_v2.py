#!/usr/bin/env python
# coding: utf-8
"""
ChangeForest Change-Point Detection — Streamlit App v2

Sidebar controls : player name | rolling window | CDP sensitivity
Tabs             :
  1. ChangeForest Result + Input Signal
  2. Before / After Eval
  3. Parameter Stability
"""

import os
from pathlib import Path

import gdown
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

try:
    from changeforest import Control, changeforest
except Exception:  # pragma: no cover — runtime dependency check
    Control = None
    changeforest = None


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SENSITIVITY_TO_MIN_SEG = {
    "Low":    0.10,
    "Medium": 0.05,
    "High":   0.02,
}

CPD_INDICATORS = [
    "hitting_decisions_score",
    "power_efficiency",
    "woba_residual",
    "launch_angle_stability_50pa",
]

INDICATOR_LABELS = {
    "hitting_decisions_score":       "Hitting Decisions Score",
    "power_efficiency":              "Power Efficiency",
    "woba_residual":                 "wOBA Residual",
    "launch_angle_stability_50pa":   "Launch Angle Stability",
}

INDICATOR_TOOLTIPS = {
    "hitting_decisions_score":     "Plate discipline. Measures swing vs. take quality. Higher is better (Elite: >3.0, League Avg: ~0.3).",
    "power_efficiency":            "Raw power. Effectiveness of converting swing effort to exit velocity. Higher is better (Elite: >0.0100, League Avg: ~0.0040).",
    "woba_residual":               "Luck vs Skill. Difference between actual results and physics-based expectation. Positive (>0.15) means outperforming physics (luck/skill); Negative (<-0.15) means 'unlucky'.",
    "launch_angle_stability_50pa": "Swing consistency. Stability of ball flight path over recent 50 PAs. Higher values indicate a more repeatable, optimized swing path.",
}

INDICATOR_COLORS = {
    "hitting_decisions_score":       "#2ca02c",
    "power_efficiency":              "#1f77b4",
    "woba_residual":                 "#ff7f0e",
    "launch_angle_stability_50pa":   "#d62728",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Palette ────────────────────────────────────────────────────────────────────
DARK       = "#FFFFFF"
PANEL      = "#F0F2F6"
BORDER     = "#D0D7DE"
GOLD       = "#855D00"
GOLD_LT    = "#B08800"
TEAL       = "#0068C9"
TEAL_LT    = "#004B91"
RED        = "#D32F2F"
RED_LT     = "#B71C1C"
GREY       = "#586069"
TEXT       = "#111418"
TEXT_MUTED = "#586069"


# ══════════════════════════════════════════════════════════════════════════════
# BUG FIX 1 — st.set_page_config() MUST be the very first Streamlit call.
#             It was buried inside main() while st.markdown(CSS) ran first.
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="MLB Batting Pulse", page_icon="⚾", layout="wide")

# ══════════════════════════════════════════════════════════════════════════════
# STYLES  (CSS injection — safe here, after set_page_config)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: {DARK}; color: {TEXT};
}}
h1,h2,h3 {{ font-family:'Bebas Neue',sans-serif; letter-spacing:0.06em; color:{TEXT}; }}
.block-container {{ padding-top:1.2rem; background-color:{DARK}; max-width:1400px; }}
[data-testid="stSidebar"] {{ background-color:{PANEL}; border-right:1px solid {BORDER}; }}
[data-testid="stSidebar"] * {{ color:{TEXT} !important; }}
.stRadio label,.stSelectbox label,.stMultiSelect label,.stSlider label {{
    color:{TEXT_MUTED} !important; font-size:0.72rem;
    text-transform:uppercase; letter-spacing:0.1em;
    font-family:'IBM Plex Mono',monospace;
}}
.card-row {{ display:flex; gap:10px; margin-bottom:1rem; flex-wrap:wrap; }}
.card {{
    background:{DARK}; border:1px solid {BORDER};
    border-top:3px solid {GOLD}; border-radius:4px;
    padding:0.9rem 1.1rem; flex:1; min-width:130px;
}}
.card.teal {{ border-top-color:{TEAL_LT}; }}
.card.red  {{ border-top-color:{RED_LT}; }}
.card.grey {{ border-top-color:{GREY}; }}
.card-label {{
    font-size:0.62rem; text-transform:uppercase; letter-spacing:0.12em;
    color:{TEXT_MUTED}; font-family:'IBM Plex Mono',monospace; margin-bottom:4px;
}}
.card-val {{
    font-size:1.6rem; font-weight:600; color:{TEXT};
    font-family:'Bebas Neue',sans-serif; letter-spacing:0.04em; line-height:1.1;
}}
.card-sub {{ font-size:0.72rem; color:{TEXT_MUTED}; font-family:'IBM Plex Mono',monospace; margin-top:2px; }}
.sec {{
    font-size:0.65rem; text-transform:uppercase; letter-spacing:0.15em;
    color:{TEAL_LT}; font-family:'IBM Plex Mono',monospace;
    border-bottom:1px solid {BORDER}; padding-bottom:4px; margin:1.1rem 0 0.7rem 0;
}}
.cpd-minor {{ display:inline-block; background:#f0f2f6; color:{TEXT_MUTED}; border-radius:3px; padding:2px 8px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; margin:2px; }}
.cpd-mod   {{ display:inline-block; background:#fff5b1; color:{GOLD_LT}; border:1px solid {GOLD}; border-radius:3px; padding:2px 8px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; margin:2px; }}
.cpd-sig   {{ display:inline-block; background:#ffeef0; color:{RED_LT}; border:1px solid {RED}; border-radius:3px; padding:2px 8px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; margin:2px; }}
.preamble {{
    background:{PANEL}; border:1px solid {BORDER}; border-left:4px solid {GOLD};
    border-radius:4px; padding:1.2rem 1.4rem; margin-bottom:1.4rem;
    font-size:0.92rem; line-height:1.7; color:{TEXT_MUTED};
}}
.preamble b {{ color:{TEXT}; }}
.narrative {{
    background:{PANEL}; border:1px solid {BORDER}; border-radius:4px;
    padding:1rem 1.2rem; font-family:'IBM Plex Mono',monospace;
    font-size:0.82rem; line-height:1.6; color:{TEXT}; margin-bottom:1rem;
}}
.rel-high {{ color:{TEAL_LT}; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }}
.rel-med  {{ color:{GOLD_LT}; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }}
.rel-low  {{ color:{RED_LT};  font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }}
</style>
""", unsafe_allow_html=True)


# ── UI helper functions ────────────────────────────────────────────────────────
def card(label, val, sub="", color="gold"):
    cls = {"gold": "card", "teal": "card teal", "red": "card red", "grey": "card grey"}[color]
    return (f'<div class="{cls}"><div class="card-label">{label}</div>'
            f'<div class="card-val">{val}</div><div class="card-sub">{sub}</div></div>')


def sec(title):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)


def mpl_fig(w=11, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)
    ax.title.set_color(TEXT)
    ax.grid(color=BORDER, linewidth=0.4, linestyle="--", alpha=0.5)
    return fig, ax


def effect_size_label(d):
    ad = abs(d)
    if ad < 0.2:
        return "minor", "cpd-minor"
    if ad < 0.5:
        return "moderate", "cpd-mod"
    return "significant", "cpd-sig"


def reliability_badge(n):
    if n >= 200:
        return f'<span class="rel-high">● High reliability</span>'
    if n >= 100:
        return f'<span class="rel-med">● Medium reliability</span>'
    return f'<span class="rel-low">● Low reliability (few pitches this season)</span>'


def games_label(n_pitches):
    return f"≈{max(1, round(n_pitches / 2.6))} games"


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading plate appearance data…")
def load_data() -> pd.DataFrame:
    DATA_FILE_ID = "1G8eA6gX8hdCwWp1YWddAMmwt6R62tcmA"
    tmp_path = "/tmp/Qualified_hitters_statcast_2021_2025_pa_master.csv"
    local_path = (
        Path(__file__).resolve().parent.parent
        / "data" / "processed"
        / "Qualified_hitters_statcast_2021_2025_pa_master.csv"
    )

    cols = [
        "batter", "player_name", "pa_uid", "game_date", "game_pk",
        "at_bat_number", "cf_seq_id",
        "hitting_decisions_score", "power_efficiency",
        "woba_residual", "launch_angle_stability_50pa",
    ]

    if local_path.exists():
        data_path = str(local_path)
    elif os.path.exists(tmp_path):
        data_path = tmp_path
    else:
        url = f"https://drive.google.com/uc?id={DATA_FILE_ID}"
        gdown.download(url, tmp_path, quiet=False, fuzzy=True)
        data_path = tmp_path

    df = pd.read_csv(data_path, usecols=cols)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# CPD pipeline
# ──────────────────────────────────────────────────────────────────────────────
def changeforest_subdataset_generator(
    df: pd.DataFrame,
    selected_player_id: int,
    window: int = 50,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Build aligned multivariate CPD subdataset for one player."""
    if min_periods is None:
        min_periods = window

    feature_cols = ["cf_seq_id"] + CPD_INDICATORS
    base_cols = ["batter", "player_name", "pa_uid", "game_date", "game_pk", "at_bat_number"]

    subdf = (
        df.loc[df["batter"] == selected_player_id, base_cols + feature_cols]
        .dropna(subset=feature_cols)
        .sort_values("cf_seq_id")
        .reset_index(drop=True)
    )

    for col in CPD_INDICATORS:
        subdf[f"{col}_rollmean_{window}"] = (
            subdf[col].rolling(window=window, min_periods=min_periods).mean()
        )

    rollmean_cols = [f"{col}_rollmean_{window}" for col in CPD_INDICATORS]
    subdf = subdf.dropna(subset=rollmean_cols).reset_index(drop=True)
    return subdf


def run_changeforest(
    subdf: pd.DataFrame,
    window: int = 50,
    use_rollmean: bool = True,
    standardize: bool = True,
    minimal_relative_segment_length: float = 0.05,
):
    """Run ChangeForest on all 4 CPD indicators."""
    if subdf is None or subdf.empty:
        return None, [], None, None

    feature_names = (
        [f"{col}_rollmean_{window}" for col in CPD_INDICATORS]
        if use_rollmean
        else list(CPD_INDICATORS)
    )

    missing = [c for c in feature_names if c not in subdf.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    X = subdf[feature_names].dropna().values
    if len(X) == 0 or np.isnan(X).any():
        raise ValueError(
            "Feature matrix is empty or contains NaN — "
            "try a smaller rolling window or a player with more data."
        )

    if standardize:
        X = StandardScaler().fit_transform(X)

    control = Control(
        model_selection_alpha=0.02,
        minimal_relative_segment_length=minimal_relative_segment_length,
    )
    result = changeforest(X, method="random_forest", control=control)
    cps = [int(cp) for cp in result.split_points() if 0 <= int(cp) < len(subdf)]
    return result, cps, X, feature_names


# ── Plot A: Input Signal 2×2 panel ────────────────────────────────────────────
def plot_input_signals(subdf: pd.DataFrame, window: int, player_name: str = "") -> plt.Figure:
    """2×2 panel: raw scatter + rolling mean for all 4 indicators."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=True)
    axes = axes.flatten()
    fig.patch.set_facecolor(DARK)

    for ax, col in zip(axes, CPD_INDICATORS):
        color = INDICATOR_COLORS[col]
        label = INDICATOR_LABELS[col]
        smooth_col = f"{col}_rollmean_{window}"
        ax.set_facecolor(PANEL)

        if subdf is None or subdf.empty:
            ax.set_title(f"{label} (no data)", fontsize=11, fontweight="bold")
            continue

        ax.scatter(
            subdf["cf_seq_id"], subdf[col],
            s=8, alpha=0.22, color=color, edgecolor="none", label="Raw",
        )
        if smooth_col in subdf.columns:
            ax.plot(
                subdf["cf_seq_id"], subdf[smooth_col],
                linewidth=2.4, color=color, alpha=0.95,
                label=f"Rolling Mean (w={window})",
            )

        ax.set_title(f"{label}  (n={len(subdf):,})", fontsize=11, fontweight="bold")
        ax.set_xlabel("CF Sequence ID")
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Input Signals — {player_name}" if player_name else "Input Signals",
        fontsize=13, fontweight="bold",
    )
    return fig


# ── Plot B: ChangeForest result ────────────────────────────────────────────────
def plot_changeforest_result(
    subdf: pd.DataFrame,
    cps: list,
    window: int,
    player_name: str,
    min_rel_seg_len: float,
) -> plt.Figure:
    """Single time-series chart with all 4 rolling-mean signals + CP dashed lines."""
    dates = pd.to_datetime(subdf["game_date"])
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(PANEL)

    label_x_offsets = [8, -68, 8, -68]

    for col, x_offset in zip(CPD_INDICATORS, label_x_offsets):
        color = INDICATOR_COLORS[col]
        label = INDICATOR_LABELS[col]
        y_col = f"{col}_rollmean_{window}"

        ax.plot(dates, subdf[y_col], linewidth=2.2, color=color, label=label, alpha=0.9)

        valid = subdf[[y_col, "game_date"]].dropna()
        if not valid.empty:
            last_idx = valid.index[-1]
            last_date = pd.to_datetime(subdf.loc[last_idx, "game_date"])
            last_val = subdf.loc[last_idx, y_col]
            ax.scatter(last_date, last_val, color=color, s=40, zorder=5)
            ax.annotate(
                label,
                xy=(last_date, last_val),
                xytext=(x_offset, 0),
                textcoords="offset points",
                color=color, fontsize=9, va="center", fontweight="bold",
            )

    for cp in cps:
        if 0 <= cp < len(subdf):
            ax.axvline(
                x=dates.iloc[cp],
                color="#c43b3b", linestyle=(0, (4, 4)), alpha=0.75, linewidth=1.8,
            )

    ax.set_xlabel("Game Date")
    ax.set_ylabel("Metric Value (standardized rolling mean)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)
    ax.set_title(
        f"ChangeForest Result  |  {player_name}"
        f"  |  window={window}  |  min_rel_seg_len={min_rel_seg_len}  |  CPs={len(cps)}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper left", frameon=False)
    return fig


# ── Before / After Eval ────────────────────────────────────────────────────────
def build_cp_eval_dfs(
    subdf: pd.DataFrame,
    cps: list,
    feature_names: list,
    compare_window: int = 50,
) -> dict:
    n = len(subdf)
    if n <= 2 * compare_window:
        return {}

    cps_set = set(cps)
    eval_dfs = {}

    for col in feature_names:
        rows = []
        for i in range(compare_window, n - compare_window):
            before = subdf[col].iloc[i - compare_window: i]
            after  = subdf[col].iloc[i: i + compare_window]
            rows.append({
                "feature":       col,
                "cp":            "Y" if i in cps_set else "N",
                "mean_before":   before.mean(),
                "mean_after":    after.mean(),
                "mean_diff":     after.mean() - before.mean(),
                "std_before":    before.std(),
                "std_after":     after.std(),
                "std_diff":      after.std() - before.std(),
                "abs_mean_diff": abs(after.mean() - before.mean()),
                "abs_std_diff":  abs(after.std() - before.std()),
            })
        eval_dfs[col] = pd.DataFrame(rows)

    return eval_dfs


def plot_cp_eval_comparison(eval_dfs: dict) -> plt.Figure | None:
    n = len(eval_dfs)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), constrained_layout=True)
    fig.patch.set_facecolor(DARK)
    if n == 1:
        axes = [axes]

    for ax, (feature, df_eval) in zip(axes, eval_dfs.items()):
        ax.set_facecolor(PANEL)
        if df_eval.empty or "cp" not in df_eval.columns:
            ax.set_title(f"{feature}\n(no data)")
            continue

        summary = df_eval.groupby("cp")[["abs_mean_diff", "abs_std_diff"]].mean()
        summary.plot(kind="bar", ax=ax, rot=0, color=["#1f77b4", "#ff7f0e"])

        base_col = feature.split("_rollmean_")[0]
        ax.set_title(INDICATOR_LABELS.get(base_col, feature), fontsize=11, fontweight="bold")
        ax.set_xlabel("Is Change Point?  (Y = detected CP, N = non-CP)")
        ax.set_ylabel("Avg Absolute Difference")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(["Abs Mean Diff", "Abs Std Diff"], frameon=False, fontsize=9)

    fig.suptitle(
        "Before / After Comparison at Detected Change Points",
        fontsize=13, fontweight="bold",
    )
    return fig


# ── Parameter Stability ────────────────────────────────────────────────────────
def run_parameter_stability(
    subdf: pd.DataFrame,
    window: int,
    min_rel_seg_len_list: list | tuple = (0.02, 0.05, 0.10),
    use_rollmean: bool = True,
    standardize: bool = True,
) -> pd.DataFrame:
    inv_map = {v: k for k, v in SENSITIVITY_TO_MIN_SEG.items()}
    rows = []

    for val in sorted(min_rel_seg_len_list):
        try:
            _, cps, _, _ = run_changeforest(
                subdf, window=window, use_rollmean=use_rollmean,
                standardize=standardize, minimal_relative_segment_length=val,
            )
            n_cp, cp_list = len(cps), cps
        except Exception as exc:
            n_cp, cp_list = None, []
            st.warning(f"Stability run failed for min_rel_seg_len={val}: {exc}")

        rows.append({
            "Sensitivity":     inv_map.get(val, str(val)),
            "min_rel_seg_len": val,
            "# Change Points": n_cp,
            "CP Indices":      str(cp_list),
        })

    return pd.DataFrame(rows)


def plot_parameter_stability(stability_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(PANEL)

    valid = stability_df.dropna(subset=["# Change Points"])
    x_labels = valid["Sensitivity"] + "\n(min=" + valid["min_rel_seg_len"].astype(str) + ")"
    bars = ax.bar(x_labels, valid["# Change Points"], color="#1f77b4", alpha=0.82, width=0.5)

    for bar, val in zip(bars, valid["# Change Points"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(int(val)),
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_xlabel("Sensitivity level  (min_rel_seg_len)")
    ax.set_ylabel("# Change Points Detected")
    ax.set_title(
        "Parameter Stability — # Change Points vs Sensitivity",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# BUG FIX 2 — Load data ONCE at module level so every page can use df,
#             min_year, and max_year.  Previously only loaded inside main().
# ══════════════════════════════════════════════════════════════════════════════
df = load_data()
df["year"] = df["game_date"].dt.year
max_year = int(df["year"].max())
min_year = int(df["year"].min())


# ══════════════════════════════════════════════════════════════════════════════
# BUG FIX 3 — Sidebar lives at module level (not inside main()).
#             A single `with st.sidebar:` block renders the nav radio +
#             page-specific controls in sequence.
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Controls")
    page = st.radio(
        "Navigation",
        ["🏠 Welcome 2: Welcome Page", "👤  Player Snapshot", "📊  ChangeForest CPD"],
        label_visibility="collapsed",
    )
    st.markdown("---")

# Pre-calculate players list with "All Players" option
player_names = sorted(df["player_name"].dropna().unique().tolist())
players_with_all = ["All Players (League Avg)"] + player_names

# ══════════════════════════════════════════════════════════════════════════════
# BUG FIX 4 — Page routing is a flat if / elif at module level.
#             Previously the routing was wrapped in def main() which was
#             never called, making ALL page content unreachable.
#             Also: the original had `if __name__ == "__main__": main()`
#             placed BETWEEN the `if "Welcome"` block and `elif "ChangeForest"`
#             block, which made the elif a syntax/logic error (elif after a
#             non-if statement).
# ══════════════════════════════════════════════════════════════════════════════

# ── PAGE: Welcome ─────────────────────────────────────────────────────────────
if "Welcome" in page:
    st.markdown("# ⚾ MLB Batting Pulse — Welcome 2")
    st.markdown("### Real-time batting performance change detection using Statcast & Machine Learning")

    # ── Motivation & Navigation ────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎯 Why it matters")
        st.markdown(f"""
        In baseball, a slump can last weeks before it's visible in traditional box score stats like Batting Average.
        
        Using **Statcast** (physics-based pitch tracking) and **Change Point Detection (CPD)**, we monitor 
        underlying performance signals in real-time. This allows us to catch performance shifts the 
        moment they happen—often long before the results catch up.
        """)
    with col2:
        st.markdown("#### 🚀 How to use this tool")
        st.markdown("""
        1. **Select a Player**: Use the sidebar to pick any qualified hitter from 2021-2025.
        2. **Tune the Window**: Adjust the 'Rolling Window' to change the sensitivity (zoom level).
        3. **Explore Tabs**:
           - **Input Signals**: See the raw data and rolling trends.
           - **Before/After Eval**: Statistical proof of the performance shift.
           - **Parameter Stability**: Ensure the detected changes are reliable.
        """)

    st.markdown("---")

    # ── Indicators Grid ───────────────────────────────────────────────────────
    st.markdown("#### 🔬 The Four Indicators")
    st.caption("We monitor these four metrics to understand the 'pulse' of a hitter's contact quality.")
    
    i1, i2 = st.columns(2)
    with i1:
        st.markdown(f"""<div class="card-row">
            {card("Hitting Decisions", "Plate Discipline", "Are they swinging at strikes and taking balls?", "gold")}
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="card-row">
            {card("wOBA Residual", "Luck vs Skill", "Is their actual performance matching the physics of their contact?", "teal")}
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown(f"""<div class="card-row">
            {card("Power Efficiency", "Raw Power", "How well are they converting swing effort into exit velocity?", "teal")}
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="card-row">
            {card("Launch Angle Stability", "Swing Consistency", "Is their swing path staying steady or wavering?", "gold")}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Dataset Stats ─────────────────────────────────────────────────────────
    sec("📊 By the Numbers")
    st.markdown(f"""<div class="card-row">
        {card("Players",        f"{df['player_name'].nunique():,}",          "unique hitters",       "gold")}
        {card("Total Records",  f"{len(df):,}",                              "plate appearances",    "teal")}
        {card("Seasons",        str(df['year'].nunique()),                   f"{min_year}–{max_year}", "grey")}
        {card(f"{max_year} PAs", f"{len(df[df['year'] == max_year]):,}",     "current season",       "teal")}
    </div>""", unsafe_allow_html=True)

# ── PAGE: Player Snapshot ─────────────────────────────────────────────────────
elif "Player Snapshot" in page:
    st.markdown("# 👤 Player Snapshot")

    # ── TOP FILTERS (Reciprocal filtering) ────────────────────────────────────
    if 'v2_snapshot_player' not in st.session_state:
        st.session_state.v2_snapshot_player = "All Players (League Avg)"
    if 'v2_snapshot_year' not in st.session_state:
        st.session_state.v2_snapshot_year = max_year

    # 1. Filter Years based on current player
    if st.session_state.v2_snapshot_player == "All Players (League Avg)":
        available_years = sorted(df["year"].unique(), reverse=True)
    else:
        available_years = sorted(df[df["player_name"] == st.session_state.v2_snapshot_player]["year"].unique(), reverse=True)
    
    if st.session_state.v2_snapshot_year not in available_years:
        st.session_state.v2_snapshot_year = available_years[0]

    # 2. Filter Players based on current year
    available_players_for_year = sorted(df[df["year"] == st.session_state.v2_snapshot_year]["player_name"].dropna().unique())
    player_options = ["All Players (League Avg)"] + available_players_for_year
    
    if st.session_state.v2_snapshot_player not in player_options:
        st.session_state.v2_snapshot_player = "All Players (League Avg)"

    # Filter Layout
    row1_c1, row1_c2 = st.columns([3, 1])
    with row1_c1:
        sel_player = st.selectbox(
            "Select Player", 
            player_options, 
            index=player_options.index(st.session_state.v2_snapshot_player),
            key="v2_snapshot_player_select"
        )
    with row1_c2:
        sel_year = st.selectbox(
            "Season", 
            available_years, 
            index=available_years.index(st.session_state.v2_snapshot_year),
            key="v2_snapshot_year_select"
        )

    # Update session state and rerun if changed
    if sel_player != st.session_state.v2_snapshot_player or sel_year != st.session_state.v2_snapshot_year:
        st.session_state.v2_snapshot_player = sel_player
        st.session_state.v2_snapshot_year = sel_year
        st.rerun()

    # Data Slicing
    is_all = (sel_player == "All Players (League Avg)")
    if is_all:
        snapshot_df = df[df["year"] == sel_year].copy()
        display_name = f"League Average — {sel_year}"
    else:
        snapshot_df = df[(df["player_name"] == sel_player) & (df["year"] == sel_year)].copy()
        display_name = f"{sel_player} — {sel_year}"

    # Calculate Metrics
    def _mean(s): return s.dropna().mean() if not s.empty else 0.0
    
    m_discipline = _mean(snapshot_df["hitting_decisions_score"])
    m_power      = _mean(snapshot_df["power_efficiency"])
    m_woba_res   = _mean(snapshot_df["woba_residual"])
    m_la_stab    = _mean(snapshot_df["launch_angle_stability_50pa"])
    m_pa_count   = len(snapshot_df)

    # Metric Cards with Tooltips
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hitting Decisions", f"{m_discipline:.2f}", help=INDICATOR_TOOLTIPS["hitting_decisions_score"])
    c2.metric("Power Efficiency",  f"{m_power:.4f}",      help=INDICATOR_TOOLTIPS["power_efficiency"])
    c3.metric("wOBA Residual",     f"{m_woba_res:.3f}",   help=INDICATOR_TOOLTIPS["woba_residual"])
    c4.metric("Launch Angle Stab.", f"{m_la_stab:.2f}",   help=INDICATOR_TOOLTIPS["launch_angle_stability_50pa"])
    c5.metric("PAs this Season",   f"{m_pa_count:,}",     help="Total records for the season. More PAs (>200) indicate higher statistical reliability.")

    # ── DYNAMIC INSIGHT ───────────────────────────────────────────────────────
    if not is_all:
        season_df = df[df["year"] == sel_year]
        
        def get_stat_insight(col, val):
            l_mean = season_df[col].mean()
            l_std  = season_df[col].std()
            if val > l_mean + l_std: return "Elite", "Performing significantly better than the league average."
            if val > l_mean: return "Above Average", "Performing better than most of the league."
            if val < l_mean - l_std: return "Struggling", "Currently performing well below the league benchmark."
            if val < l_mean: return "Below Average", "Performing slightly below the league average."
            return "Average", "Performing right in line with the league average."

        st.markdown("---")
        sec(f"💡 {sel_year} Performance Pulse")
        st.caption("We analyze these four pillars to identify a player's true performance beyond simple averages.")
        
        # 2x2 Grid for Insights
        i_col1, i_col2 = st.columns(2)
        
        with i_col1:
            # 1. Discipline
            status, desc = get_stat_insight("hitting_decisions_score", m_discipline)
            st.markdown(f"🎯 **Discipline**")
            st.caption("The ability to swing at strikes and take balls.")
            st.write(f"**{status}**: {desc}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 3. Results vs. Physics
            if m_woba_res > 0.15: status, res_text = "Outpacing Physics", "Results are better than the contact quality suggests (Potential Luck)."
            elif m_woba_res < -0.15: status, res_text = "Underperforming Physics", "Contact quality is high, but results haven't followed (Unlucky)."
            else: status, res_text = "Steady", "Actual results are closely matching the physics of the contact."
            st.markdown(f"🎲 **Results vs. Physics**")
            st.caption("Whether actual results match the quality of contact.")
            st.write(f"**{status}**: {res_text}")

        with i_col2:
            # 2. Power
            status, desc = get_stat_insight("power_efficiency", m_power)
            st.markdown(f"💥 **Power**")
            st.caption("How hard a player hits the ball relative to their effort.")
            st.write(f"**{status}**: {desc}")

            st.markdown("<br>", unsafe_allow_html=True)

            # 4. Consistency
            status, desc = get_stat_insight("launch_angle_stability_50pa", m_la_stab)
            st.markdown(f"📈 **Consistency**")
            st.caption("How repeatable and steady a player's swing path is.")
            st.write(f"**{status}**: {desc}")

    # ── VISUALIZATIONS ────────────────────────────────────────────────────────
    st.markdown("---")
    v_col1, v_col2 = st.columns([1, 1])

    with v_col1:
        sec("Player Profile (Radar Chart)")
        season_df = df[df["year"] == sel_year]
        radar_data = []
        theta_labels = []
        for col in CPD_INDICATORS:
            val = _mean(snapshot_df[col])
            s_min, s_max = season_df[col].min(), season_df[col].max()
            norm_val = (val - s_min) / (s_max - s_min + 1e-9)
            radar_data.append(norm_val)
            theta_labels.append(INDICATOR_LABELS[col])
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_data + [radar_data[0]],
            theta=theta_labels + [theta_labels[0]],
            fill='toself',
            name=sel_player,
            line_color=TEAL
        ))
        
        leag_vals = []
        for col in CPD_INDICATORS:
            l_val = _mean(season_df[col])
            s_min, s_max = season_df[col].min(), season_df[col].max()
            norm_l = (l_val - s_min) / (s_max - s_min + 1e-9)
            leag_vals.append(norm_l)
        
        fig_radar.add_trace(go.Scatterpolar(
            r=leag_vals + [leag_vals[0]],
            theta=theta_labels + [theta_labels[0]],
            name='League Average',
            line=dict(dash='dash', color=GREY)
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, height=400, margin=dict(t=40, b=40, l=40, r=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with v_col2:
        sec("Statcast-Style Percentile Rankings")
        if is_all:
            st.info("Percentile rankings represent the 50th percentile (average) when 'All Players' is selected.")
            percentiles = [50] * 4
        else:
            player_means = df[df["year"] == sel_year].groupby("player_name")[CPD_INDICATORS].mean()
            percentiles = []
            for col in CPD_INDICATORS:
                p_val = _mean(snapshot_df[col])
                pct = (player_means[col] < p_val).mean() * 100
                percentiles.append(pct)
        
        fig_pct = go.Figure(go.Bar(
            x=percentiles,
            y=[INDICATOR_LABELS[c] for c in CPD_INDICATORS],
            orientation='h',
            marker=dict(color=percentiles, colorscale='RdYlGn', cmin=0, cmax=100),
            text=[f"{p:.0f}%" for p in percentiles],
            textposition='auto',
        ))
        fig_pct.update_layout(
            xaxis=dict(range=[0, 100], title="Percentile Rank"),
            yaxis=dict(autorange="reversed"),
            height=400, margin=dict(t=40, b=40, l=100, r=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT)
        )
        st.plotly_chart(fig_pct, use_container_width=True)

    st.markdown("---")
    sec("Contact Quality Profile (Power vs Luck)")
    fig_scatter = px.scatter(
        snapshot_df.dropna(subset=["power_efficiency", "woba_residual"]),
        x="power_efficiency", y="woba_residual",
        color_discrete_sequence=[TEAL],
        opacity=0.4,
        labels={"power_efficiency": "Power Efficiency", "woba_residual": "wOBA Residual"},
        title=f"Distribution for {display_name}"
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color=GREY)
    fig_scatter.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT), height=500
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── PAGE: ChangeForest CPD ────────────────────────────────────────────────────
elif "ChangeForest CPD" in page:
    st.markdown("# ChangeForest Change-Point Detection")
    
    # ── TOP FILTERS ───────────────────────────────────────────────────────────
    row1_c1, row1_c2 = st.columns([3, 1])
    with row1_c1:
        sel_player = st.selectbox("Select Player", player_names, index=player_names.index("Trout, Mike") if "Trout, Mike" in player_names else 0)
    with row1_c2:
        cf_sens = st.radio("Sensitivity", ["Low","Medium","High"], index=1, horizontal=True)

    cf_window = st.slider("Rolling Window (PAs)", 20, 120, 50, 5)
    cf_min_seg = SENSITIVITY_TO_MIN_SEG[cf_sens]
    st.markdown("---")

    # ── Dependency check ──────────────────────────────────────────────────────
    if changeforest is None or Control is None:
        st.error(
            "The `changeforest` package is not installed. "
            "Run `pip install changeforest` and restart the app."
        )
        st.stop()

    if selected_player_id is None:
        st.warning("Player not found in dataset.")
        st.stop()

    # ── Prepare subdataset ────────────────────────────────────────────────────
    with st.spinner("Preparing player dataset…"):
        subdf = changeforest_subdataset_generator(df, selected_player_id, window=rolling_window)

    if subdf.empty:
        st.warning(
            f"No rows available for **{selected_name}** after filtering for all 4 indicators. "
            "Try a smaller rolling window."
        )
        st.stop()

    # ── Run ChangeForest ──────────────────────────────────────────────────────
    with st.spinner("Running ChangeForest random-forest CPD…"):
        try:
            _, cps, _, feature_names = run_changeforest(
                subdf,
                window=rolling_window,
                minimal_relative_segment_length=min_rel_seg_len,
                use_rollmean=True,
                standardize=True,
            )
        except Exception as exc:
            st.error(f"ChangeForest failed: {exc}")
            st.stop()

    # ── Summary metric bar ────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Player", selected_name)
    c2.metric("Batter ID", str(selected_player_id))
    c3.metric("PA Rows Used", f"{len(subdf):,}")
    c4.metric("Sensitivity", sensitivity)
    c5.metric("Detected CPs", str(len(cps)))

    if cps:
        with st.expander(f"📍 Change-point rows ({len(cps)} points)"):
            st.dataframe(
                subdf.iloc[cps][["cf_seq_id", "game_date", "pa_uid"]].reset_index(drop=True),
                use_container_width=True,
            )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📈 ChangeForest Result + Input Signal",
        "📊 Before / After Eval",
        "🔧 Parameter Stability",
    ])

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("ChangeForest Result")
        st.caption(
            "All 4 standardized rolling-mean signals on a single timeline. "
            "Dashed red vertical lines mark detected change points."
        )
        fig_result = plot_changeforest_result(subdf, cps, rolling_window, selected_name, min_rel_seg_len)
        st.pyplot(fig_result, use_container_width=True)

        st.markdown("---")
        st.subheader("Input Signals  (Raw + Rolling Mean)")
        st.caption(
            "2 × 2 panel showing each indicator's raw plate-appearance values "
            "and its rolling mean (same window as CPD)."
        )
        fig_signals = plot_input_signals(subdf, rolling_window, selected_name)
        st.pyplot(fig_signals, use_container_width=True)

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Before / After Statistical Comparison")
        st.caption(
            "For every sequence position, the metric distribution in the window **before** "
            "vs **after** is compared. Detected CPs (Y) should show larger absolute shifts "
            "in mean and std than non-CP positions (N)."
        )

        compare_window = max(5, min(50, len(subdf) // 4))

        if len(subdf) <= 2 * compare_window:
            st.warning(
                f"Not enough data for before/after evaluation "
                f"(need > {2 * compare_window} rows, have {len(subdf)}). "
                "Try a smaller rolling window."
            )
        else:
            with st.spinner("Computing before/after statistics…"):
                eval_dfs = build_cp_eval_dfs(subdf, cps, feature_names, compare_window=compare_window)

            if not eval_dfs:
                st.warning("Could not build eval dataframes.")
            else:
                fig_eval = plot_cp_eval_comparison(eval_dfs)
                if fig_eval:
                    st.pyplot(fig_eval, use_container_width=True)

                with st.expander("📋 Detailed eval summary tables"):
                    for feat, df_eval in eval_dfs.items():
                        base = feat.split("_rollmean_")[0]
                        st.write(f"**{INDICATOR_LABELS.get(base, feat)}**")
                        st.dataframe(
                            df_eval.groupby("cp")[
                                ["mean_before", "mean_after", "mean_diff",
                                 "std_before", "std_after", "std_diff",
                                 "abs_mean_diff", "abs_std_diff"]
                            ].mean().round(4),
                            use_container_width=True,
                        )

    # ── Tab 3 ─────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Parameter Stability")
        st.caption(
            "Runs ChangeForest across **all three sensitivity levels** "
            "(Low / Medium / High) on the same player + window. "
            "Stable results → similar CP counts across levels. "
            "Large variation → the detected CPs are sensitive to this parameter."
        )

        stability_vals = sorted(SENSITIVITY_TO_MIN_SEG.values())  # [0.02, 0.05, 0.10]

        with st.spinner("Running stability analysis across all sensitivity levels…"):
            stability_df = run_parameter_stability(subdf, rolling_window, min_rel_seg_len_list=stability_vals)

        fig_stability = plot_parameter_stability(stability_df)
        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            st.pyplot(fig_stability, use_container_width=True)
        with col_table:
            st.dataframe(stability_df, use_container_width=True, hide_index=True)