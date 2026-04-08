#!/usr/bin/env python
# coding: utf-8
"""
MLB Batting Pulse
==================
Single dataset: PA-level engineered features
  pa_master.csv — one row per plate appearance, 420 qualified hitters, 2021-2025

Pages
-----
1. Welcome
2. Player Snapshot
3. PA Indicator Correlations   (replaces "What Drives Outcomes")
4. Performance Index           (PELT CPD on composite score)
5. Metric Drilldown            (PELT CPD on each PA indicator)
6. Season Timeline
7. Peer Comparison
8. ChangeForest CPD            (multivariate CPD on all 4 indicators)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns
import streamlit as st
import gdown
from sklearn.preprocessing import StandardScaler

try:
    from changeforest import Control, changeforest
except Exception:
    Control = None
    changeforest = None


# ══════════════════════════════════════════════════════════════════════════════
# MUST BE THE VERY FIRST STREAMLIT CALL
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MLB Batting Pulse",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
CURRENT_SEASON  = 2025
DATA_CF_FILE_ID = "1G8eA6gX8hdCwWp1YWddAMmwt6R62tcmA"

PA_INDICATORS = [
    "hitting_decisions_score",
    "power_efficiency",
    "woba_residual",
    "launch_angle_stability_50pa",
]
PA_LABELS = {
    "hitting_decisions_score":     "Hitting Decisions",
    "power_efficiency":            "Power Efficiency",
    "woba_residual":               "wOBA Residual",
    "launch_angle_stability_50pa": "Launch Angle Stability",
}
PA_COLORS = {
    "hitting_decisions_score":     "#2ca02c",
    "power_efficiency":            "#1f77b4",
    "woba_residual":               "#ff7f0e",
    "launch_angle_stability_50pa": "#d62728",
}

SENSITIVITY_MAP        = {"Low": 8, "Medium": 3, "High": 1}
SENSITIVITY_TO_MIN_SEG = {"Low": 0.10, "Medium": 0.05, "High": 0.02}

DARK       = "#0d1117"
PANEL      = "#161b22"
BORDER     = "#30363d"
GOLD       = "#d4a017"
GOLD_LT    = "#f0c040"
TEAL_LT    = "#3eb8b8"
RED_LT     = "#e74c3c"
GREY       = "#6e7681"
TEXT       = "#e6edf3"
TEXT_MUTED = "#8b949e"


# ══════════════════════════════════════════════════════════════════════════════
# STYLES
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
    text-transform:uppercase; letter-spacing:0.1em; font-family:'IBM Plex Mono',monospace;
}}
.card-row {{ display:flex; gap:10px; margin-bottom:1rem; flex-wrap:wrap; }}
.card {{ background:{PANEL}; border:1px solid {BORDER}; border-top:3px solid {GOLD};
    border-radius:4px; padding:0.9rem 1.1rem; flex:1; min-width:130px; }}
.card.teal {{ border-top-color:{TEAL_LT}; }}
.card.grey {{ border-top-color:{GREY}; }}
.card-label {{ font-size:0.62rem; text-transform:uppercase; letter-spacing:0.12em;
    color:{TEXT_MUTED}; font-family:'IBM Plex Mono',monospace; margin-bottom:4px; }}
.card-val {{ font-size:1.6rem; font-weight:600; color:{TEXT};
    font-family:'Bebas Neue',sans-serif; letter-spacing:0.04em; line-height:1.1; }}
.card-sub {{ font-size:0.72rem; color:{TEXT_MUTED}; font-family:'IBM Plex Mono',monospace; margin-top:2px; }}
.sec {{ font-size:0.65rem; text-transform:uppercase; letter-spacing:0.15em;
    color:{TEAL_LT}; font-family:'IBM Plex Mono',monospace;
    border-bottom:1px solid {BORDER}; padding-bottom:4px; margin:1.1rem 0 0.7rem 0; }}
.cpd-minor {{ display:inline-block; background:#30363d; color:{TEXT_MUTED}; border-radius:3px; padding:2px 8px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; margin:2px; }}
.cpd-mod   {{ display:inline-block; background:#3a2f00; color:{GOLD_LT}; border:1px solid {GOLD}; border-radius:3px; padding:2px 8px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; margin:2px; }}
.cpd-sig   {{ display:inline-block; background:#3b0d0d; color:{RED_LT}; border:1px solid #c0392b; border-radius:3px; padding:2px 8px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; margin:2px; }}
.preamble {{ background:{PANEL}; border:1px solid {BORDER}; border-left:4px solid {GOLD};
    border-radius:4px; padding:1.2rem 1.4rem; margin-bottom:1.4rem;
    font-size:0.92rem; line-height:1.7; color:{TEXT_MUTED}; }}
.preamble b {{ color:{TEXT}; }}
.narrative {{ background:#161b22; border:1px solid {BORDER}; border-radius:4px;
    padding:1rem 1.2rem; font-family:'IBM Plex Mono',monospace;
    font-size:0.82rem; line-height:1.6; color:{TEXT}; margin-bottom:1rem; }}
.rel-high {{ color:{TEAL_LT}; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }}
.rel-med  {{ color:{GOLD_LT}; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }}
.rel-low  {{ color:{RED_LT};  font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def card(label, val, sub="", color="gold"):
    cls = {"gold":"card","teal":"card teal","grey":"card grey"}.get(color, "card")
    return (f'<div class="{cls}"><div class="card-label">{label}</div>'
            f'<div class="card-val">{val}</div><div class="card-sub">{sub}</div></div>')

def sec(title):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)

def mpl_fig(w=11, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8.5)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.xaxis.label.set_color(TEXT_MUTED); ax.yaxis.label.set_color(TEXT_MUTED)
    ax.title.set_color(TEXT)
    ax.grid(color=BORDER, linewidth=0.4, linestyle="--", alpha=0.5)
    return fig, ax

def effect_size_label(d):
    ad = abs(d)
    if ad < 0.2: return "minor",    "cpd-minor"
    if ad < 0.5: return "moderate", "cpd-mod"
    return "significant", "cpd-sig"

def reliability_badge(n):
    if n >= 200: return f'<span class="rel-high">&#9679; High reliability</span>'
    if n >= 100: return f'<span class="rel-med">&#9679; Medium reliability</span>'
    return f'<span class="rel-low">&#9679; Low reliability (few PAs this season)</span>'

def games_label(n_pa):
    return f"~{max(1, round(n_pa / 3.8))} games"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading PA-level data...")
def load_data() -> pd.DataFrame:
    path = "/tmp/Qualified_hitters_statcast_2021_2025_pa_master.csv"
    cols = [
        "batter", "player_name", "pa_uid", "game_date", "game_pk",
        "at_bat_number", "cf_seq_id",
        "hitting_decisions_score", "power_efficiency",
        "woba_residual", "launch_angle_stability_50pa",
    ]
    if not os.path.exists(path):
        gdown.download(
            f"https://drive.google.com/uc?id={DATA_CF_FILE_ID}",
            path, quiet=False, fuzzy=True,
        )
    df = pd.read_csv(path, usecols=cols)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["year"]  = df["game_date"].dt.year
    df["month"] = df["game_date"].dt.month
    df = df.sort_values(["player_name", "game_date"]).reset_index(drop=True)
    return df


df       = load_data()
players  = sorted(df["player_name"].dropna().unique())
min_year = int(df["year"].min())
max_year = int(df["year"].max())


# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE INDEX
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Computing performance index...")
def build_perf_index(_df: pd.DataFrame) -> pd.DataFrame:
    out = _df.copy()
    normed = pd.DataFrame(index=out.index)
    for col in PA_INDICATORS:
        if col not in out.columns:
            continue
        lo, hi = out[col].quantile(0.01), out[col].quantile(0.99)
        clipped = out[col].clip(lo, hi)
        normed[col] = (clipped - lo) / (hi - lo + 1e-9)
    out["perf_index"] = normed.mean(axis=1) * 100
    return out

df_idx = build_perf_index(df)


# ══════════════════════════════════════════════════════════════════════════════
# PELT CPD UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def detect_cpd(series, penalty):
    try:
        import ruptures as rpt
        sig  = series.values.reshape(-1, 1)
        algo = rpt.Pelt(model="rbf").fit(sig)
        bkps = algo.predict(pen=penalty)
        return [b for b in bkps[:-1] if 0 < b < len(series)]
    except Exception:
        return []

def cpd_stats(series, cp_indices):
    results = []
    bpts = [0] + cp_indices + [len(series)]
    for i in range(len(bpts) - 2):
        before = series.iloc[bpts[i]:bpts[i+1]]
        after  = series.iloc[bpts[i+1]:bpts[i+2]]
        if len(before) < 5 or len(after) < 5:
            continue
        delta  = after.mean() - before.mean()
        pooled = np.sqrt((before.std()**2 + after.std()**2) / 2 + 1e-9)
        d      = delta / pooled
        label, badge = effect_size_label(d)
        results.append({
            "cp_idx": bpts[i+1], "before_mean": before.mean(),
            "after_mean": after.mean(), "delta": delta,
            "effect_d": d, "label": label, "badge": badge,
            "direction": "improvement" if delta > 0 else "decline",
        })
    return results

def rolling_with_dates(pdf, metric, window):
    sub = pdf[pdf[metric].notna()][["game_date", metric]].sort_values("game_date")
    if len(sub) < window:
        return pd.DataFrame(columns=["game_date", metric])
    roll = sub[metric].rolling(window, min_periods=window // 2).mean()
    return pd.DataFrame({"game_date": sub["game_date"].values,
                          metric: roll.values}).dropna()

def baselines(metric, player):
    prev = df[(df["player_name"] == player) & (df["year"] == max_year - 1)][metric].dropna()
    leag = df[df["year"] == max_year][metric].dropna()
    return (
        prev.mean() if len(prev) else np.nan,
        leag.mean() if len(leag) else np.nan,
        leag.std()  if len(leag) else np.nan,
    )

def add_baselines(ax, b_last, b_leag_m, b_leag_s):
    if not np.isnan(b_last):
        ax.axhline(b_last, color=GOLD, lw=1.2, ls="--", alpha=0.8,
                   label=f"Prev season avg ({b_last:.2f})", zorder=2)
    if not np.isnan(b_leag_m):
        ax.axhline(b_leag_m, color=GREY, lw=1.0, ls="-", alpha=0.7,
                   label=f"League avg {max_year} ({b_leag_m:.2f})", zorder=2)
        if not np.isnan(b_leag_s):
            ax.axhspan(b_leag_m - b_leag_s, b_leag_m + b_leag_s,
                       alpha=0.06, color=GREY, zorder=1)

def add_cpd_markers(ax, roll_df, metric, cp_indices, stats):
    ymin, ymax = ax.get_ylim()
    for s in stats:
        idx   = min(s["cp_idx"], len(roll_df) - 1)
        date  = roll_df["game_date"].iloc[idx]
        color = {"cpd-minor": GREY, "cpd-mod": GOLD, "cpd-sig": RED_LT}[s["badge"]]
        ax.axvline(date, color=color, lw=1.4, ls="--", alpha=0.85, zorder=4)
        arrow = "^" if s["delta"] > 0 else "v"
        ax.text(date, ymin + (ymax - ymin) * 0.93,
                f" {arrow}{abs(s['delta']):.2f}",
                color=color, fontsize=7.5, va="top", fontfamily="monospace")

def add_season_dividers(ax, ymax):
    for yr in range(min_year + 1, max_year + 1):
        d = pd.Timestamp(f"{yr}-03-01")
        ax.axvline(d, color=GREY, lw=0.6, ls=":", alpha=0.4)
        ax.text(d, ymax * 0.98, str(yr), color=TEXT_MUTED, fontsize=7,
                ha="left", va="top", fontfamily="monospace")

def generate_narrative(player, metric_label, roll_df, cp_st, b_last, b_leag_m, window):
    metric_col = roll_df.columns[1]
    current    = roll_df[metric_col].iloc[-1] if len(roll_df) else np.nan
    if not cp_st:
        txt = (f"No significant change points detected for {player} on "
               f"{metric_label} over the last {games_label(window)}. "
               f"Current rolling value: {current:.2f}.")
        if not np.isnan(b_leag_m):
            diff = current - b_leag_m
            txt += (f" This is {abs(diff):.2f} "
                    f"{'above' if diff >= 0 else 'below'} the {max_year} league average.")
        return txt
    s        = cp_st[-1]
    idx      = min(s["cp_idx"], len(roll_df) - 1)
    date     = roll_df["game_date"].iloc[idx]
    date_str = date.strftime("%B %d, %Y") if hasattr(date, "strftime") else str(date)
    txt = (f"{player}'s {metric_label} {s['direction']}d {s['label']}ly around {date_str} "
           f"(Cohen's d={abs(s['effect_d']):.2f}). "
           f"Rolling average shifted from {s['before_mean']:.2f} to {s['after_mean']:.2f} "
           f"({'+' if s['delta'] > 0 else ''}{s['delta']:.2f}).")
    if len(cp_st) > 1:
        txt += f" {len(cp_st)} total change points detected."
    if not np.isnan(b_leag_m):
        diff = s["after_mean"] - b_leag_m
        txt += (f" Current level is {abs(diff):.2f} "
                f"{'above' if diff >= 0 else 'below'} the {max_year} league average.")
    if not np.isnan(b_last):
        diff2 = s["after_mean"] - b_last
        txt += (f" Previous season average was {b_last:.2f} - currently "
                f"{abs(diff2):.2f} {'above' if diff2 >= 0 else 'below'}.")
    return txt


# ══════════════════════════════════════════════════════════════════════════════
# CHANGEFOREST UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def cf_subdataset(player_name, window):
    base_cols    = ["batter","player_name","pa_uid","game_date","game_pk","at_bat_number"]
    feature_cols = ["cf_seq_id"] + PA_INDICATORS
    subdf = (
        df.loc[df["player_name"] == player_name, base_cols + feature_cols]
        .dropna(subset=PA_INDICATORS)
        .sort_values("cf_seq_id")
        .reset_index(drop=True)
    )
    for col in PA_INDICATORS:
        subdf[f"{col}_rollmean_{window}"] = (
            subdf[col].rolling(window=window, min_periods=window).mean()
        )
    rollmean_cols = [f"{col}_rollmean_{window}" for col in PA_INDICATORS]
    return subdf.dropna(subset=rollmean_cols).reset_index(drop=True)


def run_changeforest(subdf, window=50, use_rollmean=True, standardize=True,
                     minimal_relative_segment_length=0.05):
    if subdf is None or subdf.empty:
        return None, [], None, None
    feature_names = (
        [f"{col}_rollmean_{window}" for col in PA_INDICATORS]
        if use_rollmean else list(PA_INDICATORS)
    )
    missing = [c for c in feature_names if c not in subdf.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    X = subdf[feature_names].dropna().values
    if len(X) == 0 or np.isnan(X).any():
        raise ValueError("Feature matrix empty or contains NaN.")
    if standardize:
        X = StandardScaler().fit_transform(X)
    ctrl   = Control(model_selection_alpha=0.02,
                     minimal_relative_segment_length=minimal_relative_segment_length)
    result = changeforest(X, method="random_forest", control=ctrl)
    cps    = [int(cp) for cp in result.split_points() if 0 <= int(cp) < len(subdf)]
    return result, cps, X, feature_names


def plot_cf_result(subdf, cps, window, player_name, min_rel_seg_len):
    dates = pd.to_datetime(subdf["game_date"])
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    fig.patch.set_facecolor("#f7f8fa"); ax.set_facecolor("#ffffff")
    for col, x_offset in zip(PA_INDICATORS, [8, -68, 8, -68]):
        color = PA_COLORS[col]; label = PA_LABELS[col]
        y_col = f"{col}_rollmean_{window}"
        ax.plot(dates, subdf[y_col], linewidth=2.2, color=color, label=label, alpha=0.9)
        valid = subdf[[y_col,"game_date"]].dropna()
        if not valid.empty:
            last_idx  = valid.index[-1]
            last_date = pd.to_datetime(subdf.loc[last_idx,"game_date"])
            last_val  = subdf.loc[last_idx, y_col]
            ax.scatter(last_date, last_val, color=color, s=40, zorder=5)
            ax.annotate(label, xy=(last_date, last_val), xytext=(x_offset, 0),
                        textcoords="offset points", color=color, fontsize=9,
                        va="center", fontweight="bold")
    for cp in cps:
        if 0 <= cp < len(subdf):
            ax.axvline(x=dates.iloc[cp], color="#c43b3b",
                       linestyle=(0, (4, 4)), alpha=0.75, linewidth=1.8)
    ax.set_xlabel("Game Date")
    ax.set_ylabel("Metric value (standardized rolling mean)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)
    ax.set_title(
        f"ChangeForest Result  |  {player_name}"
        f"  |  window={window}  |  min_seg={min_rel_seg_len}  |  CPs={len(cps)}",
        fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    return fig


def plot_cf_input_signals(subdf, window, player_name=""):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=True)
    axes = axes.flatten(); fig.patch.set_facecolor("#f7f8fa")
    for ax, col in zip(axes, PA_INDICATORS):
        color = PA_COLORS[col]; label = PA_LABELS[col]
        smooth_col = f"{col}_rollmean_{window}"
        ax.set_facecolor("#ffffff")
        ax.scatter(subdf["cf_seq_id"], subdf[col], s=8, alpha=0.22,
                   color=color, edgecolor="none", label="Raw PA")
        if smooth_col in subdf.columns:
            ax.plot(subdf["cf_seq_id"], subdf[smooth_col], linewidth=2.4,
                    color=color, alpha=0.95, label=f"Rolling mean (w={window})")
        ax.set_title(f"{label}  (n={len(subdf):,})", fontsize=11, fontweight="bold")
        ax.set_xlabel("CF Sequence ID"); ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Input Signals - {player_name}" if player_name else "Input Signals",
                 fontsize=13, fontweight="bold")
    return fig


def build_cp_eval_dfs(subdf, cps, feature_names, compare_window=50):
    n = len(subdf)
    if n <= 2 * compare_window:
        return {}
    cps_set = set(cps); eval_dfs = {}
    for col in feature_names:
        rows = []
        for i in range(compare_window, n - compare_window):
            before = subdf[col].iloc[i - compare_window: i]
            after  = subdf[col].iloc[i: i + compare_window]
            rows.append({
                "feature": col, "cp": "Y" if i in cps_set else "N",
                "mean_before": before.mean(), "mean_after": after.mean(),
                "mean_diff": after.mean() - before.mean(),
                "std_before": before.std(), "std_after": after.std(),
                "std_diff": after.std() - before.std(),
                "abs_mean_diff": abs(after.mean() - before.mean()),
                "abs_std_diff":  abs(after.std()  - before.std()),
            })
        eval_dfs[col] = pd.DataFrame(rows)
    return eval_dfs


def plot_cp_eval_comparison(eval_dfs):
    n = len(eval_dfs)
    if n == 0: return None
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), constrained_layout=True)
    if n == 1: axes = [axes]
    for ax, (feature, df_eval) in zip(axes, eval_dfs.items()):
        if df_eval.empty or "cp" not in df_eval.columns:
            ax.set_title(f"{feature}\n(no data)"); continue
        summary = df_eval.groupby("cp")[["abs_mean_diff","abs_std_diff"]].mean()
        summary.plot(kind="bar", ax=ax, rot=0, color=["#1f77b4","#ff7f0e"])
        base_col = feature.split("_rollmean_")[0]
        ax.set_title(PA_LABELS.get(base_col, feature), fontsize=11, fontweight="bold")
        ax.set_xlabel("Is Change Point?  (Y = detected CP, N = non-CP)")
        ax.set_ylabel("Avg absolute difference")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(["Abs Mean Diff","Abs Std Diff"], frameon=False, fontsize=9)
    fig.suptitle("Before / After Comparison at Detected Change Points",
                 fontsize=13, fontweight="bold")
    return fig


def run_parameter_stability(subdf, window, min_rel_seg_len_list=(0.02, 0.05, 0.10)):
    inv_map = {v: k for k, v in SENSITIVITY_TO_MIN_SEG.items()}
    rows = []
    for val in sorted(min_rel_seg_len_list):
        try:
            _, cps, _, _ = run_changeforest(
                subdf, window=window, minimal_relative_segment_length=val)
            n_cp, cp_list = len(cps), cps
        except Exception as exc:
            n_cp, cp_list = None, []
            st.warning(f"Stability run failed for min_rel_seg_len={val}: {exc}")
        rows.append({"Sensitivity": inv_map.get(val, str(val)),
                     "min_rel_seg_len": val,
                     "# Change Points": n_cp,
                     "CP Indices": str(cp_list)})
    return pd.DataFrame(rows)


def plot_parameter_stability(stability_df):
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    fig.patch.set_facecolor("#f7f8fa"); ax.set_facecolor("#ffffff")
    valid    = stability_df.dropna(subset=["# Change Points"])
    x_labels = valid["Sensitivity"] + "\n(min=" + valid["min_rel_seg_len"].astype(str) + ")"
    bars = ax.bar(x_labels, valid["# Change Points"], color="#1f77b4", alpha=0.82, width=0.5)
    for bar, val in zip(bars, valid["# Change Points"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(int(val)), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sensitivity level  (min_rel_seg_len)")
    ax.set_ylabel("# Change Points Detected")
    ax.set_title("Parameter Stability - # Change Points vs Sensitivity",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## MLB Batting Pulse")
    st.caption("Real-time performance change detection · PA-level data")
    st.markdown("---")
    page = st.radio("Navigation", [
        "Welcome",
        "Player Snapshot",
        "PA Indicator Correlations",
        "Performance Index",
        "Metric Drilldown",
        "Season Timeline",
        "Peer Comparison",
        "ChangeForest CPD",
    ], label_visibility="collapsed")
    st.markdown("---")

    sel_player   = st.selectbox("Player", players)
    cpd_window   = st.slider("Rolling window (PAs)", 20, 100, 50, 5)
    st.caption(f"{games_label(cpd_window)} of plate appearances")
    sensitivity  = st.radio("CPD Sensitivity", ["Low","Medium","High"],
                             index=1, horizontal=True)
    penalty      = SENSITIVITY_MAP[sensitivity]
    show_history = st.toggle(f"Show full history ({min_year}-present)", value=False)
    st.markdown("---")

    if "ChangeForest" in page:
        st.caption("ChangeForest controls")
        cf_window  = st.slider("CF Rolling Window (PAs)", 20, 120, 50, 5, key="cf_w")
        cf_sens    = st.radio("CF Sensitivity", ["Low","Medium","High"],
                               horizontal=True, index=1, key="cf_s")
        cf_min_seg = SENSITIVITY_TO_MIN_SEG[cf_sens]
        st.markdown("---")
        st.caption(f"min_rel_seg_len: `{cf_min_seg}`")
        st.markdown(
            "| Level | min_seg | Effect |\n|---|---|---|\n"
            "| Low | 0.10 | fewer CPs |\n"
            "| Medium | 0.05 | balanced |\n"
            "| High | 0.02 | more CPs |"
        )

    n_curr = len(df[(df["player_name"] == sel_player) & (df["year"] == max_year)])
    n_all  = len(df[df["player_name"] == sel_player])
    st.markdown(reliability_badge(n_curr), unsafe_allow_html=True)
    st.caption(f"{n_all:,} total PAs  {n_curr:,} in {max_year}")


# Shared slices
player_df  = df[df["player_name"] == sel_player].copy()
player_idx = df_idx[df_idx["player_name"] == sel_player].copy()
cpd_df     = player_df  if show_history else player_df[player_df["year"]  == max_year].copy()
cpd_idx    = player_idx if show_history else player_idx[player_idx["year"] == max_year].copy()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 - WELCOME
# ══════════════════════════════════════════════════════════════════════════════
if "Welcome" in page:
    st.markdown("# MLB Batting Pulse")
    st.markdown("### Real-time batting performance change detection")
    st.markdown(f"""
    <div class="preamble">
    <b>The problem with traditional baseball metrics:</b> Statistics like wOBA, BABIP, and
    batting average are season-long rolling averages. By the time a decline shows up in the
    box score, a player may have been struggling for weeks.<br><br>
    <b>What this dashboard does:</b> Using plate-appearance-level Statcast data from
    {min_year} to {max_year}, we monitor four engineered contact-quality indicators in real
    time using <b>PELT</b> and <b>ChangeForest</b> change point detection.<br><br>
    [suggest that we add in a sentence or two about the intuition behind the indicators and why they are more sensitive to true performance changes than traditional stats]<br><br>

    <b>Why plate appearances, not raw pitches?</b> Pitch-level events are sparse and
    outcome-driven. PA-level aggregates capture <i>underlying contact quality</i> independent
    of luck, giving earlier and more reliable signals of true performance changes.<br><br>
    <b>The four indicators:</b><br>
    - <b>Hitting Decisions Score</b> - plate discipline and swing decision quality<br>
    - <b>Power Efficiency</b> - exit velocity relative to swing effort<br>
    - <b>wOBA Residual</b> - outcome quality above statistical expectation<br>
    - <b>Launch Angle Stability</b> - consistency of launch angle over recent PAs
    </div>
    """, unsafe_allow_html=True)

    sec("Dataset at a glance")
    st.markdown(f"""<div class="card-row">
        {card("Players",        f"{df['player_name'].nunique():,}", "qualified hitters",     "gold")}
        {card("Total PAs",      f"{len(df):,}",                    "plate appearances",      "teal")}
        {card("Seasons",        str(df['year'].nunique()),         f"{min_year}-{max_year}", "grey")}
        {card(f"{max_year} PAs",f"{len(df[df['year']==max_year]):,}", "current season",      "teal")}
    </div>""", unsafe_allow_html=True)

    sec("PA indicator distributions - all players, all seasons")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), constrained_layout=True)
    fig.patch.set_facecolor(DARK)
    for ax, col in zip(axes, PA_INDICATORS):
        color = PA_COLORS[col]
        data  = df[col].dropna()
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.hist(data, bins=60, color=color, alpha=0.8, edgecolor="none")
        ax.axvline(data.mean(), color=GOLD, lw=1.5, ls="--",
                   label=f"mean {data.mean():.2f}")
        ax.set_title(PA_LABELS[col], color=TEXT, fontsize=9, fontweight="bold")
        ax.set_xlabel("Value", color=TEXT_MUTED, fontsize=8)
        ax.set_ylabel("Count", color=TEXT_MUTED, fontsize=8)
        ax.legend(fontsize=7, framealpha=0.2, facecolor=PANEL,
                  edgecolor=BORDER, labelcolor=TEXT)
        ax.grid(color=BORDER, linewidth=0.3, linestyle="--", alpha=0.4)
    fig.suptitle(f"PA Indicator Distributions  ({min_year}-{max_year}, all players)",
                 color=TEXT, fontsize=11)
    st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 - PLAYER SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════
elif "Snapshot" in page:
    st.markdown(f"# {sel_player}")
    curr = player_df[player_df["year"] == max_year]
    leag = df[df["year"] == max_year]
    def _mean(s): return s.dropna().mean()

    sec(f"{max_year} performance vs league average")
    st.markdown(f"""<div class="card-row">
        {card("Hitting Decisions",  f"{_mean(curr['hitting_decisions_score']):.2f}",     f"League {_mean(leag['hitting_decisions_score']):.2f}",     "gold")}
        {card("Power Efficiency",   f"{_mean(curr['power_efficiency']):.2f}",            f"League {_mean(leag['power_efficiency']):.2f}",            "teal")}
        {card("wOBA Residual",      f"{_mean(curr['woba_residual']):.2f}",               f"League {_mean(leag['woba_residual']):.2f}",               "gold")}
        {card("Launch Angle Stab.", f"{_mean(curr['launch_angle_stability_50pa']):.2f}", f"League {_mean(leag['launch_angle_stability_50pa']):.2f}", "teal")}
        {card("PAs this season",    f"{len(curr):,}",                                   games_label(len(curr)),                                     "grey")}
    </div>""", unsafe_allow_html=True)

    sec("Season-by-season averages")
    summ = player_df.groupby("year").agg(
        hitting_decisions =("hitting_decisions_score",     "mean"),
        power_efficiency  =("power_efficiency",            "mean"),
        woba_residual     =("woba_residual",               "mean"),
        launch_angle_stab =("launch_angle_stability_50pa", "mean"),
        pa_count          =("pa_uid",                      "count"),
    ).round(3).reset_index().rename(columns={"year": "Season"})
    st.dataframe(summ, use_container_width=True, hide_index=True)

    sec("Power efficiency vs wOBA residual - contact quality profile by season")
    st.caption("Each dot is one plate appearance. The cloud shape reveals the balance "
               "between power generation and outcome quality.")
    plot_df = player_df[["power_efficiency","woba_residual","year"]].dropna()
    seasons = sorted(plot_df["year"].unique())
    pal     = [GOLD, TEAL_LT, RED_LT, "#a8d8a8", GREY]
    fig, ax = mpl_fig(11, 4.5)
    for i, s in enumerate(seasons):
        sub = plot_df[plot_df["year"] == s]
        ax.scatter(sub["power_efficiency"], sub["woba_residual"],
                   color=pal[i % len(pal)], alpha=0.3, s=10, label=str(s))
    ax.axhline(0, color=GREY, lw=0.8, ls="--", alpha=0.5, label="wOBA residual = 0")
    ax.set_xlabel("Power Efficiency"); ax.set_ylabel("wOBA Residual")
    ax.set_title(f"{sel_player} - Contact Quality Profile by Season",
                 color=TEXT, fontsize=11)
    ax.legend(fontsize=8, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    sec("All 4 indicators - rolling 30-PA trend")
    fig2, ax2 = mpl_fig(13, 4)
    for col in PA_INDICATORS:
        roll = rolling_with_dates(player_df, col, 30)
        if len(roll) < 5:
            continue
        ax2.plot(roll["game_date"], roll[col],
                 color=PA_COLORS[col], lw=1.8, label=PA_LABELS[col], alpha=0.9)
    ax2.set_xlabel("Date"); ax2.set_ylabel("Indicator value (30-PA rolling mean)")
    ax2.set_title(f"{sel_player} - PA Indicator Trends", color=TEXT, fontsize=11)
    ax2.legend(fontsize=8, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.tick_params(axis="x", rotation=30)
    fig2.tight_layout(); st.pyplot(fig2); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 - PA INDICATOR CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif "Correlations" in page:
    st.markdown("# PA Indicator Correlations")
    st.markdown("""<div class="preamble">
    This page answers <b>which of the four PA-level indicators matter most</b> and how
    they relate to each other. A high positive correlation with <b>wOBA Residual</b>
    means that indicator is a strong predictor of batting outcomes above expectation,
    justifying its inclusion in the Performance Index.<br><br>
    Correlations are computed across all players and all seasons so the relationships
    reflect genuine structural signals rather than individual player quirks.
    </div>""", unsafe_allow_html=True)

    corr_df = df[PA_INDICATORS].dropna()

    sec("Pairwise Pearson correlations - all 4 PA indicators")
    corr_mat = corr_df.corr()
    corr_mat.index   = [PA_LABELS[c] for c in corr_mat.index]
    corr_mat.columns = [PA_LABELS[c] for c in corr_mat.columns]
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(PANEL)
    mask = np.zeros_like(corr_mat, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(corr_mat, ax=ax, mask=mask, cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 11, "color": DARK},
                linewidths=0.5, linecolor=DARK, cbar_kws={"shrink": 0.7})
    ax.set_title("Pairwise Correlations - PA Indicators", color=TEXT, fontsize=12, pad=12)
    ax.tick_params(colors=TEXT, labelsize=9)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); st.pyplot(fig); plt.close()

    sec("Each indicator vs wOBA Residual - league-wide")
    st.caption("wOBA Residual measures outcome quality above expectation. "
               "Stronger correlation here justifies higher weight in the Performance Index.")
    other_inds = [c for c in PA_INDICATORS if c != "woba_residual"]
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    fig2.patch.set_facecolor(DARK)
    for ax2, col in zip(axes2, other_inds):
        color = PA_COLORS[col]
        sub   = corr_df[[col,"woba_residual"]].dropna().sample(
            min(5000, len(corr_df)), random_state=42)
        ax2.set_facecolor(PANEL)
        ax2.tick_params(colors=TEXT, labelsize=8)
        for sp in ax2.spines.values(): sp.set_edgecolor(BORDER)
        ax2.scatter(sub[col], sub["woba_residual"],
                    color=color, alpha=0.15, s=6, edgecolor="none")
        m, b = np.polyfit(sub[col], sub["woba_residual"], 1)
        xs = np.linspace(sub[col].min(), sub[col].max(), 100)
        r  = corr_df[[col,"woba_residual"]].corr().iloc[0,1]
        ax2.plot(xs, m * xs + b, color=GOLD, lw=2, label=f"r={r:.2f}")
        ax2.set_xlabel(PA_LABELS[col], color=TEXT_MUTED, fontsize=9)
        ax2.set_ylabel("wOBA Residual", color=TEXT_MUTED, fontsize=9)
        ax2.set_title(f"{PA_LABELS[col]} vs wOBA Residual",
                      color=TEXT, fontsize=10, fontweight="bold")
        ax2.legend(fontsize=9, framealpha=0.2, facecolor=PANEL,
                   edgecolor=BORDER, labelcolor=TEXT)
        ax2.grid(color=BORDER, linewidth=0.3, linestyle="--", alpha=0.4)
    fig2.suptitle("PA Indicators vs wOBA Residual  (random sample of 5,000 PAs)",
                  color=TEXT, fontsize=11)
    st.pyplot(fig2); plt.close()

    sec("Correlation with wOBA Residual by season")
    st.caption("Stable correlations across seasons confirm consistent predictive power.")
    season_corrs = []
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr][PA_INDICATORS].dropna()
        if len(sub) < 100:
            continue
        row = {"Season": yr}
        for col in other_inds:
            row[PA_LABELS[col]] = sub[[col,"woba_residual"]].corr().iloc[0,1]
        season_corrs.append(row)
    sc_df = pd.DataFrame(season_corrs)
    fig3, ax3 = mpl_fig(10, 4)
    for col in other_inds:
        label = PA_LABELS[col]
        if label in sc_df.columns:
            ax3.plot(sc_df["Season"], sc_df[label],
                     color=PA_COLORS[col], lw=2, marker="o", markersize=5, label=label)
    ax3.axhline(0, color=GREY, lw=0.8, ls="--", alpha=0.5)
    ax3.set_xlabel("Season"); ax3.set_ylabel("Pearson r with wOBA Residual")
    ax3.set_title("Correlation Stability by Season", color=TEXT, fontsize=11)
    ax3.legend(fontsize=9, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    fig3.tight_layout(); st.pyplot(fig3); plt.close()

    sec("Summary - mean correlation with wOBA Residual")
    summary_rows = []
    for col in other_inds:
        r_overall   = corr_df[[col,"woba_residual"]].corr().iloc[0,1]
        label       = PA_LABELS[col]
        r_by_season = sc_df[label].tolist() if label in sc_df.columns else []
        summary_rows.append({
            "Indicator":              label,
            "Overall r":              round(r_overall, 3),
            "Min r (across seasons)": round(min(r_by_season), 3) if r_by_season else "N/A",
            "Max r (across seasons)": round(max(r_by_season), 3) if r_by_season else "N/A",
            "Index weight (equal)":   "25%",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.caption("All four indicators are equally weighted in the Performance Index. "
               "A future enhancement would weight by correlation magnitude.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 - PERFORMANCE INDEX
# ══════════════════════════════════════════════════════════════════════════════
elif "Performance Index" in page:
    st.markdown("# Performance Index")
    st.markdown("""<div class="preamble">
    A <b>composite score (0-100)</b> built from equal-weighted normalised PA indicators.
    Higher = better contact quality. Monitored in real time using
    <b>PELT change point detection</b> with Cohen's d effect size labelling.
    </div>""", unsafe_allow_html=True)

    roll_idx = rolling_with_dates(cpd_idx, "perf_index", cpd_window)
    if len(roll_idx) < 10:
        st.warning("Not enough data. Try a larger window or enable full history."); st.stop()

    cp_idx_pi       = detect_cpd(roll_idx["perf_index"], penalty)
    cp_st_pi        = cpd_stats(roll_idx["perf_index"], cp_idx_pi)
    b_l, b_lm, b_ls = baselines("perf_index", sel_player)

    narrative = generate_narrative(sel_player, "Performance Index",
                                   roll_idx, cp_st_pi, b_l, b_lm, cpd_window)
    st.markdown(f'<div class="narrative">&#128202; {narrative}</div>', unsafe_allow_html=True)

    if cp_st_pi:
        sec("Detected change points")
        badges = ""
        for s in cp_st_pi:
            idx_  = min(s["cp_idx"], len(roll_idx) - 1)
            date_ = roll_idx["game_date"].iloc[idx_]
            ds    = date_.strftime("%b %d, %Y") if hasattr(date_,"strftime") else str(date_)
            badges += (f'<span class="{s["badge"]}">'
                       f'{ds} · {s["direction"]} · {s["label"]} '
                       f'(d={s["effect_d"]:.2f})</span> ')
        st.markdown(badges, unsafe_allow_html=True)

    sec("Performance index - rolling average with PELT change points")
    fig, ax = mpl_fig(13, 5)
    ax.scatter(roll_idx["game_date"], roll_idx["perf_index"],
               color=GREY, alpha=0.15, s=6, zorder=1)
    ax.plot(roll_idx["game_date"], roll_idx["perf_index"],
            color=TEAL_LT, lw=2, zorder=3, label=f"Rolling {games_label(cpd_window)}")
    ymin, ymax = ax.get_ylim()
    add_baselines(ax, b_l, b_lm, b_ls)
    add_cpd_markers(ax, roll_idx, "perf_index", cp_idx_pi, cp_st_pi)
    if show_history: add_season_dividers(ax, ymax)
    ax.legend(fontsize=8, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    ax.set_xlabel("Date"); ax.set_ylabel("Performance Index (0-100)")
    ax.set_title(f"{sel_player} - Performance Index - PELT - {sensitivity} sensitivity",
                 color=TEXT, fontsize=12)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    if cp_st_pi:
        sec("Change point summary table")
        tbl = []
        for s in cp_st_pi:
            idx_  = min(s["cp_idx"], len(roll_idx) - 1)
            date_ = roll_idx["game_date"].iloc[idx_]
            tbl.append({
                "Date":          date_.strftime("%b %d, %Y") if hasattr(date_,"strftime") else str(date_),
                "Before avg":    round(s["before_mean"], 2),
                "After avg":     round(s["after_mean"],  2),
                "Delta":         round(s["delta"],       2),
                "Effect size d": round(s["effect_d"],    2),
                "Magnitude":     s["label"],
                "Direction":     s["direction"],
            })
        st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 - METRIC DRILLDOWN
# ══════════════════════════════════════════════════════════════════════════════
elif "Drilldown" in page:
    st.markdown("# Metric Drilldown")
    st.markdown("""<div class="preamble">
    Each PA indicator is monitored independently with PELT change point detection.
    Use this to identify <b>which specific component</b> of a player's contact quality
    has shifted.
    </div>""", unsafe_allow_html=True)

    fig = plt.figure(figsize=(13, 9)); fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.3)
    pal = [TEAL_LT, GOLD, RED_LT, "#a8d8a8"]

    for idx, col in enumerate(PA_INDICATORS):
        ri, ci = divmod(idx, 2)
        ax = fig.add_subplot(gs[ri, ci]); ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.grid(color=BORDER, lw=0.3, ls="--", alpha=0.4)

        roll_m = rolling_with_dates(cpd_df, col, cpd_window)
        if len(roll_m) < 10:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center", color=TEXT_MUTED, fontsize=9)
            ax.set_title(PA_LABELS[col], color=pal[idx], fontsize=10)
            continue

        cp_m    = detect_cpd(roll_m[col], penalty)
        cp_st_m = cpd_stats(roll_m[col], cp_m)
        b_l_m, b_lm_m, b_ls_m = baselines(col, sel_player)

        ax.plot(roll_m["game_date"], roll_m[col], color=pal[idx], lw=1.8, zorder=3)
        add_baselines(ax, b_l_m, b_lm_m, b_ls_m)
        add_cpd_markers(ax, roll_m, col, cp_m, cp_st_m)

        n_cp = len(cp_st_m)
        suffix = f"  [{n_cp} CP{'s' if n_cp != 1 else ''}]" if n_cp else ""
        ax.set_title(f"{PA_LABELS[col]}{suffix}", color=pal[idx], fontsize=10)
        ax.set_xlabel("Date", color=TEXT_MUTED, fontsize=8)
        ax.set_ylabel(PA_LABELS[col], color=TEXT_MUTED, fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle(f"{sel_player} - Per-Indicator PELT CPD - {sensitivity} sensitivity",
                 color=TEXT, fontsize=12, y=1.01)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    sec("Narratives per indicator")
    for col in PA_INDICATORS:
        roll_m = rolling_with_dates(cpd_df, col, cpd_window)
        if len(roll_m) < 10: continue
        cp_m    = detect_cpd(roll_m[col], penalty)
        cp_st_m = cpd_stats(roll_m[col], cp_m)
        b_l_m, b_lm_m, _ = baselines(col, sel_player)
        narr = generate_narrative(sel_player, PA_LABELS[col], roll_m,
                                  cp_st_m, b_l_m, b_lm_m, cpd_window)
        st.markdown(f'<div class="narrative"><b>{PA_LABELS[col]}:</b> {narr}</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 - SEASON TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
elif "Timeline" in page:
    st.markdown(f"# {max_year} Season Timeline")

    tl_options = {**{PA_LABELS[c]: c for c in PA_INDICATORS}, "Performance Index": "perf_index"}
    tl_label   = st.selectbox("Metric", options=list(tl_options.keys()), index=0)
    tl_col     = tl_options[tl_label]

    tl_src  = cpd_idx if tl_col == "perf_index" else cpd_df
    roll_tl = rolling_with_dates(tl_src, tl_col, cpd_window)

    if len(roll_tl) < 10:
        st.warning("Not enough data for this metric and window."); st.stop()

    cp_tl    = detect_cpd(roll_tl[tl_col], penalty)
    cp_st_tl = cpd_stats(roll_tl[tl_col], cp_tl)
    b_l_tl, b_lm_tl, b_ls_tl = baselines(tl_col, sel_player)

    narr_tl = generate_narrative(sel_player, tl_label, roll_tl,
                                  cp_st_tl, b_l_tl, b_lm_tl, cpd_window)
    st.markdown(f'<div class="narrative">&#128203; {narr_tl}</div>', unsafe_allow_html=True)

    if cp_st_tl:
        sec("Change points detected")
        badges = ""
        for s in cp_st_tl:
            idx_  = min(s["cp_idx"], len(roll_tl) - 1)
            date_ = roll_tl["game_date"].iloc[idx_]
            ds    = date_.strftime("%b %d, %Y") if hasattr(date_,"strftime") else str(date_)
            badges += f'<span class="{s["badge"]}">{ds} - {s["direction"]} - {s["label"]}</span> '
        st.markdown(badges, unsafe_allow_html=True)

    sec(f"{tl_label} timeline")
    fig, ax = mpl_fig(13, 5)
    raw_src = cpd_idx if tl_col == "perf_index" else cpd_df
    raw_tl  = raw_src[[tl_col,"game_date"]].dropna().sort_values("game_date")
    ax.scatter(raw_tl["game_date"], raw_tl[tl_col],
               color=GREY, alpha=0.12, s=6, zorder=1)
    ax.plot(roll_tl["game_date"], roll_tl[tl_col],
            color=TEAL_LT, lw=2.2, zorder=3,
            label=f"{tl_label} - rolling {games_label(cpd_window)}")
    ymin_tl, ymax_tl = ax.get_ylim()
    add_baselines(ax, b_l_tl, b_lm_tl, b_ls_tl)
    add_cpd_markers(ax, roll_tl, tl_col, cp_tl, cp_st_tl)
    if show_history: add_season_dividers(ax, ymax_tl)
    ax.set_xlabel("Date"); ax.set_ylabel(tl_label)
    label_suffix = "full history" if show_history else str(max_year)
    ax.set_title(f"{sel_player} - {tl_label} - {label_suffix} - {sensitivity} sensitivity",
                 color=TEXT, fontsize=12)
    ax.legend(fontsize=8, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    sec(f"Monthly averages - {max_year}")
    curr_m = player_df[player_df["year"] == max_year].copy()
    month_names = {3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct"}
    monthly = curr_m.groupby("month").agg(
        hitting_decisions =("hitting_decisions_score",     "mean"),
        power_efficiency  =("power_efficiency",            "mean"),
        woba_residual     =("woba_residual",               "mean"),
        launch_angle_stab =("launch_angle_stability_50pa", "mean"),
        pa_count          =("pa_uid",                      "count"),
    ).round(3).reset_index()
    monthly["month"] = monthly["month"].map(month_names)
    st.dataframe(monthly, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 - PEER COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif "Peer" in page:
    st.markdown("# Peer Comparison")
    st.markdown("""<div class="preamble">
    Compare the selected player's rolling PA indicator against up to 3 peers.
    Dashed vertical lines mark each player's most recent significant or moderate
    change point. Use this to determine whether a decline is player-specific or
    a broader trend.
    </div>""", unsafe_allow_html=True)

    peers = st.multiselect(
        "Comparison players (up to 3)",
        [p for p in players if p != sel_player],
        default=[p for p in players if p != sel_player][:2],
        max_selections=3,
    )
    all_sel = [sel_player] + peers

    peer_options = {PA_LABELS[c]: c for c in PA_INDICATORS}
    peer_label   = st.selectbox("Metric", options=list(peer_options.keys()), index=1)
    peer_col     = peer_options[peer_label]
    pal_peer     = [GOLD, TEAL_LT, RED_LT, "#a8d8a8"]

    sec(f"{peer_label} - rolling {games_label(cpd_window)} comparison")
    fig, ax = mpl_fig(13, 5)
    peer_rows = []

    for i, pname in enumerate(all_sel):
        pdf_p  = df[df["player_name"] == pname]
        pdf_p  = pdf_p if show_history else pdf_p[pdf_p["year"] == max_year]
        roll_p = rolling_with_dates(pdf_p, peer_col, cpd_window)
        if len(roll_p) < 5: continue

        cp_p    = detect_cpd(roll_p[peer_col], penalty)
        cp_st_p = cpd_stats(roll_p[peer_col], cp_p)

        ax.plot(roll_p["game_date"], roll_p[peer_col], color=pal_peer[i],
                lw=2.5 if pname == sel_player else 1.5,
                alpha=1.0 if pname == sel_player else 0.65,
                label=pname, zorder=3 - i)

        sig_cps = [s for s in cp_st_p if s["label"] in ("moderate","significant")]
        if sig_cps:
            idx_sig  = min(sig_cps[-1]["cp_idx"], len(roll_p) - 1)
            date_sig = roll_p["game_date"].iloc[idx_sig]
            ax.axvline(date_sig, color=pal_peer[i], lw=1, ls="--", alpha=0.55)

        curr_val = roll_p[peer_col].iloc[-1]
        b_l_p, b_lm_p, _ = baselines(peer_col, pname)
        peer_rows.append({
            "Player":              pname,
            "Current rolling avg": round(curr_val, 3),
            "League avg":          round(b_lm_p, 3) if not np.isnan(b_lm_p) else "N/A",
            "Prev season avg":     round(b_l_p,  3) if not np.isnan(b_l_p)  else "N/A",
            "Significant CPDs":    len(sig_cps),
        })

    leag_p = df[df["year"] == max_year][peer_col].dropna()
    if len(leag_p):
        ax.axhline(leag_p.mean(), color=GREY, lw=1.0, alpha=0.6,
                   label=f"League avg {max_year} ({leag_p.mean():.2f})")
        ax.axhspan(leag_p.mean() - leag_p.std(), leag_p.mean() + leag_p.std(),
                   alpha=0.05, color=GREY)

    ax.set_xlabel("Date")
    ax.set_ylabel(f"{peer_label} ({games_label(cpd_window)} rolling)")
    ax.set_title(f"{peer_label} comparison - dashed = most recent significant PELT CPD",
                 color=TEXT, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    if peer_rows:
        sec("Peer summary table")
        st.dataframe(pd.DataFrame(peer_rows), use_container_width=True, hide_index=True)

    sec(f"{peer_label} distribution - {max_year}")
    from scipy.stats import gaussian_kde
    fig2, ax2 = mpl_fig(11, 4)
    for i, pname in enumerate(all_sel):
        data = df[(df["player_name"] == pname) & (df["year"] == max_year)][peer_col].dropna()
        if len(data) < 10: continue
        xs = np.linspace(data.min() - 0.5, data.max() + 0.5, 300)
        ys = gaussian_kde(data, bw_method=0.4)(xs)
        ax2.fill_between(xs, ys, alpha=0.2, color=pal_peer[i])
        ax2.plot(xs, ys, color=pal_peer[i], lw=2,
                 label=f"{pname} (mean={data.mean():.2f})")
        ax2.axvline(data.mean(), color=pal_peer[i], lw=1, ls=":", alpha=0.8)
    ax2.set_xlabel(peer_label); ax2.set_ylabel("Density")
    ax2.set_title(f"{peer_label} distribution - {max_year}", color=TEXT, fontsize=11)
    ax2.legend(fontsize=9, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    fig2.tight_layout(); st.pyplot(fig2); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 - CHANGEFOREST CPD
# ══════════════════════════════════════════════════════════════════════════════
elif "ChangeForest" in page:
    st.title("ChangeForest Change-Point Detection")
    st.markdown("""<div class="preamble">
    <b>Multivariate CPD</b> across all 4 PA indicators simultaneously using the
    ChangeForest random-forest method. Unlike PELT which monitors one signal at a time,
    ChangeForest detects change points jointly significant across the full indicator
    vector - catching subtle multi-dimensional shifts that univariate methods miss.
    </div>""", unsafe_allow_html=True)

    if changeforest is None or Control is None:
        st.error("`changeforest` package not installed. Run `pip install changeforest`.")
        st.stop()

    with st.spinner("Preparing PA dataset..."):
        subdf = cf_subdataset(sel_player, cf_window)

    if subdf.empty:
        st.warning(
            f"No rows available for **{sel_player}** after filtering. "
            "Try a smaller rolling window.")
        st.stop()

    with st.spinner("Running ChangeForest random-forest CPD..."):
        try:
            _, cps, _, feature_names = run_changeforest(
                subdf, window=cf_window,
                minimal_relative_segment_length=cf_min_seg,
                use_rollmean=True, standardize=True,
            )
        except Exception as exc:
            st.error(f"ChangeForest failed: {exc}")
            st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Player",       sel_player)
    c2.metric("PA Rows Used", f"{len(subdf):,}")
    c3.metric("CF Window",    str(cf_window))
    c4.metric("Sensitivity",  cf_sens)
    c5.metric("Detected CPs", str(len(cps)))

    if cps:
        with st.expander(f"Change-point rows ({len(cps)} points)"):
            st.dataframe(
                subdf.iloc[cps][["cf_seq_id","game_date","pa_uid"]].reset_index(drop=True),
                use_container_width=True,
            )

    tab1, tab2, tab3 = st.tabs([
        "ChangeForest Result + Input Signals",
        "Before / After Eval",
        "Parameter Stability",
    ])

    with tab1:
        st.subheader("ChangeForest Result")
        st.caption("All 4 standardized rolling-mean signals - dashed red = detected CPs")
        fig_result = plot_cf_result(subdf, cps, cf_window, sel_player, cf_min_seg)
        st.pyplot(fig_result, use_container_width=True)
        st.markdown("---")
        st.subheader("Input Signals (Raw + Rolling Mean)")
        st.caption("2x2 panel - raw PA values and rolling mean (same window as CPD)")
        fig_signals = plot_cf_input_signals(subdf, cf_window, sel_player)
        st.pyplot(fig_signals, use_container_width=True)

    with tab2:
        st.subheader("Before / After Statistical Comparison")
        compare_window = max(5, min(50, len(subdf) // 4))
        if len(subdf) <= 2 * compare_window:
            st.warning(f"Not enough data (need > {2*compare_window} rows, have {len(subdf)}).")
        else:
            with st.spinner("Computing before/after statistics..."):
                eval_dfs = build_cp_eval_dfs(subdf, cps, feature_names,
                                              compare_window=compare_window)
            if not eval_dfs:
                st.warning("Could not build eval dataframes.")
            else:
                fig_eval = plot_cp_eval_comparison(eval_dfs)
                if fig_eval: st.pyplot(fig_eval, use_container_width=True)
                with st.expander("Detailed eval summary tables"):
                    for feat, df_eval in eval_dfs.items():
                        base = feat.split("_rollmean_")[0]
                        st.write(f"**{PA_LABELS.get(base, feat)}**")
                        st.dataframe(
                            df_eval.groupby("cp")[
                                ["mean_before","mean_after","mean_diff",
                                 "std_before","std_after","std_diff",
                                 "abs_mean_diff","abs_std_diff"]
                            ].mean().round(4),
                            use_container_width=True,
                        )

    with tab3:
        st.subheader("Parameter Stability")
        st.caption("Runs ChangeForest across all three sensitivity levels. "
                   "Similar CP counts = stable, reliable detection.")
        stability_vals = sorted(SENSITIVITY_TO_MIN_SEG.values())
        with st.spinner("Running stability analysis..."):
            stability_df = run_parameter_stability(subdf, cf_window, stability_vals)
        fig_stab = plot_parameter_stability(stability_df)
        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            st.pyplot(fig_stab, use_container_width=True)
        with col_table:
            st.dataframe(stability_df, use_container_width=True, hide_index=True)