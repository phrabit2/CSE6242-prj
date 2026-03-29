"""
MLB Real-Time Batting Performance Dashboard
============================================
Pitch-by-pitch Statcast data · 2021-2025 · 420 qualified hitters
"""

import os
import warnings
warnings.filterwarnings("ignore")
import gdown
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MLB Batting Pulse",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

CURRENT_SEASON = 2025

# ── Palette ────────────────────────────────────────────────────────────────────
DARK       = "#0d1117"
PANEL      = "#161b22"
BORDER     = "#30363d"
GOLD       = "#d4a017"
GOLD_LT    = "#f0c040"
TEAL       = "#238b8b"
TEAL_LT    = "#3eb8b8"
RED        = "#c0392b"
RED_LT     = "#e74c3c"
GREY       = "#6e7681"
TEXT       = "#e6edf3"
TEXT_MUTED = "#8b949e"

# ── Feature whitelist (no outcome leakage) ─────────────────────────────────────
FEATURE_WHITELIST = [
    "exit_velocity", "launch_angle_metric",
    "release_speed", "effective_speed",
    "release_spin_rate", "release_extension", "spin_axis",
    "pfx_x", "pfx_z",
    "plate_x", "plate_z",
    "sz_top", "sz_bot",
]

# ── Events ─────────────────────────────────────────────────────────────────────
ALL_EVENTS = [
    "home_run", "single", "double", "triple",
    "field_out", "force_out", "grounded_into_double_play",
    "double_play", "fielders_choice_out", "fielders_choice",
    "field_error", "sac_fly", "sac_bunt",
    "sac_fly_double_play", "sac_bunt_double_play",
    "catcher_interf", "triple_play",
]
DEFAULT_EVENTS  = ["home_run", "single", "double", "triple", "field_out"]
SENSITIVITY_MAP = {"Low": 8, "Medium": 3, "High": 1}

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
    text-transform:uppercase; letter-spacing:0.1em;
    font-family:'IBM Plex Mono',monospace;
}}
.card-row {{ display:flex; gap:10px; margin-bottom:1rem; flex-wrap:wrap; }}
.card {{
    background:{PANEL}; border:1px solid {BORDER};
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
.cpd-minor {{ display:inline-block; background:#30363d; color:{TEXT_MUTED}; border-radius:3px; padding:2px 8px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; margin:2px; }}
.cpd-mod   {{ display:inline-block; background:#3a2f00; color:{GOLD_LT}; border:1px solid {GOLD}; border-radius:3px; padding:2px 8px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; margin:2px; }}
.cpd-sig   {{ display:inline-block; background:#3b0d0d; color:{RED_LT}; border:1px solid {RED}; border-radius:3px; padding:2px 8px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; margin:2px; }}
.preamble {{
    background:{PANEL}; border:1px solid {BORDER}; border-left:4px solid {GOLD};
    border-radius:4px; padding:1.2rem 1.4rem; margin-bottom:1.4rem;
    font-size:0.92rem; line-height:1.7; color:{TEXT_MUTED};
}}
.preamble b {{ color:{TEXT}; }}
.narrative {{
    background:#161b22; border:1px solid {BORDER}; border-radius:4px;
    padding:1rem 1.2rem; font-family:'IBM Plex Mono',monospace;
    font-size:0.82rem; line-height:1.6; color:{TEXT}; margin-bottom:1rem;
}}
.rel-high {{ color:{TEAL_LT}; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }}
.rel-med  {{ color:{GOLD_LT}; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }}
.rel-low  {{ color:{RED_LT};  font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def card(label, val, sub="", color="gold"):
    cls = {"gold":"card","teal":"card teal","red":"card red","grey":"card grey"}[color]
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
    if ad < 0.2:  return "minor",    "cpd-minor"
    if ad < 0.5:  return "moderate", "cpd-mod"
    return "significant", "cpd-sig"

def reliability_badge(n):
    if n >= 200: return f'<span class="rel-high">● High reliability</span>'
    if n >= 100: return f'<span class="rel-med">● Medium reliability</span>'
    return f'<span class="rel-low">● Low reliability (few pitches this season)</span>'

def games_label(n_pitches):
    return f"≈{max(1, round(n_pitches / 2.6))} games"


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
# @st.cache_data(show_spinner="Loading pitch data…")
# def load_data():
#     base = os.path.dirname(os.path.abspath(__file__))
#     path = os.path.join(base, "..", "data", "processed",
#                         # "qualified_hitters_statcast_2021_2025_batted_ball.csv")
#                         "filtered_mlb_data.csv")
#     df = pd.read_csv(path, low_memory=False)
#     df["game_date"] = pd.to_datetime(df["game_date"])
#     df = df.sort_values(["Name", "game_date"]).reset_index(drop=True)

#     # Canonical column aliases
#     if "exit_velocity" not in df.columns and "launch_speed" in df.columns:
#         df["exit_velocity"] = df["launch_speed"]
#     if "launch_angle_metric" not in df.columns and "launch_angle" in df.columns:
#         df["launch_angle_metric"] = df["launch_angle"]
#     if "xwoba_est" not in df.columns and "estimated_woba_using_speedangle" in df.columns:
#         df["xwoba_est"] = df["estimated_woba_using_speedangle"]

#     # Fill xwoba nulls with player-season mean (not zero — zero distorts CPD)
#     df["xwoba_est"] = df.groupby(["Name", "Season"])["xwoba_est"].transform(
#         lambda x: x.fillna(x.mean())
#     )

#     # Flags
#     if "is_batted_ball" not in df.columns:
#         df["is_batted_ball"] = df["exit_velocity"].notna().astype(int)
#     if "is_in_play" not in df.columns:
#         df["is_in_play"] = df["events"].notna().astype(int)
#     # df=df[[
#     #     "Name", "game_date", "Season", "events", 
#     #     "launch_speed", "launch_angle", "estimated_woba_using_speedangle",
#     #     "release_speed", "effective_speed", "release_spin_rate", 
#     #     "release_extension", "pfx_x", "pfx_z", "plate_x", "plate_z"
#     # # ]] 
#     # df=df[[
#     # "Name",                       # player identifier
#     # "game_date",                  # for rolling plots / timelines
#     # "Season",                     # for filtering per season
#     # "events",                     # outcome labels for ML
#     # "launch_speed",               # exit_velocity
#     # "launch_angle",               # launch_angle_metric
#     # "estimated_woba_using_speedangle",  # xwOBA est
#     # 'exit_velocity','launch_angle_metric',
#     # "plate_x",'xwoba_est','is_hard_hit',
#     # "plate_z",'is_barrel_proxy',
#     # "sz_top",
#     # "sz_bot"]]
#     return df

def load_data():
    
   
    # file_id = "1-ueHmB1xBscgx-Zqkp4vF-nRHkXcYcXx"
   
    # https://drive.google.com/file/d/1CmKAQetOiPgCZ9eL3Xg40a_BzCYGRiJN/view?usp=share_link
    # https://drive.google.com/file/d/142uq2WYlHGGmyBSzkti3fP2ipk_MtoA3/view?usp=share_link
    file_id = "142uq2WYlHGGmyBSzkti3fP2ipk_MtoA3"
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'large_mlb_data.csv'
    
    # 1. Download via gdown to bypass Google's "Virus Scan" interstitial
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False, fuzzy=True)
    
    # 2. Define ONLY the columns we need to save massive amounts of RAM
    # If your CSV has 50 columns but we only use 15, we save ~70% memory here.
    # keep_cols = [
    #     "Name", "game_date", "Season", "events", 
    #     "launch_speed", "launch_angle", "estimated_woba_using_speedangle",
    #     "release_speed", "effective_speed", "release_spin_rate", 
    #     "release_extension", "pfx_x", "pfx_z", "plate_x", "plate_z"
    # ]

    # 3. Load with explicit dtypes to prevent memory bloat
    # float32 uses half the memory of the default float64
    dtype_dict = {
        "launch_speed": "float32",
        "launch_angle": "float32",
        "release_speed": "float32",
        "Season": "int16"
    }

    try:
        df = pd.read_csv(
            output, 
            # usecols=lambda c: c in keep_cols, # Dynamic check for columns
            # dtype=dtype_dict,
            # low_memory=False
        )
        st.write(df.columns.tolist())
        # st.write("Columns actually loaded:", df.columns.tolist())
        # st.dataframe(df.head())
    
    except Exception as e:
        st.error(f"Pandas failed to load the 1.3GB file: {df.columns}")
        # st.stop()

    # 4. Canonical column aliases (Standardizing column names)
    if "exit_velocity" not in df.columns and "launch_speed" in df.columns:
        df["exit_velocity"] = df["launch_speed"]
    if "launch_angle_metric" not in df.columns and "launch_angle" in df.columns:
        df["launch_angle_metric"] = df["launch_angle"]
    if "xwoba_est" not in df.columns and "estimated_woba_using_speedangle" in df.columns:
        df["xwoba_est"] = df["estimated_woba_using_speedangle"]

    # 5. Essential cleaning
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["Name", "game_date"]).reset_index(drop=True)

    # Fill xwoba nulls with player-season mean
    df["xwoba_est"] = df.groupby(["Name", "Season"])["xwoba_est"].transform(
        lambda x: x.fillna(x.mean())
    )

    # Calculate derived flags
    df["is_hard_hit"] = (df["exit_velocity"] >= 95).astype(np.int8)
    df["is_barrel_proxy"] = ((df["exit_velocity"] >= 98) & 
                             (df["launch_angle_metric"].between(26, 30))).astype(np.int8)
    

    return df

# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Training Random Forest…")
def train_rf(df, selected_events, selected_features):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score

    mdf  = df[df["events"].isin(selected_events)].copy()
    feat = [c for c in selected_features if c in mdf.columns]
    mdf  = mdf[["events"] + feat].dropna()
    if mdf["events"].nunique() < 2 or len(mdf) < 50:
        return None, feat, None, None

    le = LabelEncoder()
    y  = le.fit_transform(mdf["events"])
    X  = mdf[feat]
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, n_jobs=-1, random_state=42)
    clf.fit(X, y)
    cv = cross_val_score(clf, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    return clf, feat, le, cv


@st.cache_data(show_spinner="Training Logistic Regression…")
def train_lr(df, selected_events, selected_features):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    mdf  = df[df["events"].isin(selected_events)].copy()
    feat = [c for c in selected_features if c in mdf.columns]
    mdf  = mdf[["events"] + feat].dropna()
    if mdf["events"].nunique() < 2 or len(mdf) < 50:
        return None, feat, None, None

    le  = LabelEncoder()
    y   = le.fit_transform(mdf["events"])
    X   = mdf[feat]
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)
    lr  = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    lr.fit(Xs, y)
    return lr, feat, le, sc


# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE INDEX
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Computing performance index…")
def build_perf_index(df, weights_tuple):
    """Weighted composite of RF-important features, normalised 0-100.
    Computed on all batted-ball contacts including fouls."""
    weights = dict(weights_tuple)
    feats   = [f for f in weights if f in df.columns]
    if not feats:
        df = df.copy(); df["perf_index"] = np.nan; return df

    sub = df[feats].copy()
    for f in feats:
        lo, hi = sub[f].quantile(0.01), sub[f].quantile(0.99)
        sub[f] = sub[f].clip(lo, hi)

    normed = sub.apply(lambda c: (c - c.min()) / (c.max() - c.min() + 1e-9))
    w = pd.Series(weights)[feats]; w = w / w.sum()
    df = df.copy()
    df["perf_index"] = (normed * w).sum(axis=1) * 100
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CPD UTILITIES
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
            "cp_idx":      bpts[i+1],
            "before_mean": before.mean(),
            "after_mean":  after.mean(),
            "delta":       delta,
            "effect_d":    d,
            "label":       label,
            "badge":       badge,
            "direction":   "▲ improvement" if delta > 0 else "▼ decline",
        })
    return results


def rolling_with_dates(pdf, metric, window):
    """Rolling mean on all rows where metric is not null (includes fouls)."""
    sub = pdf[pdf[metric].notna()][["game_date", metric]].sort_values("game_date")
    if len(sub) < window:
        return pd.DataFrame(columns=["game_date", metric])
    roll = sub[metric].rolling(window, min_periods=window // 2).mean()
    return pd.DataFrame({"game_date": sub["game_date"].values,
                          metric: roll.values}).dropna()


def baselines(df, metric, player):
    last = df[(df["Name"] == player) & (df["Season"] == CURRENT_SEASON - 1)][metric].dropna()
    leag = df[df["Season"] == CURRENT_SEASON][metric].dropna()
    return (
        last.mean() if len(last) else np.nan,
        leag.mean() if len(leag) else np.nan,
        leag.std()  if len(leag) else np.nan,
    )


def add_baselines(ax, b_last, b_leag_m, b_leag_s):
    if not np.isnan(b_last):
        ax.axhline(b_last, color=GOLD, lw=1.2, ls="--", alpha=0.8,
                   label=f"Prev season avg ({b_last:.1f})", zorder=2)
    if not np.isnan(b_leag_m):
        ax.axhline(b_leag_m, color=GREY, lw=1.0, ls="-", alpha=0.7,
                   label=f"League avg {CURRENT_SEASON} ({b_leag_m:.1f})", zorder=2)
        if not np.isnan(b_leag_s):
            ax.axhspan(b_leag_m - b_leag_s, b_leag_m + b_leag_s,
                       alpha=0.06, color=GREY, zorder=1)


def add_cpd_markers(ax, roll_df, metric, cp_indices, stats):
    ymin, ymax = ax.get_ylim()
    for s in stats:
        idx   = min(s["cp_idx"], len(roll_df) - 1)
        date  = roll_df["game_date"].iloc[idx]
        color = {
            "cpd-minor": GREY,
            "cpd-mod":   GOLD,
            "cpd-sig":   RED_LT,
        }[s["badge"]]
        ax.axvline(date, color=color, lw=1.4, ls="--", alpha=0.85, zorder=4)
        arrow = "▲" if s["delta"] > 0 else "▼"
        ax.text(date, ymin + (ymax - ymin) * 0.93,
                f" {arrow}{abs(s['delta']):.1f}",
                color=color, fontsize=7.5, va="top", fontfamily="monospace")


def add_season_dividers(ax, ymax):
    for yr in range(2022, CURRENT_SEASON + 1):
        d = pd.Timestamp(f"{yr}-03-01")
        ax.axvline(d, color=GREY, lw=0.6, ls=":", alpha=0.4)
        ax.text(d, ymax * 0.98, str(yr),
                color=GREY, fontsize=7, fontfamily="monospace", va="top")


def generate_narrative(player, metric_label, roll_df, cp_st, b_last, b_leag_m, window):
    metric_col = roll_df.columns[1]
    current    = roll_df[metric_col].iloc[-1] if len(roll_df) else np.nan

    if not cp_st:
        txt = (f"No significant change points detected for {player} on "
               f"{metric_label} over the last {games_label(window)}. "
               f"Current rolling value: {current:.1f}.")
        if not np.isnan(b_leag_m):
            diff = current - b_leag_m
            txt += f" This is {abs(diff):.1f} {'above' if diff >= 0 else 'below'} the {CURRENT_SEASON} league average."
        return txt

    s        = cp_st[-1]
    idx      = min(s["cp_idx"], len(roll_df) - 1)
    date     = roll_df["game_date"].iloc[idx]
    date_str = date.strftime("%B %d, %Y") if hasattr(date, "strftime") else str(date)
    direction = "declined" if s["delta"] < 0 else "improved"

    txt = (f"{player}'s {metric_label} {direction} {s['label']}ly around {date_str} "
           f"(effect size d={abs(s['effect_d']):.2f}). "
           f"Rolling average shifted from {s['before_mean']:.1f} to {s['after_mean']:.1f} "
           f"({'+' if s['delta'] > 0 else ''}{s['delta']:.1f}).")
    if len(cp_st) > 1:
        txt += f" {len(cp_st)} total change points detected."
    if not np.isnan(b_leag_m):
        diff = s["after_mean"] - b_leag_m
        txt += f" Current level is {abs(diff):.1f} {'above' if diff >= 0 else 'below'} the {CURRENT_SEASON} league average."
    if not np.isnan(b_last):
        diff2 = s["after_mean"] - b_last
        txt += f" Previous season average was {b_last:.1f} — currently {abs(diff2):.1f} {'above' if diff2 >= 0 else 'below'}."
    return txt


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df  = load_data()

players = sorted(df["Name"].dropna().unique())
avail_events = sorted([e for e in ALL_EVENTS if e in df["events"].dropna().unique()])
avail_feats  = [f for f in FEATURE_WHITELIST if f in df.columns]
event_counts = df["events"].value_counts()

CPD_METRICS = {k: v for k, v in {
    "Performance Index": "perf_index",
    "Exit Velocity":     "exit_velocity",
    "Launch Angle":      "launch_angle_metric",
    "xwOBA (est)":       "xwoba_est",
}.items()}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚾ MLB Batting Pulse")
    st.caption("Real-time performance change detection")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠  Welcome",
        "👤  Player Snapshot",
        "🔬  What Drives Outcomes",
        "📈  Performance Index",
        "🔍  Metric Drilldown",
        "📅  Season Timeline",
        "👥  Peer Comparison",
    ], label_visibility="collapsed")
    st.markdown("---")

    sel_player = st.selectbox("Player", players)
    cpd_window = st.slider("Rolling window (pitches)", 20, 80, 40, 5)
    st.caption(f"{games_label(cpd_window)} of plate appearances")

    sensitivity  = st.radio("CPD Sensitivity", ["Low", "Medium", "High"], index=1, horizontal=True)
    penalty      = SENSITIVITY_MAP[sensitivity]
    show_history = st.toggle("Show full history (2021–present)", value=False)

    st.markdown("---")
    n_curr = len(df[(df["Name"] == sel_player) & (df["Season"] == CURRENT_SEASON)])
    n_all  = len(df[df["Name"] == sel_player])
    st.markdown(reliability_badge(n_curr), unsafe_allow_html=True)
    st.caption(f"{n_all:,} total pitches · {n_curr:,} in {CURRENT_SEASON}")


# ══════════════════════════════════════════════════════════════════════════════
# PLAYER SLICES
# ══════════════════════════════════════════════════════════════════════════════
player_df = df[df["Name"] == sel_player].copy()
cpd_df    = player_df if show_history else player_df[player_df["Season"] == CURRENT_SEASON].copy()
# ══════════════════════════════════════════════════════════════════════════════
# SHARED: build perf index (used by pages 4, 5, 6)
# ══════════════════════════════════════════════════════════════════════════════
def get_weights():
    if "rf_weights" in st.session_state:
        return st.session_state["rf_weights"]
    return {f: 1.0 / len(avail_feats) for f in avail_feats}



def get_df_with_index():
    return build_perf_index(df,tuple(sorted(get_weights().items())))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — WELCOME
# ══════════════════════════════════════════════════════════════════════════════
if "Welcome" in page:
    st.markdown("# MLB Batting Pulse")
    st.markdown("### Real-time batting performance change detection")

    st.markdown(f"""
    <div class="preamble">
    <b>The problem with traditional baseball metrics:</b> Statistics like xwOBA, BABIP, and
    batting average are calculated as season-long rolling averages. By the time a meaningful
    decline shows up in the box score, a player may have been struggling for weeks —
    costing teams games and fans their confidence in a favourite star.<br><br>
    <b>What this dashboard does:</b> Using pitch-by-pitch Statcast data from 2021 to {CURRENT_SEASON},
    we apply machine learning to identify <b>which contact quality metrics</b> (exit velocity,
    launch angle, etc.) most strongly predict batting outcomes — then monitor those metrics
    in real time using <b>change point detection (PELT)</b> to flag when a player's performance
    has shifted, how large that shift is, and whether it's a genuine decline or noise.<br><br>
    <b>Who it's for:</b><br>
    ⚾ <b>Fans</b> — understand when your favourite player is actually in a slump, not just unlucky.<br>
    📋 <b>Coaches</b> — pinpoint exactly which contact component changed to target coaching advice.<br>
    📊 <b>Managers / GMs</b> — compare players against league baselines and peers with confidence.<br><br>
    <b>How to use it:</b> Select a player and rolling window in the sidebar, then work through the
    pages in order — or jump straight to <i>Performance Index</i> for the headline signal.
    Visit <i>What Drives Outcomes</i> first to train the model; its weights flow into the index.
    </div>
    """, unsafe_allow_html=True)

    sec("Dataset at a glance")
    st.markdown(f"""<div class="card-row">
        {card("Players",        f"{df['Name'].nunique():,}",  "qualified hitters",       "gold")}
        {card("Total pitches",  f"{len(df):,}",               "batted ball contacts",    "teal")}
        {card("Seasons",        str(df['Season'].nunique()),   f"2021–{CURRENT_SEASON}",  "grey")}
        {card(f"{CURRENT_SEASON} pitches", f"{len(df[df['Season']==CURRENT_SEASON]):,}", "current season", "teal")}
    </div>""", unsafe_allow_html=True)

    sec("Event distribution — all players · all seasons")
    ev_all  = df["events"].value_counts().head(12)
    fig, ax = mpl_fig(11, 3.8)
    bcolors = [GOLD if e in ["home_run","single","double","triple"] else GREY
               for e in ev_all.index]
    ax.barh(ev_all.index[::-1], ev_all.values[::-1],
            color=bcolors[::-1], edgecolor="none", height=0.65)
    ax.set_xlabel("Count")
    ax.set_title(f"Batted Ball Events (all players · 2021–{CURRENT_SEASON})",
                 color=TEXT, fontsize=11)
    for i, (v, n) in enumerate(zip(ev_all.values[::-1], ev_all.index[::-1])):
        ax.text(v + ev_all.max() * 0.005, i, f"{v:,}",
                va="center", color=TEXT_MUTED, fontsize=8, fontfamily="monospace")
    fig.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PLAYER SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════
elif "Snapshot" in page:
    st.markdown(f"# {sel_player}")

    curr = player_df[player_df["Season"] == CURRENT_SEASON]
    leag = df[df["Season"] == CURRENT_SEASON]

    def _mean(series): return series.dropna().mean()
    def _pct(series):  return series.dropna().mean() * 100

    sec(f"{CURRENT_SEASON} performance vs league")
    st.markdown(f"""<div class="card-row">
        {card("Exit Velocity",  f"{_mean(curr['exit_velocity']):.1f}",       f"League {_mean(leag['exit_velocity']):.1f} mph",  "gold")}
        {card("Launch Angle",   f"{_mean(curr['launch_angle_metric']):.1f}°", f"League {_mean(leag['launch_angle_metric']):.1f}°","teal")}
        {card("xwOBA (est)",    f"{_mean(curr['xwoba_est']):.3f}",            f"League {_mean(leag['xwoba_est']):.3f}",           "gold")}
        {card("Hard Hit %",     f"{_pct(curr['is_hard_hit']):.1f}%",          f"League {_pct(leag['is_hard_hit']):.1f}%",         "teal")}
        {card("Barrel %",       f"{_pct(curr['is_barrel_proxy']):.1f}%",      f"League {_pct(leag['is_barrel_proxy']):.1f}%",     "grey")}
    </div>""", unsafe_allow_html=True)

    sec("Season-by-season averages")
    summ = player_df.groupby("Season").agg(
        exit_velocity   =("exit_velocity",      "mean"),
        launch_angle    =("launch_angle_metric", "mean"),
        xwoba           =("xwoba_est",           "mean"),
        hard_hit_pct    =("is_hard_hit",         "mean"),
        barrel_pct      =("is_barrel_proxy",     "mean"),
        n_contacts      =("exit_velocity",       "count"),
    ).round(3).reset_index()
    st.dataframe(summ, use_container_width=True, hide_index=True)

    sec("Exit velocity vs launch angle by season")
    plot_df = player_df[["exit_velocity","launch_angle_metric","Season"]].dropna()
    seasons = sorted(plot_df["Season"].unique())
    pal     = [GOLD, TEAL_LT, RED_LT, "#a8d8a8", GREY]
    fig, ax = mpl_fig(11, 4.5)
    for i, s in enumerate(seasons):
        sub = plot_df[plot_df["Season"] == s]
        ax.scatter(sub["exit_velocity"], sub["launch_angle_metric"],
                   color=pal[i % len(pal)], alpha=0.3, s=10, label=str(s))
    ax.axhspan(8, 32, alpha=0.07, color=GOLD, zorder=0, label="Optimal LA (8°–32°)")
    ax.axvline(95, color=GREY, lw=0.8, ls="--", alpha=0.5, label="95 mph EV threshold")
    ax.set_xlabel("Exit Velocity (mph)"); ax.set_ylabel("Launch Angle (°)")
    ax.set_title(f"{sel_player} — Contact Profile by Season", color=TEXT, fontsize=11)
    ax.legend(fontsize=8, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    sec("Event breakdown — all seasons")
    ev_p   = player_df["events"].value_counts()
    fig2, ax2 = mpl_fig(10, 3.2)
    bc2 = [GOLD if e in ["home_run","single","double","triple"] else GREY for e in ev_p.index]
    ax2.barh(ev_p.index[::-1], ev_p.values[::-1], color=bc2[::-1], edgecolor="none", height=0.6)
    ax2.set_xlabel("Count")
    ax2.set_title(f"{sel_player} — Event Distribution", color=TEXT, fontsize=10)
    fig2.tight_layout(); st.pyplot(fig2); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — WHAT DRIVES OUTCOMES
# ══════════════════════════════════════════════════════════════════════════════
elif "Drives" in page:
    st.markdown("# What Drives Batting Outcomes?")
    st.markdown("""<div class="preamble">
    The <b>Random Forest</b> identifies which contact-quality metrics best separate batting outcomes.
    The <b>Logistic Regression</b> shows the direction and magnitude of each feature's effect per class.
    RF feature importances flow directly into the <b>Performance Index</b> used for change detection —
    so CPD monitors exactly the metrics the model says matter most.
    </div>""", unsafe_allow_html=True)

    sec("Model configuration")
    col_e, col_f = st.columns([1, 1])
    with col_e:
        ev_labels  = {e: f"{e}  ({event_counts.get(e,0):,})" for e in avail_events}
        sel_events = st.multiselect(
            "Events to classify",
            options=avail_events,
            default=[e for e in DEFAULT_EVENTS if e in avail_events],
            format_func=lambda e: ev_labels[e],
        )
    with col_f:
        with st.expander("Feature selection (all on by default)"):
            sel_feats = st.multiselect(
                "Predictor features", options=avail_feats,
                default=avail_feats, key="model_feats",
            )

    if len(sel_events) < 2: st.warning("Select at least 2 events."); st.stop()
    if len(sel_feats)  < 1: st.warning("Select at least 1 feature."); st.stop()

    clf, feat_cols, le_rf, cv = train_rf(df, tuple(sel_events), tuple(sel_feats))
    lr,  feat_lr,  le_lr, sc  = train_lr(df, tuple(sel_events), tuple(sel_feats))

    if clf is None:
        st.error("Not enough data to train. Add more events or check feature coverage.")
        st.stop()

    # RF importance
    sec("Random Forest — global feature importance")
    importances = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    top_n   = st.slider("Top N features", 5, len(importances), min(12, len(importances)), 1)
    top_imp = importances.head(top_n)

    fig, ax = mpl_fig(11, max(3.5, top_n * 0.42))
    bcolors = [GOLD if i == 0 else (TEAL_LT if i < 3 else GREY) for i in range(top_n)]
    bars = ax.barh(top_imp.index[::-1], top_imp.values[::-1],
                   color=bcolors[::-1], edgecolor="none", height=0.65)
    for bar, val in zip(bars, top_imp.values[::-1]):
        ax.text(val + importances.max() * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", color=TEXT_MUTED,
                fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title("Feature Importance — predicting batted ball outcome", color=TEXT, fontsize=11)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    model_rows = df[df["events"].isin(sel_events)][["events"] + feat_cols].dropna()
    st.markdown(f"""<div class="card-row">
        {card("CV Accuracy (mean)", f"{cv.mean():.1%}", "5-fold honest estimate", "gold")}
        {card("CV Accuracy (std)",  f"{cv.std():.1%}",  "across 5 folds",         "grey")}
        {card("Classes",            str(len(le_rf.classes_)), ", ".join(le_rf.classes_[:3])+"…","teal")}
        {card("Training rows",      f"{len(model_rows):,}", "in-play contacts",   "grey")}
    </div>""", unsafe_allow_html=True)

    # LR coefficients
    if lr is not None:
        sec("Logistic Regression — coefficients per outcome")
        st.caption("Teal = positive · Red = negative · Features standardised")
        outcomes = le_lr.classes_
        coef_df  = pd.DataFrame(lr.coef_, columns=feat_lr, index=outcomes)
        n_out    = len(outcomes)
        n_cg     = min(3, n_out)
        n_rg     = int(np.ceil(n_out / n_cg))
        pal_out  = [GOLD, TEAL_LT, RED_LT, "#a8d8a8", GOLD_LT, GREY]

        fig = plt.figure(figsize=(6 * n_cg, 4.5 * n_rg))
        fig.patch.set_facecolor(DARK)
        for i, outcome in enumerate(outcomes):
            ax = fig.add_subplot(n_rg, n_cg, i + 1)
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=TEXT, labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            ax.grid(color=BORDER, lw=0.3, ls="--", alpha=0.3, axis="x")
            row    = coef_df.loc[outcome].abs().sort_values(ascending=False).head(top_n)
            signed = coef_df.loc[outcome][row.index]
            bc     = [TEAL_LT if v >= 0 else RED_LT for v in signed.values]
            ax.barh(signed.index[::-1], signed.values[::-1], color=bc[::-1], edgecolor="none", height=0.65)
            ax.axvline(0, color=BORDER, lw=0.8)
            ax.set_title(outcome, color=pal_out[i % len(pal_out)], fontsize=9, pad=4)
            ax.set_xlabel("Coefficient", color=TEXT_MUTED, fontsize=8)
        fig.suptitle("LR Coefficients per Outcome (standardised)", color=TEXT, fontsize=11, y=1.01)
        fig.tight_layout(); st.pyplot(fig); plt.close()

        sec("Top driver per outcome")
        st.dataframe(pd.DataFrame([{
            "Outcome":     o,
            "Strongest +": coef_df.loc[o].idxmax(),
            "+ coef":      round(coef_df.loc[o].max(), 4),
            "Strongest −": coef_df.loc[o].idxmin(),
            "− coef":      round(coef_df.loc[o].min(), 4),
        } for o in outcomes]), use_container_width=True, hide_index=True)

        sec("Coefficient heatmap")
        top_f  = coef_df.abs().max(axis=0).sort_values(ascending=False).head(top_n).index
        fig2, ax2 = plt.subplots(figsize=(max(8, top_n * 0.65), max(4, n_out * 0.6)))
        fig2.patch.set_facecolor(DARK); ax2.set_facecolor(PANEL)
        sns.heatmap(coef_df[top_f], ax=ax2, cmap="RdYlGn", center=0,
                    linewidths=0.4, linecolor=DARK,
                    annot=(top_n <= 12), fmt=".2f",
                    annot_kws={"size": 7, "color": DARK},
                    cbar_kws={"shrink": 0.6})
        ax2.set_title("LR Coefficient Heatmap", color=TEXT, fontsize=11, pad=8)
        ax2.tick_params(colors=TEXT, labelsize=8)
        plt.xticks(rotation=40, ha="right")
        fig2.tight_layout(); st.pyplot(fig2); plt.close()

    # Store RF weights in session for downstream pages
    st.session_state["rf_weights"] = dict(zip(importances.index, importances.values))





# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PERFORMANCE INDEX
# ══════════════════════════════════════════════════════════════════════════════

elif "Performance Index" in page:
    st.markdown("# Performance Index")
    st.markdown("""<div class="preamble">
    A <b>weighted composite score (0–100)</b> built from RF feature importances.
    Higher = better contact quality. All batted-ball contacts including foul balls are included
    to maximise sensitivity. Visit <i>What Drives Outcomes</i> first to train the weights;
    equal weights are used as fallback.
    </div>""", unsafe_allow_html=True)

    df_idx      = get_df_with_index()
    player_idx  = df_idx[df_idx["Name"] == sel_player]
    cpd_idx     = player_idx if show_history else player_idx[player_idx["Season"] == CURRENT_SEASON]
    roll_idx    = rolling_with_dates(cpd_idx, "perf_index", cpd_window)

    if len(roll_idx) < 10:
        st.warning("Not enough data for this player / window."); st.stop()

    cp_idx_pi = detect_cpd(roll_idx["perf_index"], penalty)
    cp_st_pi  = cpd_stats(roll_idx["perf_index"], cp_idx_pi)
    b_l, b_lm, b_ls = baselines(df_idx, "perf_index", sel_player)

    narrative = generate_narrative(sel_player, "Performance Index",
                                   roll_idx, cp_st_pi, b_l, b_lm, cpd_window)
    st.markdown(f'<div class="narrative">📊 {narrative}</div>', unsafe_allow_html=True)

    if cp_st_pi:
        sec("Detected change points")
        badges = ""
        for s in cp_st_pi:
            idx_  = min(s["cp_idx"], len(roll_idx) - 1)
            date_ = roll_idx["game_date"].iloc[idx_]
            ds    = date_.strftime("%b %d, %Y") if hasattr(date_, "strftime") else str(date_)
            badges += f'<span class="{s["badge"]}">{ds} · {s["direction"]} · {s["label"]} (d={s["effect_d"]:.2f})</span> '
        st.markdown(badges, unsafe_allow_html=True)

    sec("Performance index — rolling average with change points & baselines")
    fig, ax = mpl_fig(13, 5)
    ax.scatter(roll_idx["game_date"], roll_idx["perf_index"],
               color=GREY, alpha=0.15, s=6, zorder=1)
    ax.plot(roll_idx["game_date"], roll_idx["perf_index"],
            color=TEAL_LT, lw=2, zorder=3, label=f"Rolling {games_label(cpd_window)}")
    ymin, ymax = ax.get_ylim()
    add_baselines(ax, b_l, b_lm, b_ls)
    add_cpd_markers(ax, roll_idx, "perf_index", cp_idx_pi, cp_st_pi)
    if show_history: add_season_dividers(ax, ymax)

    legend_handles = [
        Line2D([0],[0], color=TEAL_LT, lw=2,   label=f"Rolling {games_label(cpd_window)}"),
        Line2D([0],[0], color=GOLD,    lw=1.2, ls="--", label="Prev season avg"),
        Line2D([0],[0], color=GREY,    lw=1.0, label=f"League avg {CURRENT_SEASON}"),
        Line2D([0],[0], color=RED_LT,  lw=1.4, ls="--", label="Significant CPD"),
        Line2D([0],[0], color=GOLD,    lw=1.4, ls="--", label="Moderate CPD"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.2,
              facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    ax.set_xlabel("Date"); ax.set_ylabel("Performance Index (0–100)")
    ax.set_title(f"{sel_player} — Performance Index · {sensitivity} sensitivity",
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
                "Δ":             round(s["delta"],       2),
                "Effect size d": round(s["effect_d"],    2),
                "Magnitude":     s["label"],
                "Direction":     s["direction"],
            })
        st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — METRIC DRILLDOWN
# ══════════════════════════════════════════════════════════════════════════════
elif "Drilldown" in page:
    st.markdown("# Metric Drilldown")
    st.markdown("""<div class="preamble">
    Each contact-quality metric is monitored independently with its own change point detection.
    Use this to identify <b>which component</b> of a player's game has shifted —
    exit velocity, launch angle, barrel rate, or something subtler.
    </div>""", unsafe_allow_html=True)

    drill_metrics = {k: v for k, v in {
        "Exit Velocity":  "exit_velocity",
        "Launch Angle":   "launch_angle_metric",
        "xwOBA (est)":    "xwoba_est",
        "Hard Hit %":     "is_hard_hit",
        "Barrel Proxy":   "is_barrel_proxy",
    }.items() if v in cpd_df.columns}

    n_m  = len(drill_metrics)
    n_c  = 2
    n_r  = int(np.ceil(n_m / n_c))
    fig  = plt.figure(figsize=(13, 4.5 * n_r))
    fig.patch.set_facecolor(DARK)
    gs   = gridspec.GridSpec(n_r, n_c, figure=fig, hspace=0.55, wspace=0.3)
    pal  = [TEAL_LT, GOLD, RED_LT, "#a8d8a8", GOLD_LT]

    for idx, (mlabel, mcol) in enumerate(drill_metrics.items()):
        ri, ci = divmod(idx, n_c)
        ax = fig.add_subplot(gs[ri, ci])
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.grid(color=BORDER, lw=0.3, ls="--", alpha=0.4)

        roll_m = rolling_with_dates(cpd_df, mcol, cpd_window)
        if len(roll_m) < 10:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center", color=TEXT_MUTED, fontsize=9)
            ax.set_title(mlabel, color=pal[idx % len(pal)], fontsize=10); continue

        cp_m   = detect_cpd(roll_m[mcol], penalty)
        cp_st_m = cpd_stats(roll_m[mcol], cp_m)
        b_l_m, b_lm_m, b_ls_m = baselines(df, mcol, sel_player)

        ax.plot(roll_m["game_date"], roll_m[mcol],
                color=pal[idx % len(pal)], lw=1.8, zorder=3)
        add_baselines(ax, b_l_m, b_lm_m, b_ls_m)
        add_cpd_markers(ax, roll_m, mcol, cp_m, cp_st_m)

        n_cp = len(cp_st_m)
        ax.set_title(f"{mlabel}{'  ['+str(n_cp)+' CPD'+('s' if n_cp!=1 else '')+']' if n_cp else ''}",
                     color=pal[idx % len(pal)], fontsize=10)
        ax.set_xlabel("Date", color=TEXT_MUTED, fontsize=8)
        ax.set_ylabel(mlabel, color=TEXT_MUTED, fontsize=8)

    fig.suptitle(f"{sel_player} — Per-Metric CPD · {sensitivity} sensitivity",
                 color=TEXT, fontsize=12, y=1.01)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    sec("Narratives per metric")
    for mlabel, mcol in drill_metrics.items():
        roll_m = rolling_with_dates(cpd_df, mcol, cpd_window)
        if len(roll_m) < 10: continue
        cp_m    = detect_cpd(roll_m[mcol], penalty)
        cp_st_m = cpd_stats(roll_m[mcol], cp_m)
        b_l_m, b_lm_m, _ = baselines(df, mcol, sel_player)
        narr = generate_narrative(sel_player, mlabel, roll_m, cp_st_m, b_l_m, b_lm_m, cpd_window)
        st.markdown(f'<div class="narrative"><b>{mlabel}:</b> {narr}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — SEASON TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
elif "Timeline" in page:
    st.markdown(f"# {CURRENT_SEASON} Season Timeline")
    st.markdown(f"""<div class="preamble">
    Current season view with annotated change points and the baseline trinity:
    player's previous season average (dashed gold), league average ± 1 std (grey band),
    and the rolling current season line (teal). Toggle history in the sidebar to extend back to 2021.
    </div>""", unsafe_allow_html=True)

    st.markdown(f"**{n_curr:,} pitches** in {CURRENT_SEASON} · {games_label(n_curr)} · "
                + reliability_badge(n_curr), unsafe_allow_html=True)

    tl_metric_label = st.selectbox(
        "Metric",
        options=[k for k in CPD_METRICS if k != "Performance Index"] + ["Performance Index"],
        index=0,
    )
    tl_col = CPD_METRICS[tl_metric_label]

    if tl_col == "perf_index":
        df_tl      = get_df_with_index()
        tl_src     = df_tl[df_tl["Name"] == sel_player]
    else:
        df_tl      = df
        tl_src     = player_df

    tl_src_filtered = tl_src if show_history else tl_src[tl_src["Season"] == CURRENT_SEASON]
    roll_tl = rolling_with_dates(tl_src_filtered, tl_col, cpd_window)

    if len(roll_tl) < 10:
        st.warning("Not enough data for this metric and window."); st.stop()

    cp_tl    = detect_cpd(roll_tl[tl_col], penalty)
    cp_st_tl = cpd_stats(roll_tl[tl_col], cp_tl)
    b_l_tl, b_lm_tl, b_ls_tl = baselines(df_tl, tl_col, sel_player)

    narr_tl = generate_narrative(sel_player, tl_metric_label, roll_tl,
                                  cp_st_tl, b_l_tl, b_lm_tl, cpd_window)
    st.markdown(f'<div class="narrative">📋 {narr_tl}</div>', unsafe_allow_html=True)

    if cp_st_tl:
        sec("Change points detected")
        badges = ""
        for s in cp_st_tl:
            idx_  = min(s["cp_idx"], len(roll_tl) - 1)
            date_ = roll_tl["game_date"].iloc[idx_]
            ds    = date_.strftime("%b %d, %Y") if hasattr(date_,"strftime") else str(date_)
            badges += f'<span class="{s["badge"]}">{ds} · {s["direction"]} · {s["label"]}</span> '
        st.markdown(badges, unsafe_allow_html=True)

    sec(f"{tl_metric_label} timeline")
    fig, ax = mpl_fig(13, 5)
    raw_tl = tl_src_filtered[[tl_col, "game_date"]].dropna().sort_values("game_date")
    ax.scatter(raw_tl["game_date"], raw_tl[tl_col], color=GREY, alpha=0.12, s=6, zorder=1)
    ax.plot(roll_tl["game_date"], roll_tl[tl_col],
            color=TEAL_LT, lw=2.2, zorder=3,
            label=f"{tl_metric_label} · rolling {games_label(cpd_window)}")
    ymin_tl, ymax_tl = ax.get_ylim()
    add_baselines(ax, b_l_tl, b_lm_tl, b_ls_tl)
    add_cpd_markers(ax, roll_tl, tl_col, cp_tl, cp_st_tl)
    if show_history: add_season_dividers(ax, ymax_tl)

    ax.set_xlabel("Date"); ax.set_ylabel(tl_metric_label)
    ax.set_title(
        f"{sel_player} — {tl_metric_label} · "
        f"{'2021–' + str(CURRENT_SEASON) if show_history else str(CURRENT_SEASON)} · "
        f"{sensitivity} sensitivity",
        color=TEXT, fontsize=12
    )
    ax.legend(fontsize=8, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    sec(f"{CURRENT_SEASON} monthly averages")
    curr_m = player_df[player_df["Season"] == CURRENT_SEASON].copy()
    curr_m["month"] = curr_m["game_date"].dt.month
    month_names = {3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct"}
    monthly = curr_m.groupby("month").agg(
        exit_velocity   =("exit_velocity",      "mean"),
        launch_angle    =("launch_angle_metric", "mean"),
        xwoba           =("xwoba_est",           "mean"),
        hard_hit_pct    =("is_hard_hit",         "mean"),
        n               =("exit_velocity",       "count"),
    ).round(3).reset_index()
    monthly["month"] = monthly["month"].map(month_names)
    st.dataframe(monthly, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — PEER COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif "Peer" in page:
    st.markdown("# Peer Comparison")
    st.markdown("""<div class="preamble">
    Compare the selected player's rolling performance against up to 3 peers.
    Dashed vertical lines mark each player's most recent <b>significant</b> change point.
    Use this to determine whether a decline is player-specific or a broader trend.
    </div>""", unsafe_allow_html=True)

    peers = st.multiselect(
        "Comparison players (up to 3)",
        [p for p in players if p != sel_player],
        default=[p for p in players if p != sel_player][1:4],
        max_selections=3,
    )
    all_sel = [sel_player] + peers

    peer_metric_label = st.selectbox(
        "Metric",
        options=[k for k in CPD_METRICS if k != "Performance Index"],
        index=0,
    )
    peer_col = CPD_METRICS[peer_metric_label]
    pal_peer = [GOLD, TEAL_LT, RED_LT, "#a8d8a8"]

    sec(f"{peer_metric_label} — rolling {games_label(cpd_window)} comparison")
    fig, ax = mpl_fig(13, 5)
    peer_rows = []

    for i, pname in enumerate(all_sel):
        pdf_p  = df[df["Name"] == pname]
        pdf_p  = pdf_p if show_history else pdf_p[pdf_p["Season"] == CURRENT_SEASON]
        roll_p = rolling_with_dates(pdf_p, peer_col, cpd_window)
        if len(roll_p) < 5: continue

        cp_p    = detect_cpd(roll_p[peer_col], penalty)
        cp_st_p = cpd_stats(roll_p[peer_col], cp_p)

        ax.plot(roll_p["game_date"], roll_p[peer_col],
                color=pal_peer[i],
                lw=2.5 if pname == sel_player else 1.5,
                alpha=1.0 if pname == sel_player else 0.65,
                label=pname, zorder=3 - i)

        sig_cps = [s for s in cp_st_p if s["label"] in ("moderate","significant")]
        if sig_cps:
            idx_sig  = min(sig_cps[-1]["cp_idx"], len(roll_p) - 1)
            date_sig = roll_p["game_date"].iloc[idx_sig]
            ax.axvline(date_sig, color=pal_peer[i], lw=1, ls="--", alpha=0.55)

        curr_val = roll_p[peer_col].iloc[-1]
        b_l_p, b_lm_p, _ = baselines(df, peer_col, pname)
        peer_rows.append({
            "Player":              pname,
            "Current rolling avg": round(curr_val, 2),
            "League avg":          round(b_lm_p, 2) if not np.isnan(b_lm_p) else "—",
            "Prev season avg":     round(b_l_p,  2) if not np.isnan(b_l_p)  else "—",
            "Significant CPDs":    len(sig_cps),
        })

    leag_p = df[df["Season"] == CURRENT_SEASON][peer_col].dropna()
    if len(leag_p):
        ax.axhline(leag_p.mean(), color=GREY, lw=1.0, alpha=0.6,
                   label=f"League avg {CURRENT_SEASON} ({leag_p.mean():.1f})")
        ax.axhspan(leag_p.mean() - leag_p.std(), leag_p.mean() + leag_p.std(),
                   alpha=0.05, color=GREY)

    ax.set_xlabel("Date")
    ax.set_ylabel(f"{peer_metric_label} ({games_label(cpd_window)} rolling)")
    ax.set_title(
        f"{peer_metric_label} comparison — dashed = most recent significant CPD",
        color=TEXT, fontsize=11
    )
    ax.legend(fontsize=9, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    if peer_rows:
        sec("Peer summary table")
        st.dataframe(pd.DataFrame(peer_rows), use_container_width=True, hide_index=True)

    sec(f"{peer_metric_label} distribution — {CURRENT_SEASON}")
    from scipy.stats import gaussian_kde
    fig2, ax2 = mpl_fig(11, 4)
    for i, pname in enumerate(all_sel):
        data = df[(df["Name"] == pname) & (df["Season"] == CURRENT_SEASON)][peer_col].dropna()
        if len(data) < 10: continue
        xs = np.linspace(data.min() - 2, data.max() + 2, 300)
        ys = gaussian_kde(data, bw_method=0.4)(xs)
        ax2.fill_between(xs, ys, alpha=0.2, color=pal_peer[i])
        ax2.plot(xs, ys, color=pal_peer[i], lw=2, label=f"{pname} (μ={data.mean():.1f})")
        ax2.axvline(data.mean(), color=pal_peer[i], lw=1, ls=":", alpha=0.8)
    ax2.set_xlabel(peer_metric_label); ax2.set_ylabel("Density")
    ax2.set_title(f"{peer_metric_label} distribution — {CURRENT_SEASON}", color=TEXT, fontsize=11)
    ax2.legend(fontsize=9, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    fig2.tight_layout(); st.pyplot(fig2); plt.close()