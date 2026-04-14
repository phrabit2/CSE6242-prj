#!/usr/bin/env python
# coding: utf-8
"""
MLB Batting Pulse

Single dataset: PA-level engineered features
  pa_master.csv — one row per plate appearance, 420 qualified hitters, 2021-2025

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
import plotly.express as px
import plotly.graph_objects as go
import gdown
from sklearn.preprocessing import StandardScaler

try:
    from changeforest import Control, changeforest
except Exception:
    Control = None
    changeforest = None


# Streamlit page configuration

st.set_page_config(
    page_title="Team 26: Performance Inflection Dashboard",
    
    layout="wide",
    initial_sidebar_state="expanded",
)


# CONSTANTS

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
PA_TOOLTIPS = {
    "hitting_decisions_score":     "Plate discipline. Measures swing vs. take quality. Higher is better (Elite: >3.0, League Avg: ~0.3).",
    "power_efficiency":            "Raw power. Effectiveness of converting swing effort to exit velocity. Higher is better (Elite: >0.0100, League Avg: ~0.0040).",
    "woba_residual":               "Luck vs Skill. Difference between actual results and physics-based expectation. Positive (>0.15) means outperforming physics (luck/skill); Negative (<-0.15) means 'unlucky'.",
    "launch_angle_stability_50pa": "Swing consistency. Stability of ball flight path over recent 50 PAs. Higher values indicate a more repeatable, optimized swing path.",
}
PA_COLORS = {
    "hitting_decisions_score":     "#2ca02c",
    "power_efficiency":            "#1f77b4",
    "woba_residual":               "#ff7f0e",
    "launch_angle_stability_50pa": "#d62728",
}

SENSITIVITY_MAP        = {"Low": 8, "Medium": 3, "High": 1}
SENSITIVITY_TO_MIN_SEG = {"Low": 0.10, "Medium": 0.05, "High": 0.02}

DARK       = "#FFFFFF"
PANEL      = "#F0F2F6"
BORDER     = "#D0D7DE"
GOLD       = "#855D00"
GOLD_LT    = "#B08800"
TEAL       = "#0068C9"
TEAL_LT    = "#58A6FF"
RED        = "#D32F2F"
RED_LT     = "#B71C1C"
GREY       = "#586069"
TEXT       = "#111418"
TEXT_MUTED = "#586069"



# STYLES

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
.card {{ background:{DARK}; border:1px solid {BORDER}; border-top:3px solid {GOLD};
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
.cpd-minor {{ display:inline-block; background:#f0f2f6; color:{TEXT_MUTED}; border-radius:3px; padding:2px 8px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; margin:2px; }}
.cpd-mod   {{ display:inline-block; background:#fff5b1; color:{GOLD_LT}; border:1px solid {GOLD}; border-radius:3px; padding:2px 8px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; margin:2px; }}
.cpd-sig   {{ display:inline-block; background:#ffeef0; color:{RED_LT}; border:1px solid {RED}; border-radius:3px; padding:2px 8px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; margin:2px; }}
.preamble {{ background:{PANEL}; border:1px solid {BORDER}; border-left:4px solid {GOLD};
    border-radius:4px; padding:1.2rem 1.4rem; margin-bottom:1.4rem;
    font-size:0.92rem; line-height:1.7; color:{TEXT_MUTED}; }}
.preamble b {{ color:{TEXT}; }}
.narrative {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:4px;
    padding:1rem 1.2rem; font-family:'IBM Plex Mono',monospace;
    font-size:0.82rem; line-height:1.6; color:{TEXT}; margin-bottom:1rem; }}
.rel-high {{ color:{TEAL_LT}; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }}
.rel-med  {{ color:{GOLD_LT}; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }}
.rel-low  {{ color:{RED_LT};  font-family:'IBM Plex Mono',monospace; font-size:0.75rem; }}
</style>
""", unsafe_allow_html=True)



# HELPERS

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

# ── SMART ANALYZER ENGINE ─────────────────────────────────────────────────────
def get_diagnostic_insight(stats_list, player_name):
    findings = []
    indicator_summary = {}
    for col, s in stats_list.items():
        d = s['effect_d']
        label = "Stable"
        if d > 0.5: label = "Significant Gains"
        elif d > 0.2: label = "Marginal Gains"
        elif d < -0.5: label = "Significant Decline"
        elif d < -0.2: label = "Marginal Decline"
        indicator_summary[col] = label

    pwr = indicator_summary.get('power_efficiency', "Stable")
    la = indicator_summary.get('launch_angle_stability_50pa', "Stable")
    disc = indicator_summary.get('hitting_decisions_score', "Stable")
    res = indicator_summary.get('woba_residual', "Stable")

    if "Decline" in pwr and "Decline" in la:
        findings.append(f"⚠️ <b>Mechanical/Physical Shift:</b> Both Power and Consistency declined. This strongly suggests a mechanical flaw or a physical issue (fatigue/injury) affecting the swing path.")
    elif "Improvement" in pwr and "Improvement" in la:
        findings.append(f"🔥 <b>Optimized Mechanics:</b> Improvements in both Power and Consistency indicate <b>{player_name}</b> has found a repeatable, high-impact swing.")
    
    if "Improvement" in disc and "Decline" in pwr:
        findings.append(f"🧘 <b>Heightened Selectivity:</b> Discipline improved, but Power dropped. This often happens when a hitter becomes <i>too</i> selective, sacrificing aggression for better take decisions.")
    elif "Decline" in disc and "Improvement" in pwr:
        findings.append(f"⚔️ <b>Increased Plate Aggression:</b> Discipline dropped while Power rose. The hitter is likely 'selling out' for power, swinging harder at the cost of strike-zone control.")

    if "Decline" in res and "Stable" in pwr and "Stable" in la:
        findings.append(f"📉 <b>Pure Bad Luck:</b> Performance results (wOBA) dropped despite Power and Consistency remaining steady. Physics says the hitter is doing everything right—results should follow.")
    elif "Improvement" in res and "Stable" in pwr and "Stable" in la:
        findings.append(f"🍀 <b>Results Surge:</b> Results are improving faster than the underlying physics change, indicating a period of high efficiency or favorable luck.")

    if not findings:
        sorted_stats = sorted(stats_list.items(), key=lambda x: abs(x[1]['effect_d']), reverse=True)
        primary_col, primary_stat = sorted_stats[0]
        label = indicator_summary[primary_col]
        findings.append(f"📊 <b>Primary Driver:</b> The most significant shift was a <b>{label}</b> in <b>{PA_LABELS[primary_col]}</b>.")

    return findings



def render_cp_analysis(selected_date, player_name, before_data, after_data, 
                       importance_df=None):  # new parameter
    st.markdown("---")
    st.write(f"### 🔍 Deep-Dive Analysis: Shift on {selected_date}")
    st.write("Comparing the window of performance before and after this detected shift.")

    all_stats = {}
    for col in PA_INDICATORS:
        delta = after_data[col].mean() - before_data[col].mean()
        pooled = np.sqrt((before_data[col].std()**2 + after_data[col].std()**2) / 2 + 1e-9)
        d = delta / pooled
        all_stats[col] = {
            'delta': delta, 'effect_d': d,
            'before': before_data[col].mean(), 'after': after_data[col].mean()
        }

    insights = get_diagnostic_insight(all_stats, player_name)

    st.markdown(f"""
    <div class="narrative">
    <b> Smart Analyzer Hypothesis:</b><br><br>
    {"<br><br>".join(insights)}
    </div>
    """, unsafe_allow_html=True)

    

    if importance_df is not None and not importance_df.empty:
        st.write("#### 🌲 ChangeForest Feature Importance")
        st.caption(
            "Which indicators drove this shift? A Random Forest classifier trained on the "
            "before/after windows shows which metrics were most separable at this change point."
        )
        fig_imp = go.Figure(go.Bar(
            x=importance_df["Importance"],
            y=importance_df["Indicator"],
            orientation='h',
            marker_color=importance_df["Color"],
            text=[f"{v:.1%}" for v in importance_df["Importance"]],
            textposition='auto',
        ))
        fig_imp.update_layout(
            height=250,
            xaxis=dict(title="Feature Importance", tickformat=".0%", range=[0, 1]),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT), margin=dict(t=10, b=30, l=10, r=10)
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        top_feature = importance_df.iloc[-1]["Indicator"]
        top_score = importance_df.iloc[-1]["Importance"]
        st.caption(
            f"**Primary driver:** {top_feature} accounts for {top_score:.1%} of the "
            f"separability between segments — the strongest signal at this change point."
        )


    # hiding key metric shifts for change forest
    # st.write("#### 📊 Key Metric Shifts")
    # with st.expander("ℹ️ How are these shifts calculated?"):
    #     st.markdown("""
    #     - **Effect Size (Cohen’s d):** This measures the magnitude of the shift relative to the player's natural variability. 
    #         - **0.2:** Small shift (normal fluctuation).
    #         - **0.5:** Medium shift (visible performance change).
    #         - **0.8+:** Large shift (major mechanical or approach overhaul).
    #     - **Primary Driver:** We identify this by finding the metric with the **highest absolute Effect Size**. It represents the most statistically significant 'break' in the player's performance profile.
    #     """)
    
    # cols = st.columns(4)
    # for i, col_name in enumerate(PA_INDICATORS):
    #     s = all_stats[col_name]
    #     with cols[i]:
    #         st.metric(PA_LABELS[col_name], f"{s['after']:.3f}", delta=f"{s['delta']:+.3f}")
    #         st.caption(f"Effect Size: {s['effect_d']:.2f}")

    if importance_df is not None and not importance_df.empty:
        st.write("#### 📈 Distribution Shift (ChangeForest Primary Driver)")
        top_label = importance_df.iloc[-1]["Indicator"]
        top_col = next(col for col in PA_INDICATORS if PA_LABELS[col] == top_label)
    else:
        st.write("#### 📈 Distribution Shift (Cohen's d Primary Driver)")
        sorted_stats = sorted(all_stats.items(), key=lambda x: abs(x[1]['effect_d']), reverse=True)
        top_col = sorted_stats[0][0]
        
        # fig = go.Figure()
        # fig.add_trace(go.Histogram(x=before_data[top_col], name="Before Shift", marker_color=GREY, opacity=0.6))
        # fig.add_trace(go.Histogram(x=after_data[top_col], name="After Shift", marker_color=TEAL, opacity=0.6))
        # fig.update_layout(barmode='overlay', title=f"Change in {PA_LABELS[top_col]} Population", 
        #                 xaxis_title=PA_LABELS[top_col], yaxis_title="Frequency (PA Count)",
        #                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT),
        #                 height=300, margin=dict(t=40, b=40, l=40, r=40))
        # st.plotly_chart(fig, use_container_width=True)

    if before_data[top_col].dropna().empty or after_data[top_col].dropna().empty:
            st.warning(f"Not enough data to plot distribution for {PA_LABELS[top_col]}.")
    else:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=before_data[top_col], name="Before Shift", marker_color=GREY, opacity=0.6))
            fig.add_trace(go.Histogram(x=after_data[top_col], name="After Shift", marker_color=TEAL, opacity=0.6))
            fig.update_layout(barmode='overlay', title=f"Change in {PA_LABELS[top_col]} Population", 
                            xaxis_title=PA_LABELS[top_col], yaxis_title="Frequency (PA Count)",
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT),
                            height=300, margin=dict(t=40, b=40, l=40, r=40))
            st.plotly_chart(fig, use_container_width=True)



# DATA LOADING

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



# PERFORMANCE INDEX

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



# CPD UTILITIES

def detect_cpd(series, penalty):
    try:
        import ruptures as rpt
        sig  = series.values.reshape(-1, 1)
        algo = rpt.Pelt(model="rbf").fit(sig)
        bkps = algo.predict(pen=penalty)
        return [b for b in bkps[:-1] if 0 < b < len(series)]
    except Exception:
        return []

def rolling_with_dates(pdf, metric, window):
    sub = pdf[pdf[metric].notna()][["game_date", metric]].sort_values("game_date")
    if len(sub) < window:
        return pd.DataFrame(columns=["game_date", metric])
    roll = sub[metric].rolling(window, min_periods=window // 2).mean()
    return pd.DataFrame({"game_date": sub["game_date"].values,
                          metric: roll.values}).dropna()

def build_cpd_subdf(df: pd.DataFrame, player_name: str, window: int) -> pd.DataFrame:
    feature_cols = ["cf_seq_id"] + PA_INDICATORS
    base_cols = ["batter", "player_name", "pa_uid", "game_date", "game_pk", "at_bat_number"]

    subdf = (
        df.loc[df["player_name"] == player_name, base_cols + feature_cols]
        .dropna(subset=feature_cols)
        .sort_values("cf_seq_id")
        .reset_index(drop=True)
    )

    for col in PA_INDICATORS:
        subdf[f"{col}_rollmean_{window}"] = (
            subdf[col].rolling(window=window, min_periods=window).mean()
        )

    rollmean_cols = [f"{col}_rollmean_{window}" for col in PA_INDICATORS]
    return subdf.dropna(subset=rollmean_cols).reset_index(drop=True)


def run_changeforest(subdf: pd.DataFrame, window: int, min_seg: float):
    if changeforest is None or Control is None:
        st.warning("changeforest not available.")
        return None, [], None, None

    feature_names = [f"{col}_rollmean_{window}" for col in PA_INDICATORS]
    
    missing = [c for c in feature_names if c not in subdf.columns]
    if missing:
        st.warning(f"Missing columns: {missing}")
        return None, [], None, None

    X = subdf[feature_names].values.astype(np.float64)

    if len(X) < 20 or np.isnan(X).any():
        return None, [], None, None

    X = StandardScaler().fit_transform(X)

    control = Control(
        model_selection_alpha=0.02,
        minimal_relative_segment_length=min_seg,
    )
    try:
        result = changeforest(X, method="random_forest", control=control)
        cps = [int(cp) for cp in result.split_points() if 0 < int(cp) < len(subdf)]
        return result, cps, X, feature_names
    except Exception as e:
        st.warning(f"changeforest error: {e}")
        return None, [], None, None
    

def get_cp_feature_importance(subdf: pd.DataFrame, cp_idx: int, window: int) -> pd.DataFrame:
    """
    For a detected change point, train a binary RF classifier
    on before/after windows and return feature importances.
    """
    from sklearn.ensemble import RandomForestClassifier

    feature_names = [f"{col}_rollmean_{window}" for col in PA_INDICATORS]
    half = min(50, cp_idx, len(subdf) - cp_idx)
    if half < 5:
        return pd.DataFrame()

    before = subdf[feature_names].iloc[cp_idx - half : cp_idx].copy()
    after  = subdf[feature_names].iloc[cp_idx : cp_idx + half].copy()

    before["label"] = 0
    after["label"]  = 1

    combined = pd.concat([before, after]).dropna()
    if len(combined) < 10:
        return pd.DataFrame()

    X = combined[feature_names].values
    y = combined["label"].values

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    importance_df = pd.DataFrame({
        "Indicator": [PA_LABELS[col] for col in PA_INDICATORS],
        "Importance": clf.feature_importances_,
        "Color": [PA_COLORS[col] for col in PA_INDICATORS],
    }).sort_values("Importance", ascending=True)

    return importance_df

# TOP NAVIGATION BAR

if 'nav_page' not in st.session_state:
    st.session_state.nav_page = "📖 Welcome"

# Custom Top Nav CSS
st.markdown("""
    <style>
    /* Hide the sidebar by default for a cleaner look */
    [data-testid="stSidebar"] {
        display: none;
    }
    .top-nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 2rem;
        background-color: white;
        border-bottom: 1px solid #E1E4E8;
        margin-bottom: 2rem;
        position: sticky;
        top: 0;
        z-index: 999;
    }
    .nav-logo {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.8rem;
        color: #111418;
        letter-spacing: 0.05em;
    }
    div[data-testid="stHorizontalBlock"] > div:has(button) {
        display: flex;
        justify-content: center;
    }
    button[kind="secondary"] {
        background: transparent !important;
        border: none !important;
        color: #586069 !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        transition: all 0.2s ease !important;
    }
    button[kind="secondary"]:hover {
        color: #0969da !important;
        background: #f6f8fa !important;
    }
    button[kind="primary"] {
        background: transparent !important;
        border: none !important;
        color: #0969da !important;
        font-weight: 700 !important;
        padding: 0.5rem 1rem !important;
        border-bottom: 3px solid #58A6FF !important;
        border-radius: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# App Header / Logo
t_col1, t_col2 = st.columns([1, 2])
with t_col1:
    st.markdown("<div class='nav-logo'>Team 26: Performance Inflection Dashboard</div>", unsafe_allow_html=True)

# Navigation Tabs
with t_col2:
    nav_cols = st.columns(5)
    nav_items = ["Welcome", "Player Snapshot", "👥 Peer Comparison", "Univariate Change Analyzer","Multivariate Change Analyzer"]
    
    for i, item in enumerate(nav_items):
        is_active = (st.session_state.nav_page == item)
        with nav_cols[i]:
            if st.button(item, key=f"top_nav_{item}", type="primary" if is_active else "secondary"):
                st.session_state.nav_page = item
                st.rerun()

st.markdown("---")

# Routing based on session state
page = st.session_state.nav_page

# Pre-calculate players list with "All Players" option
players_with_all = ["All Players (League Avg)"] + players

# ── PAGE: Welcome ─────────────────────────────────────────────────────────────
if "Welcome" in page:
    st.markdown("# Performance Inflection Dashboard")
    st.markdown("### Advanced Hitter Performance Analytics through Statcast & Machine Learning")

    # ── Motivation & Navigation ────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Why it Matters: The Shift from Lagging to Leading Indicators")
        st.markdown(f"""
        Traditional baseball statistics like **Batting Average** or **OPS** are *lagging indicators*—by the time a slump shows up in the box score, a player may have been struggling for weeks. 
        
        Changes the game by monitoring *leading indicators*. By analyzing the underlying physics of every plate appearance—how hard the ball was hit, the decision to swing, and the consistency of the swing path—we identify performance shifts the moment they happen. 
        
        Our goal is to give you the **true performance pulse** of a hitter, separating pure luck from genuine skill changes.
        """)
    with col2:
        st.markdown("#### How to Use the Dashboard")
        st.markdown("""
        1.  **Start with the Snapshot**: Select a player to see their current performance profile and where they rank relative to the league average this season.
        2.  **Benchmark with Peers**: Select up to 3 hitters to see how their performance "fingerprints" differ and who is currently leading in contact quality.
        3.  **Diagnose the Shift**: Use the **Change Analyzer** to pinpoint the exact date a player's performance changed and let our **Smart Analyzer** explain the likely cause.
        """)

    st.markdown("---")

    # ── Indicators Grid ───────────────────────────────────────────────────────
    st.markdown("#### The Four Pillars of Performance")
    st.caption("We monitor these core indicators to track a hitter's true performance pulse.")
    
    i1, i2 = st.columns(2)
    with i1:
        st.markdown(f"""<div class="card-row">
            {card("Hitting Decisions", "Plate Discipline", "Measuring the quality of swing vs. take decisions.", "gold")}
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="card-row">
            {card("wOBA Residual", "Luck vs Skill", "Identifying performance that outpaces (or trails) contact physics.", "teal")}
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown(f"""<div class="card-row">
            {card("Power Efficiency", "Raw Power", "The efficiency of converting swing effort into ball impact.", "teal")}
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="card-row">
            {card("Launch Angle Stability", "Swing Consistency", "The repeatability and steadiness of the ball flight path.", "gold")}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Dataset Stats ─────────────────────────────────────────────────────────
    sec("📊 Dataset Overview")
    st.markdown(f"""<div class="card-row">
        {card("Total Hitters",   f"{df['player_name'].nunique():,}", "qualified players",     "gold")}
        {card("Total Records",  f"{len(df):,}",                    "plate appearances",      "teal")}
        {card("Season Span",    str(df['year'].nunique()),         f"{min_year} – {max_year}", "grey")}
        {card(f"{max_year} Data",f"{len(df[df['year']==max_year]):,}", "current season PAs",   "teal")}
    </div>""", unsafe_allow_html=True)

    sec("PA Indicator Distributions — League Benchmark")
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

    st.markdown(f"""
    <div class="narrative">
    <b>📊 Distribution Analysis:</b><br>
    The histograms above reveal the baseline performance 'environment' of the major leagues:
    <ul>
        <li><b>Decision Symmetry:</b> <i>Hitting Decisions</i> are tightly centered around zero, showing that most professional hitters maintain a balanced approach, with elite outliers (>3.0) representing the top tier of plate discipline.</li>
        <li><b>Power Scarcity:</b> <i>Power Efficiency</i> shows a 'long tail' to the right. While most PAs yield low efficiency, the elite power hitters create that secondary hump, maximizing energy transfer.</li>
        <li><b>Physics Alignment:</b> The <i>wOBA Residual</i> is a near-perfect bell curve centered on zero. This confirms that over large samples, physics-based expectations and actual results align for the majority of the league.</li>
        <li><b>Professional Floor:</b> <i>Launch Angle Stability</i> shows a high concentration around the mean (~28.0), representing the baseline consistency required to compete at the major league level.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)



# PAGE 2 - PLAYER SNAPSHOT

elif "Snapshot" in page:
    st.markdown("# Player Snapshot")
    
    # ── TOP FILTERS (Reciprocal filtering) ────────────────────────────────────
    if 'snapshot_player' not in st.session_state:
        st.session_state.snapshot_player = "All Players (League Avg)"
    if 'snapshot_year' not in st.session_state:
        st.session_state.snapshot_year = max_year

    # 1. Filter Years based on current player
    if st.session_state.snapshot_player == "All Players (League Avg)":
        available_years = sorted(df["year"].unique(), reverse=True)
    else:
        available_years = sorted(df[df["player_name"] == st.session_state.snapshot_player]["year"].unique(), reverse=True)
    
    if st.session_state.snapshot_year not in available_years:
        st.session_state.snapshot_year = available_years[0]

    # 2. Filter Players based on current year
    available_players_for_year = sorted(df[df["year"] == st.session_state.snapshot_year]["player_name"].dropna().unique())
    player_options = ["All Players (League Avg)"] + available_players_for_year
    
    if st.session_state.snapshot_player not in player_options:
        st.session_state.snapshot_player = "All Players (League Avg)"

    # Filter Layout
    row1_c1, row1_c2 = st.columns([3, 1])
    with row1_c1:
        sel_player = st.selectbox(
            "Select Player", 
            player_options, 
            index=player_options.index(st.session_state.snapshot_player),
            key="snapshot_player_select"
        )
    with row1_c2:
        sel_year = st.selectbox(
            "Season", 
            available_years, 
            index=available_years.index(st.session_state.snapshot_year),
            key="snapshot_year_select"
        )

    if sel_player != st.session_state.snapshot_player or sel_year != st.session_state.snapshot_year:
        st.session_state.snapshot_player = sel_player
        st.session_state.snapshot_year = sel_year
        st.rerun()

    is_all = (sel_player == "All Players (League Avg)")
    if is_all:
        snapshot_df = df[df["year"] == sel_year].copy()
        display_name = f"League Average — {sel_year}"
    else:
        snapshot_df = df[(df["player_name"] == sel_player) & (df["year"] == sel_year)].copy()
        display_name = f"{sel_player} — {sel_year}"

    def _mean(s): return s.dropna().mean() if not s.empty else 0.0
    
    m_discipline = _mean(snapshot_df["hitting_decisions_score"])
    m_power      = _mean(snapshot_df["power_efficiency"])
    m_woba_res   = _mean(snapshot_df["woba_residual"])
    m_la_stab    = _mean(snapshot_df["launch_angle_stability_50pa"])
    m_pa_count   = len(snapshot_df)

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hitting Decisions", f"{m_discipline:.2f}", help=PA_TOOLTIPS["hitting_decisions_score"])
    c2.metric("Power Efficiency",  f"{m_power:.4f}",      help=PA_TOOLTIPS["power_efficiency"])
    c3.metric("wOBA Residual",     f"{m_woba_res:.3f}",   help=PA_TOOLTIPS["woba_residual"])
    c4.metric("Launch Angle Stab.", f"{m_la_stab:.2f}",   help=PA_TOOLTIPS["launch_angle_stability_50pa"])
    c5.metric("PAs this Season",   f"{m_pa_count:,}",     help="Total records for the season. More PAs (>200) indicate higher statistical reliability.")

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
        sec(f"📊 {sel_year} Performance Spectrum")
        st.caption("Visualizing the hitter's standing across core pillars. League Avg is the center marker; bars represent the range between the 1st and 99th league percentiles.")
        
        def spectrum_row(icon, title, status, value_pct, color, insight_text):
            return f"""
            <div style="display: flex; align-items: flex-start; gap: 20px; margin-bottom: 24px; padding: 12px 0; border-bottom: 1px solid #f0f2f6;">
                <div style="flex: 0 0 140px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                        <span style="font-size: 1.1rem;">{icon}</span>
                        <span style="font-weight: 700; font-size: 0.8rem; text-transform: uppercase; color: #495057; letter-spacing: 0.05em;">{title}</span>
                    </div>
                    <span style="display: inline-block; padding: 2px 8px; border-radius: 12px; background: {color}22; color: {color}; font-weight: 700; font-size: 0.7rem; border: 1px solid {color}44;">{status}</span>
                </div>
                <div style="flex: 1; max-width: 300px;">
                    <div style="height: 6px; width: 100%; background: #e9ecef; border-radius: 3px; position: relative; margin-top: 8px;">
                        <div style="position: absolute; left: 50%; top: -4px; height: 14px; width: 2px; background: #adb5bd; z-index: 1;"></div>
                        <div style="height: 100%; width: {value_pct}%; background: {color}; border-radius: 3px;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.62rem; color: #adb5bd; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.05em;">
                        <span>League Min</span>
                        <span>Avg</span>
                        <span>League Max</span>
                    </div>
                </div>
                <div style="flex: 1.5;">
                    <div style="font-size: 0.85rem; color: #24292f; line-height: 1.4;">{insight_text}</div>
                </div>
            </div>
            """

        def get_pct(col, val):
            s_min, s_max = season_df[col].quantile(0.01), season_df[col].quantile(0.99)
            return min(100, max(0, (val - s_min) / (s_max - s_min + 1e-9) * 100))

        season_df = df[df["year"] == sel_year]
        
        # 1. Discipline
        status, desc = get_stat_insight("hitting_decisions_score", m_discipline)
        p = get_pct('hitting_decisions_score', m_discipline)
        insight = f"<b>{sel_player}</b> is {status.lower()} in plate discipline. Sitting near the ceiling indicates elite pitch recognition and strike-zone mastery, leading to more walks and fewer 'chase' strikeouts."
        st.markdown(spectrum_row("🎯", "Discipline", status, p, "#0969da", insight), unsafe_allow_html=True)

        # 2. Power
        status, desc = get_stat_insight("power_efficiency", m_power)
        p = get_pct('power_efficiency', m_power)
        insight = f"Currently graded as <b>{status.lower()}</b>. High power efficiency means the hitter is maximizing exit velocity relative to swing effort—a key indicator of a 'heavy' bat and efficient mechanics."
        st.markdown(spectrum_row("💥", "Power", status, p, "#58A6FF", insight), unsafe_allow_html=True)

        # 3. Results vs Physics
        wr = m_woba_res
        status = "Steady" if abs(wr)<0.15 else ("Outpacing Physics" if wr>0.15 else "Unlucky")
        p = get_pct('woba_residual', wr)
        insight = f"Status: <b>{status}</b>. Sitting far to the right (Ceiling) suggests results are currently better than the physics of the contact would predict (skill/luck), while the left (Min) indicates hard-luck regression is coming."
        st.markdown(spectrum_row("🎲", "Results", status, p, "#855D00", insight), unsafe_allow_html=True)

        # 4. Consistency
        status, desc = get_stat_insight("launch_angle_stability_50pa", m_la_stab)
        p = get_pct('launch_angle_stability_50pa', m_la_stab)
        insight = f"Graded as <b>{status.lower()}</b>. Consistency near the ceiling indicates a highly repeatable swing path, making the hitter less vulnerable to timing-related slumps and cold streaks."
        st.markdown(spectrum_row("📈", "Consistency", status, p, "#2ca02c", insight), unsafe_allow_html=True)

    st.markdown("---")
    v_col1, v_col2 = st.columns([1, 1])
    with v_col1:
        sec("Player Profile (Radar Chart)")
        season_df = df[df["year"] == sel_year]
        radar_data = []
        for col in PA_INDICATORS:
            val = _mean(snapshot_df[col])
            s_min, s_max = season_df[col].min(), season_df[col].max()
            norm_val = (val - s_min) / (s_max - s_min + 1e-9)
            radar_data.append(norm_val)
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_data + [radar_data[0]],
            theta=[PA_LABELS[c] for c in PA_INDICATORS] + [PA_LABELS[PA_INDICATORS[0]]],
            fill='toself', name=sel_player, line_color=TEAL
        ))
        leag_vals = []
        for col in PA_INDICATORS:
            l_val = _mean(season_df[col])
            s_min, s_max = season_df[col].min(), season_df[col].max()
            norm_l = (l_val - s_min) / (s_max - s_min + 1e-9)
            leag_vals.append(norm_l)
        
        fig_radar.add_trace(go.Scatterpolar(
            r=leag_vals + [leag_vals[0]],
            theta=[PA_LABELS[c] for c in PA_INDICATORS] + [PA_LABELS[PA_INDICATORS[0]]],
            name='League Average', line=dict(dash='dash', color=GREY)
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, height=400, margin=dict(t=40, b=40, l=40, r=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with v_col2:
        sec("Statcast-Style Percentile Rankings")
        if is_all:
            percentiles = [50] * 4
        else:
            player_means = df[df["year"] == sel_year].groupby("player_name")[PA_INDICATORS].mean()
            percentiles = [(player_means[col] < _mean(snapshot_df[col])).mean() * 100 for col in PA_INDICATORS]
        
        fig_pct = go.Figure(go.Bar(
            x=percentiles, y=[PA_LABELS[c] for c in PA_INDICATORS], orientation='h',
            marker=dict(color=percentiles, colorscale='RdYlGn', cmin=0, cmax=100),
            text=[f"{p:.0f}%" for p in percentiles], textposition='auto',
        ))
        fig_pct.update_layout(
            xaxis=dict(range=[0, 100], title="Percentile Rank"), yaxis=dict(autorange="reversed"),
            height=400, margin=dict(t=40, b=40, l=100, r=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT)
        )
        st.plotly_chart(fig_pct, use_container_width=True)

    st.markdown("---")
    sec("Contact Quality Profile (Power vs Luck)")
    fig_scatter = px.scatter(
        snapshot_df.dropna(subset=["power_efficiency", "woba_residual"]),
        x="power_efficiency", y="woba_residual", color_discrete_sequence=[TEAL], opacity=0.4,
        labels={"power_efficiency": "Power Efficiency", "woba_residual": "wOBA Residual"},
        title=f"Distribution for {display_name}"
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color=GREY)
    fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT), height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    if not is_all:
        sec("Season-by-Season Progression")
        summ = df[df["player_name"] == sel_player].groupby("year").agg(
            hitting_decisions =("hitting_decisions_score",     "mean"),
            power_efficiency  =("power_efficiency",            "mean"),
            woba_residual     =("woba_residual",               "mean"),
            launch_angle_stab =("launch_angle_stability_50pa", "mean"),
            pa_count          =("pa_uid",                      "count"),
        ).round(3).reset_index().rename(columns={"year": "Season"})
        st.dataframe(summ, use_container_width=True, hide_index=True)

        # Season-by-Season Evolution Narrative
        if len(summ) > 1:
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            
            def get_archetype(row):
                # Standardizing on hitting_decisions_score for the full df
                pwr_elite = row['power_efficiency'] > df['power_efficiency'].mean() + df['power_efficiency'].std()
                disc_elite = row['hitting_decisions'] > df['hitting_decisions_score'].mean() + df['hitting_decisions_score'].std()
                if pwr_elite and disc_elite: return "Elite All-Around Threat"
                if pwr_elite: return "Power-First Slugger"
                if disc_elite: return "Discipline-Driven Tactician"
                return "Steady Contributor"

            first_year = int(summ.iloc[0]["Season"])
            last_year = int(summ.iloc[-1]["Season"])
            
            evolution_story = []
            
            # 1. The Starting Point
            initial_arch = get_archetype(summ.iloc[0])
            evolution_story.append(f"🌱 <b>The Foundation ({first_year}):</b> {sel_player} entered the dataset as a <b>{initial_arch}</b>. "
                                   f"Early data established a baseline {summ.iloc[0]['hitting_decisions']:.2f} Discipline and {summ.iloc[0]['power_efficiency']:.4f} Power.")

            # 2. Middle Years / Growth
            if len(summ) > 2:
                growth_notes = []
                for i in range(1, len(summ) - 1):
                    y = int(summ.iloc[i]["Season"])
                    prev = summ.iloc[i-1]
                    curr = summ.iloc[i]
                    
                    if curr['power_efficiency'] > prev['power_efficiency'] * 1.1:
                        growth_notes.append(f"a power surge in {y}")
                    if curr['hitting_decisions'] > prev['hitting_decisions'] + 0.5:
                        growth_notes.append(f"improved strike-zone mastery in {y}")
                
                if growth_notes:
                    evolution_story.append(f"📈 <b>Growth & Maturation:</b> Middle seasons were defined by {', '.join(growth_notes)}. "
                                           f"This period saw the player refine their approach and maximize their physical tools.")

            # 3. Current State
            latest_arch = get_archetype(summ.iloc[-1])
            evolution_story.append(f"🏁 <b>Current Profile ({last_year}):</b> Today, {sel_player} has evolved into a <b>{latest_arch}</b>. "
                                   f"The data shows a career-to-date trend of {'improving' if summ['power_efficiency'].iloc[-1] > summ['power_efficiency'].iloc[0] else 'shifting'} "
                                   f"performance profiles as they adapt to league adjustments.")

            st.markdown(f"""
            <div class="narrative">
            <b>📊 Career Evolution: {sel_player} ({first_year}–{last_year})</b><br><br>
            {"<br><br>".join(evolution_story)}
            </div>
            """, unsafe_allow_html=True)



# PAGE 3 - PEER COMPARISON

elif "Peer" in page:
    st.markdown("# 👥 Peer Comparison")
    if 'peer_year' not in st.session_state: st.session_state.peer_year = max_year
    if 'peer_players' not in st.session_state: st.session_state.peer_players = ["Trout, Mike", "Ohtani, Shohei"] if "Trout, Mike" in players else []

    st.markdown("<style>span[data-baseweb='tag'] { background-color: #E8F0FE !important; color: #1A73E8 !important; border: 1px solid #D2E3FC !important; } span[data-baseweb='tag'] svg { fill: #1A73E8 !important; }</style>", unsafe_allow_html=True)

    row1_c1, row1_c2 = st.columns([1, 3])
    with row1_c1:
        yrs = sorted(df["year"].unique(), reverse=True)
        sel_year = st.selectbox("Season", yrs, index=yrs.index(st.session_state.peer_year), key="peer_y_sel")
    
    p_in_s = sorted(df[df["year"] == sel_year]["player_name"].dropna().unique())
    with row1_c2:
        valid_p = [p for p in st.session_state.peer_players if p in p_in_s]
        sel_players = st.multiselect("Select Players (up to 3)", p_in_s, default=valid_p[:3], max_selections=3, key="peer_p_sel")

    opts = {**PA_LABELS, "pa_count": "PAs this Season"}
    sel_metrics = st.multiselect("Select Metrics to Compare", list(opts.keys()), default=list(PA_LABELS.keys())[:3], format_func=lambda x: opts[x])

    if sel_year != st.session_state.peer_year or sel_players != st.session_state.peer_players:
        st.session_state.peer_year = sel_year; st.session_state.peer_players = sel_players; st.rerun()

    if not sel_players: st.warning("Please select at least one player to compare."); st.stop()

    st.markdown("---")
    comparison_data = []
    season_df = df[df["year"] == sel_year]
    radar_colors = [TEAL, GOLD, RED]
    for p_name in sel_players:
        p_df = season_df[season_df["player_name"] == p_name]
        stats = {"Player": p_name, "pa_count": len(p_df)}
        for m_col in PA_INDICATORS:
            stats[m_col] = p_df[m_col].mean()
            stats[f"{m_col}_std"] = p_df[m_col].std()
        comparison_data.append(stats)
    comp_df = pd.DataFrame(comparison_data)
    
    def peer_card(name, val, color_idx):
        c = radar_colors[color_idx % 3]
        return f'<div style="padding: 8px 12px; background: {PANEL}; border-radius: 4px; margin-bottom: 8px; border: 1px solid {BORDER}; border-left: 4px solid {c};"><div style="font-size: 0.7rem; text-transform: uppercase; color: {TEXT_MUTED}; font-weight: 600; letter-spacing: 0.05em; line-height: 1.2;">{name}</div><div style="font-size: 1.3rem; font-weight: 700; color: {TEXT}; font-family: \'Bebas Neue\', sans-serif; line-height: 1.1;">{val}</div></div>'

    ov_col1, ov_col2 = st.columns([1, 1])
    with ov_col1:
        sec(f"📊 Performance Overview — {sel_year}")
        for mk in sel_metrics:
            st.markdown(f"<div style='margin-top: 12px; margin-bottom: 4px; font-weight: 700; font-size: 0.85rem; color: {TEAL_LT}; text-transform: uppercase; letter-spacing: 0.1em;'>{opts[mk]}</div>", unsafe_allow_html=True)
            cols = st.columns(max(len(sel_players), 3))
            for i, ps in enumerate(comparison_data):
                v_str = f"{int(ps[mk]):,}" if mk == "pa_count" else f"{ps[mk]:.{(4 if mk=='power_efficiency' else (3 if mk=='woba_residual' else 2))}f}"
                with cols[i]: st.markdown(peer_card(ps["Player"], v_str, i), unsafe_allow_html=True)

    with ov_col2:
        sec("💡 Peer Comparison Insights")
        peer_insights = []
        if len(sel_players) > 1:
            m_ctx = {"hitting_decisions_score": {"why": "superior plate discipline.", "impact": "More walks, fewer strikeouts."}, "power_efficiency": {"why": "better kinetic energy transfer.", "impact": "Sustainable, elite exit velocity."}, "woba_residual": {"why": "maximizing results vs physics.", "impact": "High 'batted ball skill'."}, "launch_angle_stability_50pa": {"why": "repeatable swing path.", "impact": "Slump-resistant performance."}}
            for mk in [m for m in sel_metrics if m != "pa_count"]:
                if mk in m_ctx:
                    leader = comp_df.loc[comp_df[mk].idxmax(), "Player"]; ctx = m_ctx[mk]
                    peer_insights.append(f"🏆 <b>{opts[mk]}:</b> <b>{leader}</b> ({comp_df[mk].max():.3f})<br><i>{ctx['why']} {ctx['impact']}</i>")
            if "hitting_decisions_score" in comp_df.columns and "power_efficiency" in comp_df.columns:
                top_disc = comp_df.loc[comp_df["hitting_decisions_score"].idxmax(), "Player"]; top_pwr = comp_df.loc[comp_df["power_efficiency"].idxmax(), "Player"]
                if top_disc != top_pwr: peer_insights.append(f"⚖️ <b>Contrasting Styles:</b> <b>{top_disc}</b> (Tactician) vs. <b>{top_pwr}</b> (Slugger).")
        st.markdown(f'<div class="narrative" style="font-size: 0.85rem;">{"<br><br>".join(peer_insights if peer_insights else ["Add players/metrics for insights."])}</div>', unsafe_allow_html=True)

    st.markdown("---")
    v_c1, v_c2 = st.columns(2)
    with v_c1:
        sec("Player Profile Comparison (Radar)")
        fig_radar = go.Figure()
        for i, p_name in enumerate(sel_players):
            p_stats = comp_df[comp_df["Player"] == p_name].iloc[0]
            radar_vals = [( (p_stats[col] - season_df[col].min()) / (season_df[col].max() - season_df[col].min() + 1e-9) ) for col in PA_INDICATORS]
            fig_radar.add_trace(go.Scatterpolar(r=radar_vals + [radar_vals[0]], theta=[PA_LABELS[c] for c in PA_INDICATORS] + [PA_LABELS[PA_INDICATORS[0]]], fill='toself', name=p_name, line_color=radar_colors[i % 3]))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT))
        st.plotly_chart(fig_radar, use_container_width=True)
    with v_c2:
        sec("Metric Leaderboard")
        plot_m = [m for m in sel_metrics if m != "pa_count"]
        if plot_m:
            bar_data = [{"Player": p_name, "Metric": PA_LABELS[m], "Relative Standing (Z-Score)": (comp_df[comp_df["Player"]==p_name][m].iloc[0] - season_df[m].mean()) / (season_df[m].std() + 1e-9)} for p_name in sel_players for m in plot_m]
            fig_bar = px.bar(pd.DataFrame(bar_data), x="Metric", y="Relative Standing (Z-Score)", color="Player", barmode="group", color_discrete_sequence=radar_colors)
            fig_bar.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT))
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    sec("PA Outcome Density (League context)")
    with st.expander("ℹ️ How to interpret these density charts?"):
        st.markdown("**What are we looking at?** These charts show the 'fingerprint' of a hitter's performance over the entire season.\n- **Shaded Grey Area**: The performance of the entire league. This is the benchmark.\n- **Colored Lines**: The selected players. \n- **Peak Height (Density)**: The taller the peak, the more 'typical' that performance level is for the player.\n- **Peak Position (X-axis)**: Further to the right means a higher average performance in that category.\n- **Spread (Width)**: A narrow, tall peak indicates a highly consistent player. A wide, flat peak indicates a 'streaky' player with high variance.")
    
    pm = [m for m in sel_metrics if m != "pa_count"]
    if not pm: st.info("Select a contact quality metric above to see the outcome density.")
    else:
        from scipy.stats import gaussian_kde
        for i in range(0, len(pm), 2):
            batch = pm[i : i + 2]; st_cols = st.columns(len(batch))
            for j, m_col in enumerate(batch):
                with st_cols[j]:
                    fig_d = go.Figure(); l_data = season_df[m_col].dropna()
                    if not l_data.empty:
                        l_xs = np.linspace(l_data.min(), l_data.max(), 200); l_kde = gaussian_kde(l_data, bw_method=0.3)(l_xs)
                        fig_d.add_trace(go.Scatter(x=l_xs, y=l_kde, fill='tozeroy', name="League", line_color=GREY, opacity=0.2))
                        for p_idx, p_name in enumerate(sel_players):
                            p_data = season_df[season_df["player_name"] == p_name][m_col].dropna()
                            if len(p_data) > 5:
                                p_kde = gaussian_kde(p_data, bw_method=0.4)(l_xs)
                                fig_d.add_trace(go.Scatter(x=l_xs, y=p_kde, name=p_name, line_color=radar_colors[p_idx % 3], line_width=3))
                    fig_d.update_layout(title=f"Distribution of {PA_LABELS[m_col]}", xaxis_title=PA_LABELS[m_col], yaxis_title="Density", height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT), margin=dict(t=40, b=40, l=40, r=40), showlegend=(j == 0))
                    st.plotly_chart(fig_d, use_container_width=True)
                    
                    l_mean = l_data.mean(); p_means = {p: season_df[season_df["player_name"] == p][m_col].mean() for p in sel_players}
                    p_stds = {p: season_df[season_df["player_name"] == p][m_col].std() for p in sel_players}
                    above = [p for p, m in p_means.items() if m > l_mean]; lead = max(p_means, key=p_means.get); cons = min(p_stds, key=p_stds.get) if len(sel_players) > 1 else None
                    msg = f"**{lead}** leads this group."
                    if len(above) == len(sel_players): msg += " All selected players are performing **above league average**."
                    elif len(above) > 0: msg += f" {', '.join(above)} are performing **above league average**."
                    if cons and len(sel_players) > 1: msg += f" **{cons}** shows the tallest peak, indicating the most consistency."
                    st.caption(f"💡 {msg}")



# PAGE 4: UNIVARIATE CHANGE ANALYZER

elif "Univariate Change Analyzer" in page:
    st.markdown("# 🔍 Pruned Exact Linear Time (PELT) algorithm for univariate structural break detection")
    st.markdown("Identify significant shifts in a player's performance profile. Select a player and season to see where their performance changed, then deep-dive into the data.")

    if 'pca_player' not in st.session_state: st.session_state.pca_player = "Trout, Mike" if "Trout, Mike" in players else players[2]
    if 'pca_year' not in st.session_state: st.session_state.pca_year = max_year
    if 'pca_selected_date' not in st.session_state: st.session_state.pca_selected_date = "-- Select Date --"

    p_yrs = sorted(df[df["player_name"] == st.session_state.pca_player]["year"].unique(), reverse=True)
    if st.session_state.pca_year not in p_yrs: st.session_state.pca_year = p_yrs[0]
    p_names = sorted(df[df["year"] == st.session_state.pca_year]["player_name"].dropna().unique())
    if st.session_state.pca_player not in p_names: st.session_state.pca_player = p_names[0]

    row1_c1, row1_c2, row1_c3 = st.columns([2, 1, 1])
    with row1_c1: sel_player = st.selectbox("Select Player", p_names, index=p_names.index(st.session_state.pca_player), key="pca_p_sel")
    with row1_c2: sel_year = st.selectbox("Season", p_yrs, index=p_yrs.index(st.session_state.pca_year), key="pca_y_sel")
    with row1_c3: st.markdown("<div style='height: 38px;'></div>", unsafe_allow_html=True); show_history = st.toggle("Full Career", value=False, key="pca_hist")

    row2_c1, row2_c2 = st.columns([1, 1])
    with row2_c1: cpd_window = st.slider("Rolling Window (PAs)", 20, 100, 50, 5, key="pca_window")
    with row2_c2: sensitivity = st.radio("CPD Sensitivity", ["Low","Medium","High"], index=1, horizontal=True, key="pca_sens")

    with st.expander("ℹ️ How do these controls work?"):
        st.markdown("**Rolling Window**: Smoothing level. A smaller window (20) catches quick shifts but is noisy. A larger window (100) identifies long-term structural changes.")
        st.markdown("**CPD Sensitivity**: The statistical threshold for flagging a shift. 'High' detects small fluctuations; 'Low' only flags major, sustained changes.")

    if sel_player != st.session_state.pca_player or sel_year != st.session_state.pca_year:
        st.session_state.pca_player = sel_player; st.session_state.pca_year = sel_year; st.rerun()

    st.markdown("---")
    p_idx = df_idx[df_idx["player_name"] == sel_player].copy()
    c_idx = p_idx if show_history else p_idx[p_idx["year"] == sel_year].copy()
    roll_idx = rolling_with_dates(c_idx, "perf_index", cpd_window)
    
    if len(roll_idx) < 20: st.warning("Insufficient data for Change Point Detection."); st.stop()

    cp_indices = detect_cpd(roll_idx["perf_index"], SENSITIVITY_MAP[sensitivity])
    
    # ── MAIN VISUALIZATION (PLOTLY) ───────────────────────────────────────────
    sec(f"Performance Index Trend — {sel_player}")
    st.info("💡 **Interactive Analysis:** Click on any **red diamond marker** to instantly analyze that performance shift in the deep-dive section below.")
    
    fig = go.Figure()
    
    # 1. Performance Index Line
    fig.add_trace(go.Scatter(
        x=roll_idx["game_date"], y=roll_idx["perf_index"], 
        mode='lines', name='Performance Index', 
        line=dict(color=TEAL, width=3),
        hoverinfo='skip'
    ))
    
    # 2. Interactive Shift Markers (Actual data points for reliable clicking)
    cp_dates_raw = [roll_idx["game_date"].iloc[cp] for cp in cp_indices]
    cp_vals = [roll_idx["perf_index"].iloc[cp] for cp in cp_indices]
    
    fig.add_trace(go.Scatter(
        x=cp_dates_raw, y=cp_vals,
        mode='markers',
        name='Detected Shift',
        marker=dict(size=12, color=RED, symbol='diamond', line=dict(width=2, color=DARK)),
        hovertemplate="<b>Detected Shift</b><br>Date: %{x}<br>Index: %{y:.2f}<extra></extra>"
    ))
    
    # 3. Reference Vertical Lines
    for cp_date in cp_dates_raw:
        fig.add_vline(x=cp_date, line_dash="dash", line_color=RED, line_width=1, opacity=0.5)

    # 4. Legend Entry
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=RED, width=2, dash='dash'), name='Significant Performance Shift'))

    fig.update_layout(
        xaxis_title="Game Date", yaxis_title="Performance Index (0-100)", 
        height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color=TEXT), margin=dict(t=20, b=20, l=20, r=20), 
        hovermode="x",
        clickmode='event+select',
        dragmode=False # Disable all drag behaviors to ensure clicks are primary
    )
    
    # Render and capture selection - HIDING THE TOOLBAR COMPLETELY
    # Use on_select to capture clicks
    selected_event = st.plotly_chart(
        fig, 
        use_container_width=True, 
        on_select="rerun",
        config={
            'displayModeBar': False,
            'staticPlot': False,
            'doubleClick': 'reset',
        }
    )
    
    # --- AUTOMATIC SELECTION LOGIC ---
    if selected_event and "selection" in selected_event:
        sel_points = selected_event["selection"].get("points", [])
        if sel_points:
            raw_x = sel_points[0]["x"]
            clicked_str = raw_x[:10] if isinstance(raw_x, str) else raw_x.strftime("%Y-%m-%d")
            
            if cp_dates_raw:
                target_dt = pd.to_datetime(clicked_str)
                nearest_cp = min(cp_dates_raw, key=lambda d: abs(d - target_dt))
                if abs((nearest_cp - target_dt).days) <= 7:
                    new_date = nearest_cp.strftime("%Y-%m-%d")
                    if st.session_state.pca_selected_date != new_date:
                        st.session_state.pca_selected_date = new_date
                        st.rerun() # Force immediate update of the deep-dive section

    # ── INTERACTIVE DRILLDOWN ─────────────────────────────────────────────────
    if cp_indices:
        cp_dates_str = [d.strftime("%Y-%m-%d") for d in cp_dates_raw]
        if st.session_state.pca_selected_date not in ["-- Select Date --"] + cp_dates_str:
            st.session_state.pca_selected_date = "-- Select Date --"
        
        if st.session_state.pca_selected_date != "-- Select Date --":
            selected_cp = st.session_state.pca_selected_date
            actual_cp_idx = cp_indices[cp_dates_str.index(selected_cp)]
            render_cp_analysis(selected_cp, sel_player, c_idx.iloc[max(0, actual_cp_idx - 50) : actual_cp_idx], c_idx.iloc[actual_cp_idx : min(len(c_idx), actual_cp_idx + 50)],importance_df=None  # PELT page — no feature importance
                               )
    else:
        st.info("No significant performance shifts detected with current settings.")




# PAGE 5: PERFORMANCE CHANGE ANALYZER

elif "Multivariate Change Analyzer" in page:
    st.markdown("# ChangeForest for capturing multivariate distributional shifts")
    st.markdown("Multivariate change point detection across all 4 indicators simultaneously.")

    if 'cf_player' not in st.session_state: st.session_state.cf_player = "Trout, Mike" if "Trout, Mike" in players else players[2]
    if 'cf_year' not in st.session_state: st.session_state.cf_year = max_year
    if 'cf_selected_date' not in st.session_state: st.session_state.cf_selected_date = "-- Select Date --"

    p_yrs = sorted(df[df["player_name"] == st.session_state.cf_player]["year"].unique(), reverse=True)
    if st.session_state.cf_year not in p_yrs: st.session_state.cf_year = p_yrs[0]
    p_names = sorted(df[df["year"] == st.session_state.cf_year]["player_name"].dropna().unique())
    if st.session_state.cf_player not in p_names: st.session_state.cf_player = p_names[0]

    row1_c1, row1_c2, row1_c3 = st.columns([2, 1, 1])
    with row1_c1: sel_player = st.selectbox("Select Player", p_names, index=p_names.index(st.session_state.cf_player), key="cf_p_sel")
    with row1_c2: sel_year = st.selectbox("Season", p_yrs, index=p_yrs.index(st.session_state.cf_year), key="cf_y_sel")
    with row1_c3: st.markdown("<div style='height: 38px;'></div>", unsafe_allow_html=True); show_history = st.toggle("Full Career", value=False, key="cf_hist")

    row2_c1, row2_c2 = st.columns([1, 1])
    with row2_c1: cpd_window = st.slider("Rolling Window (PAs)", 20, 100, 50, 5, key="cf_window")
    with row2_c2: sensitivity = st.radio("CPD Sensitivity", ["Low", "Medium", "High"], index=1, horizontal=True, key="cf_sens")

    if sel_player != st.session_state.cf_player or sel_year != st.session_state.cf_year:
        st.session_state.cf_player = sel_player; st.session_state.cf_year = sel_year; st.rerun()

    st.markdown("---")
    source_df = df if show_history else df[df["year"] == sel_year]

    with st.spinner("Preparing player dataset..."):
        subdf = build_cpd_subdf(source_df, sel_player, cpd_window)

    if subdf.empty or len(subdf) < 20:
        st.warning("Insufficient data for Change Point Detection. Try a smaller rolling window.")
        st.stop()

    with st.spinner("Running ChangeForest random-forest CPD..."):
        try:
            _, cps, _, feature_names = run_changeforest(
                subdf, cpd_window, SENSITIVITY_TO_MIN_SEG[sensitivity]
            )
        except Exception as exc:
            st.error(f"ChangeForest failed: {exc}")
            st.stop()

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Player", sel_player)
    c2.metric("PA Rows Used", f"{len(subdf):,}")
    c3.metric("Sensitivity", sensitivity)
    c4.metric("Detected CPs", str(len(cps)))

    if cps:
        with st.expander(f"📍 Change-point rows ({len(cps)} points)"):
            st.dataframe(
                subdf.iloc[cps][["cf_seq_id", "game_date", "pa_uid"]].reset_index(drop=True),
                use_container_width=True,
            )

    tab1, tab2, tab3 = st.tabs([
        "📈 ChangeForest Result + Input Signal",
        "📊 Before / After Eval",
        "🔧 Parameter Stability",
    ])

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        sec("ChangeForest Result")
        st.caption("4 rolling-mean signals on a shared timeline. Red dashed lines mark detected change points. Click a diamond to analyze that shift.")

        from plotly.subplots import make_subplots

        dates = pd.to_datetime(subdf["game_date"])
        cp_dates_raw = [subdf["game_date"].iloc[cp] for cp in cps]
        cp_dates_str = [d.strftime("%Y-%m-%d") for d in cp_dates_raw]

        fig_r = make_subplots(
            rows=5, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.23, 0.23, 0.23, 0.23, 0.08],
            subplot_titles=[PA_LABELS[col] for col in PA_INDICATORS] + [""]
        )

        for i, col in enumerate(PA_INDICATORS, start=1):
            y_col = f"{col}_rollmean_{cpd_window}"
            fig_r.add_trace(go.Scatter(
                x=dates, y=subdf[y_col],
                mode='lines', name=PA_LABELS[col],
                line=dict(color=PA_COLORS[col], width=2),
                hoverinfo='skip', showlegend=False,
            ), row=i, col=1)

        for cp_date in cp_dates_raw:
            for i in range(1, 6):
                fig_r.add_vline(
                    x=cp_date, line_dash="dot",
                    line_color=RED, line_width=1.5, opacity=0.7
                )

        if cps:
            fig_r.add_trace(go.Scatter(
                x=cp_dates_raw, y=[0] * len(cp_dates_raw),
                mode='markers', name='Click to Analyze',
                marker=dict(size=12, color=RED, symbol='diamond', line=dict(width=2, color=DARK)),
                hovertemplate="<b>Click to Analyze</b><br>Date: %{x}<extra></extra>",
            ), row=5, col=1)

        fig_r.update_yaxes(visible=False, row=5, col=1)
        fig_r.update_layout(
            height=750,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT),
            margin=dict(t=40, b=20, l=60, r=20),
            hovermode="x", clickmode='event+select', dragmode=False,
            title_text=f"{sel_player}  |  window={cpd_window}  |  CPs={len(cps)}",
            title_font=dict(color=TEXT, size=11),
        )
        fig_r.update_xaxes(tickformat="%Y-%m", tickangle=35)

        selected_event = st.plotly_chart(
            fig_r, use_container_width=True, on_select="rerun",
            config={'displayModeBar': False, 'staticPlot': False}
        )

        if selected_event and "selection" in selected_event:
            sel_points = selected_event["selection"].get("points", [])
            if sel_points:
                raw_x = sel_points[0]["x"]
                clicked_str = raw_x[:10] if isinstance(raw_x, str) else raw_x.strftime("%Y-%m-%d")
                if cp_dates_raw:
                    target_dt = pd.to_datetime(clicked_str)
                    nearest_cp = min(cp_dates_raw, key=lambda d: abs(d - target_dt))
                    if abs((nearest_cp - target_dt).days) <= 7:
                        new_date = nearest_cp.strftime("%Y-%m-%d")
                        if st.session_state.cf_selected_date != new_date:
                            st.session_state.cf_selected_date = new_date
                            st.rerun()

        if cps:
            if st.session_state.cf_selected_date not in ["-- Select Date --"] + cp_dates_str:
                st.session_state.cf_selected_date = "-- Select Date --"
            if st.session_state.cf_selected_date != "-- Select Date --":
                actual_cp_idx = cps[cp_dates_str.index(st.session_state.cf_selected_date)]
                importance_df = get_cp_feature_importance(subdf, actual_cp_idx, cpd_window)

                render_cp_analysis(
                    st.session_state.cf_selected_date, sel_player,
                    subdf.iloc[max(0, actual_cp_idx - 50) : actual_cp_idx],
                    subdf.iloc[actual_cp_idx : min(len(subdf), actual_cp_idx + 50)],importance_df=importance_df)
            else:
                st.info("💡 Click on any red diamond marker to analyze that shift.")
        else:
            st.info("No significant performance shifts detected with current settings.")
    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        sec("Before / After Statistical Comparison")
        st.caption("Detected CPs (Y) should show larger absolute shifts in mean and std than non-CP positions (N).")

        compare_window = max(5, min(50, len(subdf) // 4))

        if len(subdf) <= 2 * compare_window:
            st.warning(f"Not enough data for before/after evaluation (need > {2 * compare_window} rows, have {len(subdf)}).")
        else:
            with st.spinner("Computing before/after statistics..."):
                cps_set = set(cps)
                eval_dfs = {}
                for col in feature_names:
                    rows = []
                    for i in range(compare_window, len(subdf) - compare_window):
                        before = subdf[col].iloc[i - compare_window : i]
                        after  = subdf[col].iloc[i : i + compare_window]
                        rows.append({
                            "cp":            "Y" if i in cps_set else "N",
                            "abs_mean_diff": abs(after.mean() - before.mean()),
                            "abs_std_diff":  abs(after.std() - before.std()),
                        })
                    eval_dfs[col] = pd.DataFrame(rows)

            if eval_dfs:
                fig_e, axes = plt.subplots(1, len(eval_dfs), figsize=(5 * len(eval_dfs), 4), constrained_layout=True)
                fig_e.patch.set_facecolor(DARK)
                if len(eval_dfs) == 1: axes = [axes]
                for ax, (feat, df_eval) in zip(axes, eval_dfs.items()):
                    ax.set_facecolor(PANEL)
                    summary = df_eval.groupby("cp")[["abs_mean_diff", "abs_std_diff"]].mean()
                    summary.plot(kind="bar", ax=ax, rot=0, color=[TEAL, GOLD])
                    base = feat.split("_rollmean_")[0]
                    ax.set_title(PA_LABELS.get(base, feat), color=TEXT, fontsize=10, fontweight="bold")
                    ax.set_xlabel("Is Change Point?", color=TEXT_MUTED)
                    ax.set_ylabel("Avg Absolute Difference", color=TEXT_MUTED)
                    ax.tick_params(colors=TEXT)
                    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
                    ax.grid(axis="y", alpha=0.3, color=BORDER)
                    ax.legend(["Abs Mean Diff", "Abs Std Diff"], frameon=False, fontsize=9, labelcolor=TEXT)
                st.pyplot(fig_e, use_container_width=True); plt.close()

                with st.expander("📋 Detailed eval summary tables"):
                    for feat, df_eval in eval_dfs.items():
                        base = feat.split("_rollmean_")[0]
                        st.write(f"**{PA_LABELS.get(base, feat)}**")
                        st.dataframe(df_eval.groupby("cp")[["abs_mean_diff", "abs_std_diff"]].mean().round(4), use_container_width=True)

    # ── Tab 3 ─────────────────────────────────────────────────────────────────
    with tab3:
        sec("Parameter Stability")
        st.caption("Runs ChangeForest across all three sensitivity levels on the same player + window. Stable results → similar CP counts across levels.")

        with st.spinner("Running stability analysis across all sensitivity levels..."):
            stability_rows = []
            inv_map = {v: k for k, v in SENSITIVITY_TO_MIN_SEG.items()}
            for val in sorted(SENSITIVITY_TO_MIN_SEG.values()):
                try:
                    _, s_cps, _, _ = run_changeforest(subdf, cpd_window, val)
                    n_cp = len(s_cps)
                except Exception as exc:
                    n_cp = None
                    s_cps = []
                    st.warning(f"Stability run failed for min_rel_seg_len={val}: {exc}")
                stability_rows.append({
                    "Sensitivity": inv_map.get(val, str(val)),
                    "min_rel_seg_len": val,
                    "# Change Points": n_cp,
                    "CP Indices": str(s_cps),
                })
            stability_df = pd.DataFrame(stability_rows)

        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            fig_st, ax_st = plt.subplots(figsize=(7, 4), constrained_layout=True)
            fig_st.patch.set_facecolor(DARK); ax_st.set_facecolor(PANEL)
            valid = stability_df.dropna(subset=["# Change Points"])
            x_labels = valid["Sensitivity"] + "\n(min=" + valid["min_rel_seg_len"].astype(str) + ")"
            bars = ax_st.bar(x_labels, valid["# Change Points"], color=TEAL, alpha=0.82, width=0.5)
            for bar, val in zip(bars, valid["# Change Points"]):
                ax_st.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(int(val)), ha="center", va="bottom", fontsize=12, fontweight="bold", color=TEXT)
            ax_st.set_xlabel("Sensitivity", color=TEXT_MUTED)
            ax_st.set_ylabel("# Change Points", color=TEXT_MUTED)
            ax_st.tick_params(colors=TEXT)
            for sp in ax_st.spines.values(): sp.set_edgecolor(BORDER)
            ax_st.grid(axis="y", alpha=0.3, color=BORDER)
            ax_st.set_title("Change Points vs Sensitivity", color=TEXT, fontsize=11)
            st.pyplot(fig_st, use_container_width=True); plt.close()
        with col_table:
            st.dataframe(stability_df, use_container_width=True, hide_index=True)