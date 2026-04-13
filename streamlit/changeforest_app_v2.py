#!/usr/bin/env python
# coding: utf-8
"""
Diamond Insight: Performance Shift Analytics
============================================
Single dataset: PA-level engineered features
  pa_master.csv — one row per plate appearance, 420 qualified hitters, 2021-2025

Active Pages:
1. Welcome 2: Welcome Page
2. Player Snapshot
3. Peer Comparison
4. Performance Change Analyzer
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


# ══════════════════════════════════════════════════════════════════════════════
# MUST BE THE VERY FIRST STREAMLIT CALL
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Diamond Insight",
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
[data-testid="stSidebar"] {{ display: none; }}
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

/* Top Nav Styles */
.nav-logo {{ font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem; color: #111418; letter-spacing: 0.05em; }}
button[kind="secondary"] {{
    background: transparent !important;
    border: none !important;
    color: #586069 !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: all 0.2s ease !important;
}}
button[kind="secondary"]:hover {{
    color: #0969da !important;
    background: #f6f8fa !important;
}}
button[kind="primary"] {{
    background: transparent !important;
    border: none !important;
    color: #0969da !important;
    font-weight: 700 !important;
    padding: 0.5rem 1rem !important;
    border-bottom: 3px solid #58A6FF !important;
    border-radius: 0 !important;
}}
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

def games_label(n_pa):
    return f"~{max(1, round(n_pa / 3.8))} games"

def get_diagnostic_insight(stats_list, player_name):
    findings = []
    indicator_summary = {}
    for col, s in stats_list.items():
        d = s['effect_d']
        label = "Stable"
        if d > 0.5: label = "Large Improvement"
        elif d > 0.2: label = "Small Improvement"
        elif d < -0.5: label = "Large Decline"
        elif d < -0.2: label = "Small Decline"
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
        findings.append(f"🧘 <b>Passive Approach:</b> Discipline improved, but Power dropped. This often happens when a hitter becomes <i>too</i> selective, sacrificing aggression for better take decisions.")
    elif "Decline" in disc and "Improvement" in pwr:
        findings.append(f"⚔️ <b>Aggressive Shift:</b> Discipline dropped while Power rose. The hitter is likely 'selling out' for power, swinging harder at the cost of strike-zone control.")

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

def render_cp_analysis(selected_date, player_name, before_data, after_data):
    st.markdown("---")
    st.write(f"### 🔍 Deep-Dive Analysis: Shift on {selected_date}")
    st.write(f"Comparing the window of performance before and after this detected shift.")
    
    all_stats = {}
    for col in PA_INDICATORS:
        delta = after_data[col].mean() - before_data[col].mean()
        pooled = np.sqrt((before_data[col].std()**2 + after_data[col].std()**2) / 2 + 1e-9)
        d = delta / pooled
        all_stats[col] = {'delta': delta, 'effect_d': d, 'before': before_data[col].mean(), 'after': after_data[col].mean()}

    insights = get_diagnostic_insight(all_stats, player_name)
    st.markdown(f'<div class="narrative"><b>🧠 Smart Analyzer Hypothesis:</b><br><br>{"<br><br>".join(insights)}</div>', unsafe_allow_html=True)

    st.write("#### 📊 Key Metric Shifts")
    with st.expander("ℹ️ How are these shifts calculated?"):
        st.markdown("""
        - **Effect Size (Cohen’s d):** This measures the magnitude of the shift relative to the player's natural variability. 
            - **0.2:** Small shift (normal fluctuation).
            - **0.5:** Medium shift (visible performance change).
            - **0.8+:** Large shift (major mechanical or approach overhaul).
        - **Primary Driver:** We identify this by finding the metric with the **highest absolute Effect Size**. It represents the most statistically significant 'break' in the player's performance profile.
        """)
    
    cols = st.columns(4)
    for i, col_name in enumerate(PA_INDICATORS):
        s = all_stats[col_name]
        with cols[i]:
            st.metric(PA_LABELS[col_name], f"{s['after']:.3f}", delta=f"{s['delta']:+.3f}")
            st.caption(f"Effect Size: {s['effect_d']:.2f}")

    st.write("#### 📈 Distribution Shift (Primary Driver)")
    sorted_stats = sorted(all_stats.items(), key=lambda x: abs(x[1]['effect_d']), reverse=True)
    top_col = sorted_stats[0][0]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=before_data[top_col], name="Before Shift", marker_color=GREY, opacity=0.6))
    fig.add_trace(go.Histogram(x=after_data[top_col], name="After Shift", marker_color=TEAL, opacity=0.6))
    fig.update_layout(barmode='overlay', title=f"Change in {PA_LABELS[top_col]} Population", 
                      xaxis_title=PA_LABELS[top_col], yaxis_title="Frequency (PA Count)",
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT),
                      height=300, margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading PA-level data...")
def load_data() -> pd.DataFrame:
    path = "/tmp/Qualified_hitters_statcast_2021_2025_pa_master.csv"
    if not os.path.exists(path):
        gdown.download(f"https://drive.google.com/uc?id={DATA_CF_FILE_ID}", path, quiet=False, fuzzy=True)
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["year"]  = df["game_date"].dt.year
    df["month"] = df["game_date"].dt.month
    return df.sort_values(["player_name", "game_date"]).reset_index(drop=True)

df       = load_data()
players  = sorted(df["player_name"].dropna().unique())
min_year = int(df["year"].min())
max_year = int(df["year"].max())
df_idx   = build_perf_index(df)

# ══════════════════════════════════════════════════════════════════════════════
# TOP NAVIGATION BAR
# ══════════════════════════════════════════════════════════════════════════════
if 'nav_page' not in st.session_state: st.session_state.nav_page = "📖 Welcome"
t_col1, t_col2 = st.columns([1, 2])
with t_col1: st.markdown("<div class='nav-logo'>💎 DIAMOND INSIGHT</div>", unsafe_allow_html=True)
with t_col2:
    nav_cols = st.columns(4)
    nav_items = ["📖 Welcome", "🎯 Player Snapshot", "👥 Peer Comparison", "🔍 Change Analyzer"]
    for i, item in enumerate(nav_items):
        if st.button(item, key=f"top_nav_{item}", type="primary" if st.session_state.nav_page == item else "secondary"):
            st.session_state.nav_page = item; st.rerun()
st.markdown("---")
page = st.session_state.nav_page
players_with_all = ["All Players (League Avg)"] + players

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 - WELCOME
# ══════════════════════════════════════════════════════════════════════════════
if "Welcome" in page:
    st.markdown("# 📖 Diamond Insight — Welcome")
    st.markdown("### Advanced Hitter Performance Analytics through Statcast & Machine Learning")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎯 Why it Matters")
        st.markdown("Traditional stats like BA or OPS are lagging indicators. Diamond Insight identifies performance shifts the moment they happen by analyzing the physics of every plate appearance—separating luck from genuine skill changes.")
    with col2:
        st.markdown("#### 🚀 How to Use Diamond Insight")
        st.markdown("1. **Snapshot**: Start with a single player profile.\n2. **Peer Comparison**: Benchmark hitters side-by-side.\n3. **Change Analyzer**: Diagnose exactly when and why performance shifted.")
    st.markdown("---")
    st.markdown("#### 🔬 The Four Pillars of Performance")
    i1, i2 = st.columns(2)
    with i1:
        st.markdown(f'<div class="card-row">{card("Hitting Decisions", "Plate Discipline", "Quality of swing vs. take decisions.", "gold")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card-row">{card("wOBA Residual", "Luck vs Skill", "Matching performance to contact physics.", "teal")}</div>', unsafe_allow_html=True)
    with i2:
        st.markdown(f'<div class="card-row">{card("Power Efficiency", "Raw Power", "Converting swing effort to ball impact.", "teal")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card-row">{card("Launch Angle Stability", "Swing Consistency", "Repeatability of ball flight path.", "gold")}</div>', unsafe_allow_html=True)
    st.markdown("---")
    sec("📊 Dataset Overview")
    st.markdown(f'<div class="card-row">{card("Hitters", f"{len(players):,}", "qualified players", "gold")}{card("Total Records", f"{len(df):,}", "plate appearances", "teal")}{card("Season Span", str(df.year.nunique()), f"{min_year}-{max_year}", "grey")}{card(f"{max_year} Data", f"{len(df[df.year==max_year]):,}", "current season", "teal")}</div>', unsafe_allow_html=True)
    sec("PA Indicator Distributions — League Benchmark")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), constrained_layout=True)
    fig.patch.set_facecolor(DARK)
    for ax, col in zip(axes, PA_INDICATORS):
        data = df[col].dropna(); ax.set_facecolor(PANEL); ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.hist(data, bins=60, color=PA_COLORS[col], alpha=0.8); ax.axvline(data.mean(), color=GOLD, lw=1.5, ls="--")
        ax.set_title(PA_LABELS[col], color=TEXT, fontsize=9, fontweight="bold")
        ax.grid(color=BORDER, linewidth=0.3, linestyle="--", alpha=0.4)
    st.pyplot(fig); plt.close()
    st.markdown(f'<div class="narrative"><b>📊 Distribution Analysis:</b> Indicators show league-wide "normal" performance. Decisions are centered on zero, while Power shows a "long tail" of elite hitters. Residual confirms results align with physics over large samples.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 - PLAYER SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════
elif "Snapshot" in page:
    st.markdown("# 🎯 Player Snapshot")
    if 'snapshot_player' not in st.session_state: st.session_state.snapshot_player = "All Players (League Avg)"
    if 'snapshot_year' not in st.session_state: st.session_state.snapshot_year = max_year
    if st.session_state.snapshot_player == "All Players (League Avg)": av_yrs = sorted(df.year.unique(), reverse=True)
    else: av_yrs = sorted(df[df.player_name == st.session_state.snapshot_player].year.unique(), reverse=True)
    if st.session_state.snapshot_year not in av_yrs: st.session_state.snapshot_year = av_yrs[0]
    av_p_yr = sorted(df[df.year == st.session_state.snapshot_year].player_name.dropna().unique())
    p_opts = ["All Players (League Avg)"] + av_p_yr
    r1c1, r1c2 = st.columns([3, 1])
    with r1c1: sel_p = st.selectbox("Select Player", p_opts, index=p_opts.index(st.session_state.snapshot_player), key="ps_p")
    with r1c2: sel_y = st.selectbox("Season", av_yrs, index=av_yrs.index(st.session_state.snapshot_year), key="ps_y")
    if sel_p != st.session_state.snapshot_player or sel_y != st.session_state.snapshot_year:
        st.session_state.snapshot_player = sel_p; st.session_state.snapshot_year = sel_y; st.rerun()
    is_all = (sel_p == "All Players (League Avg)")
    snap_df = df[df.year == sel_y].copy() if is_all else df[(df.player_name == sel_p) & (df.year == sel_y)].copy()
    def _m(col): return snap_df[col].dropna().mean() if not snap_df[col].empty else 0.0
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hitting Decisions", f"{_m('hitting_decisions_score'):.2f}", help=PA_TOOLTIPS["hitting_decisions_score"])
    c2.metric("Power Efficiency", f"{_m('power_efficiency'):.4f}", help=PA_TOOLTIPS["power_efficiency"])
    c3.metric("wOBA Residual", f"{_m('woba_residual'):.3f}", help=PA_TOOLTIPS["woba_residual"])
    c4.metric("Launch Angle Stab.", f"{_m('launch_angle_stability_50pa'):.2f}", help=PA_TOOLTIPS["launch_angle_stability_50pa"])
    c5.metric("PAs this Season", f"{len(snap_df):,}", help="Statistical reliability check.")

    if not is_all:
        s_df = df[df.year == sel_y]
        def get_ins(col, val):
            lm, ls = s_df[col].mean(), s_df[col].std()
            if val > lm + ls: return "Elite", "Significantly better than average."
            if val > lm: return "Above Average", "Better than most of the league."
            if val < lm - ls: return "Struggling", "Well below benchmark."
            return "Average", "In line with league average."
        st.markdown("---")
        st.markdown("---")
        sec(f"📊 {sel_y} Performance Spectrum")
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
            s_min, s_max = df[df.year==sel_y][col].quantile(0.01), df[df.year==sel_y][col].quantile(0.99)
            return min(100, max(0, (val - s_min) / (s_max - s_min + 1e-9) * 100))

        # 1. Discipline
        status, desc = get_ins('hitting_decisions_score', _m('hitting_decisions_score'))
        p = get_pct('hitting_decisions_score', _m('hitting_decisions_score'))
        insight = f"<b>{sel_p}</b> is {status.lower()} in plate discipline. Sitting near the ceiling indicates elite pitch recognition and strike-zone mastery, leading to more walks and fewer 'chase' strikeouts."
        st.markdown(spectrum_row("🎯", "Discipline", status, p, "#0969da", insight), unsafe_allow_html=True)

        # 2. Power
        status, desc = get_ins('power_efficiency', _m('power_efficiency'))
        p = get_pct('power_efficiency', _m('power_efficiency'))
        insight = f"Currently graded as <b>{status.lower()}</b>. High power efficiency means the hitter is maximizing exit velocity relative to swing effort—a key indicator of a 'heavy' bat and efficient mechanics."
        st.markdown(spectrum_row("💥", "Power", status, p, "#58A6FF", insight), unsafe_allow_html=True)

        # 3. Results vs Physics
        wr = _m('woba_residual')
        status = "Steady" if abs(wr)<0.15 else ("Outpacing Physics" if wr>0.15 else "Unlucky")
        p = get_pct('woba_residual', wr)
        insight = f"Status: <b>{status}</b>. Sitting far to the right (Ceiling) suggests results are currently better than the physics of the contact would predict (skill/luck), while the left (Min) indicates hard-luck regression is coming."
        st.markdown(spectrum_row("🎲", "Results", status, p, "#855D00", insight), unsafe_allow_html=True)

        # 4. Consistency
        status, desc = get_ins('launch_angle_stability_50pa', _m('launch_angle_stability_50pa'))
        p = get_pct('launch_angle_stability_50pa', _m('launch_angle_stability_50pa'))
        insight = f"Graded as <b>{status.lower()}</b>. Consistency near the ceiling indicates a highly repeatable swing path, making the hitter less vulnerable to timing-related slumps and cold streaks."
        st.markdown(spectrum_row("📈", "Consistency", status, p, "#2ca02c", insight), unsafe_allow_html=True)

    st.markdown("---")
    v1, v2 = st.columns(2)
    with v1:
        sec("Player Profile (Radar)"); fig = go.Figure()
        theta = [PA_LABELS[c] for c in PA_INDICATORS]
        r_p = [(_m(c)-df[df.year==sel_y][c].min())/(df[df.year==sel_y][c].max()-df[df.year==sel_y][c].min()+1e-9) for c in PA_INDICATORS]
        fig.add_trace(go.Scatterpolar(r=r_p+[r_p[0]], theta=theta+[theta[0]], fill='toself', name=sel_p, line_color=TEAL))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT))
        st.plotly_chart(fig, use_container_width=True)
    with v2:
        sec("Percentile Rankings")
        if is_all: pcts = [50]*4
        else:
            p_m = df[df.year==sel_y].groupby("player_name")[PA_INDICATORS].mean()
            pcts = [(p_m[c] < _m(c)).mean()*100 for c in PA_INDICATORS]
        fig = go.Figure(go.Bar(x=pcts, y=theta, orientation='h', marker=dict(color=pcts, colorscale='RdYlGn', cmin=0, cmax=100)))
        fig.update_layout(xaxis=dict(range=[0, 100]), yaxis=dict(autorange="reversed"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    sec("Contact Quality Profile (Power vs Luck)")
    fig_scatter = px.scatter(
        snap_df.dropna(subset=["power_efficiency", "woba_residual"]),
        x="power_efficiency", y="woba_residual", color_discrete_sequence=[TEAL], opacity=0.4,
        labels={"power_efficiency": "Power Efficiency", "woba_residual": "wOBA Residual"},
        title=f"Distribution for {sel_p if not is_all else 'League'}"
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color=GREY)
    fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT), height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    if not is_all:
        sec("Season-by-Season Progression")
        summ = df[df["player_name"] == sel_p].groupby("year").agg(
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
            
            initial_arch = get_archetype(summ.iloc[0])
            evolution_story.append(f"🌱 <b>The Foundation ({first_year}):</b> {sel_p} entered the dataset as a <b>{initial_arch}</b>. Early data established a baseline {summ.iloc[0]['hitting_decisions']:.2f} Discipline and {summ.iloc[0]['power_efficiency']:.4f} Power.")

            if len(summ) > 2:
                growth_notes = []
                for i in range(1, len(summ) - 1):
                    y = int(summ.iloc[i]["Season"])
                    prev, curr = summ.iloc[i-1], summ.iloc[i]
                    if curr['power_efficiency'] > prev['power_efficiency'] * 1.1: growth_notes.append(f"a power surge in {y}")
                    if curr['hitting_decisions'] > prev['hitting_decisions'] + 0.5: growth_notes.append(f"improved strike-zone mastery in {y}")
                if growth_notes:
                    evolution_story.append(f"📈 <b>Growth & Maturation:</b> Middle seasons were defined by {', '.join(growth_notes)}. This period saw the player refine their approach and maximize their physical tools.")

            latest_arch = get_archetype(summ.iloc[-1])
            evolution_story.append(f"🏁 <b>Current Profile ({last_year}):</b> Today, {sel_p} has evolved into a <b>{latest_arch}</b>. The data shows a career-to-date trend of {'improving' if summ['power_efficiency'].iloc[-1] > summ['power_efficiency'].iloc[0] else 'shifting'} performance profiles as they adapt to league adjustments.")

            st.markdown(f'<div class="narrative"><b>📊 Career Evolution: {sel_p} ({first_year}–{last_year})</b><br><br>{"<br><br>".join(evolution_story)}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 - PEER COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif "Peer" in page:
    st.markdown("# 👥 Peer Comparison")
    if 'peer_year' not in st.session_state: st.session_state.peer_year = max_year
    if 'peer_players' not in st.session_state: st.session_state.peer_players = ["Trout, Mike", "Ohtani, Shohei"] if "Trout, Mike" in players else []
    st.markdown("<style>span[data-baseweb='tag'] { background-color: #E8F0FE !important; color: #1A73E8 !important; border: 1px solid #D2E3FC !important; }</style>", unsafe_allow_html=True)
    r1c1, r1c2 = st.columns([1, 3])
    with r1c1: sel_y = st.selectbox("Season", sorted(df.year.unique(), reverse=True), index=sorted(df.year.unique(), reverse=True).index(st.session_state.peer_year), key="pc_y")
    p_in_s = sorted(df[df.year == sel_y].player_name.dropna().unique())
    with r1c2: sel_p = st.multiselect("Select Players", p_in_s, default=[p for p in st.session_state.peer_players if p in p_in_s][:3], max_selections=3, key="pc_p")
    opts = {**PA_LABELS, "pa_count": "PAs this Season"}
    sel_m = st.multiselect("Select Metrics", list(opts.keys()), default=list(PA_LABELS.keys())[:3], format_func=lambda x: opts[x])
    if sel_y != st.session_state.peer_year or sel_p != st.session_state.peer_players:
        st.session_state.peer_year = sel_y; st.session_state.peer_players = sel_p; st.rerun()
    if not sel_p: st.warning("Select players."); st.stop()
    s_df = df[df.year == sel_y]; radar_colors = [TEAL, GOLD, RED]
    comp_data = []
    for p in sel_p:
        pdf = s_df[s_df.player_name == p]
        d = {"Player": p, "pa_count": len(pdf)}
        for c in PA_INDICATORS: d[c] = pdf[c].mean(); d[f"{c}_std"] = pdf[c].std()
        comp_data.append(d)
    comp_df = pd.DataFrame(comp_data)
    def p_card(n, v, idx): return f'<div style="padding: 8px 12px; background: {PANEL}; border-radius: 4px; margin-bottom: 8px; border: 1px solid {BORDER}; border-left: 4px solid {radar_colors[idx%3]};"><div style="font-size: 0.7rem; text-transform: uppercase; color: {TEXT_MUTED}; font-weight: 600;">{n}</div><div style="font-size: 1.3rem; font-weight: 700; color: {TEXT}; font-family: \'Bebas Neue\', sans-serif;">{v}</div></div>'
    st.markdown("---")
    ov1, ov2 = st.columns([1, 1])
    with ov1:
        sec(f"📊 Overview — {sel_y}")
        for mk in sel_m:
            st.markdown(f"<div style='margin-top: 12px; color: {TEAL_LT}; font-weight: 700; font-size: 0.85rem;'>{opts[mk].upper()}</div>", unsafe_allow_html=True)
            cols = st.columns(max(len(sel_p), 3))
            for i, ps in enumerate(comp_data):
                v_s = f"{int(ps[mk]):,}" if mk=="pa_count" else f"{ps[mk]:.{(4 if mk=='power_efficiency' else (3 if mk=='woba_residual' else 2))}f}"
                with cols[i]: st.markdown(p_card(ps["Player"], v_s, i), unsafe_allow_html=True)
    with ov2:
        sec("💡 Insights")
        pi = []
        if len(sel_p) > 1:
            m_ctx = {"hitting_decisions_score": {"w": "superior discipline.", "i": "Higher OBP floor."}, "power_efficiency": {"w": "better energy transfer.", "i": "Sustainable power."}, "woba_residual": {"w": "maximizing physics.", "i": "High batted ball skill."}, "launch_angle_stability_50pa": {"w": "repeatable swing path.", "i": "Slump-resistant."}}
            for mk in [m for m in sel_m if m != "pa_count"]:
                if mk in m_ctx:
                    leader = comp_df.loc[comp_df[mk].idxmax(), "Player"]
                    pi.append(f"🏆 <b>{opts[mk]}:</b> <b>{leader}</b> ({comp_df[mk].max():.3f})<br><i>{m_ctx[mk]['w']} {m_ctx[mk]['i']}</i>")
        st.markdown(f'<div class="narrative" style="font-size: 0.85rem;">{"<br><br>".join(pi if pi else ["Add players/metrics."])}</div>', unsafe_allow_html=True)
    st.markdown("---")
    sec("PA Outcome Density"); from scipy.stats import gaussian_kde
    pm = [m for m in sel_m if m != "pa_count"]
    for i in range(0, len(pm), 2):
        btch = pm[i:i+2]; stc = st.columns(len(btch))
        for j, mc in enumerate(btch):
            with stc[j]:
                fig = go.Figure(); ld = s_df[mc].dropna()
                if not ld.empty:
                    lx = np.linspace(ld.min(), ld.max(), 200); lk = gaussian_kde(ld, bw_method=0.3)(lx)
                    fig.add_trace(go.Scatter(x=lx, y=lk, fill='tozeroy', name="League", line_color=GREY, opacity=0.2))
                    for k, p in enumerate(sel_p):
                        pd_ = s_df[s_df.player_name == p][mc].dropna()
                        if len(pd_) > 5: px_ = np.linspace(ld.min(), ld.max(), 200); pk = gaussian_kde(pd_, bw_method=0.4)(px_); fig.add_trace(go.Scatter(x=px_, y=pk, name=p, line_color=radar_colors[k%3], line_width=3))
                fig.update_layout(title=f"{opts[mc]} Distribution", height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT), margin=dict(t=40,b=40,l=40,r=40), showlegend=(j==0))
                st.plotly_chart(fig, use_container_width=True)
                lead = max({p: s_df[s_df.player_name==p][mc].mean() for p in sel_p}, key={p: s_df[s_df.player_name==p][mc].mean() for p in sel_p}.get)
                st.caption(f"💡 **{lead}** leads this group in {opts[mc]}.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 - PERFORMANCE CHANGE ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
elif "Change Analyzer" in page:
    st.markdown("# 🔍 Performance Change Analyzer")
    if 'pca_p' not in st.session_state: st.session_state.pca_p = "Trout, Mike" if "Trout, Mike" in players else players[0]
    if 'pca_y' not in st.session_state: st.session_state.pca_y = max_year
    if 'pca_selected_v2_date' not in st.session_state: st.session_state.pca_selected_v2_date = "-- Select Date --"
    p_yrs = sorted(df[df.player_name == st.session_state.pca_p].year.unique(), reverse=True)
    if st.session_state.pca_y not in p_yrs: st.session_state.pca_y = p_yrs[0]
    p_n = sorted(df[df.year == st.session_state.pca_y].player_name.dropna().unique())
    if st.session_state.pca_p not in p_n: st.session_state.pca_p = p_n[0]
    r1c1, r1c2, r1c3 = st.columns([2, 1, 1])
    with r1c1: sel_p = st.selectbox("Select Player", p_n, index=p_n.index(st.session_state.pca_p), key="pca_p_sel")
    with r1c2: sel_y = st.selectbox("Season", p_yrs, index=p_yrs.index(st.session_state.pca_y), key="pca_y_sel")
    with r1c3: st.markdown("<div style='height: 38px;'></div>", unsafe_allow_html=True); show_h = st.toggle("Full Career", value=False, key="pca_h")
    r2c1, r2c2 = st.columns([1, 1])
    with r2c1: c_w = st.slider("Rolling Window (PAs)", 20, 100, 50, 5, key="pca_w")
    with r2c2: sens = st.radio("CPD Sensitivity", ["Low","Medium","High"], index=1, horizontal=True, key="pca_s")
    if sel_p != st.session_state.pca_p or sel_y != st.session_state.pca_y:
        st.session_state.pca_p = sel_p; st.session_state.pca_y = sel_y; st.rerun()
    st.markdown("---")
    p_i = df_idx[df_idx.player_name == sel_p].copy()
    c_i = p_i if show_h else p_i[p_i.year == sel_y].copy()
    roll_i = rolling_with_dates(c_i, "perf_index", c_w)
    if len(roll_i) < 20: st.warning("Insufficient data."); st.stop()

    # ── MAIN VISUALIZATION (PLOTLY) ───────────────────────────────────────────
    sec(f"Performance Index Trend — {sel_p}")
    st.info("💡 **Interactive Analysis:** Click on any **red diamond marker** to instantly analyze that performance shift in the deep-dive section below.")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roll_i.game_date, y=roll_i.perf_index, mode='lines', name='Performance Index', line=dict(color=TEAL, width=3), hoverinfo='skip'))
    
    cp_dates_raw = [roll_i.game_date.iloc[cp] for cp in cp_idx]
    cp_vals = [roll_i.perf_index.iloc[cp] for cp in cp_idx]
    fig.add_trace(go.Scatter(x=cp_dates_raw, y=cp_vals, mode='markers', name='Detected Shift', marker=dict(size=12, color=RED, symbol='diamond', line=dict(width=2, color=DARK)), hovertemplate="<b>Detected Shift</b><br>Date: %{x}<extra></extra>"))
    
    for cp_date in cp_dates_raw:
        fig.add_vline(x=cp_date, line_dash="dash", line_color=RED, line_width=1, opacity=0.5)
    
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=RED, width=2, dash='dash'), name='Significant Performance Shift'))

    fig.update_layout(xaxis_title="Date", yaxis_title="Index (0-100)", height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT), margin=dict(t=20,b=20,l=20,r=20), hovermode="x", clickmode='event+select', dragmode=False)
    
    sel_ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun", config={'displayModeBar': False})
    if sel_ev and "selection" in sel_ev:
        pts = sel_ev["selection"].get("points", [])
        if pts:
            raw_x = pts[0]["x"]
            clk_d_s = raw_x[:10] if isinstance(raw_x, str) else raw_x.strftime("%Y-%m-%d")
            if cp_dates_raw:
                target = pd.to_datetime(clk_d_s)
                near_cp = min(cp_dates_raw, key=lambda d: abs(d - target))
                if abs((near_cp - target).days) <= 7:
                    new_date = near_cp.strftime("%Y-%m-%d")
                    if st.session_state.pca_selected_v2_date != new_date:
                        st.session_state.pca_selected_v2_date = new_date
                        st.rerun()

    if cp_idx:
        cp_dates_str = [d.strftime("%Y-%m-%d") for d in cp_dates_raw]
        if st.session_state.pca_selected_v2_date not in ["-- Select Date --"] + cp_dates_str:
            st.session_state.pca_selected_v2_date = "-- Select Date --"
        
        if st.session_state.pca_selected_v2_date != "-- Select Date --":
            sel_cp = st.session_state.pca_selected_v2_date
            actual_idx = cp_idx[cp_dates_str.index(sel_cp)]
            render_cp_analysis(sel_cp, sel_p, c_i.iloc[max(0, actual_idx-50):actual_idx], c_i.iloc[actual_idx:min(len(c_i), actual_idx+50)])
    else: st.info("No shifts detected.")
