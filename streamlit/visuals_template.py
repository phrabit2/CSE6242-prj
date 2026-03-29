import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Change Point Detection",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ──────────────────────────────────────────────────────────────────────
DARK       = "#12172b"
PANEL      = "#1a2238"
PURPLE     = "#6c4fa3"
PURPLE_LT  = "#9b7fd4"
GOLD       = "#e8b84b"
TEAL       = "#3eb8c0"
RED        = "#e05c72"
TEXT       = "#e8e4f0"
MUTED      = "#5a647a"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono&display=swap');
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; background-color: {DARK}; color: {TEXT}; }}
h1,h2,h3 {{ color: {TEXT}; }}
.block-container {{ padding-top: 1.5rem; background-color: {DARK}; }}
[data-testid="stSidebar"] {{ background-color: {PANEL}; }}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
.stRadio label, .stSelectbox label, .stMultiSelect label {{ color: {TEXT} !important; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.07em; }}
.metric-row {{ display:flex; gap:12px; margin-bottom:1.2rem; }}
.metric-box {{ background:{PANEL}; border-left:3px solid {PURPLE}; border-radius:6px; padding:0.8rem 1.1rem; flex:1; }}
.metric-box.gold {{ border-left-color:{GOLD}; }}
.metric-box.teal {{ border-left-color:{TEAL}; }}
.metric-box.red  {{ border-left-color:{RED}; }}
.metric-label {{ font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:{MUTED}; }}
.metric-val {{ font-size:1.7rem; font-weight:700; color:{TEXT}; line-height:1.15; }}
.metric-delta {{ font-size:0.78rem; color:{MUTED}; }}
.section-title {{ font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; color:{PURPLE_LT}; border-bottom:1px solid {PANEL}; padding-bottom:4px; margin-bottom:12px; margin-top:1rem; }}
.cpd-badge {{ display:inline-block; background:{PURPLE}22; border:1px solid {PURPLE}; color:{PURPLE_LT}; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-family:'DM Mono',monospace; }}
</style>
""", unsafe_allow_html=True)

# ── Data & CPD ─────────────────────────────────────────────────────────────────
S3_BUCKET = "team26-cpd-data-294342039761"
S3_KEY = "data/processed/qualified_hitters_statcast_2021_2025_batted_ball.csv"

@st.cache_data
def load_data():
    import boto3
    from io import StringIO
    s3 = boto3.client("s3", region_name="ap-northeast-2")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
    df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")), low_memory=False)
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['Name', 'game_date']).reset_index(drop=True)
    return df

@st.cache_data
def compute_cpd(df, player, metric, window=30, penalty=3):
    """Detect dominant change point via PELT (ruptures) or variance fallback."""
    sub = df[df['Name'] == player][['game_date', metric]].dropna().sort_values('game_date')
    if len(sub) < window * 2:
        return None, sub
    roll = sub[metric].rolling(window, min_periods=window//2).mean().dropna()
    dates = sub['game_date'].iloc[len(sub) - len(roll):]

    try:
        import ruptures as rpt
        signal = roll.values.reshape(-1, 1)
        algo = rpt.Pelt(model="rbf").fit(signal)
        bkps = algo.predict(pen=penalty)
        idx = bkps[-2] if len(bkps) >= 2 else bkps[0] // 2
    except ImportError:
        rsd = roll.rolling(window, center=True).std()
        idx = int(rsd.idxmax()) if not rsd.isna().all() else len(roll) // 2
        idx = min(max(idx, 0), len(roll) - 1)

    cpd_date = dates.iloc[min(idx, len(dates)-1)]
    return cpd_date, pd.DataFrame({'game_date': dates.values, metric: roll.values})

def mpl_fig(w=10, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(color=MUTED, linewidth=0.3, linestyle='--', alpha=0.4)
    return fig, ax

def metric_box(label, val, delta="", color="purple"):
    cls = {"purple":"metric-box","gold":"metric-box gold","teal":"metric-box teal","red":"metric-box red"}[color]
    return f"""<div class="{cls}">
        <div class="metric-label">{label}</div>
        <div class="metric-val">{val}</div>
        <div class="metric-delta">{delta}</div>
    </div>"""

# ── Load ───────────────────────────────────────────────────────────────────────
df = load_data()
players = sorted(df['Name'].dropna().unique())
METRICS = {
    'Exit Velocity': 'exit_velocity',
    'Launch Angle':  'launch_angle_metric',
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚾ CPD Dashboard")
    st.markdown("---")
    page = st.radio("", [
        "Overview",
        "Trajectory & CPD",
        "Before / After",
        "Two-Path Distribution",
        "Peer Comparison",
    ], label_visibility="collapsed")
    st.markdown("---")
    sel_player = st.selectbox("Player", players)
    sel_metric_label = st.selectbox("Metric", list(METRICS.keys()))
    sel_metric = METRICS[sel_metric_label]
    cpd_window = st.slider("Rolling window (pitches)", 15, 60, 30, 5)
    st.markdown("---")
    st.caption(f"{len(df[df['Name']==sel_player]):,} batted ball events for {sel_player}")

cpd_date, roll_df = compute_cpd(df, sel_player, sel_metric, window=cpd_window)
player_df = df[df['Name'] == sel_player].copy()
raw = player_df[['game_date', sel_metric]].dropna().sort_values('game_date')

if cpd_date is not None:
    before = raw[raw['game_date'] < cpd_date][sel_metric]
    after  = raw[raw['game_date'] >= cpd_date][sel_metric]
else:
    mid    = raw['game_date'].median()
    before = raw[raw['game_date'] < mid][sel_metric]
    after  = raw[raw['game_date'] >= mid][sel_metric]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown(f"# {sel_player}")
    st.markdown('<div class="section-title">Batted ball profile · Change Point Detection</div>', unsafe_allow_html=True)
    if cpd_date is not None:
        st.markdown(f'Detected change point: <span class="cpd-badge">{cpd_date.strftime("%b %d, %Y")}</span>', unsafe_allow_html=True)
    st.markdown("")

    ev_mean = player_df['exit_velocity'].dropna().mean()
    la_mean = player_df['launch_angle_metric'].dropna().mean()
    b_mean  = before.mean() if len(before) else 0
    a_mean  = after.mean()  if len(after)  else 0
    delta   = a_mean - b_mean
    delta_str = f"{'▲' if delta >= 0 else '▼'} {abs(delta):.1f} after CPD"

    st.markdown(f"""<div class="metric-row">
        {metric_box("Avg Exit Velocity",  f"{ev_mean:.1f} mph", f"{player_df['exit_velocity'].dropna().shape[0]} pitches", "gold")}
        {metric_box("Avg Launch Angle",   f"{la_mean:.1f}°",    f"{len(player_df):,} batted balls", "teal")}
        {metric_box(f"Pre-CPD {sel_metric_label}",  f"{b_mean:.1f}", f"{len(before)} events", "purple")}
        {metric_box(f"Post-CPD {sel_metric_label}", f"{a_mean:.1f}", delta_str, "red" if delta < 0 else "teal")}
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Exit Velocity vs Launch Angle</div>', unsafe_allow_html=True)
    plot_df = player_df[['exit_velocity','launch_angle_metric','Season']].dropna()
    fig, ax = mpl_fig(10, 4)
    seasons  = sorted(plot_df['Season'].unique())
    pal      = [PURPLE_LT, GOLD, TEAL, RED, "#a8d8a8"]
    for i, s in enumerate(seasons):
        sub = plot_df[plot_df['Season'] == s]
        ax.scatter(sub['exit_velocity'], sub['launch_angle_metric'],
                   color=pal[i % len(pal)], alpha=0.35, s=12, label=str(s))
    ax.axhspan(8, 32, alpha=0.07, color=GOLD, zorder=0)
    ax.axvline(95, color=MUTED, linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_xlabel("Exit Velocity (mph)")
    ax.set_ylabel("Launch Angle (°)")
    ax.legend(fontsize=8, framealpha=0.3, facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Season Averages</div>', unsafe_allow_html=True)
    summ = player_df.groupby('Season').agg(
        exit_velocity=('exit_velocity','mean'),
        launch_angle=('launch_angle_metric','mean'),
        batted_balls=('exit_velocity','count'),
    ).round(2).reset_index()
    st.dataframe(summ, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Trajectory & CPD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Trajectory & CPD":
    st.markdown("# Trajectory & Change Point")
    st.markdown(f'<div class="section-title">{sel_player} · {sel_metric_label} · rolling {cpd_window}-pitch avg</div>', unsafe_allow_html=True)
    if cpd_date:
        st.markdown(f'Change point: <span class="cpd-badge">{cpd_date.strftime("%b %d, %Y")}</span>', unsafe_allow_html=True)
    st.markdown("")

    fig, ax = mpl_fig(12, 4.2)
    ax.scatter(raw['game_date'], raw[sel_metric], color=MUTED, alpha=0.18, s=6, zorder=1)
    if len(roll_df):
        ax.plot(roll_df['game_date'], roll_df[sel_metric],
                color=PURPLE_LT, linewidth=2, zorder=3, label=f"{cpd_window}-pitch rolling avg")
    if cpd_date is not None:
        ax.axvline(cpd_date, color=GOLD, linewidth=2, linestyle='--', zorder=4,
                   label=f"CPD: {cpd_date.strftime('%b %Y')}")
        ymin, ymax = ax.get_ylim()
        ax.fill_betweenx([ymin, ymax], raw['game_date'].min(), cpd_date, alpha=0.05, color=TEAL)
        ax.fill_betweenx([ymin, ymax], cpd_date, raw['game_date'].max(), alpha=0.05, color=PURPLE)
    ax.set_xlabel("Date"); ax.set_ylabel(sel_metric_label)
    ax.legend(fontsize=9, framealpha=0.3, facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Before CPD</div>', unsafe_allow_html=True)
        st.markdown(metric_box("Mean",   f"{before.mean():.2f}", f"{len(before)} pitches", "teal"), unsafe_allow_html=True)
        st.markdown(metric_box("Std Dev",f"{before.std():.2f}",  "",                       "purple"), unsafe_allow_html=True)
        st.markdown(metric_box("Median", f"{before.median():.2f}","",                      "purple"), unsafe_allow_html=True)
    with col2:
        delta = after.mean() - before.mean()
        st.markdown('<div class="section-title">After CPD</div>', unsafe_allow_html=True)
        st.markdown(metric_box("Mean",   f"{after.mean():.2f}", f"{'▲' if delta>=0 else '▼'} {abs(delta):.2f}", "gold" if delta>=0 else "red"), unsafe_allow_html=True)
        st.markdown(metric_box("Std Dev",f"{after.std():.2f}",  "", "purple"), unsafe_allow_html=True)
        st.markdown(metric_box("Median", f"{after.median():.2f}","","purple"), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Before / After Grid  (Slide 5 style)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Before / After":
    st.markdown("# Before / After CPD")
    st.markdown(f'<div class="section-title">{sel_player} · all metrics</div>', unsafe_allow_html=True)
    if cpd_date:
        st.markdown(f'Change point: <span class="cpd-badge">{cpd_date.strftime("%b %d, %Y")}</span>', unsafe_allow_html=True)
    st.markdown("")

    all_metrics = {k: v for k, v in {
        'Exit Velocity': 'exit_velocity',
        'Launch Angle':  'launch_angle_metric',
        'xwOBA':         'xwoba_est',
        'Hit Distance':  'hit_distance',
    }.items() if v in player_df.columns}

    fig = plt.figure(figsize=(12, 2.8 * len(all_metrics)))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(len(all_metrics), 2, figure=fig, hspace=0.55, wspace=0.3)

    from scipy.stats import gaussian_kde
    for row, (mlabel, mcol) in enumerate(all_metrics.items()):
        raw_m = player_df[['game_date', mcol]].dropna().sort_values('game_date')
        if cpd_date is not None:
            bef = raw_m[raw_m['game_date'] < cpd_date][mcol]
            aft = raw_m[raw_m['game_date'] >= cpd_date][mcol]
        else:
            mid2 = raw_m['game_date'].median()
            bef  = raw_m[raw_m['game_date'] < mid2][mcol]
            aft  = raw_m[raw_m['game_date'] >= mid2][mcol]

        for col_i, (data, title, color) in enumerate([
            (bef, f"Before · {mlabel}", TEAL),
            (aft, f"After · {mlabel}",  GOLD),
        ]):
            ax = fig.add_subplot(gs[row, col_i])
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=TEXT, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(MUTED)
            ax.grid(color=MUTED, linewidth=0.3, linestyle='--', alpha=0.3)
            if len(data) > 5:
                xs = np.linspace(data.min(), data.max(), 200)
                ys = gaussian_kde(data)(xs)
                ax.fill_between(xs, ys, alpha=0.35, color=color)
                ax.plot(xs, ys, color=color, linewidth=1.5)
                ax.axvline(data.mean(), color='white', linewidth=1.2, linestyle='--', alpha=0.7)
            ax.set_title(f"{title}  μ={data.mean():.1f}", color=TEXT, fontsize=9, pad=4)
            ax.set_xlabel(mlabel, color=MUTED, fontsize=8)
            ax.set_ylabel("Density", color=MUTED, fontsize=8)

    fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Two-Path Distribution  (Slide 2 style)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Two-Path Distribution":
    st.markdown("# Two-Path Distribution")
    st.markdown(f'<div class="section-title">{sel_player} · {sel_metric_label} · before vs after CPD</div>', unsafe_allow_html=True)
    if cpd_date:
        st.markdown(f'Change point: <span class="cpd-badge">{cpd_date.strftime("%b %d, %Y")}</span>', unsafe_allow_html=True)
    st.markdown("")

    from scipy.stats import gaussian_kde
    fig, ax = mpl_fig(12, 5)
    if len(before) > 5 and len(after) > 5:
        xs = np.linspace(raw[sel_metric].min()-3, raw[sel_metric].max()+3, 300)
        ys_b = gaussian_kde(before, bw_method=0.4)(xs)
        ys_a = gaussian_kde(after,  bw_method=0.4)(xs)
        ax.fill_between(xs, ys_b, alpha=0.3, color=TEAL, label=f"Before CPD  μ={before.mean():.1f}")
        ax.fill_between(xs, ys_a, alpha=0.3, color=GOLD, label=f"After CPD   μ={after.mean():.1f}")
        ax.plot(xs, ys_b, color=TEAL, linewidth=2.2)
        ax.plot(xs, ys_a, color=GOLD, linewidth=2.2)
        ax.axvline(before.mean(), color=TEAL, linewidth=1.2, linestyle=':', alpha=0.85)
        ax.axvline(after.mean(),  color=GOLD, linewidth=1.2, linestyle=':', alpha=0.85)
        ax.fill_between(xs, np.minimum(ys_b, ys_a), alpha=0.15, color='white', label="Overlap")
    ax.set_xlabel(sel_metric_label, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"{sel_player} — {sel_metric_label} distribution shift", color=TEXT, fontsize=12)
    ax.legend(fontsize=10, framealpha=0.3, facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Season breakdown</div>', unsafe_allow_html=True)
    seasons = sorted(player_df['Season'].dropna().unique())
    pal     = [PURPLE_LT, GOLD, TEAL, RED, "#a8d8a8"]
    fig2, axes = plt.subplots(1, len(seasons), figsize=(3*len(seasons), 3.5), sharey=True)
    fig2.patch.set_facecolor(DARK)
    if len(seasons) == 1:
        axes = [axes]
    for i, (s, axi) in enumerate(zip(seasons, axes)):
        data = player_df[player_df['Season'] == s][sel_metric].dropna()
        axi.set_facecolor(PANEL)
        axi.tick_params(colors=TEXT, labelsize=8)
        for spine in axi.spines.values():
            spine.set_edgecolor(MUTED)
        if len(data) > 5:
            xs2 = np.linspace(data.min()-2, data.max()+2, 200)
            ys2 = gaussian_kde(data, bw_method=0.4)(xs2)
            axi.fill_between(xs2, ys2, alpha=0.4, color=pal[i % len(pal)])
            axi.plot(xs2, ys2, color=pal[i % len(pal)], linewidth=1.8)
            axi.axvline(data.mean(), color='white', linewidth=1, linestyle='--', alpha=0.7)
        title_color = GOLD if (cpd_date and cpd_date.year == s) else TEXT
        axi.set_title(str(s), color=title_color, fontsize=10)
        axi.set_xlabel(sel_metric_label, color=MUTED, fontsize=8)
        if i == 0:
            axi.set_ylabel("Density", color=MUTED, fontsize=8)
    fig2.suptitle(f"{sel_player} · {sel_metric_label} by season", color=TEXT, fontsize=11, y=1.02)
    fig2.tight_layout(); st.pyplot(fig2); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Peer Comparison  (Slide 3/4 style)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Peer Comparison":
    st.markdown("# Peer Comparison")
    st.markdown(f'<div class="section-title">{sel_player} vs peers · {sel_metric_label}</div>', unsafe_allow_html=True)

    peers = st.multiselect(
        "Add comparison players (up to 3)",
        [p for p in players if p != sel_player],
        default=[p for p in players if p != sel_player][:2],
        max_selections=3,
    )
    all_sel = [sel_player] + peers
    pal     = [GOLD, TEAL, PURPLE_LT, RED]

    fig, ax = mpl_fig(12, 4.5)
    for i, pname in enumerate(all_sel):
        p_cpd, p_roll = compute_cpd(df, pname, sel_metric, window=cpd_window)
        if len(p_roll):
            ax.plot(p_roll['game_date'], p_roll[sel_metric],
                    color=pal[i], linewidth=2.5 if pname == sel_player else 1.5,
                    alpha=1.0 if pname == sel_player else 0.65,
                    label=pname, zorder=3-i)
        if p_cpd is not None:
            ax.axvline(p_cpd, color=pal[i], linewidth=1, linestyle='--', alpha=0.55)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{sel_metric_label} ({cpd_window}-pitch rolling avg)")
    ax.set_title(f"{sel_metric_label} trajectories — dashed lines = each player's CPD", color=TEXT, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.3, facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">CPD Summary Table</div>', unsafe_allow_html=True)
    rows = []
    for pname in all_sel:
        cpd_d, _ = compute_cpd(df, pname, sel_metric, window=cpd_window)
        p_raw = df[df['Name'] == pname][['game_date', sel_metric]].dropna().sort_values('game_date')
        if cpd_d is not None:
            bef = p_raw[p_raw['game_date'] < cpd_d][sel_metric]
            aft = p_raw[p_raw['game_date'] >= cpd_d][sel_metric]
            rows.append({
                'Player':                          pname,
                'CPD Date':                        cpd_d.strftime('%b %d, %Y'),
                f'Pre-CPD {sel_metric_label}':     round(bef.mean(), 2) if len(bef) else None,
                f'Post-CPD {sel_metric_label}':    round(aft.mean(), 2) if len(aft) else None,
                'Δ':                               round(aft.mean() - bef.mean(), 2) if len(bef) and len(aft) else None,
                'N events':                        len(p_raw),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)