import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Batted Ball Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global style ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700&family=Barlow:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Barlow', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Barlow Condensed', sans-serif;
        letter-spacing: 0.02em;
    }
    .block-container { padding-top: 2rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f1923;
    }
    [data-testid="stSidebar"] * {
        color: #e8dcc8 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.05rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Metric cards */
    .metric-card {
        background: #0f1923;
        border-left: 4px solid #c8a951;
        border-radius: 4px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.5rem;
    }
    .metric-card .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8a9bb0;
        font-family: 'Barlow Condensed', sans-serif;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Barlow Condensed', sans-serif;
        color: #e8dcc8;
        line-height: 1.1;
    }
    .metric-card .sub {
        font-size: 0.8rem;
        color: #5a6a7a;
    }

    /* Section headers */
    .section-header {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #c8a951;
        border-bottom: 1px solid #1e2d3d;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Data loader ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(
        base_dir, '..', 'data', 'processed',
        'qualified_hitters_statcast_2021_2025_batted_ball.csv'
    )
    return pd.read_csv(file_path)

df = load_data()

# ── Matplotlib theme ───────────────────────────────────────────────────────────
DARK_BG   = "#0f1923"
PANEL_BG  = "#141f2b"
GOLD      = "#c8a951"
TEXT      = "#e8dcc8"
MUTED     = "#4a5a6a"

def apply_theme(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED)
    ax.grid(color=MUTED, linewidth=0.4, linestyle='--', alpha=0.5)
    return fig, ax

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚾ MLB Batted Ball")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "Season Trends", "Batting Metrics", "Player Comparison", "Custom Filter"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    # Global season filter (used across all pages)
    if 'Season' in df.columns:
        seasons = sorted(df['Season'].unique())
        selected_seasons = st.multiselect("Seasons", seasons, default=seasons)
        df_filtered = df[df['Season'].isin(selected_seasons)]
    else:
        df_filtered = df

    st.caption(f"{len(df_filtered):,} player-seasons loaded")

# ── Helper ─────────────────────────────────────────────────────────────────────
def metric_card(label, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("# MLB Batted Ball Dashboard")
    st.markdown("Qualified hitters · 2021–2025 · Statcast")
    st.markdown("---")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Players", f"{df_filtered['Name'].nunique():,}" if 'Name' in df_filtered.columns else "—")
    with col2:
        metric_card("Player-Seasons", f"{len(df_filtered):,}")
    with col3:
        avg_hr = f"{df_filtered['HR'].mean():.1f}" if 'HR' in df_filtered.columns else "—"
        metric_card("Avg HR / Season", avg_hr)
    with col4:
        avg_avg = f"{df_filtered['AVG'].mean():.3f}" if 'AVG' in df_filtered.columns else "—"
        metric_card("Avg Batting Avg", avg_avg)

    st.markdown("---")

    # Distribution of a key metric
    st.markdown('<div class="section-header">Exit Velocity Distribution</div>', unsafe_allow_html=True)
    ev_col = next((c for c in df_filtered.columns if 'exit' in c.lower() and 'velo' in c.lower()), None)
    if ev_col:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        sns.histplot(df_filtered[ev_col].dropna(), bins=40, color=GOLD, alpha=0.85, ax=ax, edgecolor='none')
        ax.set_xlabel("Exit Velocity (mph)")
        ax.set_ylabel("Count")
        apply_theme(fig, ax)
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No exit velocity column found — update the column name in the code.")

    # Raw data preview
    st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_filtered.head(50), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Season Trends
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Season Trends":
    st.markdown("# Season Trends")
    st.markdown("League-wide averages by season")
    st.markdown("---")

    if 'Season' not in df_filtered.columns:
        st.warning("No 'Season' column found in dataset.")
    else:
        numeric_cols = df_filtered.select_dtypes(include='number').columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'Season']

        col_left, col_right = st.columns([1, 3])
        with col_left:
            metric = st.selectbox("Metric to plot", numeric_cols)
            agg_func = st.radio("Aggregation", ["Mean", "Median", "Sum"])

        agg = {'Mean': 'mean', 'Median': 'median', 'Sum': 'sum'}[agg_func]
        trend = df_filtered.groupby('Season')[metric].agg(agg).reset_index()

        with col_right:
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.lineplot(data=trend, x='Season', y=metric, color=GOLD,
                         linewidth=2.5, marker='o', markersize=7, ax=ax)
            ax.fill_between(trend['Season'], trend[metric], alpha=0.1, color=GOLD)
            ax.set_title(f"{agg_func} {metric} by Season", fontsize=13)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            apply_theme(fig, ax)
            st.pyplot(fig)
            plt.close()

        st.markdown('<div class="section-header">Season Summary Table</div>', unsafe_allow_html=True)
        summary = df_filtered.groupby('Season')[numeric_cols].mean().round(3).reset_index()
        st.dataframe(summary, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Batting Metrics Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Batting Metrics":
    st.markdown("# Batting Metrics Explorer")
    st.markdown("Explore relationships between any two metrics")
    st.markdown("---")

    numeric_cols = df_filtered.select_dtypes(include='number').columns.tolist()

    col1, col2, col3 = st.columns(3)
    with col1:
        x_col = st.selectbox("X Axis", numeric_cols, index=0)
    with col2:
        y_col = st.selectbox("Y Axis", numeric_cols, index=min(1, len(numeric_cols)-1))
    with col3:
        color_col = st.selectbox("Color by", ["None"] + [c for c in df_filtered.columns if df_filtered[c].nunique() < 20])

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = df_filtered[[x_col, y_col]].dropna()

    if color_col != "None":
        plot_df = df_filtered[[x_col, y_col, color_col]].dropna()
        palette = sns.color_palette("YlOrBr", n_colors=plot_df[color_col].nunique())
        sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=color_col,
                        palette=palette, alpha=0.7, s=40, ax=ax)
        ax.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT,
                  facecolor=PANEL_BG, edgecolor=MUTED)
    else:
        sns.scatterplot(data=plot_df, x=x_col, y=y_col,
                        color=GOLD, alpha=0.6, s=40, ax=ax)

    # Regression line
    sns.regplot(data=plot_df, x=x_col, y=y_col, scatter=False,
                color="#e05c5c", line_kws={"linewidth": 1.5, "linestyle": "--"}, ax=ax)

    ax.set_title(f"{x_col} vs {y_col}", fontsize=13)
    apply_theme(fig, ax)
    st.pyplot(fig)
    plt.close()

    # Correlation
    corr = plot_df[[x_col, y_col]].corr().iloc[0, 1]
    st.caption(f"Pearson correlation: **{corr:.3f}**")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Player Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Player Comparison":
    st.markdown("# Player Comparison")
    st.markdown("Compare up to 4 players side by side")
    st.markdown("---")

    if 'Name' not in df_filtered.columns:
        st.warning("No 'Name' column found in dataset.")
    else:
        all_players = sorted(df_filtered['Name'].unique())
        selected_players = st.multiselect(
            "Select players (up to 4)",
            all_players,
            max_selections=4,
            default=all_players[:2] if len(all_players) >= 2 else all_players
        )

        if not selected_players:
            st.info("Select at least one player above.")
        else:
            player_df = df_filtered[df_filtered['Name'].isin(selected_players)]
            numeric_cols = player_df.select_dtypes(include='number').columns.tolist()
            season_col_present = 'Season' in player_df.columns

            # Summary stats table per player
            st.markdown('<div class="section-header">Career Averages</div>', unsafe_allow_html=True)
            summary = player_df.groupby('Name')[numeric_cols].mean().round(3).T
            st.dataframe(summary, use_container_width=True)

            # Line chart per player over seasons
            if season_col_present:
                st.markdown('<div class="section-header">Metric Over Seasons</div>', unsafe_allow_html=True)
                metric = st.selectbox("Metric", numeric_cols)
                fig, ax = plt.subplots(figsize=(10, 4))
                palette = [GOLD, "#5ec4e0", "#e05c5c", "#7ec87e"]
                for idx, player in enumerate(selected_players):
                    pdata = player_df[player_df['Name'] == player].groupby('Season')[metric].mean().reset_index()
                    sns.lineplot(data=pdata, x='Season', y=metric,
                                 label=player, color=palette[idx % 4],
                                 linewidth=2, marker='o', markersize=6, ax=ax)
                ax.legend(fontsize=9, framealpha=0.3, labelcolor=TEXT,
                          facecolor=PANEL_BG, edgecolor=MUTED)
                ax.set_title(f"{metric} by Season", fontsize=13)
                ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                apply_theme(fig, ax)
                st.pyplot(fig)
                plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Custom Filter
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Custom Filter":
    st.markdown("# Custom Filter")
    st.markdown("Slice the dataset by any numeric column")
    st.markdown("---")

    numeric_cols = df_filtered.select_dtypes(include='number').columns.tolist()

    filters = {}
    col1, col2 = st.columns(2)
    cols_left = numeric_cols[:len(numeric_cols)//2]
    cols_right = numeric_cols[len(numeric_cols)//2:]

    for col_name, container in zip(cols_left, [col1]*len(cols_left)):
        with container:
            mn = float(df_filtered[col_name].min())
            mx = float(df_filtered[col_name].max())
            if mn < mx:
                filters[col_name] = st.slider(col_name, mn, mx, (mn, mx), key=col_name)

    for col_name, container in zip(cols_right, [col2]*len(cols_right)):
        with container:
            mn = float(df_filtered[col_name].min())
            mx = float(df_filtered[col_name].max())
            if mn < mx:
                filters[col_name] = st.slider(col_name, mn, mx, (mn, mx), key=col_name)

    result = df_filtered.copy()
    for col_name, (lo, hi) in filters.items():
        result = result[(result[col_name] >= lo) & (result[col_name] <= hi)]

    st.markdown("---")
    st.markdown(f'<div class="section-header">Results — {len(result):,} players match</div>', unsafe_allow_html=True)

    if 'Name' in result.columns:
        st.dataframe(result.sort_values('Name'), use_container_width=True)
    else:
        st.dataframe(result, use_container_width=True)

    # Download button
    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button("Download filtered data as CSV", csv, "filtered_players.csv", "text/csv")