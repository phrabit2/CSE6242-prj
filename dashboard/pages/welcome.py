import matplotlib.pyplot as plt
import streamlit as st

from config import (PA_INDICATORS, PA_LABELS, PA_COLORS,
                    DARK, PANEL, BORDER, GOLD, TEXT, TEXT_MUTED)
from ui_components import card, sec


def render(df, min_year: int, max_year: int) -> None:
    st.markdown("# Performance Inflection Dashboard")
    st.markdown("### Advanced Hitter Performance Analytics through Statcast & Machine Learning")

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

    sec("📊 Dataset Overview")
    st.markdown(f"""<div class="card-row">
        {card("Total Hitters",   f"{df['player_name'].nunique():,}", "qualified players",       "gold")}
        {card("Total Records",  f"{len(df):,}",                    "plate appearances",        "teal")}
        {card("Season Span",    str(df['year'].nunique()),         f"{min_year} – {max_year}", "grey")}
        {card(f"{max_year} Data", f"{len(df[df['year']==max_year]):,}", "current season PAs",  "teal")}
    </div>""", unsafe_allow_html=True)

    sec("PA Indicator Distributions — League Benchmark")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), constrained_layout=True)
    fig.patch.set_facecolor(DARK)
    for ax, col in zip(axes, PA_INDICATORS):
        color = PA_COLORS[col]
        data  = df[col].dropna()
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.hist(data, bins=60, color=color, alpha=0.8, edgecolor="none")
        ax.axvline(data.mean(), color=GOLD, lw=1.5, ls="--", label=f"mean {data.mean():.2f}")
        ax.set_title(PA_LABELS[col], color=TEXT, fontsize=9, fontweight="bold")
        ax.set_xlabel("Value", color=TEXT_MUTED, fontsize=8)
        ax.set_ylabel("Count", color=TEXT_MUTED, fontsize=8)
        ax.legend(fontsize=7, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
        ax.grid(color=BORDER, linewidth=0.3, linestyle="--", alpha=0.4)
    fig.suptitle(f"PA Indicator Distributions  ({min_year}-{max_year}, all players)",
                 color=TEXT, fontsize=11)
    st.pyplot(fig)
    plt.close()

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
