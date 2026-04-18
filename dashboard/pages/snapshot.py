import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (PA_INDICATORS, PA_LABELS, PA_TOOLTIPS,
                    DARK, PANEL, BORDER, GOLD, TEAL, GREY, TEXT, TEXT_MUTED)
from ui_components import card, sec


def render(df, players: list, max_year: int) -> None:
    st.markdown("# Player Snapshot")

    # ── Session state init ────────────────────────────────────────────────────
    if "snapshot_player" not in st.session_state:
        st.session_state.snapshot_player = "All Players (League Avg)"
    if "snapshot_year" not in st.session_state:
        st.session_state.snapshot_year = max_year

    # ── Reciprocal filtering ──────────────────────────────────────────────────
    if st.session_state.snapshot_player == "All Players (League Avg)":
        available_years = sorted(df["year"].unique(), reverse=True)
    else:
        available_years = sorted(
            df[df["player_name"] == st.session_state.snapshot_player]["year"].unique(),
            reverse=True,
        )

    if st.session_state.snapshot_year not in available_years:
        st.session_state.snapshot_year = available_years[0]

    available_players_for_year = sorted(
        df[df["year"] == st.session_state.snapshot_year]["player_name"].dropna().unique()
    )
    player_options = ["All Players (League Avg)"] + available_players_for_year

    if st.session_state.snapshot_player not in player_options:
        st.session_state.snapshot_player = "All Players (League Avg)"

    row1_c1, row1_c2 = st.columns([3, 1])
    with row1_c1:
        sel_player = st.selectbox(
            "Select Player", player_options,
            index=player_options.index(st.session_state.snapshot_player),
            key="snapshot_player_select",
        )
    with row1_c2:
        sel_year = st.selectbox(
            "Season", available_years,
            index=available_years.index(st.session_state.snapshot_year),
            key="snapshot_year_select",
        )

    if sel_player != st.session_state.snapshot_player or sel_year != st.session_state.snapshot_year:
        st.session_state.snapshot_player = sel_player
        st.session_state.snapshot_year   = sel_year
        st.rerun()

    is_all = sel_player == "All Players (League Avg)"
    if is_all:
        snapshot_df  = df[df["year"] == sel_year].copy()
        display_name = f"League Average — {sel_year}"
    else:
        snapshot_df  = df[(df["player_name"] == sel_player) & (df["year"] == sel_year)].copy()
        display_name = f"{sel_player} — {sel_year}"

    def _mean(s):
        return s.dropna().mean() if not s.empty else 0.0

    m_discipline = _mean(snapshot_df["hitting_decisions_score"])
    m_power      = _mean(snapshot_df["power_efficiency"])
    m_woba_res   = _mean(snapshot_df["woba_residual"])
    m_la_stab    = _mean(snapshot_df["launch_angle_stability_50pa"])
    m_pa_count   = len(snapshot_df)

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hitting Decisions",  f"{m_discipline:.2f}",  help=PA_TOOLTIPS["hitting_decisions_score"])
    c2.metric("Power Efficiency",   f"{m_power:.4f}",        help=PA_TOOLTIPS["power_efficiency"])
    c3.metric("wOBA Residual",      f"{m_woba_res:.3f}",     help=PA_TOOLTIPS["woba_residual"])
    c4.metric("Launch Angle Stab.", f"{m_la_stab:.2f}",      help=PA_TOOLTIPS["launch_angle_stability_50pa"])
    c5.metric("PAs this Season",    f"{m_pa_count:,}",       help="Total records for the season.")

    if not is_all:
        season_df = df[df["year"] == sel_year]

        def get_stat_insight(col, val):
            l_mean = season_df[col].mean()
            l_std  = season_df[col].std()
            if val > l_mean + l_std:  return "Elite",         "Performing significantly better than the league average."
            if val > l_mean:          return "Above Average",  "Performing better than most of the league."
            if val < l_mean - l_std:  return "Struggling",     "Currently performing well below the league benchmark."
            if val < l_mean:          return "Below Average",  "Performing slightly below the league average."
            return "Average", "Performing right in line with the league average."

        st.markdown("---")
        sec(f"📊 {sel_year} Performance Spectrum")
        st.caption("Visualizing the hitter's standing across core pillars. League Avg is the center marker.")

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
                        <span>League Min</span><span>Avg</span><span>League Max</span>
                    </div>
                </div>
                <div style="flex: 1.5;">
                    <div style="font-size: 0.85rem; color: #24292f; line-height: 1.4;">{insight_text}</div>
                </div>
            </div>
            """

        def get_pct(col, val):
            s_min = season_df[col].quantile(0.01)
            s_max = season_df[col].quantile(0.99)
            return min(100, max(0, (val - s_min) / (s_max - s_min + 1e-9) * 100))

        status, _ = get_stat_insight("hitting_decisions_score", m_discipline)
        p = get_pct("hitting_decisions_score", m_discipline)
        insight = f"<b>{sel_player}</b> is {status.lower()} in plate discipline."
        st.markdown(spectrum_row("🎯", "Discipline", status, p, "#0969da", insight), unsafe_allow_html=True)

        status, _ = get_stat_insight("power_efficiency", m_power)
        p = get_pct("power_efficiency", m_power)
        insight = f"Currently graded as <b>{status.lower()}</b>. High power efficiency means the hitter is maximizing exit velocity relative to swing effort."
        st.markdown(spectrum_row("💥", "Power", status, p, "#58A6FF", insight), unsafe_allow_html=True)

        wr = m_woba_res
        status = "Steady" if abs(wr) < 0.15 else ("Outpacing Physics" if wr > 0.15 else "Unlucky")
        p = get_pct("woba_residual", wr)
        insight = f"Status: <b>{status}</b>. Far right means results better than physics predicts; far left indicates regression coming."
        st.markdown(spectrum_row("🎲", "Results", status, p, "#855D00", insight), unsafe_allow_html=True)

        status, _ = get_stat_insight("launch_angle_stability_50pa", m_la_stab)
        p = get_pct("launch_angle_stability_50pa", m_la_stab)
        insight = f"Graded as <b>{status.lower()}</b>. Consistency near the ceiling indicates a highly repeatable swing path."
        st.markdown(spectrum_row("📈", "Consistency", status, p, "#2ca02c", insight), unsafe_allow_html=True)

    st.markdown("---")
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        sec("Player Profile (Radar Chart)")
        season_df = df[df["year"] == sel_year]
        radar_data = []
        for col in PA_INDICATORS:
            val = _mean(snapshot_df[col])
            s_min = season_df[col].min()
            s_max = season_df[col].max()
            radar_data.append((val - s_min) / (s_max - s_min + 1e-9))

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_data + [radar_data[0]],
            theta=[PA_LABELS[c] for c in PA_INDICATORS] + [PA_LABELS[PA_INDICATORS[0]]],
            fill="toself", name=sel_player, line_color=TEAL,
        ))
        leag_vals = []
        for col in PA_INDICATORS:
            l_val = _mean(season_df[col])
            s_min = season_df[col].min()
            s_max = season_df[col].max()
            leag_vals.append((l_val - s_min) / (s_max - s_min + 1e-9))
        fig_radar.add_trace(go.Scatterpolar(
            r=leag_vals + [leag_vals[0]],
            theta=[PA_LABELS[c] for c in PA_INDICATORS] + [PA_LABELS[PA_INDICATORS[0]]],
            name="League Average", line=dict(dash="dash", color=GREY),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, height=400, margin=dict(t=40, b=40, l=40, r=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with v_col2:
        sec("Statcast-Style Percentile Rankings")
        if is_all:
            percentiles = [50] * 4
        else:
            player_means = df[df["year"] == sel_year].groupby("player_name")[PA_INDICATORS].mean()
            percentiles  = [(player_means[col] < _mean(snapshot_df[col])).mean() * 100 for col in PA_INDICATORS]

        fig_pct = go.Figure(go.Bar(
            x=percentiles, y=[PA_LABELS[c] for c in PA_INDICATORS], orientation="h",
            marker=dict(color=percentiles, colorscale="RdYlGn", cmin=0, cmax=100),
            text=[f"{p:.0f}%" for p in percentiles], textposition="auto",
        ))
        fig_pct.update_layout(
            xaxis=dict(range=[0, 100], title="Percentile Rank"), yaxis=dict(autorange="reversed"),
            height=400, margin=dict(t=40, b=40, l=100, r=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT),
        )
        st.plotly_chart(fig_pct, use_container_width=True)

    st.markdown("---")
    sec("Contact Quality Profile (Power vs Luck)")
    fig_scatter = px.scatter(
        snapshot_df.dropna(subset=["power_efficiency", "woba_residual"]),
        x="power_efficiency", y="woba_residual",
        color_discrete_sequence=[TEAL], opacity=0.4,
        labels={"power_efficiency": "Power Efficiency", "woba_residual": "wOBA Residual"},
        title=f"Distribution for {display_name}",
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color=GREY)
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT), height=500,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    if not is_all:
        sec("Season-by-Season Progression")
        summ = df[df["player_name"] == sel_player].groupby("year").agg(
            hitting_decisions=("hitting_decisions_score",     "mean"),
            power_efficiency =("power_efficiency",            "mean"),
            woba_residual    =("woba_residual",               "mean"),
            launch_angle_stab=("launch_angle_stability_50pa", "mean"),
            pa_count         =("pa_uid",                      "count"),
        ).round(3).reset_index().rename(columns={"year": "Season"})
        st.dataframe(summ, use_container_width=True, hide_index=True)

        if len(summ) > 1:
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

            def get_archetype(row):
                pwr_elite  = row["power_efficiency"]  > df["power_efficiency"].mean()  + df["power_efficiency"].std()
                disc_elite = row["hitting_decisions"]  > df["hitting_decisions_score"].mean() + df["hitting_decisions_score"].std()
                if pwr_elite and disc_elite: return "Elite All-Around Threat"
                if pwr_elite:                return "Power-First Slugger"
                if disc_elite:               return "Discipline-Driven Tactician"
                return "Steady Contributor"

            first_year = int(summ.iloc[0]["Season"])
            last_year  = int(summ.iloc[-1]["Season"])
            evolution_story = []

            initial_arch = get_archetype(summ.iloc[0])
            evolution_story.append(
                f"🌱 <b>The Foundation ({first_year}):</b> {sel_player} entered the dataset as a <b>{initial_arch}</b>. "
                f"Early data established a baseline {summ.iloc[0]['hitting_decisions']:.2f} Discipline and {summ.iloc[0]['power_efficiency']:.4f} Power."
            )

            if len(summ) > 2:
                growth_notes = []
                for i in range(1, len(summ) - 1):
                    y    = int(summ.iloc[i]["Season"])
                    prev = summ.iloc[i - 1]
                    curr = summ.iloc[i]
                    if curr["power_efficiency"]  > prev["power_efficiency"]  * 1.1:
                        growth_notes.append(f"a power surge in {y}")
                    if curr["hitting_decisions"] > prev["hitting_decisions"] + 0.5:
                        growth_notes.append(f"improved strike-zone mastery in {y}")
                if growth_notes:
                    evolution_story.append(
                        f"📈 <b>Growth & Maturation:</b> Middle seasons were defined by {', '.join(growth_notes)}."
                    )

            latest_arch = get_archetype(summ.iloc[-1])
            trend = "improving" if summ["power_efficiency"].iloc[-1] > summ["power_efficiency"].iloc[0] else "shifting"
            evolution_story.append(
                f"🏁 <b>Current Profile ({last_year}):</b> Today, {sel_player} has evolved into a <b>{latest_arch}</b>. "
                f"The data shows a career-to-date trend of {trend} performance profiles."
            )

            st.markdown(f"""
            <div class="narrative">
            <b>📊 Career Evolution: {sel_player} ({first_year}–{last_year})</b><br><br>
            {"<br><br>".join(evolution_story)}
            </div>
            """, unsafe_allow_html=True)
