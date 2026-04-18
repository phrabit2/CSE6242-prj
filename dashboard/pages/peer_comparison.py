import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (PA_INDICATORS, PA_LABELS,
                    PANEL, BORDER, TEAL, TEAL_LT, GOLD, RED,
                    GREY, TEXT, TEXT_MUTED)
from ui_components import sec


def render(df, players: list, max_year: int) -> None:
    st.markdown("# 👥 Peer Comparison")

    if "peer_year"    not in st.session_state: st.session_state.peer_year    = max_year
    if "peer_players" not in st.session_state:
        st.session_state.peer_players = (
            ["Trout, Mike", "Ohtani, Shohei"] if "Trout, Mike" in players else []
        )

    st.markdown(
        "<style>span[data-baseweb='tag'] { background-color: #E8F0FE !important; "
        "color: #1A73E8 !important; border: 1px solid #D2E3FC !important; } "
        "span[data-baseweb='tag'] svg { fill: #1A73E8 !important; }</style>",
        unsafe_allow_html=True,
    )

    row1_c1, row1_c2 = st.columns([1, 3])
    with row1_c1:
        yrs     = sorted(df["year"].unique(), reverse=True)
        sel_year = st.selectbox("Season", yrs, index=yrs.index(st.session_state.peer_year), key="peer_y_sel")

    p_in_s = sorted(df[df["year"] == sel_year]["player_name"].dropna().unique())
    with row1_c2:
        valid_p    = [p for p in st.session_state.peer_players if p in p_in_s]
        sel_players = st.multiselect("Select Players (up to 3)", p_in_s, default=valid_p[:3], max_selections=3, key="peer_p_sel")

    opts       = {**PA_LABELS, "pa_count": "PAs this Season"}
    sel_metrics = st.multiselect(
        "Select Metrics to Compare", list(opts.keys()),
        default=list(PA_LABELS.keys())[:3],
        format_func=lambda x: opts[x],
    )

    if sel_year != st.session_state.peer_year or sel_players != st.session_state.peer_players:
        st.session_state.peer_year    = sel_year
        st.session_state.peer_players = sel_players
        st.rerun()

    if not sel_players:
        st.warning("Please select at least one player to compare.")
        st.stop()

    st.markdown("---")
    comparison_data = []
    season_df   = df[df["year"] == sel_year]
    radar_colors = [TEAL, GOLD, RED]

    for p_name in sel_players:
        p_df  = season_df[season_df["player_name"] == p_name]
        stats = {"Player": p_name, "pa_count": len(p_df)}
        for m_col in PA_INDICATORS:
            stats[m_col]              = p_df[m_col].mean()
            stats[f"{m_col}_std"]     = p_df[m_col].std()
        comparison_data.append(stats)

    import pandas as pd
    comp_df = pd.DataFrame(comparison_data)

    def peer_card(name, val, color_idx):
        c = radar_colors[color_idx % 3]
        return (
            f'<div style="padding: 8px 12px; background: {PANEL}; border-radius: 4px; '
            f'margin-bottom: 8px; border: 1px solid {BORDER}; border-left: 4px solid {c};">'
            f'<div style="font-size: 0.7rem; text-transform: uppercase; color: {TEXT_MUTED}; '
            f'font-weight: 600; letter-spacing: 0.05em; line-height: 1.2;">{name}</div>'
            f'<div style="font-size: 1.3rem; font-weight: 700; color: {TEXT}; '
            f'font-family: \'Bebas Neue\', sans-serif; line-height: 1.1;">{val}</div></div>'
        )

    ov_col1, ov_col2 = st.columns(2)
    with ov_col1:
        sec(f"📊 Performance Overview — {sel_year}")
        for mk in sel_metrics:
            st.markdown(
                f"<div style='margin-top: 12px; margin-bottom: 4px; font-weight: 700; "
                f"font-size: 0.85rem; color: {TEAL_LT}; text-transform: uppercase; "
                f"letter-spacing: 0.1em;'>{opts[mk]}</div>",
                unsafe_allow_html=True,
            )
            cols = st.columns(max(len(sel_players), 3))
            for i, ps in enumerate(comparison_data):
                if mk == "pa_count":
                    v_str = f"{int(ps[mk]):,}"
                elif mk == "power_efficiency":
                    v_str = f"{ps[mk]:.4f}"
                elif mk == "woba_residual":
                    v_str = f"{ps[mk]:.3f}"
                else:
                    v_str = f"{ps[mk]:.2f}"
                with cols[i]:
                    st.markdown(peer_card(ps["Player"], v_str, i), unsafe_allow_html=True)

    with ov_col2:
        sec("💡 Peer Comparison Insights")
        peer_insights = []
        if len(sel_players) > 1:
            m_ctx = {
                "hitting_decisions_score":     {"why": "superior plate discipline.",        "impact": "More walks, fewer strikeouts."},
                "power_efficiency":             {"why": "better kinetic energy transfer.",   "impact": "Sustainable, elite exit velocity."},
                "woba_residual":                {"why": "maximizing results vs physics.",    "impact": "High 'batted ball skill'."},
                "launch_angle_stability_50pa":  {"why": "repeatable swing path.",            "impact": "Slump-resistant performance."},
            }
            for mk in [m for m in sel_metrics if m != "pa_count"]:
                if mk in m_ctx:
                    leader = comp_df.loc[comp_df[mk].idxmax(), "Player"]
                    ctx    = m_ctx[mk]
                    peer_insights.append(
                        f"🏆 <b>{opts[mk]}:</b> <b>{leader}</b> ({comp_df[mk].max():.3f})<br>"
                        f"<i>{ctx['why']} {ctx['impact']}</i>"
                    )
            if "hitting_decisions_score" in comp_df.columns and "power_efficiency" in comp_df.columns:
                top_disc = comp_df.loc[comp_df["hitting_decisions_score"].idxmax(), "Player"]
                top_pwr  = comp_df.loc[comp_df["power_efficiency"].idxmax(), "Player"]
                if top_disc != top_pwr:
                    peer_insights.append(f"⚖️ <b>Contrasting Styles:</b> <b>{top_disc}</b> (Tactician) vs. <b>{top_pwr}</b> (Slugger).")
        st.markdown(
            f'<div class="narrative" style="font-size: 0.85rem;">'
            f'{"<br><br>".join(peer_insights if peer_insights else ["Add players/metrics for insights."])}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    v_c1, v_c2 = st.columns(2)
    with v_c1:
        sec("Player Profile Comparison (Radar)")
        fig_radar = go.Figure()
        for i, p_name in enumerate(sel_players):
            p_stats   = comp_df[comp_df["Player"] == p_name].iloc[0]
            radar_vals = [
                (p_stats[col] - season_df[col].min()) / (season_df[col].max() - season_df[col].min() + 1e-9)
                for col in PA_INDICATORS
            ]
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=[PA_LABELS[c] for c in PA_INDICATORS] + [PA_LABELS[PA_INDICATORS[0]]],
                fill="toself", name=p_name, line_color=radar_colors[i % 3],
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with v_c2:
        sec("Metric Leaderboard")
        plot_m = [m for m in sel_metrics if m != "pa_count"]
        if plot_m:
            bar_data = [
                {
                    "Player": p_name,
                    "Metric": PA_LABELS[m],
                    "Relative Standing (Z-Score)": (
                        comp_df[comp_df["Player"] == p_name][m].iloc[0] - season_df[m].mean()
                    ) / (season_df[m].std() + 1e-9),
                }
                for p_name in sel_players
                for m in plot_m
            ]
            import pandas as pd
            fig_bar = px.bar(
                pd.DataFrame(bar_data), x="Metric", y="Relative Standing (Z-Score)",
                color="Player", barmode="group", color_discrete_sequence=radar_colors,
            )
            fig_bar.update_layout(
                height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    sec("PA Outcome Density (League context)")
    with st.expander("ℹ️ How to interpret these density charts?"):
        st.markdown(
            "**Shaded Grey Area**: League benchmark. **Colored Lines**: Selected players. "
            "**Peak height**: How typical that level is. **Spread**: Narrow=consistent, Wide=streaky."
        )

    pm = [m for m in sel_metrics if m != "pa_count"]
    if not pm:
        st.info("Select a contact quality metric above to see the outcome density.")
    else:
        from scipy.stats import gaussian_kde
        for i in range(0, len(pm), 2):
            batch   = pm[i : i + 2]
            st_cols = st.columns(len(batch))
            for j, m_col in enumerate(batch):
                with st_cols[j]:
                    fig_d  = go.Figure()
                    l_data = season_df[m_col].dropna()
                    if not l_data.empty:
                        l_xs  = np.linspace(l_data.min(), l_data.max(), 200)
                        l_kde = gaussian_kde(l_data, bw_method=0.3)(l_xs)
                        fig_d.add_trace(go.Scatter(
                            x=l_xs, y=l_kde, fill="tozeroy",
                            name="League", line_color=GREY, opacity=0.2,
                        ))
                        for p_idx, p_name in enumerate(sel_players):
                            p_data = season_df[season_df["player_name"] == p_name][m_col].dropna()
                            if len(p_data) > 5:
                                p_kde = gaussian_kde(p_data, bw_method=0.4)(l_xs)
                                fig_d.add_trace(go.Scatter(
                                    x=l_xs, y=p_kde, name=p_name,
                                    line_color=radar_colors[p_idx % 3], line_width=3,
                                ))
                    fig_d.update_layout(
                        title=f"Distribution of {PA_LABELS[m_col]}",
                        xaxis_title=PA_LABELS[m_col], yaxis_title="Density",
                        height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=TEXT), margin=dict(t=40, b=40, l=40, r=40),
                        showlegend=(j == 0),
                    )
                    st.plotly_chart(fig_d, use_container_width=True)

                    l_mean  = l_data.mean()
                    p_means = {p: season_df[season_df["player_name"] == p][m_col].mean() for p in sel_players}
                    p_stds  = {p: season_df[season_df["player_name"] == p][m_col].std()  for p in sel_players}
                    above   = [p for p, m in p_means.items() if m > l_mean]
                    lead    = max(p_means, key=p_means.get)
                    cons    = min(p_stds, key=p_stds.get) if len(sel_players) > 1 else None

                    msg = f"**{lead}** leads this group."
                    if len(above) == len(sel_players):
                        msg += " All selected players are performing **above league average**."
                    elif above:
                        msg += f" {', '.join(above)} are performing **above league average**."
                    if cons and len(sel_players) > 1:
                        msg += f" **{cons}** shows the tallest peak, indicating the most consistency."
                    st.caption(f"💡 {msg}")
