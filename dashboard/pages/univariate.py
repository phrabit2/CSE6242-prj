import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import SENSITIVITY_MAP, DARK, RED, TEAL, TEXT, BORDER
from cpd_engine import detect_cpd, rolling_with_dates
from ui_components import sec, render_cp_analysis


def render(df, df_idx, players: list, max_year: int) -> None:
    st.markdown("# Univariate Change Point Analyzer (PELT)")
    st.markdown(
        "Identify significant shifts in a player's performance profile. "
        "Select a player and season to see where their performance changed, "
        "then deep-dive into the data."
    )

    # ── Session state init ────────────────────────────────────────────────────
    if "pca_player"        not in st.session_state:
        st.session_state.pca_player        = "Trout, Mike" if "Trout, Mike" in players else players[2]
    if "pca_year"          not in st.session_state:
        st.session_state.pca_year          = max_year
    if "pca_selected_date" not in st.session_state:
        st.session_state.pca_selected_date = "-- Select Date --"

    p_yrs = sorted(df[df["player_name"] == st.session_state.pca_player]["year"].unique(), reverse=True)
    if st.session_state.pca_year not in p_yrs:
        st.session_state.pca_year = p_yrs[0]

    p_names = sorted(df[df["year"] == st.session_state.pca_year]["player_name"].dropna().unique())
    if st.session_state.pca_player not in p_names:
        st.session_state.pca_player = p_names[0]

    # ── Controls ──────────────────────────────────────────────────────────────
    row1_c1, row1_c2, row1_c3 = st.columns([2, 1, 1])
    with row1_c1:
        sel_player = st.selectbox("Select Player", p_names,
                                  index=p_names.index(st.session_state.pca_player), key="pca_p_sel")
    with row1_c2:
        sel_year = st.selectbox("Season", p_yrs,
                                index=p_yrs.index(st.session_state.pca_year), key="pca_y_sel")
    with row1_c3:
        st.markdown("<div style='height: 38px;'></div>", unsafe_allow_html=True)
        show_history = st.toggle("Full Career", value=False, key="pca_hist")

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        cpd_window  = st.slider("Rolling Window (PAs)", 20, 100, 50, 5, key="pca_window")
    with row2_c2:
        sensitivity = st.radio("CPD Sensitivity", ["Low", "Medium", "High"], index=1,
                               horizontal=True, key="pca_sens")

    with st.expander("ℹ️ How do these controls work?"):
        st.markdown("**Rolling Window**: Smoothing level. Smaller = catches quick shifts but noisy. "
                    "Larger = identifies long-term structural changes.")
        st.markdown("**CPD Sensitivity**: Statistical threshold for flagging a shift. "
                    "'High' detects small fluctuations; 'Low' only flags major, sustained changes.")

    if sel_player != st.session_state.pca_player or sel_year != st.session_state.pca_year:
        st.session_state.pca_player = sel_player
        st.session_state.pca_year   = sel_year
        st.rerun()

    st.markdown("---")

    # ── Data prep ─────────────────────────────────────────────────────────────
    p_idx    = df_idx[df_idx["player_name"] == sel_player].copy()
    c_idx    = p_idx if show_history else p_idx[p_idx["year"] == sel_year].copy()
    roll_idx = rolling_with_dates(c_idx, "perf_index", cpd_window)

    if len(roll_idx) < 20:
        st.warning("Insufficient data for Change Point Detection.")
        st.stop()

    cp_indices = detect_cpd(roll_idx["perf_index"], SENSITIVITY_MAP[sensitivity])

    # ── Main visualization ────────────────────────────────────────────────────
    sec(f"Performance Index Trend — {sel_player}")
    st.info("💡 **Interactive Analysis:** Click on any **red diamond marker** to instantly analyze that performance shift below.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roll_idx["game_date"], y=roll_idx["perf_index"],
        mode="lines", name="Performance Index",
        line=dict(color=TEAL, width=3), hoverinfo="skip",
    ))

    cp_dates_raw = [roll_idx["game_date"].iloc[cp] for cp in cp_indices]
    cp_vals      = [roll_idx["perf_index"].iloc[cp] for cp in cp_indices]
    fig.add_trace(go.Scatter(
        x=cp_dates_raw, y=cp_vals, mode="markers", name="Detected Shift",
        marker=dict(size=12, color=RED, symbol="diamond", line=dict(width=2, color=DARK)),
        hovertemplate="<b>Detected Shift</b><br>Date: %{x}<br>Index: %{y:.2f}<extra></extra>",
    ))
    for cp_date in cp_dates_raw:
        fig.add_vline(x=cp_date, line_dash="dash", line_color=RED, line_width=1, opacity=0.5)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=RED, width=2, dash="dash"), name="Significant Performance Shift",
    ))
    fig.update_layout(
        xaxis_title="Game Date", yaxis_title="Performance Index (0-100)",
        height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT), margin=dict(t=20, b=20, l=20, r=20),
        hovermode="x", clickmode="event+select", dragmode=False,
    )

    selected_event = st.plotly_chart(
        fig, use_container_width=True, on_select="rerun",
        config={"displayModeBar": False, "staticPlot": False, "doubleClick": "reset"},
    )

    # ── Click handler ─────────────────────────────────────────────────────────
    if selected_event and "selection" in selected_event:
        sel_points = selected_event["selection"].get("points", [])
        if sel_points:
            raw_x       = sel_points[0]["x"]
            clicked_str = raw_x[:10] if isinstance(raw_x, str) else raw_x.strftime("%Y-%m-%d")
            if cp_dates_raw:
                target_dt  = pd.to_datetime(clicked_str)
                nearest_cp = min(cp_dates_raw, key=lambda d: abs(d - target_dt))
                if abs((nearest_cp - target_dt).days) <= 7:
                    new_date = nearest_cp.strftime("%Y-%m-%d")
                    if st.session_state.pca_selected_date != new_date:
                        st.session_state.pca_selected_date = new_date
                        st.rerun()

    # ── Deep-dive drilldown ───────────────────────────────────────────────────
    if cp_indices:
        cp_dates_str = [d.strftime("%Y-%m-%d") for d in cp_dates_raw]
        if st.session_state.pca_selected_date not in ["-- Select Date --"] + cp_dates_str:
            st.session_state.pca_selected_date = "-- Select Date --"

        if st.session_state.pca_selected_date != "-- Select Date --":
            selected_cp    = st.session_state.pca_selected_date
            actual_cp_idx  = cp_indices[cp_dates_str.index(selected_cp)]
            render_cp_analysis(
                selected_cp, sel_player,
                c_idx.iloc[max(0, actual_cp_idx - 50) : actual_cp_idx],
                c_idx.iloc[actual_cp_idx : min(len(c_idx), actual_cp_idx + 50)],
                importance_df=None,
            )
    else:
        st.info("No significant performance shifts detected with current settings.")
