import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import (PA_INDICATORS, PA_LABELS, PA_COLORS,
                    SENSITIVITY_TO_MIN_SEG,
                    DARK, PANEL, BORDER, TEAL, GOLD, RED, TEXT, TEXT_MUTED)
from cpd_engine import build_cpd_subdf, run_changeforest, get_cp_feature_importance
from ui_components import sec, render_cp_analysis


def render(df, players: list, max_year: int) -> None:
    st.markdown("# Multivariate Change Point Analyzer (ChangeForest)")
    st.markdown("Multivariate change point detection across all 4 indicators simultaneously.")

    # ── Session state init ────────────────────────────────────────────────────
    if "cf_player"        not in st.session_state:
        st.session_state.cf_player        = "Trout, Mike" if "Trout, Mike" in players else players[2]
    if "cf_year"          not in st.session_state:
        st.session_state.cf_year          = max_year
    if "cf_selected_date" not in st.session_state:
        st.session_state.cf_selected_date = "-- Select Date --"

    p_yrs = sorted(df[df["player_name"] == st.session_state.cf_player]["year"].unique(), reverse=True)
    if st.session_state.cf_year not in p_yrs:
        st.session_state.cf_year = p_yrs[0]

    p_names = sorted(df[df["year"] == st.session_state.cf_year]["player_name"].dropna().unique())
    if st.session_state.cf_player not in p_names:
        st.session_state.cf_player = p_names[0]

    # ── Controls ──────────────────────────────────────────────────────────────
    row1_c1, row1_c2, row1_c3 = st.columns([2, 1, 1])
    with row1_c1:
        sel_player = st.selectbox("Select Player", p_names,
                                  index=p_names.index(st.session_state.cf_player), key="cf_p_sel")
    with row1_c2:
        sel_year = st.selectbox("Season", p_yrs,
                                index=p_yrs.index(st.session_state.cf_year), key="cf_y_sel")
    with row1_c3:
        st.markdown("<div style='height: 38px;'></div>", unsafe_allow_html=True)
        show_history = st.toggle("Full Career", value=False, key="cf_hist")

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        cpd_window  = st.slider("Rolling Window (PAs)", 20, 100, 50, 5, key="cf_window")
    with row2_c2:
        sensitivity = st.radio("CPD Sensitivity", ["Low", "Medium", "High"], index=1,
                               horizontal=True, key="cf_sens")

    if sel_player != st.session_state.cf_player or sel_year != st.session_state.cf_year:
        st.session_state.cf_player = sel_player
        st.session_state.cf_year   = sel_year
        st.rerun()

    st.markdown("---")

    # ── Data prep ─────────────────────────────────────────────────────────────
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

    # ── Summary metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Player",       sel_player)
    c2.metric("PA Rows Used", f"{len(subdf):,}")
    c3.metric("Sensitivity",  sensitivity)
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

    # ── Tab 1: ChangeForest timeline ──────────────────────────────────────────
    with tab1:
        sec("ChangeForest Result")
        st.caption("4 rolling-mean signals on a shared timeline. Red dotted lines mark detected change points.")

        dates        = pd.to_datetime(subdf["game_date"])
        cp_dates_raw = [subdf["game_date"].iloc[cp] for cp in cps]
        cp_dates_str = [d.strftime("%Y-%m-%d") for d in cp_dates_raw]

        fig_r = make_subplots(
            rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.23, 0.23, 0.23, 0.23, 0.08],
            subplot_titles=[PA_LABELS[col] for col in PA_INDICATORS] + [""],
        )

        for i, col in enumerate(PA_INDICATORS, start=1):
            y_col = f"{col}_rollmean_{cpd_window}"
            fig_r.add_trace(go.Scatter(
                x=dates, y=subdf[y_col], mode="lines", name=PA_LABELS[col],
                line=dict(color=PA_COLORS[col], width=2),
                hoverinfo="skip", showlegend=False,
            ), row=i, col=1)

        for cp_date in cp_dates_raw:
            for i in range(1, 6):
                fig_r.add_vline(x=cp_date, line_dash="dot",
                                line_color=RED, line_width=1.5, opacity=0.7)

        if cps:
            fig_r.add_trace(go.Scatter(
                x=cp_dates_raw, y=[0] * len(cp_dates_raw),
                mode="markers", name="Click to Analyze",
                marker=dict(size=12, color=RED, symbol="diamond", line=dict(width=2, color=DARK)),
                hovertemplate="<b>Click to Analyze</b><br>Date: %{x}<extra></extra>",
            ), row=5, col=1)

        fig_r.update_yaxes(visible=False, row=5, col=1)
        fig_r.update_layout(
            height=750,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT), margin=dict(t=40, b=20, l=60, r=20),
            hovermode="x", clickmode="event+select", dragmode=False,
            title_text=f"{sel_player}  |  window={cpd_window}  |  CPs={len(cps)}",
            title_font=dict(color=TEXT, size=11),
        )
        fig_r.update_xaxes(tickformat="%Y-%m", tickangle=35)

        selected_event = st.plotly_chart(
            fig_r, use_container_width=True, on_select="rerun",
            config={"displayModeBar": False, "staticPlot": False},
        )

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
                    subdf.iloc[actual_cp_idx : min(len(subdf), actual_cp_idx + 50)],
                    importance_df=importance_df,
                )
            else:
                st.info("💡 Click on any red diamond marker to analyze that shift.")
        else:
            st.info("No significant performance shifts detected with current settings.")

    # ── Tab 2: Before / After evaluation ─────────────────────────────────────
    with tab2:
        sec("Before / After Statistical Comparison")
        st.caption("Detected CPs (Y) should show larger absolute shifts in mean and std than non-CP positions (N).")

        compare_window = max(5, min(50, len(subdf) // 4))

        if len(subdf) <= 2 * compare_window:
            st.warning(f"Not enough data (need > {2 * compare_window} rows, have {len(subdf)}).")
        else:
            with st.spinner("Computing before/after statistics..."):
                cps_set  = set(cps)
                eval_dfs = {}
                for col in feature_names:
                    rows = []
                    for i in range(compare_window, len(subdf) - compare_window):
                        before = subdf[col].iloc[i - compare_window : i]
                        after  = subdf[col].iloc[i : i + compare_window]
                        rows.append({
                            "cp":            "Y" if i in cps_set else "N",
                            "abs_mean_diff": abs(after.mean() - before.mean()),
                            "abs_std_diff":  abs(after.std()  - before.std()),
                        })
                    eval_dfs[col] = pd.DataFrame(rows)

            if eval_dfs:
                fig_e, axes = plt.subplots(1, len(eval_dfs), figsize=(5 * len(eval_dfs), 4),
                                           constrained_layout=True)
                fig_e.patch.set_facecolor(DARK)
                if len(eval_dfs) == 1:
                    axes = [axes]
                for ax, (feat, df_eval) in zip(axes, eval_dfs.items()):
                    ax.set_facecolor(PANEL)
                    summary = df_eval.groupby("cp")[["abs_mean_diff", "abs_std_diff"]].mean()
                    summary.plot(kind="bar", ax=ax, rot=0, color=[TEAL, GOLD])
                    base = feat.split("_rollmean_")[0]
                    ax.set_title(PA_LABELS.get(base, feat), color=TEXT, fontsize=10, fontweight="bold")
                    ax.set_xlabel("Is Change Point?", color=TEXT_MUTED)
                    ax.set_ylabel("Avg Absolute Difference", color=TEXT_MUTED)
                    ax.tick_params(colors=TEXT)
                    for sp in ax.spines.values():
                        sp.set_edgecolor(BORDER)
                    ax.grid(axis="y", alpha=0.3, color=BORDER)
                    ax.legend(["Abs Mean Diff", "Abs Std Diff"], frameon=False, fontsize=9, labelcolor=TEXT)
                st.pyplot(fig_e, use_container_width=True)
                plt.close()

                with st.expander("📋 Detailed eval summary tables"):
                    for feat, df_eval in eval_dfs.items():
                        base = feat.split("_rollmean_")[0]
                        st.write(f"**{PA_LABELS.get(base, feat)}**")
                        st.dataframe(
                            df_eval.groupby("cp")[["abs_mean_diff", "abs_std_diff"]].mean().round(4),
                            use_container_width=True,
                        )

    # ── Tab 3: Parameter stability ────────────────────────────────────────────
    with tab3:
        sec("Parameter Stability")
        st.caption("Runs ChangeForest across all three sensitivity levels. Stable results → similar CP counts.")

        with st.spinner("Running stability analysis across all sensitivity levels..."):
            stability_rows = []
            inv_map = {v: k for k, v in SENSITIVITY_TO_MIN_SEG.items()}
            for val in sorted(SENSITIVITY_TO_MIN_SEG.values()):
                try:
                    _, s_cps, _, _ = run_changeforest(subdf, cpd_window, val)
                    n_cp = len(s_cps)
                except Exception as exc:
                    n_cp  = None
                    s_cps = []
                    st.warning(f"Stability run failed for min_rel_seg_len={val}: {exc}")
                stability_rows.append({
                    "Sensitivity":     inv_map.get(val, str(val)),
                    "min_rel_seg_len": val,
                    "# Change Points": n_cp,
                    "CP Indices":      str(s_cps),
                })
            stability_df = pd.DataFrame(stability_rows)

        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            fig_st, ax_st = plt.subplots(figsize=(7, 4), constrained_layout=True)
            fig_st.patch.set_facecolor(DARK)
            ax_st.set_facecolor(PANEL)
            valid   = stability_df.dropna(subset=["# Change Points"])
            x_labels = valid["Sensitivity"] + "\n(min=" + valid["min_rel_seg_len"].astype(str) + ")"
            bars    = ax_st.bar(x_labels, valid["# Change Points"], color=TEAL, alpha=0.82, width=0.5)
            for bar, val in zip(bars, valid["# Change Points"]):
                ax_st.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                           str(int(val)), ha="center", va="bottom",
                           fontsize=12, fontweight="bold", color=TEXT)
            ax_st.set_xlabel("Sensitivity", color=TEXT_MUTED)
            ax_st.set_ylabel("# Change Points", color=TEXT_MUTED)
            ax_st.tick_params(colors=TEXT)
            for sp in ax_st.spines.values():
                sp.set_edgecolor(BORDER)
            ax_st.grid(axis="y", alpha=0.3, color=BORDER)
            ax_st.set_title("Change Points vs Sensitivity", color=TEXT, fontsize=11)
            st.pyplot(fig_st, use_container_width=True)
            plt.close()
        with col_table:
            st.dataframe(stability_df, use_container_width=True, hide_index=True)
