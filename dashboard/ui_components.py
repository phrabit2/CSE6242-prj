import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from config import (PA_INDICATORS, PA_LABELS, PA_COLORS,
                    DARK, PANEL, BORDER, GOLD, GOLD_LT,
                    TEAL, TEAL_LT, RED, RED_LT, GREY, TEXT, TEXT_MUTED)


# ── Basic UI primitives ───────────────────────────────────────────────────────

def card(label: str, val: str, sub: str = "", color: str = "gold") -> str:
    """Return an HTML card block."""
    cls = {"gold": "card", "teal": "card teal", "grey": "card grey"}.get(color, "card")
    return (f'<div class="{cls}"><div class="card-label">{label}</div>'
            f'<div class="card-val">{val}</div><div class="card-sub">{sub}</div></div>')


def sec(title: str) -> None:
    """Render a section header."""
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)


def mpl_fig(w: float = 11, h: float = 4):
    """Create a styled matplotlib figure."""
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)
    ax.title.set_color(TEXT)
    ax.grid(color=BORDER, linewidth=0.4, linestyle="--", alpha=0.5)
    return fig, ax


def effect_size_label(d: float) -> tuple:
    ad = abs(d)
    if ad < 0.2:
        return "minor", "cpd-minor"
    if ad < 0.5:
        return "moderate", "cpd-mod"
    return "significant", "cpd-sig"


def reliability_badge(n: int) -> str:
    if n >= 200:
        return f'<span class="rel-high">&#9679; High reliability</span>'
    if n >= 100:
        return f'<span class="rel-med">&#9679; Medium reliability</span>'
    return f'<span class="rel-low">&#9679; Low reliability (few PAs this season)</span>'


def games_label(n_pa: int) -> str:
    return f"~{max(1, round(n_pa / 3.8))} games"


# ── Smart Analyzer Engine ─────────────────────────────────────────────────────

def get_diagnostic_insight(stats_list: dict, player_name: str) -> list:
    """Interpret multivariate effect sizes and generate narrative insights."""
    findings = []
    indicator_summary = {}
    for col, s in stats_list.items():
        d = s["effect_d"]
        label = "Stable"
        if d > 0.5:   label = "Significant Gain"
        elif d > 0.2: label = "Marginal Gain"
        elif d < -0.5: label = "Significant Decline"
        elif d < -0.2: label = "Marginal Decline"
        indicator_summary[col] = label

    pwr  = indicator_summary.get("power_efficiency", "Stable")
    la   = indicator_summary.get("launch_angle_stability_50pa", "Stable")
    disc = indicator_summary.get("hitting_decisions_score", "Stable")
    res  = indicator_summary.get("woba_residual", "Stable")

    if "Decline" in pwr and "Decline" in la:
        findings.append("⚠️ <b>Mechanical/Physical Shift:</b> Both Power and Consistency declined. "
                        "This strongly suggests a mechanical flaw or a physical issue (fatigue/injury) affecting the swing path.")
    elif "Gain" in pwr and "Gain" in la:
        findings.append(f"🔥 <b>Optimized Mechanics:</b> Improvements in both Power and Consistency indicate "
                        f"<b>{player_name}</b> has found a repeatable, high-impact swing.")

    if "Gain" in disc and "Decline" in pwr:
        findings.append("🧘 <b>Heightened Selectivity:</b> Discipline improved, but Power dropped. "
                        "This often happens when a hitter becomes <i>too</i> selective, sacrificing aggression for better take decisions.")
    elif "Decline" in disc and "Gain" in pwr:
        findings.append("⚔️ <b>Increased Plate Aggression:</b> Discipline dropped while Power rose. "
                        "The hitter is likely 'selling out' for power, swinging harder at the cost of strike-zone control.")

    if "Decline" in res and "Stable" in pwr and "Stable" in la:
        findings.append("📉 <b>Pure Bad Luck:</b> Performance results (wOBA) dropped despite Power and Consistency "
                        "remaining steady. Physics says the hitter is doing everything right—results should follow.")
    elif "Gain" in res and "Stable" in pwr and "Stable" in la:
        findings.append("🍀 <b>Results Surge:</b> Results are improving faster than the underlying physics change, "
                        "indicating a period of high efficiency or favorable luck.")

    if not findings:
        sorted_stats = sorted(stats_list.items(), key=lambda x: abs(x[1]["effect_d"]), reverse=True)
        primary_col, primary_stat = sorted_stats[0]
        label = indicator_summary[primary_col]
        findings.append(f"📊 <b>Primary Driver:</b> The most significant shift was a <b>{label}</b> in "
                        f"<b>{PA_LABELS[primary_col]}</b>.")

    return findings


# ── Change-Point Deep-Dive ────────────────────────────────────────────────────

def render_cp_analysis(selected_date, player_name, before_data, after_data,
                       importance_df=None) -> None:
    """Render the full deep-dive analysis panel for a detected change point."""
    st.markdown("---")
    st.write(f"### 🔍 Deep-Dive Analysis: Shift on {selected_date}")
    st.write("Comparing the window of performance before and after this detected shift.")

    all_stats = {}
    for col in PA_INDICATORS:
        delta  = after_data[col].mean() - before_data[col].mean()
        pooled = np.sqrt((before_data[col].std() ** 2 + after_data[col].std() ** 2) / 2 + 1e-9)
        d      = delta / pooled
        all_stats[col] = {
            "delta": delta, "effect_d": d,
            "before": before_data[col].mean(), "after": after_data[col].mean(),
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
            "Which indicators drove this shift? A Random Forest classifier ranks each indicator "
            "by how much it contributed to this change point."
        )
        fig_imp = go.Figure(go.Bar(
            x=importance_df["Importance"],
            y=importance_df["Indicator"],
            orientation="h",
            marker_color=importance_df["Color"],
            text=[f"{v:.1%}" for v in importance_df["Importance"]],
            textposition="auto",
        ))
        fig_imp.update_layout(
            height=250,
            xaxis=dict(title="Feature Importance", tickformat=".0%", range=[0, 1]),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT), margin=dict(t=10, b=30, l=10, r=10),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        top_feature = importance_df.iloc[-1]["Indicator"]
        top_score   = importance_df.iloc[-1]["Importance"]
        st.caption(
            f"**Primary driver:** {top_feature} accounts for {top_score:.1%} of the "
            f"total feature importance — the strongest contributor at this change point."
        )

    if importance_df is None:
        st.write("#### 📊 Key Metric Shifts")
        with st.expander("ℹ️ How are these shifts calculated?"):
            st.markdown("""
            - **Effect Size (Cohen's d):** This measures the magnitude of the shift relative to the player's natural variability.
                - **0.2:** Small shift (normal fluctuation).
                - **0.5:** Medium shift (visible performance change).
                - **0.8+:** Large shift (major mechanical or approach overhaul).
            - **Primary Driver:** We identify this by finding the metric with the **highest absolute Effect Size**.
            """)
        cols = st.columns(4)
        for i, col_name in enumerate(PA_INDICATORS):
            s = all_stats[col_name]
            with cols[i]:
                st.metric(PA_LABELS[col_name], f"{s['after']:.3f}", delta=f"{s['delta']:+.3f}")
                st.caption(f"Effect Size: {s['effect_d']:.2f}")

    if importance_df is not None and not importance_df.empty:
        st.write(f"#### 📈 Distribution Shift - {top_feature}")
        top_label = importance_df.iloc[-1]["Indicator"]
        top_col   = next(col for col in PA_INDICATORS if PA_LABELS[col] == top_label)
    else:
        st.write("#### 📈 Distribution Shift (Cohen's d Primary Driver)")
        sorted_stats = sorted(all_stats.items(), key=lambda x: abs(x[1]["effect_d"]), reverse=True)
        top_col = sorted_stats[0][0]

    if before_data[top_col].dropna().empty or after_data[top_col].dropna().empty:
        st.warning(f"Not enough data to plot distribution for {PA_LABELS[top_col]}.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=before_data[top_col], name="Before Shift", marker_color=GREY, opacity=0.6))
        fig.add_trace(go.Histogram(x=after_data[top_col],  name="After Shift",  marker_color=TEAL, opacity=0.6))
        fig.update_layout(
            barmode="overlay",
            title=f"{PA_LABELS[top_col]}: Before vs. After Shift",
            xaxis_title=PA_LABELS[top_col], yaxis_title="Frequency (PA Count)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT), height=300, margin=dict(t=40, b=40, l=40, r=40),
        )
        st.plotly_chart(fig, use_container_width=True)
