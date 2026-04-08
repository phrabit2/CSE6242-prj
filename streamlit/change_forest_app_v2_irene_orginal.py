#!/usr/bin/env python
# coding: utf-8
"""
ChangeForest Change-Point Detection — Streamlit App v2

Sidebar controls : player name | rolling window | CDP sensitivity
Tabs             :
  1. ChangeForest Result + Input Signal
  2. Before / After Eval
  3. Parameter Stability
"""

import os
from pathlib import Path

import gdown
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

try:
    from changeforest import Control, changeforest
except Exception:  # pragma: no cover — runtime dependency check
    Control = None
    changeforest = None


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SENSITIVITY_TO_MIN_SEG = {
    "Low":    0.10,
    "Medium": 0.05,
    "High":   0.02,
}

CPD_INDICATORS = [
    "hitting_decisions_score",
    "power_efficiency",
    "woba_residual",
    "launch_angle_stability_50pa",
]

INDICATOR_LABELS = {
    "hitting_decisions_score":       "Hitting Decisions Score",
    "power_efficiency":              "Power Efficiency",
    "woba_residual":                 "wOBA Residual",
    "launch_angle_stability_50pa":   "Launch Angle Stability",
}

INDICATOR_COLORS = {
    "hitting_decisions_score":       "#2ca02c",
    "power_efficiency":              "#1f77b4",
    "woba_residual":                 "#ff7f0e",
    "launch_angle_stability_50pa":   "#d62728",
}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading plate appearance data…")
def load_data() -> pd.DataFrame:
    DATA_FILE_ID = "1G8eA6gX8hdCwWp1YWddAMmwt6R62tcmA"
    tmp_path = "/tmp/Qualified_hitters_statcast_2021_2025_pa_master.csv"
    local_path = (
        Path(__file__).resolve().parent.parent
        / "data" / "processed"
        / "Qualified_hitters_statcast_2021_2025_pa_master.csv"
    )

    cols = [
        "batter", "player_name", "pa_uid", "game_date", "game_pk",
        "at_bat_number", "cf_seq_id",
        "hitting_decisions_score", "power_efficiency",
        "woba_residual", "launch_angle_stability_50pa",
    ]

    if local_path.exists():
        data_path = str(local_path)
    elif os.path.exists(tmp_path):
        data_path = tmp_path
    else:
        url = f"https://drive.google.com/uc?id={DATA_FILE_ID}"
        gdown.download(url, tmp_path, quiet=False, fuzzy=True)
        data_path = tmp_path

    df = pd.read_csv(data_path, usecols=cols)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# CPD pipeline  (ported from cdp_changeforest.py)
# ──────────────────────────────────────────────────────────────────────────────

def changeforest_subdataset_generator(
    df: pd.DataFrame,
    selected_player_id: int,
    window: int = 50,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Build aligned multivariate CPD subdataset for one player."""
    if min_periods is None:
        min_periods = window

    feature_cols = ["cf_seq_id"] + CPD_INDICATORS
    base_cols = ["batter", "player_name", "pa_uid", "game_date", "game_pk", "at_bat_number"]

    subdf = (
        df.loc[df["batter"] == selected_player_id, base_cols + feature_cols]
        .dropna(subset=feature_cols)
        .sort_values("cf_seq_id")
        .reset_index(drop=True)
    )

    for col in CPD_INDICATORS:
        subdf[f"{col}_rollmean_{window}"] = (
            subdf[col].rolling(window=window, min_periods=min_periods).mean()
        )

    rollmean_cols = [f"{col}_rollmean_{window}" for col in CPD_INDICATORS]
    subdf = subdf.dropna(subset=rollmean_cols).reset_index(drop=True)
    return subdf


def run_changeforest(
    subdf: pd.DataFrame,
    window: int = 50,
    use_rollmean: bool = True,
    standardize: bool = True,
    minimal_relative_segment_length: float = 0.05,
):
    """Run ChangeForest on all 4 CPD indicators."""
    if subdf is None or subdf.empty:
        return None, [], None, None

    feature_names = (
        [f"{col}_rollmean_{window}" for col in CPD_INDICATORS]
        if use_rollmean
        else list(CPD_INDICATORS)
    )

    missing = [c for c in feature_names if c not in subdf.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    X = subdf[feature_names].dropna().values
    if len(X) == 0 or np.isnan(X).any():
        raise ValueError(
            "Feature matrix is empty or contains NaN — "
            "try a smaller rolling window or a player with more data."
        )

    if standardize:
        X = StandardScaler().fit_transform(X)

    control = Control(
        model_selection_alpha=0.02,
        minimal_relative_segment_length=minimal_relative_segment_length,
    )
    result = changeforest(X, method="random_forest", control=control)
    cps = [int(cp) for cp in result.split_points() if 0 <= int(cp) < len(subdf)]
    return result, cps, X, feature_names


# ── Plot A: Input Signal 2×2 panel ────────────────────────────────────────────

def plot_input_signals(subdf: pd.DataFrame, window: int, player_name: str = "") -> plt.Figure:
    """2×2 panel: raw scatter + rolling mean for all 4 indicators."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=True)
    axes = axes.flatten()
    fig.patch.set_facecolor("#f7f8fa")

    for ax, col in zip(axes, CPD_INDICATORS):
        color = INDICATOR_COLORS[col]
        label = INDICATOR_LABELS[col]
        smooth_col = f"{col}_rollmean_{window}"
        ax.set_facecolor("#ffffff")

        if subdf is None or subdf.empty:
            ax.set_title(f"{label} (no data)", fontsize=11, fontweight="bold")
            continue

        ax.scatter(
            subdf["cf_seq_id"], subdf[col],
            s=8, alpha=0.22, color=color, edgecolor="none", label="Raw",
        )
        if smooth_col in subdf.columns:
            ax.plot(
                subdf["cf_seq_id"], subdf[smooth_col],
                linewidth=2.4, color=color, alpha=0.95,
                label=f"Rolling Mean (w={window})",
            )

        ax.set_title(f"{label}  (n={len(subdf):,})", fontsize=11, fontweight="bold")
        ax.set_xlabel("CF Sequence ID")
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Input Signals — {player_name}" if player_name else "Input Signals",
        fontsize=13, fontweight="bold",
    )
    return fig


# ── Plot B: ChangeForest result (all 4 signals + CP vertical lines) ───────────

def plot_changeforest_result(
    subdf: pd.DataFrame,
    cps: list,
    window: int,
    player_name: str,
    min_rel_seg_len: float,
) -> plt.Figure:
    """Single time-series chart with all 4 rolling-mean signals + CP dashed lines."""
    dates = pd.to_datetime(subdf["game_date"])
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    fig.patch.set_facecolor("#f7f8fa")
    ax.set_facecolor("#ffffff")

    label_x_offsets = [8, -68, 8, -68]

    for col, x_offset in zip(CPD_INDICATORS, label_x_offsets):
        color = INDICATOR_COLORS[col]
        label = INDICATOR_LABELS[col]
        y_col = f"{col}_rollmean_{window}"

        ax.plot(dates, subdf[y_col], linewidth=2.2, color=color, label=label, alpha=0.9)

        valid = subdf[[y_col, "game_date"]].dropna()
        if not valid.empty:
            last_idx = valid.index[-1]
            last_date = pd.to_datetime(subdf.loc[last_idx, "game_date"])
            last_val = subdf.loc[last_idx, y_col]
            ax.scatter(last_date, last_val, color=color, s=40, zorder=5)
            ax.annotate(
                label,
                xy=(last_date, last_val),
                xytext=(x_offset, 0),
                textcoords="offset points",
                color=color, fontsize=9, va="center", fontweight="bold",
            )

    for cp in cps:
        if 0 <= cp < len(subdf):
            ax.axvline(
                x=dates.iloc[cp],
                color="#c43b3b", linestyle=(0, (4, 4)), alpha=0.75, linewidth=1.8,
            )

    ax.set_xlabel("Game Date")
    ax.set_ylabel("Metric Value (standardized rolling mean)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)
    ax.set_title(
        f"ChangeForest Result  |  {player_name}"
        f"  |  window={window}  |  min_rel_seg_len={min_rel_seg_len}  |  CPs={len(cps)}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper left", frameon=False)
    return fig


# ── Before / After Eval ────────────────────────────────────────────────────────

def build_cp_eval_dfs(
    subdf: pd.DataFrame,
    cps: list,
    feature_names: list,
    compare_window: int = 50,
) -> dict:
    """
    For every position i in [compare_window, n-compare_window], compute
    before/after statistics and flag whether i is a detected CP.
    """
    n = len(subdf)
    if n <= 2 * compare_window:
        return {}

    cps_set = set(cps)
    eval_dfs = {}

    for col in feature_names:
        rows = []
        for i in range(compare_window, n - compare_window):
            before = subdf[col].iloc[i - compare_window : i]
            after  = subdf[col].iloc[i : i + compare_window]
            rows.append({
                "feature":       col,
                "cp":            "Y" if i in cps_set else "N",
                "mean_before":   before.mean(),
                "mean_after":    after.mean(),
                "mean_diff":     after.mean() - before.mean(),
                "std_before":    before.std(),
                "std_after":     after.std(),
                "std_diff":      after.std() - before.std(),
                "abs_mean_diff": abs(after.mean() - before.mean()),
                "abs_std_diff":  abs(after.std() - before.std()),
            })
        eval_dfs[col] = pd.DataFrame(rows)

    return eval_dfs


def plot_cp_eval_comparison(eval_dfs: dict) -> plt.Figure | None:
    n = len(eval_dfs)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (feature, df_eval) in zip(axes, eval_dfs.items()):
        if df_eval.empty or "cp" not in df_eval.columns:
            ax.set_title(f"{feature}\n(no data)")
            continue

        summary = df_eval.groupby("cp")[["abs_mean_diff", "abs_std_diff"]].mean()
        summary.plot(kind="bar", ax=ax, rot=0, color=["#1f77b4", "#ff7f0e"])

        base_col = feature.split("_rollmean_")[0]
        ax.set_title(INDICATOR_LABELS.get(base_col, feature), fontsize=11, fontweight="bold")
        ax.set_xlabel("Is Change Point?  (Y = detected CP, N = non-CP)")
        ax.set_ylabel("Avg Absolute Difference")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(["Abs Mean Diff", "Abs Std Diff"], frameon=False, fontsize=9)

    fig.suptitle(
        "Before / After Comparison at Detected Change Points",
        fontsize=13, fontweight="bold",
    )
    return fig


# ── Parameter Stability ────────────────────────────────────────────────────────

def run_parameter_stability(
    subdf: pd.DataFrame,
    window: int,
    min_rel_seg_len_list: list | tuple = (0.02, 0.05, 0.10),
    use_rollmean: bool = True,
    standardize: bool = True,
) -> pd.DataFrame:
    """Run ChangeForest for each sensitivity level and collect CP counts."""
    inv_map = {v: k for k, v in SENSITIVITY_TO_MIN_SEG.items()}
    rows = []

    for val in sorted(min_rel_seg_len_list):
        try:
            _, cps, _, _ = run_changeforest(
                subdf, window=window, use_rollmean=use_rollmean,
                standardize=standardize, minimal_relative_segment_length=val,
            )
            n_cp, cp_list = len(cps), cps
        except Exception as exc:
            n_cp, cp_list = None, []
            st.warning(f"Stability run failed for min_rel_seg_len={val}: {exc}")

        rows.append({
            "Sensitivity":      inv_map.get(val, str(val)),
            "min_rel_seg_len":  val,
            "# Change Points":  n_cp,
            "CP Indices":       str(cp_list),
        })

    return pd.DataFrame(rows)


def plot_parameter_stability(stability_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    fig.patch.set_facecolor("#f7f8fa")
    ax.set_facecolor("#ffffff")

    valid = stability_df.dropna(subset=["# Change Points"])
    x_labels = valid["Sensitivity"] + "\n(min=" + valid["min_rel_seg_len"].astype(str) + ")"
    bars = ax.bar(x_labels, valid["# Change Points"], color="#1f77b4", alpha=0.82, width=0.5)

    for bar, val in zip(bars, valid["# Change Points"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(int(val)),
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_xlabel("Sensitivity level  (min_rel_seg_len)")
    ax.set_ylabel("# Change Points Detected")
    ax.set_title(
        "Parameter Stability — # Change Points vs Sensitivity",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main Streamlit App
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="ChangeForest CPD v2", page_icon="🌲", layout="wide")
    st.title("🌲 ChangeForest Change-Point Detection")
    st.caption(
        "Multi-feature CPD for MLB hitters · 4 indicators · rolling window · sensitivity tuning"
    )

    # ── Dependency check ─────────────────────────────────────────────────────
    if changeforest is None or Control is None:
        st.error(
            "The `changeforest` package is not installed. "
            "Run `pip install changeforest` and restart the app."
        )
        st.stop()

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_data()
    if df.empty:
        st.error("No data loaded.")
        st.stop()

    # ── Build player name list ─────────────────────────────────────────────────
    name_id_map = (
        df[["player_name", "batter"]]
        .dropna()
        .drop_duplicates()
        .sort_values("player_name")
    )
    player_names = sorted(name_id_map["player_name"].unique().tolist())

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Controls")

        default_idx = player_names.index("Trout, Mike") if "Trout, Mike" in player_names else 0
        selected_name = st.selectbox("Player Name", options=player_names, index=default_idx)

        matching_ids = name_id_map.loc[
            name_id_map["player_name"] == selected_name, "batter"
        ].tolist()
        selected_player_id = int(matching_ids[0]) if matching_ids else None

        rolling_window = st.slider(
            "Rolling Window (PAs)", min_value=20, max_value=120, value=50, step=5
        )

        sensitivity = st.radio(
            "CDP Sensitivity",
            options=["Low", "Medium", "High"],
            horizontal=True,
            index=1,
        )
        min_rel_seg_len = SENSITIVITY_TO_MIN_SEG[sensitivity]

        st.markdown("---")
        st.caption(f"**Batter ID:** `{selected_player_id}`")
        st.caption(f"**min_rel_seg_len:** `{min_rel_seg_len}`")
        st.markdown(
            "**Sensitivity guide:**\n"
            "| Level | min_rel_seg_len | Effect |\n"
            "|---|---|---|\n"
            "| Low | 0.10 | fewer CPs |\n"
            "| Medium | 0.05 | balanced |\n"
            "| High | 0.02 | more CPs |"
        )

    if selected_player_id is None:
        st.warning("Player not found in dataset.")
        st.stop()

    # ── Prepare subdataset ────────────────────────────────────────────────────
    with st.spinner("Preparing player dataset…"):
        subdf = changeforest_subdataset_generator(
            df, selected_player_id, window=rolling_window
        )

    if subdf.empty:
        st.warning(
            f"No rows available for **{selected_name}** after filtering for all 4 indicators. "
            "Try a smaller rolling window."
        )
        st.stop()

    # ── Run ChangeForest ──────────────────────────────────────────────────────
    with st.spinner("Running ChangeForest random-forest CPD…"):
        try:
            _, cps, _, feature_names = run_changeforest(
                subdf,
                window=rolling_window,
                minimal_relative_segment_length=min_rel_seg_len,
                use_rollmean=True,
                standardize=True,
            )
        except Exception as exc:
            st.error(f"ChangeForest failed: {exc}")
            st.stop()

    # ── Summary metric bar ────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Player", selected_name)
    c2.metric("Batter ID", str(selected_player_id))
    c3.metric("PA Rows Used", f"{len(subdf):,}")
    c4.metric("Sensitivity", sensitivity)
    c5.metric("Detected CPs", str(len(cps)))

    if cps:
        with st.expander(f"📍 Change-point rows ({len(cps)} points)"):
            st.dataframe(
                subdf.iloc[cps][["cf_seq_id", "game_date", "pa_uid"]].reset_index(drop=True),
                use_container_width=True,
            )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        [
            "📈 ChangeForest Result + Input Signal",
            "📊 Before / After Eval",
            "🔧 Parameter Stability",
        ]
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 1 — ChangeForest Result + Input Signal
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("ChangeForest Result")
        st.caption(
            "All 4 standardized rolling-mean signals on a single timeline. "
            "Dashed red vertical lines mark detected change points."
        )
        fig_result = plot_changeforest_result(
            subdf, cps, rolling_window, selected_name, min_rel_seg_len
        )
        st.pyplot(fig_result, use_container_width=True)

        st.markdown("---")
        st.subheader("Input Signals  (Raw + Rolling Mean)")
        st.caption(
            "2 × 2 panel showing each indicator's raw plate-appearance values "
            "and its rolling mean (same window as CPD)."
        )
        fig_signals = plot_input_signals(subdf, rolling_window, selected_name)
        st.pyplot(fig_signals, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 2 — Before / After Eval
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Before / After Statistical Comparison")
        st.caption(
            "For every sequence position, the metric distribution in the window **before** "
            "vs **after** is compared. Detected CPs (Y) should show larger absolute shifts "
            "in mean and std than non-CP positions (N)."
        )

        compare_window = max(5, min(50, len(subdf) // 4))

        if len(subdf) <= 2 * compare_window:
            st.warning(
                f"Not enough data for before/after evaluation "
                f"(need > {2 * compare_window} rows, have {len(subdf)}). "
                "Try a smaller rolling window."
            )
        else:
            with st.spinner("Computing before/after statistics…"):
                eval_dfs = build_cp_eval_dfs(
                    subdf, cps, feature_names, compare_window=compare_window
                )

            if not eval_dfs:
                st.warning("Could not build eval dataframes.")
            else:
                fig_eval = plot_cp_eval_comparison(eval_dfs)
                if fig_eval:
                    st.pyplot(fig_eval, use_container_width=True)

                with st.expander("📋 Detailed eval summary tables"):
                    for feat, df_eval in eval_dfs.items():
                        base = feat.split("_rollmean_")[0]
                        st.write(f"**{INDICATOR_LABELS.get(base, feat)}**")
                        st.dataframe(
                            df_eval.groupby("cp")[
                                [
                                    "mean_before", "mean_after", "mean_diff",
                                    "std_before",  "std_after",  "std_diff",
                                    "abs_mean_diff", "abs_std_diff",
                                ]
                            ].mean().round(4),
                            use_container_width=True,
                        )

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 3 — Parameter Stability
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Parameter Stability")
        st.caption(
            "Runs ChangeForest across **all three sensitivity levels** "
            "(Low / Medium / High) on the same player + window. "
            "Stable results → similar CP counts across levels. "
            "Large variation → the detected CPs are sensitive to this parameter."
        )

        stability_vals = sorted(SENSITIVITY_TO_MIN_SEG.values())  # [0.02, 0.05, 0.10]

        with st.spinner("Running stability analysis across all sensitivity levels…"):
            stability_df = run_parameter_stability(
                subdf, rolling_window, min_rel_seg_len_list=stability_vals
            )

        fig_stability = plot_parameter_stability(stability_df)
        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            st.pyplot(fig_stability, use_container_width=True)
        with col_table:
            st.dataframe(stability_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
