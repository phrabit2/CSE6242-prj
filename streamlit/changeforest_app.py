import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

try:
    from changeforest import Control, changeforest
except Exception:  # pragma: no cover - runtime dependency check
    Control = None
    changeforest = None


SENSITIVITY_TO_MIN_SEG = {
    "Low": 0.10,
    "Medium": 0.05,
    "High": 0.02,
}


def get_data_path() -> Path:
    base = Path(__file__).resolve().parent
    return base.parent / "data" / "processed" / "Qualified_hitters_statcast_2021_2025_pa_master.csv"


@st.cache_data(show_spinner="Loading plate appearance data...")
def load_data() -> pd.DataFrame:
    cols = [
        "batter",
        "pa_uid",
        "game_date",
        "game_pk",
        "at_bat_number",
        "power_woba_seq_id",
        "power_efficiency",
        "woba_residual",
    ]
    df = pd.read_csv(get_data_path(), usecols=cols)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def changeforest_subdataset_generator(
    df: pd.DataFrame,
    selected_player_id: int,
    window: int = 50,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """min_periods defaults to window to avoid noisy estimates from partial windows."""
    if min_periods is None:
        min_periods = window

    base_cols = ["batter", "pa_uid", "game_date", "game_pk", "at_bat_number"]
    feature_cols = ["power_woba_seq_id", "power_efficiency", "woba_residual"]
    available_cols = [c for c in base_cols + feature_cols if c in df.columns]

    subdf = (
        df.loc[df["batter"] == selected_player_id, available_cols]
        .dropna(subset=["power_woba_seq_id", "power_efficiency", "woba_residual"])
        .sort_values("power_woba_seq_id")
        .reset_index(drop=True)
    )

    subdf[f"power_efficiency_rollmean_{window}"] = (
        subdf["power_efficiency"].rolling(window=window, min_periods=min_periods).mean()
    )
    subdf[f"woba_residual_rollmean_{window}"] = (
        subdf["woba_residual"].rolling(window=window, min_periods=min_periods).mean()
    )

    # Drop rows where rolling mean is NaN (first window-1 rows when min_periods=window)
    rollmean_cols = [f"power_efficiency_rollmean_{window}", f"woba_residual_rollmean_{window}"]
    subdf = subdf.dropna(subset=rollmean_cols).reset_index(drop=True)
    return subdf


def changeforest_subdataset_graph_generator(
    subdf: pd.DataFrame,
    selected_player_id: int | None = None,
    window: int = 50,
    figsize: tuple = (14, 5),
):
    """Plot the 2 ChangeForest input signals as a 1x2 panel (raw + rolling mean)."""
    plot_specs = [
        ("power_woba_seq_id", "power_efficiency", "Power Efficiency", "#1f77b4"),
        ("power_woba_seq_id", "woba_residual", "wOBA Residual", "#ff7f0e"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor("#f7f8fa")

    for ax, (x_col, y_col, title, color) in zip(axes, plot_specs):
        ax.set_facecolor("#ffffff")

        if subdf is None or subdf.empty:
            ax.set_title(f"{title} (no data)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Plate Appearance Sequence")
            ax.set_ylabel(y_col)
            ax.grid(axis="y", alpha=0.25, linewidth=0.8)
            continue

        smooth_col = f"{y_col}_rollmean_{window}"
        ax.scatter(subdf[x_col], subdf[y_col], s=10, alpha=0.22, color=color, edgecolor="none", label="Raw")

        if smooth_col in subdf.columns:
            ax.plot(subdf[x_col], subdf[smooth_col], linewidth=2.6, color=color, alpha=0.95, label=f"Rolling Mean (w={window})")

        ax.set_title(f"{title} (n={len(subdf):,})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Plate Appearance Sequence")
        ax.set_ylabel(y_col)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#b8bcc2")
        ax.spines["bottom"].set_color("#b8bcc2")
        ax.legend(frameon=False, fontsize=9, loc="best")

    fig.suptitle(
        "ChangeForest Input Signals"
        if selected_player_id is None
        else f"ChangeForest Input Signals | Player ID: {selected_player_id}",
        fontsize=14,
        fontweight="bold",
    )
    return fig, axes


def run_changeforest(
    subdf: pd.DataFrame,
    window: int,
    minimal_relative_segment_length: float,
    use_rollmean: bool = True,
    standardize: bool = True,
):
    if subdf is None or subdf.empty:
        return None, [], None, None

    if use_rollmean:
        feature_names = [
            f"power_efficiency_rollmean_{window}",
            f"woba_residual_rollmean_{window}",
        ]
    else:
        feature_names = ["power_efficiency", "woba_residual"]

    missing_cols = [c for c in feature_names if c not in subdf.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in subdf: {missing_cols}")

    X_used = subdf[feature_names].dropna().values

    if len(X_used) == 0:
        raise ValueError("All rows contain NaN after dropping — check rolling window / min_periods")

    if np.isnan(X_used).any():
        raise ValueError("X_used contains NaN — check rolling window / min_periods")

    if standardize:
        X_used = StandardScaler().fit_transform(X_used)

    control = Control(
        model_selection_alpha=0.02,
        minimal_relative_segment_length=minimal_relative_segment_length,
    )

    result = changeforest(X_used, method="random_forest", control=control)
    cps = [int(cp) for cp in result.split_points() if 0 <= int(cp) < len(subdf)]
    return result, cps, X_used, feature_names


def plot_changeforest_result(
    subdf: pd.DataFrame,
    cps: list[int],
    window: int,
    player_id: int,
    min_rel_seg_len: float,
    use_rollmean: bool = True,
    figsize: tuple = (14, 6),
):
    """Single combined chart with both signals, CP vlines, and end-of-line annotations."""
    if use_rollmean:
        y_cols = [
            f"power_efficiency_rollmean_{window}",
            f"woba_residual_rollmean_{window}",
        ]
    else:
        y_cols = ["power_efficiency", "woba_residual"]

    line_specs = [
        (y_cols[0], "Power Efficiency", "#1f77b4"),
        (y_cols[1], "wOBA Residual", "#ff7f0e"),
    ]

    dates = pd.to_datetime(subdf["game_date"])
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor("#f7f8fa")
    ax.set_facecolor("#ffffff")

    label_offsets = [8, -65]  # horizontal offsets to reduce overlap
    for (y_col, label, color), x_offset in zip(line_specs, label_offsets):
        ax.plot(dates, subdf[y_col], linewidth=2.5, color=color, label=label, alpha=0.95)

        valid = subdf[[y_col, "game_date"]].dropna()
        if not valid.empty:
            last_idx = valid.index[-1]
            last_date = pd.to_datetime(subdf.loc[last_idx, "game_date"])
            last_value = subdf.loc[last_idx, y_col]
            ax.scatter(last_date, last_value, color=color, s=42, zorder=5)
            ax.annotate(
                label,
                xy=(last_date, last_value),
                xytext=(x_offset, 0),
                textcoords="offset points",
                color=color,
                fontsize=10,
                va="center",
                fontweight="bold",
            )

    cp_dates = []
    for cp in cps:
        if 0 <= cp < len(subdf):
            cp_date = dates.iloc[cp]
            cp_dates.append(cp_date)
            ax.axvline(x=cp_date, color="#c43b3b", linestyle=(0, (4, 4)), alpha=0.7, linewidth=1.8)

    ax.set_xlabel("Game Date")
    ax.set_ylabel("Metric Value")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b8bcc2")
    ax.spines["bottom"].set_color("#b8bcc2")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)
    ax.set_title(
        f"ChangeForest Result | alpha=0.02 | Player ID: {player_id}"
        f" | min_rel_seg_len={min_rel_seg_len} | window={window} | change points={len(cp_dates)}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper left", frameon=False)
    return fig


# ------------------------------------------------------------------ #
# Evaluation helpers (ported from cdp_advanced2_changeforest.py)      #
# ------------------------------------------------------------------ #

def build_cp_eval_dfs(
    subdf: pd.DataFrame,
    cps: list[int],
    feature_names: list[str],
    compare_window: int = 50,
) -> dict:
    """Build one evaluation DataFrame per feature (before/after each index)."""
    n = len(subdf)

    missing_cols = [c for c in feature_names if c not in subdf.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in subdf: {missing_cols}")

    if n <= 2 * compare_window:
        raise ValueError(
            f"subdf too short ({n} rows) for compare_window={compare_window}. "
            f"Need more than {2 * compare_window} rows."
        )

    cps_set = set(cps)
    eval_dfs = {}

    for col in feature_names:
        rows = []
        for i in range(compare_window, n - compare_window):
            before = subdf[col].iloc[i - compare_window:i]
            after = subdf[col].iloc[i:i + compare_window]
            rows.append({
                "feature": col,
                "cp": "Y" if i in cps_set else "N",
                "mean_before": before.mean(),
                "mean_after": after.mean(),
                "mean_diff": after.mean() - before.mean(),
                "std_before": before.std(),
                "std_after": after.std(),
                "std_diff": after.std() - before.std(),
                "abs_mean_diff": abs(after.mean() - before.mean()),
                "abs_std_diff": abs(after.std() - before.std()),
            })
        eval_dfs[col] = pd.DataFrame(rows)

    return eval_dfs


def plot_cp_eval_comparison(eval_dfs: dict, figsize: tuple = (12, 4)):
    """Bar chart: CP vs non-CP average absolute difference per feature."""
    n_features = len(eval_dfs)
    fig, axes = plt.subplots(1, n_features, figsize=figsize, constrained_layout=True)
    if n_features == 1:
        axes = [axes]

    for ax, (feature, df_eval) in zip(axes, eval_dfs.items()):
        if df_eval.empty:
            ax.set_title(f"{feature}\n(no data)")
            continue
        summary = df_eval.groupby("cp")[["abs_mean_diff", "abs_std_diff"]].mean()
        summary.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(feature.replace("_rollmean_", "\nrollmean="))
        ax.set_xlabel("Detected Change Point")
        ax.set_ylabel("Average Absolute Difference")
        ax.grid(alpha=0.3)

    return fig, axes


def run_parameter_stability_on_subdf(
    subdf: pd.DataFrame,
    window: int,
    min_rel_seg_len_list: tuple = (0.01, 0.05, 0.10),
    use_rollmean: bool = True,
    standardize: bool = True,
) -> pd.DataFrame:
    """Evaluate ChangeForest sensitivity to minimal_relative_segment_length."""
    rows = []
    for min_rel_seg_len in min_rel_seg_len_list:
        _, cps, _, _ = run_changeforest(
            subdf,
            window=window,
            use_rollmean=use_rollmean,
            standardize=standardize,
            minimal_relative_segment_length=min_rel_seg_len,
        )
        rows.append({
            "window": window,
            "minimal_relative_segment_length": min_rel_seg_len,
            "n_change_points": len(cps),
            "change_points": cps,
        })
    return pd.DataFrame(rows).sort_values("minimal_relative_segment_length").reset_index(drop=True)


def main():
    st.set_page_config(page_title="ChangeForest CPD", page_icon="\U0001F332", layout="wide")
    st.title("ChangeForest Streamlit App")
    st.caption("Player ID-first CPD with rolling window and sensitivity mapping")

    if changeforest is None or Control is None:
        st.error(
            "The 'changeforest' package is not installed in this environment. "
            "Install it first, then rerun this app."
        )
        st.stop()

    df = load_data()
    if df.empty:
        st.error("No data loaded.")
        st.stop()

    player_ids = sorted(df["batter"].dropna().astype(int).unique().tolist())

    with st.sidebar:
        st.header("Controls")
        selected_player_id = st.selectbox("Player ID", options=player_ids, index=0)
        rolling_window = st.slider("Rolling window", min_value=20, max_value=120, value=50, step=5)
        sensitivity = st.radio("Sensitivity", options=["Low", "Medium", "High"], horizontal=True, index=1)
        min_rel_seg_len = SENSITIVITY_TO_MIN_SEG[sensitivity]

    subdf = changeforest_subdataset_generator(
        df=df,
        selected_player_id=selected_player_id,
        window=rolling_window,
    )

    if subdf.empty:
        st.warning("No rows available for this player after required feature filtering.")
        st.stop()

    result, cps, _, feature_names = run_changeforest(
        subdf=subdf,
        window=rolling_window,
        minimal_relative_segment_length=min_rel_seg_len,
        use_rollmean=True,
        standardize=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Player ID", f"{selected_player_id}")
    c2.metric("Rows used", f"{len(subdf):,}")
    c3.metric("Sensitivity", sensitivity)
    c4.metric("Detected CPs", f"{len(cps)}")

    tab_main, tab_signals, tab_eval, tab_stability = st.tabs(
        ["Result", "Input Signals", "Before/After Eval", "Parameter Stability"]
    )

    with tab_main:
        st.write("**Features used:**", feature_names)
        st.write("**Detected change points:**", cps)

        if len(cps) > 0:
            cp_cols = [c for c in ["power_woba_seq_id", "game_date", "pa_uid"] if c in subdf.columns]
            st.dataframe(subdf.iloc[cps][cp_cols].reset_index(drop=True), use_container_width=True)

        fig = plot_changeforest_result(
            subdf=subdf,
            cps=cps,
            window=rolling_window,
            player_id=selected_player_id,
            min_rel_seg_len=min_rel_seg_len,
        )
        st.pyplot(fig)

    with tab_signals:
        st.caption("Raw observations and rolling mean for each input feature.")
        fig_sig, _ = changeforest_subdataset_graph_generator(
            subdf=subdf,
            selected_player_id=selected_player_id,
            window=rolling_window,
        )
        st.pyplot(fig_sig)

    with tab_eval:
        st.caption("Average absolute difference in mean and std before vs after each index. CP=Y means a detected change point.")
        try:
            eval_dfs = build_cp_eval_dfs(
                subdf=subdf,
                cps=cps,
                feature_names=feature_names,
                compare_window=100,
            )
            fig_eval, _ = plot_cp_eval_comparison(eval_dfs)
            st.pyplot(fig_eval)
        except ValueError as e:
            st.warning(f"Evaluation skipped: {e}")

    with tab_stability:
        st.caption("Number of detected change points across different min_rel_seg_len values.")
        with st.spinner("Running parameter stability sweep..."):
            stability_df = run_parameter_stability_on_subdf(subdf=subdf, window=rolling_window)
        st.dataframe(stability_df, use_container_width=True)

    with st.expander("Sensitivity mapping used"):
        st.write(SENSITIVITY_TO_MIN_SEG)


if __name__ == "__main__":
    main()
