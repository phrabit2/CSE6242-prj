#!/usr/bin/env python
# coding: utf-8

# ============================================================
# changeforest — Change Point Detection
# ============================================================

# ============================================================
# 0. Imports and data loading
# ============================================================
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ruptures as rpt
from sklearn.preprocessing import StandardScaler
from changeforest import changeforest, Control


# ============================================================
# Step 0. Load data
# ============================================================
DATA_FILE = "Qualified_hitters_statcast_2021_2025_pa_master.csv"
csv_path = Path(__file__).resolve().parent.parent / "data" / "processed" / DATA_FILE
df = pd.read_csv(csv_path)

# Data availability summary for the four CPD indicators
CPD_INDICATORS = [
    "hitting_decisions_score",
    "power_efficiency",
    "woba_residual",
    "launch_angle_stability_50pa",
]

summary_df = pd.DataFrame({
    "indicator": CPD_INDICATORS,
    "rows_with_value": [df[col].notna().sum() for col in CPD_INDICATORS],
})
summary_df["pct_of_total"] = (summary_df["rows_with_value"] / len(df) * 100).round(2)

print(f"Total rows: {len(df):,}  |  Total batters: {df['batter'].nunique()}")
print(summary_df.to_string(index=False))


# ============================================================
# Step 1: Data Loading & Subdataset Preparation
# ============================================================

def changeforest_subdataset_generator(df, selected_player_id, window=50, min_periods=None):
    """
    Generate one aligned multivariate CPD subdataset for ChangeForest
    using power_efficiency + woba_residual.

    min_periods defaults to `window` so the rolling mean is only computed
    once a full window of data is available, avoiding noisy estimates near
    the start of the series.  Pass an explicit value to override.
    """
    if min_periods is None:
        min_periods = window

    base_cols = ["batter", "pa_uid", "game_date", "game_pk", "at_bat_number"]
    feature_cols = ["power_woba_seq_id", "power_efficiency", "woba_residual"]

    subdf = (
        df.loc[df["batter"] == selected_player_id, base_cols + feature_cols]
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

    # Drop rows where rolling mean is NaN (first `window-1` rows when min_periods=window)
    rollmean_cols = [f"power_efficiency_rollmean_{window}", f"woba_residual_rollmean_{window}"]
    subdf = subdf.dropna(subset=rollmean_cols).reset_index(drop=True)

    return subdf


def changeforest_subdataset_graph_generator(subdf, selected_player_id=None, window=50, figsize=(14, 5)):
    """
    Plot the 2 ChangeForest indicators as a 1x2 panel,
    showing raw signal and rolling mean.
    """
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

        # Raw observations are de-emphasized so trend lines are easier to read.
        ax.scatter(
            subdf[x_col],
            subdf[y_col],
            s=10,
            alpha=0.22,
            color=color,
            edgecolor="none",
            label="Raw",
        )

        if smooth_col in subdf.columns:
            ax.plot(
                subdf[x_col],
                subdf[smooth_col],
                linewidth=2.6,
                color=color,
                alpha=0.95,
                label=f"Rolling Mean (w={window})",
            )

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


# ============================================================
# Step 2: Changeforest
# ============================================================

def run_changeforest(
    subdf,
    window=50,
    use_rollmean=True,
    standardize=True,
    minimal_relative_segment_length=0.05,
):
    """
    Run ChangeForest on one player's aligned 2-feature dataset.

    Fixed:
    ------
    model_selection_alpha = 0.02

    User can change:
    ----------------
    - window
    - minimal_relative_segment_length
    """
    if subdf is None or subdf.empty:
        return None, [], None, None

    if use_rollmean:
        feature_names = [
            f"power_efficiency_rollmean_{window}",
            f"woba_residual_rollmean_{window}",
        ]
    else:
        feature_names = [
            "power_efficiency",
            "woba_residual",
        ]

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

    result = changeforest(
        X_used,
        method="random_forest",
        control=control,
    )

    cps = result.split_points()

    return result, cps, X_used, feature_names


def plot_changeforest_result(
    subdf,
    cps,
    window=50,
    use_rollmean=True,
    selected_player_id=None,
    minimal_relative_segment_length=None,
    figsize=(14, 6),
):
    """
    Plot both ChangeForest signals on a single time-series chart using game_date as X-axis.
    """
    if use_rollmean:
        y_cols = [
            f"power_efficiency_rollmean_{window}",
            f"woba_residual_rollmean_{window}",
        ]
    else:
        y_cols = [
            "power_efficiency",
            "woba_residual",
        ]

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
        ax.plot(
            dates,
            subdf[y_col],
            linewidth=2.5,
            color=color,
            label=label,
            alpha=0.95,
        )

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
            ax.axvline(
                x=cp_date,
                color="#c43b3b",
                linestyle=(0, (4, 4)),
                alpha=0.7,
                linewidth=1.8,
            )

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

    main_title = "ChangeForest Result | alpha=0.02"
    if selected_player_id is not None:
        main_title += f" | Player ID: {selected_player_id}"
    if minimal_relative_segment_length is not None:
        main_title += f" | min_rel_seg_len={minimal_relative_segment_length}"
    main_title += f" | window={window}"
    main_title += f" | change points={len(cp_dates)}"

    ax.set_title(main_title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=False)

    return fig, ax


# ============================================================
# Step 3: Evaluate
#   1. Before vs After Statistical Difference
#   2. Robustness / Parameter Stability
#   3. Domain Validation — do detected CPs align with real baseball events?
# ============================================================

# 1. Before vs After Statistical Difference
# Validate change points by comparing statistical properties before and after
# each detected point, ensuring significant shifts in mean or variance.

def build_cp_eval_dfs(subdf, cps, feature_names, compare_window=50):
    """
    Build one evaluation dataframe per feature.
    compare_window is the before/after comparison width,
    not the rolling window used in feature construction.
    """
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


def plot_cp_eval_comparison(eval_dfs, figsize=(12, 4)):
    """
    Plot CP vs non-CP comparison for each feature.
    """
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

        clean_title = feature.replace("_rollmean_", "\nrollmean=")
        ax.set_title(clean_title)
        ax.set_xlabel("Detected Change Point")
        ax.set_ylabel("Average Absolute Difference")
        ax.grid(alpha=0.3)

    return fig, axes


# 2. Robustness / Parameter Stability

def run_parameter_stability_on_subdf(
    subdf,
    window,
    min_rel_seg_len_list=(0.01, 0.05, 0.10),
    use_rollmean=True,
    standardize=True,
):
    """
    Evaluate ChangeForest sensitivity to minimal_relative_segment_length
    for a fixed subdf/window that is already prepared.
    """
    rows = []

    for min_rel_seg_len in min_rel_seg_len_list:
        result, cps, X_used, feature_names = run_changeforest(
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


# ============================================================
# Implement
# ============================================================

if __name__ == "__main__":
    selected_player_id = 592450
    window = 50
    min_rel_seg_len = 0.05

    subdf = changeforest_subdataset_generator(
        df,
        selected_player_id=selected_player_id,
        window=window,
    )

    result, cps, X_used, feature_names = run_changeforest(
        subdf,
        window=window,
        use_rollmean=True,
        standardize=True,
        minimal_relative_segment_length=min_rel_seg_len,
    )

    print("Features used:", feature_names)
    print("Detected change points (no.):", len(cps))
    print("Detected change points:", cps)

    if len(cps) > 0:
        print(subdf.iloc[cps][["power_woba_seq_id", "game_date", "pa_uid"]])

    fig, axes = plot_changeforest_result(
        subdf,
        cps,
        window=window,
        use_rollmean=True,
        selected_player_id=selected_player_id,
        minimal_relative_segment_length=min_rel_seg_len,
    )
    plt.show()

    # 1. Before vs After Statistical Difference
    # evaluation should follow actual feature_names
    eval_dfs = build_cp_eval_dfs(
        subdf=subdf,
        cps=cps,
        feature_names=feature_names,
        compare_window=100,  # this is evaluation window, not rolling window
    )

    fig, axes = plot_cp_eval_comparison(eval_dfs)
    plt.show()

    # 2. Robustness / Parameter Stability
    cp_summary = run_parameter_stability_on_subdf(subdf=subdf, window=window)
    print(cp_summary.to_string(index=False))
