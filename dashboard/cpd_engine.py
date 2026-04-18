import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from config import PA_INDICATORS, PA_LABELS, PA_COLORS

try:
    from changeforest import Control, changeforest
except Exception:
    Control = None
    changeforest = None


def detect_cpd(series: pd.Series, penalty: int) -> list:
    """Run PELT change-point detection on a univariate series."""
    try:
        import ruptures as rpt
        sig  = series.values.reshape(-1, 1)
        algo = rpt.Pelt(model="rbf").fit(sig)
        bkps = algo.predict(pen=penalty)
        return [b for b in bkps[:-1] if 0 < b < len(series)]
    except Exception:
        return []


def rolling_with_dates(pdf: pd.DataFrame, metric: str, window: int) -> pd.DataFrame:
    """Compute a rolling mean of *metric* and return a date-aligned DataFrame."""
    sub = pdf[pdf[metric].notna()][["game_date", metric]].sort_values("game_date")
    if len(sub) < window:
        return pd.DataFrame(columns=["game_date", metric])
    roll = sub[metric].rolling(window, min_periods=window // 2).mean()
    return pd.DataFrame({"game_date": sub["game_date"].values,
                          metric: roll.values}).dropna()


def build_cpd_subdf(df: pd.DataFrame, player_name: str, window: int) -> pd.DataFrame:
    """Subset and prepare a per-player DataFrame for ChangeForest analysis."""
    feature_cols = ["cf_seq_id"] + PA_INDICATORS
    base_cols    = ["batter", "player_name", "pa_uid", "game_date", "game_pk", "at_bat_number"]

    subdf = (
        df.loc[df["player_name"] == player_name, base_cols + feature_cols]
        .dropna(subset=feature_cols)
        .sort_values("cf_seq_id")
        .reset_index(drop=True)
    )

    for col in PA_INDICATORS:
        subdf[f"{col}_rollmean_{window}"] = (
            subdf[col].rolling(window=window, min_periods=window).mean()
        )

    rollmean_cols = [f"{col}_rollmean_{window}" for col in PA_INDICATORS]
    return subdf.dropna(subset=rollmean_cols).reset_index(drop=True)


def run_changeforest(subdf: pd.DataFrame, window: int, min_seg: float):
    """Run the ChangeForest multivariate CPD algorithm."""
    if changeforest is None or Control is None:
        st.warning("changeforest not available.")
        return None, [], None, None

    feature_names = [f"{col}_rollmean_{window}" for col in PA_INDICATORS]
    missing = [c for c in feature_names if c not in subdf.columns]
    if missing:
        st.warning(f"Missing columns: {missing}")
        return None, [], None, None

    X = subdf[feature_names].values.astype(np.float64)
    if len(X) < 20 or np.isnan(X).any():
        return None, [], None, None

    X = StandardScaler().fit_transform(X)
    control = Control(
        model_selection_alpha=0.02,
        minimal_relative_segment_length=min_seg,
    )
    try:
        result = changeforest(X, method="random_forest", control=control)
        cps = [int(cp) for cp in result.split_points() if 0 < int(cp) < len(subdf)]
        return result, cps, X, feature_names
    except Exception as e:
        st.warning(f"changeforest error: {e}")
        return None, [], None, None


def get_cp_feature_importance(subdf: pd.DataFrame, cp_idx: int, window: int) -> pd.DataFrame:
    """Train a binary RF classifier on before/after windows and return feature importances."""
    feature_names = [f"{col}_rollmean_{window}" for col in PA_INDICATORS]
    half = min(50, cp_idx, len(subdf) - cp_idx)
    if half < 5:
        return pd.DataFrame()

    before = subdf[feature_names].iloc[cp_idx - half : cp_idx].copy()
    after  = subdf[feature_names].iloc[cp_idx : cp_idx + half].copy()
    before["label"] = 0
    after["label"]  = 1

    combined = pd.concat([before, after]).dropna()
    if len(combined) < 10:
        return pd.DataFrame()

    X = combined[feature_names].values
    y = combined["label"].values

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    importance_df = pd.DataFrame({
        "Indicator": [PA_LABELS[col] for col in PA_INDICATORS],
        "Importance": clf.feature_importances_,
        "Color":      [PA_COLORS[col] for col in PA_INDICATORS],
    }).sort_values("Importance", ascending=True)

    return importance_df
