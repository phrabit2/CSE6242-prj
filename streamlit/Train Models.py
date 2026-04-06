"""
train_models.py
===============
Run ONCE offline to train, tune, and pickle the RF + LR models.
The Streamlit app loads the pickle at startup — zero training at runtime.

Usage:
    python train_models.py

Output:
    ../models/batting_models.pkl

Tuning strategy
---------------
Random Forest:
    GridSearchCV (5-fold, scoring=accuracy) over:
        n_estimators  : [200, 300, 500]
        max_depth     : [6, 8, 12, None]
        min_samples_leaf: [1, 5, 10]

Logistic Regression:
    AIC-based C selection.
    For each candidate C we fit LR on the full training set, compute the
    multinomial log-likelihood, count the effective number of parameters
    (non-zero coefficients), and pick the C that minimises AIC.
    AIC = 2k - 2*ln(L)
    where k = number of non-zero coefficients + intercepts.

Pickle bundle keys
------------------
    rf_clf            – best RandomForestClassifier (refit on full data)
    rf_best_params    – dict of best hyperparameters
    rf_importances    – pd.Series (feature → importance, sorted desc)
    rf_feat_cols      – list[str]
    rf_le             – fitted LabelEncoder
    rf_cv_mean        – float, best CV accuracy (mean)
    rf_cv_std         – float, best CV accuracy (std)
    rf_classes        – list[str]
    rf_cv_results     – pd.DataFrame, full grid search results

    lr_clf            – best LogisticRegression (fit on full data)
    lr_best_C         – float, C chosen by AIC
    lr_best_aic       – float, AIC at best C
    lr_aic_table      – pd.DataFrame, C → k, log_likelihood, AIC
    lr_feat_cols      – list[str]
    lr_le             – fitted LabelEncoder
    lr_sc             – fitted StandardScaler
    lr_classes        – list[str]
    lr_coef_df        – pd.DataFrame (outcomes × features)

    selected_events   – list[str]
    selected_features – list[str]
    trained_at        – ISO timestamp string
"""

import os
import pickle
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "processed",
                           "qualified_hitters_statcast_2021_2025_batted_ball.csv")
MODEL_DIR  = os.path.join(SCRIPT_DIR, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "batting_models.pkl")

SELECTED_EVENTS = [
    "home_run", "single", "double", "triple", "field_out",
]

FEATURE_WHITELIST = [
    "exit_velocity", "launch_angle_metric",
    "release_speed", "effective_speed",
    "release_spin_rate", "release_extension", "spin_axis",
    "pfx_x", "pfx_z",
    "plate_x", "plate_z",
    "sz_top", "sz_bot",
]

CV_FOLDS = 5

# RF hyperparameter grid — reduced to be feasible on a MacBook Air.
# 369k rows × 180 fits × parallel workers causes joblib semaphore leaks on
# macOS Python 3.13. We cut the grid to 12 combinations and subsample for CV.
RF_PARAM_GRID = {
    "n_estimators":     [200, 300],
    "max_depth":        [6, 8, None],
    "min_samples_leaf": [1, 10],
}

# Rows to sample for the CV stage only (speeds up grid search significantly).
# The final model is refit on the FULL dataset after the best params are found.
RF_CV_SAMPLE = 80_000

# LR candidate regularisation strengths (log-spaced, wide range)
LR_C_CANDIDATES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data(path):
    print(f"\nLoading data from:\n  {path}")
    df = pd.read_csv(path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["Name", "game_date"]).reset_index(drop=True)

    if "exit_velocity" not in df.columns and "launch_speed" in df.columns:
        df["exit_velocity"] = df["launch_speed"]
    if "launch_angle_metric" not in df.columns and "launch_angle" in df.columns:
        df["launch_angle_metric"] = df["launch_angle"]
    if "xwoba_est" not in df.columns and "estimated_woba_using_speedangle" in df.columns:
        df["xwoba_est"] = df["estimated_woba_using_speedangle"]

    print(f"  {len(df):,} rows · {df['Name'].nunique()} players")
    return df


def prepare_xy(df, selected_events, feature_whitelist):
    """Filter, intersect features, drop nulls. Returns X, y, le, feat_cols."""
    mdf  = df[df["events"].isin(selected_events)].copy()
    feat = [c for c in feature_whitelist if c in mdf.columns]
    mdf  = mdf[["events"] + feat].dropna()
    print(f"  Model rows: {len(mdf):,} · features: {len(feat)} · classes: {mdf['events'].nunique()}")
    le = LabelEncoder()
    y  = le.fit_transform(mdf["events"])
    return mdf[feat].values, y, le, feat


# ══════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST — GridSearchCV
# ══════════════════════════════════════════════════════════════════════════════
def tune_random_forest(X, y, feat_cols):
    """
    Grid search over RF_PARAM_GRID using stratified 5-fold CV on a subsample,
    then refit the best params on the FULL dataset.

    Subsampling rationale: CV accuracy converges well before 369k rows.
    80k rows is enough to rank hyperparameter combinations reliably while
    keeping each of the 60 fits (12 combos × 5 folds) under ~30 seconds.

    n_jobs=1 avoids joblib semaphore leaks on macOS Python 3.13.
    """
    print("\n── Random Forest hyperparameter search ─────────────────────────────")
    n_combos = (len(RF_PARAM_GRID["n_estimators"])
                * len(RF_PARAM_GRID["max_depth"])
                * len(RF_PARAM_GRID["min_samples_leaf"]))
    print(f"  Grid: {RF_PARAM_GRID}")
    print(f"  {n_combos} combinations × {CV_FOLDS} folds = {n_combos * CV_FOLDS} fits")
    print(f"  CV subsample: {RF_CV_SAMPLE:,} rows (final model uses full {len(X):,})")

    # Stratified subsample for CV
    rng    = np.random.default_rng(42)
    sample = min(RF_CV_SAMPLE, len(X))
    idx    = rng.choice(len(X), size=sample, replace=False)
    X_sub  = X[idx]
    y_sub  = y[idx]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=1),   # n_jobs=1: avoid macOS semaphore leak
        param_grid=RF_PARAM_GRID,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,          # outer loop also sequential — safe on macOS Python 3.13
        verbose=1,
        refit=False,       # we refit manually on FULL data below
        return_train_score=False,
    )
    gs.fit(X_sub, y_sub)

    best_params  = gs.best_params_
    best_idx     = gs.best_index_
    bscore_mean  = gs.cv_results_["mean_test_score"][best_idx]
    bscore_std   = gs.cv_results_["std_test_score"][best_idx]

    print(f"\n  Best params (from CV subsample) : {best_params}")
    print(f"  CV accuracy                     : {bscore_mean:.4f} ± {bscore_std:.4f}")
    print(f"\n  Refitting best model on full {len(X):,} rows …")

    # Refit on full data
    rf_best = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    rf_best.fit(X, y)
    print("  Done.")

    cv_results_df = pd.DataFrame(gs.cv_results_)[[
        "param_n_estimators", "param_max_depth", "param_min_samples_leaf",
        "mean_test_score", "std_test_score", "rank_test_score",
    ]].sort_values("rank_test_score").reset_index(drop=True)

    return rf_best, best_params, bscore_mean, bscore_std, cv_results_df


# ══════════════════════════════════════════════════════════════════════════════
# LOGISTIC REGRESSION — AIC-based C selection
# ══════════════════════════════════════════════════════════════════════════════
def _multinomial_log_likelihood(lr_model, X_scaled, y):
    """
    Compute the multinomial log-likelihood of a fitted LR model.
    Uses predict_proba — numerically stable log via clipping.
    """
    proba = lr_model.predict_proba(X_scaled)
    n     = len(y)
    # One-hot encode y for vectorised dot product
    n_classes = proba.shape[1]
    y_oh = np.zeros((n, n_classes))
    y_oh[np.arange(n), y] = 1.0
    log_proba = np.log(np.clip(proba, 1e-15, 1.0))
    return float((y_oh * log_proba).sum())


def _count_effective_params(lr_model):
    """
    Count effective parameters: non-zero coefficients + intercepts.
    With L2 regularisation all coefficients are technically non-zero,
    so we count the full coefficient matrix + intercept vector.
    k = n_classes * n_features + n_classes
    """
    coef = lr_model.coef_          # shape (n_classes, n_features)
    intercept = lr_model.intercept_ # shape (n_classes,)
    return int(coef.size + intercept.size)


def tune_logistic_regression(X_scaled, y):
    """
    Fit LR for each candidate C on the full training set, compute AIC,
    refit the best model and return it with the AIC table.

    AIC = 2k - 2*ln(L)
        k = effective number of parameters
        L = multinomial log-likelihood on training data

    Note: AIC on training data is appropriate here for model selection
    (penalises complexity), complementing CV accuracy from the RF stage.
    """
    print("\n── Logistic Regression AIC-based C selection ───────────────────────")
    print(f"  Candidates: C ∈ {LR_C_CANDIDATES}")

    rows = []
    for C in LR_C_CANDIDATES:
        lr = LogisticRegression(C=C, max_iter=2000, solver="lbfgs",
                                random_state=42, n_jobs=-1)
        lr.fit(X_scaled, y)
        ll  = _multinomial_log_likelihood(lr, X_scaled, y)
        k   = _count_effective_params(lr)
        aic = 2 * k - 2 * ll
        rows.append({"C": C, "k": k, "log_likelihood": ll, "AIC": aic})
        print(f"  C={C:<8}  k={k}  log_L={ll:>12.1f}  AIC={aic:>14.1f}")

    aic_table  = pd.DataFrame(rows)
    best_row   = aic_table.loc[aic_table["AIC"].idxmin()]
    best_C     = float(best_row["C"])
    best_aic   = float(best_row["AIC"])

    print(f"\n  Best C : {best_C}  (AIC = {best_aic:.1f})")

    # Refit on full data with best C
    lr_best = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs",
                                 random_state=42, n_jobs=-1)
    lr_best.fit(X_scaled, y)
    return lr_best, best_C, best_aic, aic_table


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def train(df):
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n══ Preparing training data ══════════════════════════════════════════")
    X_raw, y, le, feat_cols = prepare_xy(df, SELECTED_EVENTS, FEATURE_WHITELIST)
    X = pd.DataFrame(X_raw, columns=feat_cols)   # keep as DataFrame for RF

    # ── RF ────────────────────────────────────────────────────────────────────
    rf_clf, rf_best_params, rf_cv_mean, rf_cv_std, rf_cv_results = \
        tune_random_forest(X.values, y, feat_cols)

    importances = pd.Series(
        rf_clf.feature_importances_, index=feat_cols
    ).sort_values(ascending=False)

    # ── LR ────────────────────────────────────────────────────────────────────
    # LR uses the same feature set but scaled; share LabelEncoder with RF
    sc   = StandardScaler()
    Xs   = sc.fit_transform(X)

    # LR gets its own LabelEncoder in case class order differs (it won't here,
    # but keeps the bundle self-consistent)
    le_lr  = LabelEncoder()
    y_lr   = le_lr.fit_transform(
        le.inverse_transform(y)          # decode back to strings then re-encode
    )

    lr_clf, lr_best_C, lr_best_aic, lr_aic_table = \
        tune_logistic_regression(Xs, y_lr)

    coef_df = pd.DataFrame(
        lr_clf.coef_, columns=feat_cols, index=le_lr.classes_
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n══ Summary ══════════════════════════════════════════════════════════")
    print(f"  RF  best params : {rf_best_params}")
    print(f"  RF  CV accuracy : {rf_cv_mean:.4f} ± {rf_cv_std:.4f}")
    print(f"  LR  best C      : {lr_best_C}")
    print(f"  LR  best AIC    : {lr_best_aic:.1f}")
    print(f"  Top RF features : {list(importances.head(5).index)}")

    # ── Pickle ────────────────────────────────────────────────────────────────
    bundle = {
        # Random Forest
        "rf_clf":         rf_clf,
        "rf_best_params": rf_best_params,
        "rf_importances": importances,
        "rf_feat_cols":   feat_cols,
        "rf_le":          le,
        "rf_cv_mean":     rf_cv_mean,
        "rf_cv_std":      rf_cv_std,
        "rf_classes":     list(le.classes_),
        "rf_cv_results":  rf_cv_results,

        # Logistic Regression
        "lr_clf":         lr_clf,
        "lr_best_C":      lr_best_C,
        "lr_best_aic":    lr_best_aic,
        "lr_aic_table":   lr_aic_table,
        "lr_feat_cols":   feat_cols,
        "lr_le":          le_lr,
        "lr_sc":          sc,
        "lr_classes":     list(le_lr.classes_),
        "lr_coef_df":     coef_df,

        # Metadata
        "selected_events":   SELECTED_EVENTS,
        "selected_features": feat_cols,
        "trained_at":        datetime.datetime.utcnow().isoformat() + "Z",
    }
    print(f"DEBUG: Attempting to save to: {os.path.abspath(MODEL_PATH)}")


    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✓ Bundle saved → {MODEL_PATH}\n")

    # Print AIC table for inspection
    print("LR AIC table:")
    print(lr_aic_table.to_string(index=False))
    print()

    # Print top RF grid results
    print("RF top 5 CV results:")
    print(rf_cv_results.head(5).to_string(index=False))

    return bundle


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    train(df)