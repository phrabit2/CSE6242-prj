import os
from pathlib import Path
import gdown
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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

#comment out in favor of pulling data from google drive.
# @st.cache_data(show_spinner="Loading plate appearance data...")
# def load_data() -> pd.DataFrame:
#     cols = [
#         "batter",
#         "pa_uid",
#         "game_date",
#         "power_woba_seq_id",
#         "power_efficiency",
#         "woba_residual",
#     ]
#     df = pd.read_csv(get_data_path(), usecols=cols)
#     df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
#     return df
#comment out in favor of pulling data from google drive.
#load data from google drive, with caching


# ══════════════════════════════════════════════════════════════════════════════
# DATA — download from Google Drive, cache in /tmp so reruns skip the download
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading pitch data…")
def load_data():
    DATA_FILE_ID  = "1G8eA6gX8hdCwWp1YWddAMmwt6R62tcmA"
    # /tmp persists for the lifetime of the Streamlit Cloud container run
    data_path = "/tmp/mlb_data.csv"
    cols = [
        "batter",
        "pa_uid",
        "game_date",
        "power_woba_seq_id",
        "power_efficiency",
        "woba_residual",
    ]
    if not os.path.exists(data_path):
        url = f"https://drive.google.com/uc?id={DATA_FILE_ID}"
        gdown.download(url, data_path, quiet=False, fuzzy=True)

    df = pd.read_csv(data_path, usecols=cols)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def changeforest_subdataset_generator(
    df: pd.DataFrame,
    selected_player_id: int,
    window: int = 50,
    min_periods: int = 1,
) -> pd.DataFrame:
    subdf = (
        df.loc[df["batter"] == selected_player_id].copy()
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
    return subdf


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

    X_used = subdf[feature_names].copy().values
    if standardize:
        X_used = StandardScaler().fit_transform(X_used)

    control = Control(
        model_selection_alpha=0.02,
        minimal_relative_segment_length=minimal_relative_segment_length,
    )

    result = changeforest(X_used, method="random_forest", control=control)

    # Some implementations may include the final index; keep valid in-range indices only.
    cps = [int(cp) for cp in result.split_points() if 0 <= int(cp) < len(subdf)]
    return result, cps, X_used, feature_names


def plot_changeforest_result(
    subdf: pd.DataFrame,
    cps: list[int],
    window: int,
    player_id: int,
    min_rel_seg_len: float,
):
    y_cols = [
        f"power_efficiency_rollmean_{window}",
        f"woba_residual_rollmean_{window}",
    ]
    titles = ["Power Efficiency", "wOBA Residual"]

    dates = pd.to_datetime(subdf["game_date"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, y_col, title in zip(axes, y_cols, titles):
        ax.plot(dates, subdf[y_col], linewidth=1.8, color="steelblue")
        for cp in cps:
            cp_date = dates.iloc[cp]
            ax.axvline(x=cp_date, color="crimson", linestyle="--", alpha=0.8)

        ax.set_title(title)
        ax.set_xlabel("Game Date")
        ax.set_ylabel(y_col)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(
        "ChangeForest Result"
        + f" | player_id={player_id}"
        + f" | window={window}"
        + f" | min_rel_seg_len={min_rel_seg_len}",
        fontsize=14,
    )
    return fig


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

    st.write("Features used:", feature_names)
    st.write("Detected change points:", cps)

    if len(cps) > 0:
        st.subheader("Change-point rows")
        st.dataframe(
            subdf.iloc[cps][["power_woba_seq_id", "game_date", "pa_uid"]].reset_index(drop=True),
            use_container_width=True,
        )

    fig = plot_changeforest_result(
        subdf=subdf,
        cps=cps,
        window=rolling_window,
        player_id=selected_player_id,
        min_rel_seg_len=min_rel_seg_len,
    )
    st.pyplot(fig)

    with st.expander("Sensitivity mapping used"):
        st.write(SENSITIVITY_TO_MIN_SEG)


if __name__ == "__main__":
    main()
