import os

import gdown
import pandas as pd
import streamlit as st

from config import DATA_CF_FILE_ID, PA_INDICATORS


@st.cache_data(show_spinner="Loading PA-level data...")
def load_data() -> pd.DataFrame:
    path = "/tmp/Qualified_hitters_statcast_2021_2025_pa_master.csv"
    cols = [
        "batter", "player_name", "pa_uid", "game_date", "game_pk",
        "at_bat_number", "cf_seq_id",
        "hitting_decisions_score", "power_efficiency",
        "woba_residual", "launch_angle_stability_50pa",
    ]
    if not os.path.exists(path):
        gdown.download(
            f"https://drive.google.com/uc?id={DATA_CF_FILE_ID}",
            path, quiet=False, fuzzy=True,
        )
    df = pd.read_csv(path, usecols=cols)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["year"]  = df["game_date"].dt.year
    df["month"] = df["game_date"].dt.month
    df = df.sort_values(["player_name", "game_date"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Computing performance index...")
def build_perf_index(_df: pd.DataFrame) -> pd.DataFrame:
    out = _df.copy()
    normed = pd.DataFrame(index=out.index)
    for col in PA_INDICATORS:
        if col not in out.columns:
            continue
        lo, hi = out[col].quantile(0.01), out[col].quantile(0.99)
        clipped = out[col].clip(lo, hi)
        normed[col] = (clipped - lo) / (hi - lo + 1e-9)
    out["perf_index"] = normed.mean(axis=1) * 100
    return out
