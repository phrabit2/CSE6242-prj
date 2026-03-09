"""
Data preprocessing and cleaning utilities for Statcast data.

Handles:
- Missing value imputation
- Feature engineering (rolling averages, per-game aggregation)
- Player-level time series construction
"""

import os
import pandas as pd
import numpy as np

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

# Core Statcast metrics for CPD analysis
OFFENSE_METRICS = [
    "launch_speed",       # Exit Velocity
    "launch_angle",       # Launch Angle
    "barrel",             # Barrel indicator
    "estimated_ba_using_speedangle",   # xBA
    "estimated_slg_using_speedangle",  # xSLG
    "estimated_woba_using_speedangle", # xwOBA
]

PITCHING_METRICS = [
    "release_spin_rate",
    "release_extension",
    "pfx_x",              # Horizontal Break
    "pfx_z",              # Induced Vertical Break
]


def load_raw_data(filename: str = "statcast_raw.csv") -> pd.DataFrame:
    """Load raw Statcast CSV."""
    path = os.path.join(DATA_RAW_DIR, filename)
    return pd.read_csv(path, parse_dates=["game_date"])


def build_player_timeseries(df: pd.DataFrame, player_id: int, metrics: list[str]) -> pd.DataFrame:
    """
    Build a per-game aggregated time series for a specific player.

    Args:
        df: Raw Statcast DataFrame.
        player_id: MLB player ID.
        metrics: List of metric columns to aggregate.

    Returns:
        DataFrame indexed by game_date with aggregated metrics.
    """
    player_df = df[df["batter"] == player_id].copy()
    available_metrics = [m for m in metrics if m in player_df.columns]
    timeseries = player_df.groupby("game_date")[available_metrics].mean()
    timeseries = timeseries.sort_index().dropna(how="all")
    return timeseries


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Save processed DataFrame to CSV."""
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(DATA_PROCESSED_DIR, filename)
    df.to_csv(output_path)
    print(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
