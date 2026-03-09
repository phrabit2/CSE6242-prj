"""
Fetch Statcast data from Baseball Savant via pybaseball.

Usage:
    python -m src.data.fetch_statcast
"""

import os
import pandas as pd
from pybaseball import statcast

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def fetch_statcast_data(start_date: str, end_date: str, output_filename: str = "statcast_raw.csv") -> pd.DataFrame:
    """
    Fetch Statcast data for a given date range and save to CSV.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        output_filename: Name of the output CSV file.

    Returns:
        DataFrame with raw Statcast data.
    """
    print(f"Fetching Statcast data from {start_date} to {end_date}...")
    df = statcast(start_dt=start_date, end_dt=end_date)

    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    output_path = os.path.join(DATA_RAW_DIR, output_filename)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")

    return df


if __name__ == "__main__":
    # Example: fetch 2024 season data
    fetch_statcast_data("2024-03-28", "2024-09-29")
