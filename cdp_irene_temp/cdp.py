#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data_path = Path("../data/processed/qualified_hitters_statcast_2021_2025_batted_ball.csv")
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:")
print(df.columns)
display(df.head())

player_counts = df.groupby("Name").size().reset_index(name="num_pitches")
print(f"Total unique players: {player_counts['Name'].nunique()}")
player_counts.sort_values("num_pitches", ascending=False)


# In[ ]:


def plot_player_time_series(df, player_name, index, figsize=(14, 5)):
    """
    Plot a time series of a given metric for a specific player.

    Parameters
    ----------
    df          : pd.DataFrame  – full dataset
    player_name : str           – player name matching the 'Name' column
    index       : str           – metric column to plot, e.g. 'launch_angle'
    figsize     : tuple         – figure size (default (14, 5))
    """
    player_df = df[df["Name"] == player_name].copy()

    if player_df.empty:
        raise ValueError(f"No data found for player: '{player_name}'")
    if index not in player_df.columns:
        raise ValueError(f"Column '{index}' not found in DataFrame.")

    player_df["game_date"] = pd.to_datetime(player_df["game_date"])
    player_df = player_df.sort_values("game_date").dropna(subset=[index])

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(player_df["game_date"], player_df[index],
               alpha=0.5, s=20, color="steelblue")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")

    ax.set_title(f"{player_name} — {index} over time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(index)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


# In[24]:


# Example usage
plot_player_time_series(df, player_name="Freddie Freeman", index="launch_angle")


# In[ ]:


def cusum_changepoint(df, player_name, index, target=None, threshold=5.0):
    """
    Detect change points using CUSUM (Cumulative Sum Control Chart).

    Parameters
    ----------
    df        : pd.DataFrame  – full dataset
    player_name : str         – player name
    index     : str           – metric column to analyze
    target    : float         – target/baseline mean (if None, uses mean of data)
    threshold : float         – decision threshold for change point (default 5.0)

    Returns
    -------
    dict with keys:
        - 'cusum_pos': cumulative sum above target
        - 'cusum_neg': cumulative sum below target
        - 'changepoints': indices where significant deviations occur
        - 'dates': corresponding dates
        - 'values': metric values
        - 'dates_changepoint': dates of detected change points
    """
    player_df = df[df["Name"] == player_name].copy()

    if player_df.empty:
        raise ValueError(f"No data found for player: '{player_name}'")
    if index not in player_df.columns:
        raise ValueError(f"Column '{index}' not found in DataFrame.")

    player_df["game_date"] = pd.to_datetime(player_df["game_date"])
    player_df = player_df.sort_values("game_date").dropna(subset=[index])

    values = player_df[index].values
    dates = player_df["game_date"].values

    if target is None:
        target = np.mean(values)

    # Calculate standard deviation for scaling
    std_dev = np.std(values)
    if std_dev == 0:
        std_dev = 1

    # Normalize deviations
    normalized = (values - target) / std_dev

    # Calculate CUSUM
    cusum_pos = np.zeros_like(normalized)
    cusum_neg = np.zeros_like(normalized)

    for i in range(len(normalized)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + normalized[i]) if i > 0 else max(0, normalized[i])
        cusum_neg[i] = min(0, cusum_neg[i-1] + normalized[i]) if i > 0 else min(0, normalized[i])

    # Detect change points
    changepoints = np.where((np.abs(cusum_pos) > threshold) | (np.abs(cusum_neg) > threshold))[0]

    return {
        'cusum_pos': cusum_pos,
        'cusum_neg': cusum_neg,
        'changepoints': changepoints,
        'dates': dates,
        'values': values,
        'dates_changepoint': dates[changepoints] if len(changepoints) > 0 else [],
        'target': target,
        'threshold': threshold
    }


def plot_cusum(cusum_result, player_name, index, figsize=(14, 6)):
    """
    Plot CUSUM chart with change points highlighted.

    Parameters
    ----------
    cusum_result : dict – output from cusum_changepoint()
    player_name  : str  – player name for title
    index        : str  – metric name for title
    figsize      : tuple – figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    dates = cusum_result['dates']
    cusum_pos = cusum_result['cusum_pos']
    cusum_neg = cusum_result['cusum_neg']
    threshold = cusum_result['threshold']
    changepoints = cusum_result['changepoints']

    ax.plot(dates, cusum_pos, color='green', linewidth=2, label='CUSUM+')
    ax.plot(dates, cusum_neg, color='red', linewidth=2, label='CUSUM-')
    ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.6, label=f'Threshold (±{threshold})')
    ax.axhline(y=-threshold, color='gray', linestyle='--', alpha=0.6)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    # Highlight change points
    if len(changepoints) > 0:
        ax.scatter(dates[changepoints], cusum_pos[changepoints], color='orange', s=100, 
                  marker='X', zorder=5, label='Change Points')

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")

    ax.set_title(f"CUSUM Chart: {player_name} — {index}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("CUSUM Value")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Change points detected: {len(changepoints)}")
    if len(changepoints) > 0:
        print("Dates of change points:")
        for cp_date in cusum_result['dates_changepoint']:
            print(f"  {pd.Timestamp(cp_date).strftime('%Y-%m-%d')}")


# In[31]:


# Example: CUSUM analysis
cusum_result = cusum_changepoint(df, player_name="Freddie Freeman", index="launch_angle", threshold=70.0)
plot_cusum(cusum_result, player_name="Freddie Freeman", index="launch_angle")


# In[ ]:




