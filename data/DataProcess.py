import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "qualified_hitters_statcast_2021_2025.csv")
OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "Qualified_hitters_statcast_2021_2025_pa_master.csv"
)

print("Loading raw CSV...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print("Raw shape:", df.shape)

# 2.1 Schema Reduction
# Keep only the raw columns needed for:
# 1) cleaning
# 2) pitch-level feature engineering
# 3) PA-level aggregation
# =====================================================

df = df.rename(columns={
    "Season": "season",
})

RAW_COLS = [
    "season",
    "game_date",
    "game_pk",
    "batter",
    "at_bat_number",
    "pitch_number",
    "balls",
    "strikes",
    "stand",
    "p_throws",
    "description",
    "events",
    "zone",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "estimated_woba_using_speedangle",
    "woba_value",
]

missing_cols = [c for c in RAW_COLS if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df = df[RAW_COLS].copy()

print("After schema reduction:", df.shape)
print("Retained raw columns:", len(df.columns))

# 2.2 Data Cleaning
# =====================================================
rows_before = len(df)

df = df.drop_duplicates()

df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

key_cols = [
    "season",
    "game_date",
    "game_pk",
    "batter",
    "at_bat_number",
    "pitch_number",
]
df = df.dropna(subset=key_cols)

df = df[~(df["description"].isna() & df["events"].isna())]

df = df[
    (df["launch_speed"].isna()) |
    ((df["launch_speed"] >= 0) & (df["launch_speed"] <= 120))
]

df = df[
    (df["launch_angle"].isna()) |
    ((df["launch_angle"] >= -90) & (df["launch_angle"] <= 90))
]

df = df.sort_values(
    ["batter", "game_date", "game_pk", "at_bat_number", "pitch_number"]
).reset_index(drop=True)

rows_after = len(df)

print("Rows removed during cleaning:", rows_before - rows_after)

# 2.2.1 Basic Input Statistics
# =====================================================
print("Basic input statistics...")
print("Pitch-level rows after cleaning:", len(df))
print("Unique batters:", df["batter"].nunique())
print("Unique games:", df["game_pk"].nunique())
print(
    "Unique PA candidates:",
    df[["season", "game_pk", "batter", "at_bat_number"]]
    .drop_duplicates()
    .shape[0]
)
print("Missing launch_speed:", int(df["launch_speed"].isna().sum()))
print("Missing launch_angle:", int(df["launch_angle"].isna().sum()))
print(
    "Missing estimated_woba_using_speedangle:",
    int(df["estimated_woba_using_speedangle"].isna().sum())
)
print("Missing woba_value:", int(df["woba_value"].isna().sum()))

# 2.3 Pitch-Level Feature Engineering
# =====================================================
desc = df["description"].fillna("").astype(str)

swing_desc = [
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
]

whiff_desc = [
    "swinging_strike",
    "swinging_strike_blocked",
]

called_strike_desc = [
    "called_strike",
]

ball_desc = [
    "ball",
    "blocked_ball",
    "pitchout",
]

inplay_desc = [
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
]

df["is_swing"] = desc.isin(swing_desc).astype(int)
df["is_whiff"] = desc.isin(whiff_desc).astype(int)
df["is_called_strike"] = desc.isin(called_strike_desc).astype(int)
df["is_ball"] = desc.isin(ball_desc).astype(int)
df["is_in_play"] = desc.isin(inplay_desc).astype(int)
df["is_zone"] = df["zone"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(int)

# +1 / -2 decision matrix
# swing at strike = +1
# swing at ball = -2
# take strike = -2
# take ball = +1
decision_conditions = [
    (df["is_swing"] == 1) & (df["is_zone"] == 1),
    (df["is_swing"] == 1) & (df["is_zone"] == 0),
    (df["is_swing"] == 0) & (df["is_zone"] == 1),
    (df["is_swing"] == 0) & (df["is_zone"] == 0),
]
decision_values = [1, -2, -2, 1]

df["decision_score_pitch"] = np.select(
    decision_conditions,
    decision_values,
    default=0,
)

print("Pitch-level feature engineering completed")

# 2.4 Intermediate Reduction Before PA Aggregation
# description and zone are process-only columns.
# Keep only what is still needed for PA aggregation.
# =====================================================
PA_WORK_COLS = [
    "season",
    "game_date",
    "game_pk",
    "batter",
    "at_bat_number",
    "pitch_number",
    "balls",
    "strikes",
    "stand",
    "p_throws",
    "events",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "estimated_woba_using_speedangle",
    "woba_value",
    "is_swing",
    "is_whiff",
    "is_called_strike",
    "is_ball",
    "is_in_play",
    "is_zone",
    "decision_score_pitch",
]

pa_df = df[PA_WORK_COLS].copy()

print("PA aggregation input shape:", pa_df.shape)
print("Columns entering PA aggregation:", len(pa_df.columns))


# 2.5 PA Aggregation
# Group key:
# season + game_date + game_pk + batter + at_bat_number
# =====================================================
def first_value(x):
    return x.iloc[0]


def last_value(x):
    return x.iloc[-1]


group_cols = [
    "season",
    "game_date",
    "game_pk",
    "batter",
    "at_bat_number",
]

agg_dict = {
    "pitch_number": "max",
    "balls": last_value,
    "strikes": last_value,
    "stand": first_value,
    "p_throws": last_value,
    "events": last_value,
    "launch_speed": last_value,
    "launch_angle": last_value,
    "hit_distance_sc": last_value,
    "estimated_woba_using_speedangle": last_value,
    "woba_value": last_value,
    "is_swing": "sum",
    "is_whiff": "sum",
    "is_called_strike": "sum",
    "is_ball": "sum",
    "is_in_play": "max",
    "is_zone": "sum",
    "decision_score_pitch": "sum",
}

pa_df = pa_df.groupby(group_cols, as_index=False).agg(agg_dict)

pa_df = pa_df.rename(
    columns={
        "pitch_number": "num_pitches",
        "balls": "pa_end_balls",
        "strikes": "pa_end_strikes",
        "events": "pa_result",
        "is_swing": "swing_count",
        "is_whiff": "whiff_count",
        "is_called_strike": "called_strike_count",
        "is_ball": "ball_count",
        "is_in_play": "is_batted_ball_pa",
        "is_zone": "zone_pitch_count",
        "decision_score_pitch": "hitting_decisions_score",
    }
)

pa_df = pa_df.sort_values(
    ["batter", "game_date", "game_pk", "at_bat_number"]
).reset_index(drop=True)

print("After PA aggregation:", pa_df.shape)

# 2.6 PA-Level Derived Columns
# Four indicator structure:
# 1) power_efficiency
# 2) launch_angle_stability_50pa
# 3) hitting_decisions_score
# 4) woba_residual
# =====================================================
pa_df["pa_uid"] = (
    pa_df["season"].astype(str)
    + "_"
    + pa_df["game_pk"].astype(str)
    + "_"
    + pa_df["batter"].astype(str)
    + "_"
    + pa_df["at_bat_number"].astype(str)
)

pa_df["pa_seq_id"] = pa_df.groupby("batter").cumcount() + 1

# power_efficiency and woba_residual share the same validity rule
valid_power_woba = (
    pa_df["launch_speed"].notna() &
    (pa_df["launch_speed"] > 0) &
    pa_df["estimated_woba_using_speedangle"].notna() &
    pa_df["woba_value"].notna()
)

pa_df["power_efficiency"] = np.where(
    valid_power_woba,
    pa_df["estimated_woba_using_speedangle"] / pa_df["launch_speed"],
    np.nan,
)

pa_df["woba_residual"] = np.where(
    valid_power_woba,
    pa_df["woba_value"] - pa_df["estimated_woba_using_speedangle"],
    np.nan,
)

pa_df["is_hard_hit"] = (
    pa_df["launch_speed"].notna() &
    (pa_df["launch_speed"] >= 95)
).astype(int)

pa_df["is_barrel_proxy"] = (
    pa_df["launch_speed"].notna() &
    pa_df["launch_angle"].notna() &
    (pa_df["launch_speed"] >= 98) &
    (pa_df["launch_angle"].between(26, 30))
).astype(int)

print("PA-level derived columns added")

# 2.6.1 Indicator-Specific Sequence IDs for CPD
# =====================================================
pa_df["power_woba_seq_id"] = pd.Series([pd.NA] * len(pa_df), dtype="Int64")
pa_df["launch_angle_seq_id"] = pd.Series([pd.NA] * len(pa_df), dtype="Int64")

for batter_id, g in pa_df.groupby("batter"):
    power_idx = g[valid_power_woba.loc[g.index]].index
    pa_df.loc[power_idx, "power_woba_seq_id"] = np.arange(1, len(power_idx) + 1)

    la_idx = g[g["launch_angle"].notna()].index
    pa_df.loc[la_idx, "launch_angle_seq_id"] = np.arange(1, len(la_idx) + 1)

print("Indicator-specific sequence IDs added")

# 2.6.2 Launch Angle Stability by Rolling 50 Valid PAs
# =====================================================
pa_df["launch_angle_stability_50pa"] = np.nan

for batter_id, g in pa_df.groupby("batter"):
    valid_idx = g[g["launch_angle"].notna()].index
    valid_la = g.loc[valid_idx, "launch_angle"]

    rolling_sd = valid_la.rolling(window=50, min_periods=50).std()

    pa_df.loc[valid_idx, "launch_angle_stability_50pa"] = rolling_sd.values

print("Launch angle rolling stability added")

# 2.7 Final PA MASTER Reduction
# Keep only final analysis-ready columns.
# Tail feature order follows the requested layout.
# =====================================================
FINAL_COLS = [
    "season",
    "game_date",
    "game_pk",
    "batter",
    "at_bat_number",
    "pa_uid",
    "pa_seq_id",
    "stand",
    "p_throws",
    "num_pitches",
    "pa_end_balls",
    "pa_end_strikes",
    "pa_result",
    "swing_count",
    "whiff_count",
    "called_strike_count",
    "ball_count",
    "zone_pitch_count",
    "is_batted_ball_pa",
    "is_hard_hit",
    "is_barrel_proxy",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "woba_value",
    "estimated_woba_using_speedangle",
    "power_woba_seq_id",
    "power_efficiency",
    "woba_residual",
    "launch_angle_seq_id",
    "launch_angle_stability_50pa",
    "hitting_decisions_score",
]

pa_df = pa_df[FINAL_COLS].copy()

if pd.api.types.is_datetime64_any_dtype(pa_df["game_date"]):
    pa_df["game_date"] = pa_df["game_date"].dt.strftime("%Y-%m-%d")

print("Final PA_MASTER shape:", pa_df.shape)
print("Unique PA rows:", len(pa_df))
print("Unique batters in PA_MASTER:", pa_df["batter"].nunique())
print("Batted-ball PA count:", int(pa_df["is_batted_ball_pa"].sum()))
print("Average pitches per PA:", round(pa_df["num_pitches"].mean(), 4))
print(
    "Valid power/woba rows:",
    int(pa_df["power_woba_seq_id"].notna().sum())
)
print(
    "Valid launch-angle rows:",
    int(pa_df["launch_angle_seq_id"].notna().sum())
)
print(
    "Valid launch-angle rolling-50 rows:",
    int(pa_df["launch_angle_stability_50pa"].notna().sum())
)

# 2.8 Save Final Output
# =====================================================
print("Saving PA_MASTER CSV...")
pa_df.to_csv(OUTPUT_FILE, index=False)

print("PA_MASTER output completed")
print("Output file:", OUTPUT_FILE)