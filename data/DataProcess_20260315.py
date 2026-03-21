import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "qualified_hitters_statcast_2021_2025.csv")

OUTPUT_REDUCED = os.path.join(BASE_DIR, "Qualified_hitters_statcast_2021_2025_reduced_columns.csv")
OUTPUT_CLEAN_FULL = os.path.join(BASE_DIR, "Qualified_hitters_statcast_2021_2025_clean_full.csv")
OUTPUT_ALL_PITCHES = os.path.join(BASE_DIR, "Qualified_hitters_statcast_2021_2025_all_pitches.csv")
OUTPUT_BATTED_BALL = os.path.join(BASE_DIR, "Qualified_hitters_statcast_2021_2025_batted_ball.csv")

print("Loading raw CSV...")
df = pd.read_csv(INPUT_FILE, low_memory = False)
print("Raw shape:", df.shape)


# 2.1 Schema Reduction
columns_keep = [
    "season",
    "Season",
    "game_date",
    "game_pk",
    "batter",
    "pitcher",
    "player_name",
    "Name",
    "IDfg",
    "mlbam_id",
    "home_team",
    "away_team",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "balls",
    "strikes",
    "stand",
    "p_throws",
    "pitch_type",
    "pitch_name",
    "description",
    "events",
    "zone",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "babip_value",
    "iso_value",
    "hc_x",
    "hc_y",
    "bb_type",
    "on_1b",
    "on_2b",
    "on_3b",
]

columns_keep = [
	c for c in columns_keep if c in df.columns
]

df = df[columns_keep].copy()

print("After schema reduction:", df.shape)

df.to_csv(OUTPUT_REDUCED, index = False)

# 2.2 Data Cleaning(ensure data quality) 

rows_before = len(df)
df = df.drop_duplicates()

if "game_date" in df.columns:
	df["game_date"] = pd.to_datetime(df["game_date"], errors = "coerce")

required_cols = [
	c for c in ["game_date", "game_pk", "batter"] if c in df.columns
]

df = df.dropna(subset = required_cols)

if "description" in df.columns and "events" in df.columns:
	df = df[~(df["description"].isna() & df["events"].isna())]

if "launch_speed" in df.columns:
	df = df[(df["launch_speed"].isna()) | (df["launch_speed"] <= 120)]

if "launch_angle" in df.columns:
	df = df[(df["launch_angle"].isna()) | (df["launch_angle"] >= -90)]

sort_cols = [
	c for c in ["batter", "game_date", "game_pk", "at_bat_number", "pitch_number"]
	if c in df.columns
]

df = df.sort_values(sort_cols)

if "Name" in df.columns and "player_name" in df.columns:
	df["batter_name_final"] = df["Name"].fillna(df["player_name"])
elif "player_name" in df.columns:
	df["batter_name_final"] = df["player_name"]
elif "Name" in df.columns:
	df["batter_name_final"] = df["Name"]

rows_after = len(df)

print("Rows removed:", rows_before - rows_after)

# 2.3 CPD Feature Engineering
desc = df["description"].fillna("").astype(str)
events = df["events"].fillna("").astype(str)

swing_desc = [
	"swinging_strike",
	"swinging_strike_blocked",
	"foul",
	"foul_tip",
	"hit_into_play",
	"hit_into_play_no_out",
	"hit_into_play_score"
]

whiff_desc = [
	"swinging_strike",
	"swinging_strike_blocked"
]

called_strike_desc = ["called_strike"]

ball_desc = ["ball", "blocked_ball", "pitchout"]

inplay_desc = [
	"hit_into_play",
	"hit_into_play_no_out",
	"hit_into_play_score"
]

df["is_pitch"] = 1
df["is_swing"] = desc.isin(swing_desc).astype(int)
df["is_whiff"] = desc.isin(whiff_desc).astype(int)
df["is_called_strike"] = desc.isin(called_strike_desc).astype(int)
df["is_ball"] = desc.isin(ball_desc).astype(int)
df["is_in_play"] = desc.isin(inplay_desc).astype(int)

if "zone" in df.columns:
	df["is_zone"] = df["zone"].isin([1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(int)
else :
	df["is_zone"] = 0

df["is_out_of_zone"] = (df["is_zone"] == 0).astype(int)

if "launch_speed" in df.columns:
	df["is_batted_ball"] = df["launch_speed"].notna().astype(int)
else:
	df["is_batted_ball"] = 0

df["valid_launch_speed"] = df["launch_speed"].notna().astype(int)
df["valid_launch_angle"] = df["launch_angle"].notna().astype(int)
df["valid_hit_distance_sc"] = df["hit_distance_sc"].notna().astype(int)
df["valid_xwoba"] = df["estimated_woba_using_speedangle"].notna().astype(int)
df["is_hard_hit"] = (df["launch_speed"] >= 95).astype(int)
df["is_barrel_proxy"] = (
	(df["launch_speed"] >= 98) &
	(df["launch_angle"].between(26, 30))
).astype(int)
df["exit_velocity"] = df["launch_speed"]
df["launch_angle_metric"] = df["launch_angle"]
df["xwoba_est"] = df["estimated_woba_using_speedangle"]
df["woba_on_play"] = df["woba_value"]
df["hit_distance"] = df["hit_distance_sc"]

print("Saving clean full dataset...")
df.to_csv(OUTPUT_CLEAN_FULL, index = False)

# 2.4 Dataset Subsets
df.to_csv(OUTPUT_ALL_PITCHES, index = False)
batted_df = df[df["is_batted_ball"] == 1].copy()
batted_df.to_csv(OUTPUT_BATTED_BALL, index = False)

print("All outputs completed")