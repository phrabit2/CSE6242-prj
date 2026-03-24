# Players: Qualified hitters (PA >= 400)
# Pitch events: All pitches
# Seasons: 2021–2025 (2020 was a shortened season because of COVID19)

import os
import time
import math
import pandas as pd
from pybaseball import batting_stats, statcast, playerid_reverse_lookup, playerid_lookup


START_SEASON = 2021
END_SEASON = 2025
MIN_PA = 400

OUTPUT_DIR = "mlb_qualified_hitters_statcast_2021_2025"
LOOKUP_FILE = "qualified_hitters_lookup_2021_2025.csv"
PLAYER_SEASON_FILE = "qualified_hitters_player_season_2021_2025.csv"
FINAL_FILE = "qualified_hitters_statcast_2021_2025.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Functions
def safe_int_series(series):
    return pd.to_numeric(series, errors="coerce").dropna().astype(int)

def split_name(full_name):
    full_name = str(full_name).strip()
    parts = full_name.split()
    if len(parts) < 2:
        return None, None
    first = parts[0]
    last = " ".join(parts[1:])
    return first, last

def get_season_date_range(season):
    # Regular season approx window; enough for Statcast season pull
    start_date = f"{season}-03-01"
    end_date = f"{season}-11-30"
    return start_date, end_date

# 1. Pull qualified hitters for each season

all_hitters = []

print("Step 1. Pulling qualified hitters by season...")

for season in range(START_SEASON, END_SEASON + 1):
    season_df = batting_stats(season, qual=MIN_PA).copy()
    season_df["Season"] = season
    keep_cols = [c for c in ["Season", "Name", "IDfg", "PA", "Team", "Age"] if c in season_df.columns]
    season_df = season_df[keep_cols].copy()

    season_df["IDfg"] = pd.to_numeric(season_df["IDfg"], errors="coerce")
    season_df = season_df.dropna(subset=["IDfg"]).copy()
    season_df["IDfg"] = season_df["IDfg"].astype(int)

    print(f"Qualified hitters ({season}): {len(season_df)}")
    all_hitters.append(season_df)

hitters_df = pd.concat(all_hitters, ignore_index=True)
hitters_df = hitters_df.drop_duplicates(subset=["Season", "IDfg"]).copy()

print(f"Total qualified hitter rows: {len(hitters_df)}")
print(f"Unique qualified hitters across seasons: {hitters_df['IDfg'].nunique()}")

# Save player-season list
player_season_path = os.path.join(OUTPUT_DIR, PLAYER_SEASON_FILE)
hitters_df.to_csv(player_season_path, index=False)
print(f"Saved player-season file: {player_season_path}")


# 2. Fast batch mapping: Fangraphs ID -> MLBAM ID
print("\nStep 2. Building lookup table from Fangraphs IDs...")

unique_fg_ids = sorted(hitters_df["IDfg"].dropna().unique().tolist())

reverse_lookup_df = playerid_reverse_lookup(unique_fg_ids, key_type="fangraphs").copy()

reverse_lookup_df = reverse_lookup_df.rename(
    columns={
        "key_fangraphs": "IDfg",
        "key_mlbam": "mlbam_id",
        "name_first": "first_name",
        "name_last": "last_name",
    }
)

keep_cols = [c for c in ["IDfg", "mlbam_id", "first_name", "last_name"] if c in reverse_lookup_df.columns]
reverse_lookup_df = reverse_lookup_df[keep_cols].copy()

reverse_lookup_df["IDfg"] = pd.to_numeric(reverse_lookup_df["IDfg"], errors="coerce")
reverse_lookup_df["mlbam_id"] = pd.to_numeric(reverse_lookup_df["mlbam_id"], errors="coerce")
reverse_lookup_df = reverse_lookup_df.dropna(subset=["IDfg"]).copy()
reverse_lookup_df["IDfg"] = reverse_lookup_df["IDfg"].astype(int)

# Deduplicate in case reverse lookup returns more than one row and merge them
reverse_lookup_df = reverse_lookup_df.sort_values(["IDfg"]).drop_duplicates(subset=["IDfg"], keep="first").copy()
mapped_df = hitters_df.merge(reverse_lookup_df, on="IDfg", how="left")


# 3. Fallback lookup only for missing MLBAM IDs
print("\nStep 3. Fallback lookup for missing MLBAM IDs...")

missing_players = (
    mapped_df.loc[mapped_df["mlbam_id"].isna(), ["Name", "IDfg"]]
    .drop_duplicates()
    .copy()
)

print(f"Missing after reverse lookup: {len(missing_players)}")

fallback_rows = []

for idx, row in missing_players.iterrows():
    full_name = row["Name"]
    idfg = row["IDfg"]
    first, last = split_name(full_name)

    if not first or not last:
        print(f"Skipped fallback lookup for invalid name: {full_name}")
        continue

    try:
        temp = playerid_lookup(last, first)
        if temp is not None and not temp.empty:
            temp = temp.copy()

            if "key_fangraphs" in temp.columns:
                temp["key_fangraphs"] = pd.to_numeric(temp["key_fangraphs"], errors="coerce")
                matched = temp[temp["key_fangraphs"] == idfg].copy()
                if not matched.empty:
                    best = matched.iloc[0]
                else:
                    best = temp.iloc[0]
            else:
                best = temp.iloc[0]

            fallback_rows.append(
                {
                    "IDfg": int(idfg),
                    "mlbam_id_fallback": pd.to_numeric(best.get("key_mlbam"), errors="coerce"),
                }
            )
            print(f"Fallback matched: {full_name}")
        else:
            print(f"Fallback no result: {full_name}")

    except Exception as e:
        print(f"Fallback lookup failed for {full_name}: {e}")

    time.sleep(0.05)

if fallback_rows:
    fallback_df = pd.DataFrame(fallback_rows).drop_duplicates(subset=["IDfg"]).copy()
    mapped_df = mapped_df.merge(fallback_df, on="IDfg", how="left")
    mapped_df["mlbam_id"] = mapped_df["mlbam_id"].fillna(mapped_df["mlbam_id_fallback"])
    mapped_df = mapped_df.drop(columns=["mlbam_id_fallback"])

# Manual update mapping missing Fangraphs -> MLBAM
manual_patch = pd.DataFrame([
    {"Name": "Wenceel Perez", "IDfg": 22857, "mlbam_id": 672761},
    {"Name": "Carlos Narvaez", "IDfg": 19722, "mlbam_id": 665966},
    {"Name": "Agustin Ramirez", "IDfg": 26546, "mlbam_id": 682663},
    {"Name": "Angel Martinez", "IDfg": 26540, "mlbam_id": 682657},
])

mapped_df = mapped_df.merge(
    manual_patch[["IDfg", "mlbam_id"]].rename(columns={"mlbam_id": "mlbam_id_manual"}),
    on="IDfg",
    how="left",
)

mapped_df["mlbam_id"] = mapped_df["mlbam_id"].fillna(mapped_df["mlbam_id_manual"])
mapped_df = mapped_df.drop(columns=["mlbam_id_manual"])

mapped_df["mlbam_id"] = pd.to_numeric(mapped_df["mlbam_id"], errors="coerce")
mapped_df = mapped_df.dropna(subset=["mlbam_id"]).copy()
mapped_df["mlbam_id"] = mapped_df["mlbam_id"].astype(int)

# Final player lookup table
lookup_df = (
    mapped_df[["Name", "IDfg", "mlbam_id"]]
    .drop_duplicates(subset=["IDfg", "mlbam_id"])
    .sort_values(["Name", "IDfg"])
    .reset_index(drop=True)
)

lookup_path = os.path.join(OUTPUT_DIR, LOOKUP_FILE)
lookup_df.to_csv(lookup_path, index=False)
print(f"Saved lookup file: {lookup_path}")

print(f"Mapped player-season rows: {len(mapped_df)}")
print(f"Unique mapped MLBAM IDs: {mapped_df['mlbam_id'].nunique()}")


# 4. Pull Statcast by season and filter to qualified hitters
print("\nStep 4. Pulling Statcast data by season...")

all_statcast_parts = []

for season in range(START_SEASON, END_SEASON + 1):
    start_date, end_date = get_season_date_range(season)

    print(f"\nPulling Statcast for {season}: {start_date} to {end_date}")
    season_statcast = statcast(start_dt=start_date, end_dt=end_date).copy()

    if season_statcast.empty:
        print(f"No Statcast data returned for {season}")
        continue

    print(f"Raw Statcast rows ({season}): {len(season_statcast)}")

    # Standardize batter id
    if "batter" not in season_statcast.columns:
        print(f"'batter' column not found in Statcast data for {season}. Skipped.")
        continue

    season_statcast["batter"] = pd.to_numeric(season_statcast["batter"], errors="coerce")
    season_statcast = season_statcast.dropna(subset=["batter"]).copy()
    season_statcast["batter"] = season_statcast["batter"].astype(int)
    season_statcast["Season"] = season

    # Qualified players only for that season
    season_players = (
        mapped_df.loc[mapped_df["Season"] == season, ["Season", "Name", "IDfg", "mlbam_id"]]
        .drop_duplicates(subset=["Season", "mlbam_id"])
        .copy()
    )

    season_players["mlbam_id"] = pd.to_numeric(season_players["mlbam_id"], errors="coerce")
    season_players = season_players.dropna(subset=["mlbam_id"]).copy()
    season_players["mlbam_id"] = season_players["mlbam_id"].astype(int)

    qualified_ids = set(season_players["mlbam_id"].tolist())

    filtered_statcast = season_statcast[season_statcast["batter"].isin(qualified_ids)].copy()
    print(f"Filtered qualified-hitter Statcast rows ({season}): {len(filtered_statcast)}")

    # Join player info back
    filtered_statcast = filtered_statcast.merge(
        season_players,
        left_on=["Season", "batter"],
        right_on=["Season", "mlbam_id"],
        how="left",
    )

    # Save season file
    season_file = os.path.join(OUTPUT_DIR, f"statcast_qualified_hitters_{season}.csv")
    filtered_statcast.to_csv(season_file, index=False)
    print(f"Saved season file: {season_file}")

    all_statcast_parts.append(filtered_statcast)


# 5. Combine all seasons
print("\nStep 5. Combining all season files...")

if all_statcast_parts:
    final_df = pd.concat(all_statcast_parts, ignore_index=True)

    # Optional sort
    sort_cols = [c for c in ["Season", "game_date", "game_pk", "at_bat_number", "pitch_number"] if c in final_df.columns]
    if sort_cols:
        final_df = final_df.sort_values(sort_cols).reset_index(drop=True)

    final_path = os.path.join(OUTPUT_DIR, FINAL_FILE)
    final_df.to_csv(final_path, index=False)

    print(f"Final combined rows: {len(final_df)}")
    print(f"Saved final combined file: {final_path}")
else:
    print("No season data was combined because no filtered Statcast rows were returned.")

print("\nDone.")