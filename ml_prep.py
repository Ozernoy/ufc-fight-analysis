import pandas as pd
from typing import Dict  
import logging  

# Add logger configuration
logger = logging.getLogger(__name__)

def map_suffix(
        df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        columns = list(df.columns)
        renamed_columns = {col: f"{col}_{suffix}" for col in columns}
        return df[columns].rename(columns=renamed_columns)

def add_fight_history_features(events_data: pd.DataFrame) -> pd.DataFrame:
        """Add cumulative fight history features for each fighter."""
        df = events_data.copy()
        df = df.sort_values("Event_Date")

        # Initialize columns with simple A/B suffixes
        for suffix in ["A", "B"]:
            df[f"fights_{suffix}"] = 0
            df[f"wins_{suffix}"] = 0
            df[f"losses_{suffix}"] = 0
            df[f"current_win_streak_{suffix}"] = 0
            df[f"current_loss_streak_{suffix}"] = 0
            df[f"longest_win_streak_{suffix}"] = 0
            df[f"longest_loss_streak_{suffix}"] = 0

        fighter_history: Dict[str, Dict[str, int]] = {}

        for i, row in df.iterrows():
            for fighter_col, suffix in [("Fighter_A", "A"), ("Fighter_B", "B")]:
                fighter_name = row[fighter_col]

                if pd.isnull(fighter_name):
                    logger.info(
                        f"Skipped row {i} while adding fight history features because fighter name is missing"
                    )
                    continue

                if fighter_name not in fighter_history:
                    fighter_history[fighter_name] = {
                        "fights": 0,
                        "wins": 0,
                        "losses": 0,
                        "win_streak": 0,
                        "loss_streak": 0,
                        "longest_win_streak": 0,
                        "longest_loss_streak": 0,
                    }

                stats = fighter_history[fighter_name]

                # Record pre-fight stats in DataFrame using simple suffix
                df.at[i, f"fights_{suffix}"] = stats["fights"]
                df.at[i, f"wins_{suffix}"] = stats["wins"]
                df.at[i, f"losses_{suffix}"] = stats["losses"]
                df.at[i, f"current_win_streak_{suffix}"] = stats["win_streak"]
                df.at[i, f"current_loss_streak_{suffix}"] = stats["loss_streak"]
                df.at[i, f"longest_win_streak_{suffix}"] = stats["longest_win_streak"]
                df.at[i, f"longest_loss_streak_{suffix}"] = stats["longest_loss_streak"]

                # Determine result of current fight
                is_win = (
                    (row["W_L"] == "win")
                    if (fighter_col == "Fighter_A")
                    else (row["W_L"] == "loss")
                )

                # Update stats with the result of current fight
                stats["fights"] += 1
                if is_win:
                    stats["wins"] += 1
                    stats["win_streak"] += 1
                    stats["loss_streak"] = 0
                    stats["longest_win_streak"] = max(
                        stats["longest_win_streak"], stats["win_streak"]
                    )
                else:
                    stats["losses"] += 1
                    stats["loss_streak"] += 1
                    stats["win_streak"] = 0
                    stats["longest_loss_streak"] = max(
                        stats["longest_loss_streak"], stats["loss_streak"]
                    )

        return df

def mirror_half_dataset(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Mirror a random half of the dataset while maintaining class balance.
    
    Args:
        df: Input DataFrame
        random_state: Random seed for reproducibility
    Returns:
        DataFrame with randomly selected half of rows mirrored
    """
    # Select random half of the dataset
    half_to_mirror = df.sample(frac=0.5, random_state=random_state)
    unchanged_half = df.drop(half_to_mirror.index)

    # Get columns to swap
    fighter_a_cols = [col for col in df.columns if "_A" in col]
    fighter_b_cols = [col for col in df.columns if "_B" in col]
    column_mapping = {
        **{a: a.replace("_A", "_B") for a in fighter_a_cols},
        **{b: b.replace("_B", "_A") for b in fighter_b_cols},
    }

    # Mirror the selected half
    mirrored_half = half_to_mirror.copy()
    mirrored_half.rename(columns=column_mapping, inplace=True)
    mirrored_half["Winner"] = 1 - mirrored_half["Winner"]

    # Combine mirrored and unchanged portions
    return pd.concat(
        [unchanged_half, mirrored_half],
        ignore_index=True,
    )

def mirror_whole_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """Mirror entire dataset by swapping fighter A and B."""
        mirrored_df = df.copy()

        # Swap Fighter A and B columns
        fighter_a_cols = [col for col in df.columns if "_A" in col]
        fighter_b_cols = [col for col in df.columns if "_B" in col]

        column_mapping = {
            **{a: a.replace("_A", "_B") for a in fighter_a_cols},
            **{b: b.replace("_B", "_A") for b in fighter_b_cols},
        }

        mirrored_df.rename(columns=column_mapping, inplace=True)
        mirrored_df["Winner"] = 1 - df["Winner"]

        return pd.concat([df, mirrored_df], ignore_index=True)

def add_experience_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds experience-based features such as fight difference and win percentage."""
    df["fights_diff"] = df["fights_A"] - df["fights_B"]
    df["win_percentage_A"] = df["wins_A"] / (df["fights_A"].replace(0, 1))
    df["win_percentage_B"] = df["wins_B"] / (df["fights_B"].replace(0, 1))
    df["win_loss_streak_ratio_A"] = (df["longest_win_streak_A"] / (df["longest_loss_streak_A"].replace(0, 1)))
    df["win_loss_streak_ratio_B"] = (df["longest_win_streak_B"] / (df["longest_loss_streak_B"].replace(0, 1)))
    return df

def add_age_related_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds age-related features like age difference and age group indicator."""
    df["age_diff"] = df["Age_A"] - df["Age_B"]
    # Categorize ages into groups
    df["age_group_A"] = pd.cut(df["Age_A"], bins=[0, 25, 30, 35, float("inf")], labels=["Under 25", "25-30", "30-35", "Over 35"])
    df["age_group_B"] = pd.cut(df["Age_B"], bins=[0, 25, 30, 35, float("inf")], labels=["Under 25", "25-30", "30-35", "Over 35"])
    return df

def add_physical_attributes_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Adds physical attributes like reach, weight, and height differences."""
    df["reach_advantage"] = df["Reach_A"] - df["Reach_B"]
    df["weight_diff"] = df["Weight_A"] - df["Weight_B"]
    df["height_diff"] = df["Height_A"] - df["Height_B"]
    return df

def add_stance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds stance-related features such as stance match indicator and stance win rates."""
    df["stance_match"] = (df["STANCE_A"] == df["STANCE_B"]).astype(int)
    return df

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds momentum indicators like recent form metric and current win/loss streak difference."""
    # Assuming 'current_win_streak' represents recent form in current streak
    df["current_streak_diff"] = df["current_win_streak_A"] - df["current_win_streak_B"]
    df["recent_form_A"] = df["wins_A"] / (df["fights_A"].replace(0, 1))
    df["recent_form_B"] = df["wins_B"] / (df["fights_B"].replace(0, 1))
    return df

def add_win_loss_streak_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a streak continuation probability based on recent streaks divided by total fights."""
    df["streak_continuation_prob_A"] = df["current_win_streak_A"] / (df["fights_A"].replace(0, 1))
    df["streak_continuation_prob_B"] = df["current_win_streak_B"] / (df["fights_B"].replace(0, 1))
    return df

def add_weighted_history_features(df: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """Adds exponentially weighted averages for win, loss, and streaks for both fighters."""
    df["weighted_wins_A"] = df["wins_A"].ewm(alpha=alpha).mean()
    df["weighted_losses_A"] = df["losses_A"].ewm(alpha=alpha).mean()
    df["weighted_streak_A"] = df["current_win_streak_A"].ewm(alpha=alpha).mean()
    
    df["weighted_wins_B"] = df["wins_B"].ewm(alpha=alpha).mean()
    df["weighted_losses_B"] = df["losses_B"].ewm(alpha=alpha).mean()
    df["weighted_streak_B"] = df["current_win_streak_B"].ewm(alpha=alpha).mean()
    
    return df

def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Adds interaction terms between key features, such as reach and age."""
    df["reach_age_interaction_A"] = df["Reach_A"] * df["Age_A"]
    df["reach_age_interaction_B"] = df["Reach_B"] * df["Age_B"]
    
    df["reach_win_streak_interaction_A"] = df["Reach_A"] * df["longest_win_streak_A"]
    df["reach_win_streak_interaction_B"] = df["Reach_B"] * df["longest_win_streak_B"]
    
    df["weight_reach_interaction_A"] = df["Weight_A"] * df["reach_advantage"]
    df["weight_reach_interaction_B"] = df["Weight_B"] * df["reach_advantage"]
    
    return df



events_df = pd.read_csv(
    r"C:\Users\d1411\Документы\Python Projects\Final Project\data\csv\events_processed\events_combined.csv"
)
fighters_df = pd.read_csv(
    r"C:\Users\d1411\Документы\Python Projects\Final Project\data\csv\fighters_processed\fighters_combined.csv"
)

df = events_df.copy()

# Convert date columns to datetime
df["Event_Date"] = pd.to_datetime(df["Event_Date"])
fighters_df["DOB"] = pd.to_datetime(fighters_df["DOB"])

fighters_a = map_suffix(
    fighters_df,
    suffix="A",
)
fighters_b = map_suffix(
    fighters_df,
    suffix="B",
)

# Merge events with fighter data
df = df.merge(fighters_a, left_on="Fighter_A", right_on="Name_A", how="left").merge(
    fighters_b, left_on="Fighter_B", right_on="Name_B", how="left"
)

print(df.shape)

df["Age_A"] = (df["Event_Date"] - df["DOB_A"]).dt.days / 365.25
df["Age_B"] = (df["Event_Date"] - df["DOB_B"]).dt.days / 365.25

df = add_fight_history_features(df)
df["Winner"] = df["W_L"].apply(lambda x: 1 if x == "win" else 0)
df = mirror_whole_dataset(df)
#df = mirror_half_dataset(df)

df = add_experience_based_features(df)
df = add_age_related_features(df)
df = add_physical_attributes_ratios(df)
df = add_stance_features(df)
df = add_momentum_indicators(df)
df = add_win_loss_streak_trends(df)
df = add_weighted_history_features(df)
df = add_interaction_terms(df)


'''df.drop(columns=['Event_Date', 'Event_Location', 'W_L', 'Fighter_A', 'Fighter_B',
                       'fighter_id_A', 'fighter_id_B', 'KD_A', 'KD_B', 'STR_A', 'STR_B',
                       'TD_A', 'TD_B', 'SUB_A', 'SUB_B', 'Method', 'Method_Detail', 'Round',
                       'Time', 'event_id', 'Time_seconds', 'Name_A', 'Name_B', 'Record_A',
                       'Record_B', 'DOB_A', 'DOB_B', 'SLpM_A', 'SLpM_B', 'Str. Acc._A',
                       'Str. Acc._B', 'SApM_A', 'SApM_B', 'Str. Def_A', 'Str. Def_B',
                       'TD Avg._A', 'TD Avg._B', 'TD Acc._A', 'TD Acc._B', 'TD Def._A',
                       'TD Def._B', 'Sub. Avg._A', 'Sub. Avg._B'], inplace=True)'''

df.to_csv(r"C:\Users\d1411\Документы\Python Projects\Final Project\data\csv\df_processed\df_combined.csv",
          index=False)


