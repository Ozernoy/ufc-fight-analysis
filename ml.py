import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from experiments import get_experiments

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UFCFeatureEngineering:
    """Class for handling feature engineering tasks for UFC fight data."""

    @staticmethod
    def create_fighter_features(
        df: pd.DataFrame, suffix: str, exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Create fighter features with specified suffix."""
        exclude_columns = exclude_columns or []
        columns = [col for col in df.columns if col not in exclude_columns]
        renamed_columns = {col: f"{col}_{suffix}" for col in columns}
        return df[columns].rename(columns=renamed_columns)

    @staticmethod
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


class UFCDataMirror:
    """Class for handling data mirroring operations."""

    @staticmethod
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

    @staticmethod
    def mirror_half_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """Mirror half of the dataset while maintaining class balance."""
        wins_a = df[df["Winner"] == 1]
        wins_b = df[df["Winner"] == 0]

        half_wins_a = wins_a.sample(frac=0.5, random_state=42)
        half_wins_b = wins_b.sample(frac=0.5, random_state=42)

        fighter_a_cols = [col for col in df.columns if "_A" in col]
        fighter_b_cols = [col for col in df.columns if "_B" in col]
        column_mapping = {
            **{a: a.replace("_A", "_B") for a in fighter_a_cols},
            **{b: b.replace("_B", "_A") for b in fighter_b_cols},
        }

        mirrored_half_a = half_wins_a.copy()
        mirrored_half_b = half_wins_b.copy()

        for half_df in [mirrored_half_a, mirrored_half_b]:
            half_df.rename(columns=column_mapping, inplace=True)
            half_df["Winner"] = 1 - half_df["Winner"]

        return pd.concat(
            [
                wins_a.drop(half_wins_a.index),
                wins_b.drop(half_wins_b.index),
                mirrored_half_a,
                mirrored_half_b,
            ],
            ignore_index=True,
        )


class UFCModelTrainer:
    """Class for training and evaluating UFC fight prediction models."""

    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.feature_columns: Optional[List[str]] = None

    def prepare_features(
        self,
        df: pd.DataFrame,
        columns_to_keep: List[str],
        encoding_columns: Dict[str, Tuple[str, Dict[str, Any]]],
        diff_columns: Dict[str, bool],
        mirror: Optional[str] = None,
        scale_features: bool = False,
        missing_value_strategy: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training."""
        processed_df = df.copy()

        # Apply mirroring if specified
        if mirror == "whole":
            processed_df = UFCDataMirror.mirror_whole_dataset(processed_df)
        elif mirror == "half":
            processed_df = UFCDataMirror.mirror_half_dataset(processed_df)
        # Export the processed DataFrame to CSV
        processed_df.to_csv('df.csv', index=False)
        # Calculate differences for specified columns
        for feature, apply_diff in diff_columns.items():
            if apply_diff:
                if (
                    f"{feature}_A" in processed_df.columns
                    and f"{feature}_B" in processed_df.columns
                ):
                    processed_df[f"{feature}_Diff"] = (
                        processed_df[f"{feature}_A"] - processed_df[f"{feature}_B"]
                    )
                    processed_df.drop(
                        [f"{feature}_A", f"{feature}_B"], axis=1, inplace=True
                    )
                else:
                    logger.warning(
                        f"Feature columns for {feature} not found in DataFrame."
                    )

        # Apply encodings to specified columns first
        if encoding_columns:
            processed_df = self._apply_encodings(processed_df, encoding_columns)

                # Get all columns that will be used in the final dataset
        relevant_columns = set(columns_to_keep)
        relevant_columns.update(
            [f"{col}_Diff" for col in diff_columns if diff_columns[col]]
        )
        # Update relevant columns based on encoding method
        for col, (encoding_method, _) in encoding_columns.items():
            if encoding_method == 'onehot':
                # Add all columns that start with the original column name
                relevant_columns.update([c for c in processed_df.columns if c.startswith(col)])
            elif encoding_method == 'label':
                # Column name stays the same
                relevant_columns.add(col)
            elif encoding_method == 'frequency':
                # Add the frequency column
                relevant_columns.add(f"{col}_freq")


        # Drop all columns except those specified
        columns_to_drop = [col for col in processed_df.columns if col not in relevant_columns]
        processed_df.drop(columns=columns_to_drop, inplace=True)

        # Handle missing values with specified strategy
        processed_df = self.handle_missing_values(processed_df, missing_value_strategy)

        # Scale features if requested
        if scale_features:
            # Get numeric columns excluding 'Winner'
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            numeric_cols = numeric_cols[numeric_cols != 'Winner']
            
            scaler = StandardScaler()
            # Scale the numeric columns
            scaled_data = scaler.fit_transform(processed_df[numeric_cols])
            # Create new column names with _scaled suffix
            scaled_cols = [f"{col}_scaled" for col in numeric_cols]
            # Drop original numeric columns and add scaled columns with new names
            processed_df = processed_df.drop(columns=numeric_cols)
            processed_df[scaled_cols] = scaled_data

        

        # Prepare features and target
        if "Winner" not in processed_df.columns:
            raise ValueError("The 'Winner' column is missing from the DataFrame.")
        X = processed_df.drop("Winner", axis=1)
        y = processed_df["Winner"]

        self.feature_columns = X.columns.tolist()

        return X, y

    def _apply_encodings(
        self, X: pd.DataFrame, encoding_params: Dict[str, Tuple[str, Dict[str, Any]]]
    ) -> pd.DataFrame:
        """Apply various encoding methods to features."""
        X_encoded = X.copy()

        for column, (encoding_method, params) in encoding_params.items():
            if column in X_encoded.columns:
                if encoding_method == "label":
                    le = LabelEncoder()
                    X_encoded[column] = le.fit_transform(X_encoded[column].astype(str))

                elif encoding_method == "onehot":
                    dummies = pd.get_dummies(X_encoded[column], prefix=column)
                    X_encoded = pd.concat([X_encoded, dummies], axis=1)
                    X_encoded.drop(column, axis=1, inplace=True)

                elif encoding_method == "frequency":
                    frequency_map = (
                        X_encoded[column].value_counts(normalize=True).to_dict()
                    )
                    X_encoded[f"{column}_freq"] = X_encoded[column].map(frequency_map)
                    X_encoded.drop(column, axis=1, inplace=True)
                else:
                    logger.warning(
                        f"Unknown encoding method '{encoding_method}' for column '{column}'."
                    )
            else:
                logger.warning(
                    f"Column '{column}' not found in DataFrame for encoding."
                )

        return X_encoded

    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> Dict[str, Any]:
        """Train model and evaluate performance."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if param_grid is None:
            param_grid = {
                "n_estimators": [10, 25, 50, 100],
                "max_depth": [5, 10, 20, 30],
                "min_samples_split": [2, 5, 10, 15],
            }

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        y_pred = self.model.predict(X_test)

        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "cv_score": float(grid_search.best_score_),
            "best_params": grid_search.best_params_,
            "classification_report": classification_report(y_test, y_pred),
            "feature_importance": pd.DataFrame(
                {
                    "feature": self.feature_columns,
                    "importance": self.model.feature_importances_,
                }
            ).sort_values("importance", ascending=False),
        }

    def handle_missing_values(
        self, df: pd.DataFrame, strategy: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame using specified strategies.

        Args:
            df: Input DataFrame
            strategy: Dictionary with format:
                {
                    'default': str,  # default strategy ('mean', 'median', 'mode', 'constant')
                    'columns': {     # column-specific strategies
                        'column_name': {
                            'method': str,  # 'mean', 'median', 'mode', 'constant'
                            'value': Any    # only needed if method is 'constant'
                        }
                    }
                }
        """
        if strategy is None:
            strategy = {"default": "auto"}

        processed_df = df.copy()
        default_strategy = strategy.get("default", "auto")
        column_strategies = strategy.get("columns", {})

        # Handle missing values column by column
        for column in processed_df.columns:
            if processed_df[column].isnull().sum() == 0:
                continue

            # Get strategy for this column
            col_strategy = column_strategies.get(column, {"method": default_strategy})
            method = col_strategy.get("method", default_strategy)

            # Apply the specified method
            if method == "constant":
                if "value" not in col_strategy:
                    raise ValueError(
                        f"Constant method specified for {column} but no value provided"
                    )
                fill_value = col_strategy["value"]
                processed_df[column] = processed_df[column].fillna(fill_value)
                logger.info(
                    f"Filled missing values in {column} with constant: {fill_value}"
                )

            elif method in ["mean", "median", "mode"] or method == "auto":
                if pd.api.types.is_numeric_dtype(processed_df[column]):
                    if method == "mean" or (
                        method == "auto" and not processed_df[column].isnull().all()
                    ):
                        fill_value = processed_df[column].mean()
                        method_name = "mean"
                    elif method == "median":
                        fill_value = processed_df[column].median()
                        method_name = "median"
                    else:
                        fill_value = processed_df[column].mode()[0]
                        method_name = "mode"
                else:
                    fill_value = processed_df[column].mode()[0]
                    method_name = "mode"

                processed_df[column] = processed_df[column].fillna(fill_value)
                logger.info(
                    f"Filled missing values in {column} using {method_name}: {fill_value}"
                )

            else:
                raise ValueError(f"Unknown missing value handling method: {method}")

        return processed_df


class UFCExperimentRunner:
    """Class for running and managing UFC fight prediction experiments."""

    def __init__(self, results_dir: Union[str, Path] = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.trainer = UFCModelTrainer()

    def run_experiment(
        self, df: pd.DataFrame, experiment_name: str, experiment_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single experiment with given parameters."""
        logger.info(f"Running experiment: {experiment_name}")

        X, y = self.trainer.prepare_features(
            df,
            experiment_params["columns_to_keep"],
            experiment_params["encoding_columns"],
            experiment_params["diff_columns"],
            experiment_params.get("mirror"),
            experiment_params.get("x_scale", False),
            experiment_params.get("missing_value_strategy"),
        )

        results = self.trainer.train_and_evaluate(X, y)
        results["experiment_name"] = experiment_name
        results["parameters"] = experiment_params

        return results

    def save_results(self, results: Dict[str, Any], timestamp: str) -> None:
        """Save experiment results to a single CSV file."""
        filename = self.results_dir / "model_results.csv"

        try:
            # Create a dictionary of feature importances
            feature_importance_dict = dict(zip(
                results["feature_importance"]["feature"],
                results["feature_importance"]["importance"]
            ))

            # Get mirroring method from parameters
            mirror_method = results["parameters"].get("mirror", "none")

            # Flatten the results dictionary
            flat_results = {
                "cv_score": results["cv_score"],
                "n_estimators": results["best_params"]["n_estimators"],
                "max_depth": results["best_params"]["max_depth"],
                "min_samples_split": results["best_params"]["min_samples_split"],
                "columns": sorted(self.trainer.feature_columns),
                "mirror_method": mirror_method,
                "feature_importance": feature_importance_dict
            }

            # Create DataFrame with single row
            results_df = pd.DataFrame([flat_results])

            # If file exists, append without header. If not, create new file with header
            if filename.exists():
                results_df.to_csv(filename, mode="a", header=False, index=False)
            else:
                results_df.to_csv(filename, index=False)

            logger.info(f"Results saved to: {filename}")

        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise


def main():
    # Load data
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

    # Create fighter A and B dataframes with appropriate suffixes
    fighters_a = UFCFeatureEngineering.create_fighter_features(
        fighters_df,
        suffix="A",
    )
    fighters_b = UFCFeatureEngineering.create_fighter_features(
        fighters_df,
        suffix="B",
    )

    # Merge events with fighter data
    df = df.merge(fighters_a, left_on="Fighter_A", right_on="Name_A", how="left").merge(
        fighters_b, left_on="Fighter_B", right_on="Name_B", how="left"
    )

    df["Age_A"] = (df["Event_Date"] - df["DOB_A"]).dt.days / 365.25
    df["Age_B"] = (df["Event_Date"] - df["DOB_B"]).dt.days / 365.25

    # Add fight history features
    df = UFCFeatureEngineering.add_fight_history_features(df)

    # Add Winner column
    df["Winner"] = df["W_L"].apply(lambda x: 1 if x == "win" else 0)
    
    # Log initial missing values
    initial_nulls = df.isnull().sum()
    logger.info("Initial missing values:")
    for col, null_count in initial_nulls[initial_nulls > 0].items():
        logger.info(f"{col}: {null_count}")

    # Get experiments
    experiments = get_experiments()

    # Create experiment runner and run experiments
    runner = UFCExperimentRunner()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for experiment in experiments:
        results = runner.run_experiment(
            df.copy(), experiment["name"], experiment["params"]
        )
        runner.save_results(results, timestamp)


if __name__ == "__main__":
    main()
