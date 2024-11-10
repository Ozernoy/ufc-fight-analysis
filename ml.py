import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple, Any
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UFCFeatureEngineering:
    """Class for handling feature engineering tasks for UFC fight data."""
    
    @staticmethod
    def create_fighter_features(df: pd.DataFrame, suffix: str, exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Create fighter features with specified suffix."""
        exclude_columns = exclude_columns or []
        columns = [col for col in df.columns if col not in exclude_columns]
        renamed_columns = {col: f'{col}_{suffix}' for col in columns}
        return df[columns].rename(columns=renamed_columns)

    @staticmethod
    def add_fight_history_features(events_data: pd.DataFrame) -> pd.DataFrame:
        """Add cumulative fight history features for each fighter."""
        df = events_data.copy()
        df['Event_Date'] = pd.to_datetime(df['Event_Date'])
        df = df.sort_values('Event_Date')
        history_columns = [
            'fights', 'wins', 'losses', 'current_win_streak', 'current_loss_streak',
            'longest_win_streak', 'longest_loss_streak'
        ]
        
        # Initialize columns
        for col in history_columns:
            df[f'Fighter_A_{col}'] = 0
            df[f'Fighter_B_{col}'] = 0

        fighter_history: Dict[str, Dict[str, int]] = {}

        for i, row in df.iterrows():
            for fighter_col in ['Fighter_A', 'Fighter_B']:
                fighter_name = row[fighter_col]

                if pd.isnull(fighter_name):
                    continue  # Skip if fighter name is missing

                if fighter_name not in fighter_history:
                    fighter_history[fighter_name] = {
                        'fights': 0, 'wins': 0, 'losses': 0,
                        'win_streak': 0, 'loss_streak': 0,
                        'longest_win_streak': 0, 'longest_loss_streak': 0
                    }

                stats = fighter_history[fighter_name]

                # Record pre-fight stats in DataFrame
                df.at[i, f'{fighter_col}_fights'] = stats['fights']
                df.at[i, f'{fighter_col}_wins'] = stats['wins']
                df.at[i, f'{fighter_col}_losses'] = stats['losses']
                df.at[i, f'{fighter_col}_current_win_streak'] = stats['win_streak']
                df.at[i, f'{fighter_col}_current_loss_streak'] = stats['loss_streak']
                df.at[i, f'{fighter_col}_longest_win_streak'] = stats['longest_win_streak']
                df.at[i, f'{fighter_col}_longest_loss_streak'] = stats['longest_loss_streak']

                # Determine result of current fight
                is_win = (row['W_L'] == 'win') if (fighter_col == 'Fighter_A') else (row['W_L'] == 'loss')

                # Update stats with the result of current fight
                stats['fights'] += 1
                if is_win:
                    stats['wins'] += 1
                    stats['win_streak'] += 1
                    stats['loss_streak'] = 0
                    stats['longest_win_streak'] = max(stats['longest_win_streak'], stats['win_streak'])
                else:
                    stats['losses'] += 1
                    stats['loss_streak'] += 1
                    stats['win_streak'] = 0
                    stats['longest_loss_streak'] = max(stats['longest_loss_streak'], stats['loss_streak'])

        return df


class UFCDataMirror:
    """Class for handling data mirroring operations."""
    
    @staticmethod
    def mirror_whole_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """Mirror entire dataset by swapping fighter A and B."""
        mirrored_df = df.copy()
        
        # Swap Fighter A and B columns
        fighter_a_cols = [col for col in df.columns if '_A' in col]
        fighter_b_cols = [col for col in df.columns if '_B' in col]
        
        column_mapping = {
            **{a: a.replace('_A', '_B') for a in fighter_a_cols},
            **{b: b.replace('_B', '_A') for b in fighter_b_cols}
        }
        
        mirrored_df.rename(columns=column_mapping, inplace=True)
        mirrored_df['Winner'] = 1 - df['Winner']
        
        return pd.concat([df, mirrored_df], ignore_index=True)

    @staticmethod
    def mirror_half_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """Mirror half of the dataset while maintaining class balance."""
        wins_a = df[df['Winner'] == 1]
        wins_b = df[df['Winner'] == 0]
        
        half_wins_a = wins_a.sample(frac=0.5, random_state=42)
        half_wins_b = wins_b.sample(frac=0.5, random_state=42)
        
        fighter_a_cols = [col for col in df.columns if '_A' in col]
        fighter_b_cols = [col for col in df.columns if '_B' in col]
        column_mapping = {
            **{a: a.replace('_A', '_B') for a in fighter_a_cols},
            **{b: b.replace('_B', '_A') for b in fighter_b_cols}
        }
        
        mirrored_half_a = half_wins_a.copy()
        mirrored_half_b = half_wins_b.copy()
        
        for half_df in [mirrored_half_a, mirrored_half_b]:
            half_df.rename(columns=column_mapping, inplace=True)
            half_df['Winner'] = 1 - half_df['Winner']
        
        return pd.concat([
            wins_a.drop(half_wins_a.index),
            wins_b.drop(half_wins_b.index),
            mirrored_half_a,
            mirrored_half_b
        ], ignore_index=True)


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
        scale_features: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training."""
        processed_df = df.copy()
        
        # Debug: Print all columns in the DataFrame
        logger.info("Available columns in DataFrame:")
        logger.info(processed_df.columns.tolist())
        
        
        # Handle missing values before any transformations
        null_counts_before = processed_df.isnull().sum()
        
        # Get all columns that will be used in the final dataset
        relevant_columns = set(columns_to_keep)
        relevant_columns.update([f"{col}_A" for col in diff_columns if diff_columns[col]])
        relevant_columns.update([f"{col}_B" for col in diff_columns if diff_columns[col]])
        relevant_columns.update(encoding_columns.keys())  # Add encoding columns to relevant columns
        
        # Handle missing values
        for column in relevant_columns:
            if column not in processed_df.columns:
                continue
                
            null_count = processed_df[column].isnull().sum()
            if null_count > 0:
                if pd.api.types.is_numeric_dtype(processed_df[column]):
                    fill_value = processed_df[column].mean()
                    processed_df[column].fillna(fill_value, inplace=True)
                    logger.info(f"Filled {null_count} missing values in {column} with mean value {fill_value:.2f}")
                else:
                    fill_value = processed_df[column].mode()[0]
                    processed_df[column].fillna(fill_value, inplace=True)
                    logger.info(f"Filled {null_count} missing values in {column} with mode value '{fill_value}'")
        
        # Print summary of changes
        null_counts_after = processed_df.isnull().sum()
        total_nulls_before = null_counts_before.sum()
        total_nulls_after = null_counts_after.sum()
        logger.info(f"\nTotal missing values filled: {total_nulls_before - total_nulls_after}")
        
        # Apply mirroring if specified
        if mirror == 'whole':
            processed_df = UFCDataMirror.mirror_whole_dataset(processed_df)
        elif mirror == 'half':
            processed_df = UFCDataMirror.mirror_half_dataset(processed_df)
        
        # Calculate differences for specified columns
        for feature, apply_diff in diff_columns.items():
            if apply_diff:
                if f'{feature}_A' in processed_df.columns and f'{feature}_B' in processed_df.columns:
                    processed_df[f'{feature}_Diff'] = processed_df[f'{feature}_A'] - processed_df[f'{feature}_B']
                    processed_df.drop([f'{feature}_A', f'{feature}_B'], axis=1, inplace=True)
                else:
                    logger.warning(f"Feature columns for {feature} not found in DataFrame.")
        
        # Prepare features and target
        if 'Winner' not in processed_df.columns:
            raise ValueError("The 'Winner' column is missing from the DataFrame.")
        
        X = processed_df.drop('Winner', axis=1)
        y = processed_df['Winner']
        
        # Apply encodings to specified columns first
        if encoding_columns:
            X = self._apply_encodings(X, encoding_columns)
        
        # Keep only specified columns and encoded columns
        all_features = set(columns_to_keep)
        all_features.update([f"{col}_Diff" for col in diff_columns if diff_columns[col]])
        all_features.update(encoding_columns.keys())  # Ensure encoding columns are included
        
        # Add any new columns created by encoding (like one-hot encoded columns)
        encoded_columns = set(X.columns) - set(processed_df.columns)
        all_features.update(encoded_columns)
        
        # Drop all columns except those specified
        columns_to_drop = [col for col in X.columns if col not in all_features]
        X.drop(columns=columns_to_drop, inplace=True)
            
        # Scale features if requested
        if scale_features:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        
        self.feature_columns = X.columns.tolist()
        return X, y
    
    def _apply_encodings(
        self,
        X: pd.DataFrame,
        encoding_params: Dict[str, Tuple[str, Dict[str, Any]]]
    ) -> pd.DataFrame:
        """Apply various encoding methods to features."""
        X_encoded = X.copy()
        
        for column, (encoding_method, params) in encoding_params.items():
            if column in X_encoded.columns:
                if encoding_method == 'label':
                    le = LabelEncoder()
                    X_encoded[column] = le.fit_transform(X_encoded[column].astype(str))
                
                elif encoding_method == 'onehot':
                    dummies = pd.get_dummies(X_encoded[column], prefix=column)
                    X_encoded = pd.concat([X_encoded, dummies], axis=1)
                    X_encoded.drop(column, axis=1, inplace=True)
                
                elif encoding_method == 'frequency':
                    frequency_map = X_encoded[column].value_counts(normalize=True).to_dict()
                    X_encoded[f"{column}_freq"] = X_encoded[column].map(frequency_map)
                    X_encoded.drop(column, axis=1, inplace=True)
                else:
                    logger.warning(f"Unknown encoding method '{encoding_method}' for column '{column}'.")
            else:
                logger.warning(f"Column '{column}' not found in DataFrame for encoding.")
        
        return X_encoded
    
    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict[str, List[Any]]] = None
    ) -> Dict[str, Any]:
        """Train model and evaluate performance."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        y_pred = self.model.predict(X_test)
        
        return {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'cv_score': float(grid_search.best_score_),
            'best_params': grid_search.best_params_,
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        }


class UFCExperimentRunner:
    """Class for running and managing UFC fight prediction experiments."""
    
    def __init__(self, results_dir: Union[str, Path] = 'results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.trainer = UFCModelTrainer()
    
    def run_experiment(
        self,
        df: pd.DataFrame,
        experiment_name: str,
        experiment_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single experiment with given parameters."""
        logger.info(f"Running experiment: {experiment_name}")
        
        X, y = self.trainer.prepare_features(
            df,
            experiment_params['columns_to_keep'],
            experiment_params['encoding_columns'],
            experiment_params['diff_columns'],
            experiment_params.get('mirror'),
            experiment_params.get('x_scale', False)
        )
        
        results = self.trainer.train_and_evaluate(X, y)
        results['experiment_name'] = experiment_name
        results['parameters'] = experiment_params
        
        return results
    
    def save_results(self, results: Dict[str, Any], timestamp: str) -> None:
        """Save experiment results to a single CSV file."""
        filename = self.results_dir / "model_results.csv"
        
        try:
            # Flatten the results dictionary
            flat_results = {
                'experiment_name': results['experiment_name'],
                'accuracy': results['accuracy'],
                'cv_score': results['cv_score'],
                'timestamp': timestamp,
                'columns_to_keep': ', '.join(results['parameters']['columns_to_keep']),
                'encoding_columns': str(results['parameters']['encoding_columns']),
                'diff_columns': str(results['parameters']['diff_columns']),
                'mirror': results['parameters'].get('mirror', ''),
                'x_scale': str(results['parameters'].get('x_scale', False))
            }
            
            # Add best parameters
            for param, value in results['best_params'].items():
                flat_results[f'best_{param}'] = value
            
            # Create DataFrame with single row
            results_df = pd.DataFrame([flat_results])
            
            # If file exists, append without header. If not, create new file with header
            if filename.exists():
                results_df.to_csv(filename, mode='a', header=False, index=False)
            else:
                results_df.to_csv(filename, index=False)
            
            # Save feature importance to a single file, appending new results
            importance_df = results['feature_importance'].copy()
            importance_df['experiment_name'] = results['experiment_name']
            importance_df['timestamp'] = timestamp
            
            importance_filename = self.results_dir / "feature_importance.csv"
            if importance_filename.exists():
                importance_df.to_csv(importance_filename, mode='a', header=False, index=False)
            else:
                importance_df.to_csv(importance_filename, index=False)
            
            logger.info(f"Results saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise


def main():
    # Load data
    events_df = pd.read_csv(r'C:\Users\d1411\Документы\Python Projects\Final Project\data\csv\events_processed\events_combined.csv')
    fighters_df = pd.read_csv(r'C:\Users\d1411\Документы\Python Projects\Final Project\data\csv\fighters_processed\fighters_combined.csv')
    
    df = events_df.copy()

    # Convert date columns to datetime
    df['Event_Date'] = pd.to_datetime(df['Event_Date'])
    fighters_df['DOB'] = pd.to_datetime(fighters_df['DOB'])
    
    # Create fighter A and B dataframes with appropriate suffixes
    fighters_a = UFCFeatureEngineering.create_fighter_features(
        fighters_df, 
        suffix='A',
    )
    fighters_b = UFCFeatureEngineering.create_fighter_features(
        fighters_df, 
        suffix='B',
    )
        
    # Merge events with fighter data
    df = df.merge(
        fighters_a, left_on='Fighter_A', right_on='Name_A', how='left'
    ).merge(
        fighters_b, left_on='Fighter_B', right_on='Name_B', how='left'
    )
    
    # Calculate age of Fighter A and B at the time of the event
    df['DOB_A'] = pd.to_datetime(df['DOB_A'])
    df['DOB_B'] = pd.to_datetime(df['DOB_B'])
    
    df['Age_A'] = (df['Event_Date'] - df['DOB_A']).dt.days / 365.25
    df['Age_B'] = (df['Event_Date'] - df['DOB_B']).dt.days / 365.25

    
    # Drop unnecessary columns
    df.drop(columns=['Name_A', 'Name_B', 'DOB_A', 'DOB_B'], inplace=True)
    
    # Add fight history features
    df = UFCFeatureEngineering.add_fight_history_features(df)
    
    # Add Winner column
    df['Winner'] = df['W_L'].apply(lambda x: 1 if x == 'win' else 0)
    
    # Define experiments with new parameter structure
    experiments = [
        {
            'name': 'baseline_extended_features',
            'params': {
                'columns_to_keep': [
                    # Fight historyf
                    'Fighter_A_fights', 'Fighter_B_fights',
                    'Fighter_A_wins', 'Fighter_B_wins',
                    'Fighter_A_losses', 'Fighter_B_losses',
                    'Fighter_A_current_win_streak', 'Fighter_B_current_win_streak',
                    'Fighter_A_longest_win_streak', 'Fighter_B_longest_win_streak',
                    'Fighter_A_longest_loss_streak', 'Fighter_B_longest_loss_streak',
                    # Physical attributes
                    'Height_A', 'Height_B',
                    'Weight_A', 'Weight_B',
                    'Reach_A', 'Reach_B',
                    'Age_A', 'Age_B'
                ],
                'encoding_columns': {
                    'STANCE_A': ('label', {}),
                    'STANCE_B': ('label', {}),
                    'Weight_Class': ('onehot', {})
                },
                'diff_columns': {
                    'Height': True,
                    'Weight': True,
                    'Reach': True,
                    'Age': True
                },
                'mirror': 'half',
                'x_scale': True
            }
        },
        {
            'name': 'physical_and_streaks',
            'params': {
                'columns_to_keep': [
                    # Streaks and basic stats
                    'Fighter_A_current_win_streak', 'Fighter_B_current_win_streak',
                    'Fighter_A_longest_win_streak', 'Fighter_B_longest_win_streak',
                    'Fighter_A_wins', 'Fighter_B_wins',
                    # Physical attributes
                    'Height_A', 'Height_B',
                    'Weight_A', 'Weight_B',
                    'Reach_A', 'Reach_B'
                ],
                'encoding_columns': {
                    'STANCE_A': ('onehot', {}),
                    'STANCE_B': ('onehot', {}),
                    'Weight_Class': ('onehot', {})
                },
                'diff_columns': {
                    'Height': True,
                    'Weight': True,
                    'Reach': True
                },
                'mirror': 'whole',
                'x_scale': True
            }
        },
        {
            'name': 'minimal_features',
            'params': {
                'columns_to_keep': [
                    'Fighter_A_wins', 'Fighter_B_wins',
                    'Fighter_A_current_win_streak', 'Fighter_B_current_win_streak',
                    'Fighter_A_fights', 'Fighter_B_fights'
                ],
                'encoding_columns': {
                    'Weight_Class': ('onehot', {})
                },
                'diff_columns': {
                    'fights': True
                },
                'mirror': 'half',
                'x_scale': True
            }
        },
        {
            'name': 'stance_focused',
            'params': {
                'columns_to_keep': [
                    'Fighter_A_wins', 'Fighter_B_wins',
                    'Fighter_A_current_win_streak', 'Fighter_B_current_win_streak',
                    'Fighter_A_fights', 'Fighter_B_fights',
                    'STANCE_A', 'STANCE_B'
                ],
                'encoding_columns': {
                    'STANCE_A': ('onehot', {}),
                    'STANCE_B': ('onehot', {}),
                    'Weight_Class': ('onehot', {})
                },
                'diff_columns': {
                    'fights': True
                },
                'mirror': 'half',
                'x_scale': True
            }
        },
        {
            'name': 'physical_attributes',
            'params': {
                'columns_to_keep': [
                    'Height_A', 'Height_B',
                    'Weight_A', 'Weight_B',
                    'Reach_A', 'Reach_B',
                    'Age_A', 'Age_B',
                    'Fighter_A_wins', 'Fighter_B_wins',
                    'Fighter_A_fights', 'Fighter_B_fights'
                ],
                'encoding_columns': {
                    'Weight_Class': ('onehot', {})
                },
                'diff_columns': {
                    'Height': True,
                    'Weight': True,
                    'Reach': True,
                    'Age': True,
                    'fights': True
                },
                'mirror': 'whole',
                'x_scale': True
            }
        }
    ]
    
    # Create experiment runner and run experiments
    runner = UFCExperimentRunner()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for experiment in experiments:
        results = runner.run_experiment(df.copy(), experiment['name'], experiment['params'])
        runner.save_results(results, timestamp)

if __name__ == "__main__":
    main()