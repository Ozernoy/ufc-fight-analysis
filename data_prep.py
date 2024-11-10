import os
from typing import Optional
from datetime import datetime
import pandas as pd
from webscrapper import CSVFileHandler

class UFCDataPrep:
    """Class to prepare UFC data by combining CSV files."""

    def __init__(self, base_folder: str = "data"):
        """Initialize with path to data folder."""
        self.csv_handler = CSVFileHandler(base_folder)

        # Setup additional folders for unprocessed data
        self.events_processed_folder = os.path.join(self.csv_handler.csv_folder, "events_processed")
        self.fighters_processed_folder = os.path.join(self.csv_handler.csv_folder, "fighters_processed")
        os.makedirs(self.events_processed_folder, exist_ok=True)
        os.makedirs(self.fighters_processed_folder, exist_ok=True)

        # Initialize dataframes
        self.events_df: Optional[pd.DataFrame] = None
        self.fighters_df: Optional[pd.DataFrame] = None

    def load_data(self, data_type: str) -> pd.DataFrame:
        """Load and combine all CSV files."""
        print(f"Loading {data_type} data...")
        if data_type == 'events':
            folder = self.csv_handler.csv_events_folder
        elif data_type == 'fighters':
            folder = self.csv_handler.csv_fighters_folder
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        df_list = []
        files = [f for f in os.listdir(folder) if f.endswith('.csv')]

        for file in files:
            try:
                file_path = os.path.join(folder, file)
                df = pd.read_csv(file_path)

                # Add id column from filename
                id = file.replace(f'{data_type}_', '').replace('.csv', '')
                df[f'{data_type[:-1]}_id'] = id

                df_list.append(df)

            except Exception as e:
                print(f"Error loading {data_type} file {file}: {str(e)}")
                continue

        if not df_list:
            raise ValueError(f"No {data_type} data was loaded")

        # Combine all data into single dataframe
        combined_df = pd.concat(df_list, ignore_index=True)
        print(f"Successfully loaded {len(combined_df)} entries from {len(files)} {data_type}")

        # Store in appropriate attribute
        if data_type == 'events':
            self.events_df = combined_df
        elif data_prep == 'fighters':
            self.fighters_df = combined_df

        return combined_df

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get both dataframes."""
        if self.events_df is None:
            self.events_df = self.load_data(data_type='events')
        if self.fighters_df is None:
            self.fighters_df = self.load_data(data_type='fighters')

        return self.events_df, self.fighters_df

    def export_to_csv(self, df: pd.DataFrame, data_type: str) -> None:
        """
        Export DataFrame to CSV file in the appropriate subfolder, overwriting if exists.
        
        Args:
            df: DataFrame to export
            data_type: Type of data ('events' or 'fighters')
        """
        if data_type not in ['events', 'fighters']:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Create processed folders if they don't exist
        processed_folder = os.path.join(self.csv_handler.csv_folder, f"{data_type}_processed")
        os.makedirs(processed_folder, exist_ok=True)
        
        # Save to the appropriate subfolder, mode='w' forces overwrite
        filename = os.path.join(processed_folder, f"{data_type}_combined.csv")
        
        # Remove file if it exists
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"Removed existing file: {filename}")
            except Exception as e:
                print(f"Warning: Could not remove existing file {filename}: {str(e)}")
        
        # Save new file
        df.to_csv(filename, index=False)
        print(f"Exported: {filename}")

    def process_events(self, events_df) -> pd.DataFrame:

        df = events_df.copy()

        # Replace any  '--' values with NaN
        df = df.replace(['--', ''], pd.NA)
        # Fill missing values in 'KD_A', 'KD_B', 'STR_A', 'STR_B', 'TD_A', 'TD_B', 'SUB_A', and 'SUB_B' with 0

        # Convert 'Time' to seconds for easier calculations
        # Split the 'Time' column by ':' to extract minutes and seconds, then calculate total seconds
        df['Time_seconds'] = df['Time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

        int_columns = ['KD_A', 'KD_B', 'STR_A', 'STR_B', 'TD_A', 'TD_B', 'SUB_A',]
        df[int_columns] = df[int_columns].apply(pd.to_numeric, errors='coerce').astype('Int64')
        
        df['Event_Date'] = pd.to_datetime(df['Event_Date'], errors='coerce')

        return df

    def process_fighters(self, fighters_df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean fighters DataFrame."""
        # Make a copy to avoid modifying original
        df = fighters_df.copy()

        # Replace any  '--' values with NaN
        df = df.replace(['--', ''], pd.NA)

        # 1. Strip whitespace from all string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        
        # Process specific columns first, before general NA replacement
        # This ensures the string processing happens while the '--' is still present
        
        # Process Weight and convert to kilograms
        if 'Weight' in df.columns:
            df['Weight'] = df['Weight'].replace('--', pd.NA).str.extract(r'(\d+)').astype(float)
            df['Weight'] = df['Weight'] * 0.453592  # Convert lbs to kg
        
        # Process Height and convert to centimeters
        if 'Height' in df.columns:
            def height_to_cm(height_str):
                if pd.isna(height_str) or height_str == '--':
                    return pd.NA
                try:
                    feet, inches = height_str.replace('"', '').split("'")
                    total_inches = int(feet.strip()) * 12 + int(inches.strip())
                    return total_inches * 2.54  # Convert inches to centimeters
                except:
                    return pd.NA
            df['Height'] = df['Height'].apply(height_to_cm)
            df['Height'] = pd.to_numeric(df['Height'], errors='coerce')  # Convert to float, coercing errors to NaN
        
        # Process percentage columns
        percentage_cols = ['Str. Acc.', 'Str. Def', 'TD Acc.', 'TD Def.']
        for col in percentage_cols:
            if col in df.columns:
                df[col] = df[col].replace('--', pd.NA).str.rstrip('%').astype(float) / 100
        
        # Process Reach and convert to centimeters
        if 'Reach' in df.columns:
            def reach_to_cm(reach_str):
                if pd.isna(reach_str) or reach_str == '--':
                    return pd.NA
                try:
                    inches = int(reach_str.replace('"', '').strip())
                    return inches * 2.54  # Convert inches to centimeters
                except:
                    return pd.NA
            df['Reach'] = df['Reach'].apply(reach_to_cm)
            df['Reach'] = df['Reach'].astype('float', errors='ignore')  # Convert to float, ignoring errors
        
        # Process DOB last since it's a different type of conversion
        if 'DOB' in df.columns:
            df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')

        df.drop(columns=['Unnamed: 11'], inplace=True)
        
        return df



if __name__ == "__main__":
    data_prep = UFCDataPrep()

    try:
        # 1. Load raw datasets
        events_df_raw, fighters_df_raw = data_prep.get_data()

        # 2 & 3. Process each dataset and store in new DataFrames
        events_df_processed = data_prep.process_events(events_df_raw)
        fighters_df_processed = data_prep.process_fighters(fighters_df_raw)

        # 4. Export processed datasets to separate CSV files
        data_prep.export_to_csv(events_df_processed, 'events')
        data_prep.export_to_csv(fighters_df_processed, 'fighters')

        print("Data processing and export completed successfully")
        print(f"Saved: events_processed.csv and fighters_processed.csv")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
