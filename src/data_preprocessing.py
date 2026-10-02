import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    """Loads the player profiles dataset and applies data cleaning & feature engineering."""
    # Load dataset with robust encoding for special characters
    df = pd.read_csv(file_path, encoding='latin1')
    
    # Parse date columns
    current_year = 2026
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    df['joined'] = pd.to_datetime(df['joined'], errors='coerce')
    df['contract_expires'] = pd.to_datetime(df['contract_expires'], errors='coerce')
    
    # Engineer numerical features
    df['age'] = current_year - df['date_of_birth'].dt.year
    df['contract_duration_days'] = (df['contract_expires'] - df['joined']).dt.days
    
    # Handle numeric anomalies & missing values
    df['height'] = pd.to_numeric(df['height'], errors='coerce')
    df['height'] = df['height'].fillna(df['height'].median())
    
    # Drop rows missing the core target variable
    df_model = df.dropna(subset=['main_position']).copy()
    
    # Fill categorical missing values
    cat_cols = ['position', 'foot', 'citizenship']
    for col in cat_cols:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna('Unknown')
            
    return df_model