import pandas as pd
import numpy as np
from typing import List, Dict, Any

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess solar data
    """
    df = pd.read_csv(file_path)
    
    # Convert timestamp
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    return df

def calculate_summary_stats(df: pd.DataFrame, groupby_col: str = None) -> pd.DataFrame:
    """
    Calculate summary statistics for the dataset
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if groupby_col and groupby_col in df.columns:
        summary = df.groupby(groupby_col)[numeric_cols].agg(['mean', 'median', 'std'])
    else:
        summary = df[numeric_cols].agg(['mean', 'median', 'std'])
    
    return summary

def detect_outliers_zscore(df: pd.DataFrame, columns: List[str], threshold: float = 3) -> pd.DataFrame:
    """
    Detect outliers using Z-score method
    """
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean = df_clean[z_scores <= threshold]
    
    return df_clean

def resample_time_data(df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
    """
    Resample time series data
    """
    if 'Timestamp' not in df.columns:
        return df
    
    df_ts = df.set_index('Timestamp')
    numeric_cols = df_ts.select_dtypes(include=[np.number]).columns
    
    return df_ts[numeric_cols].resample(freq).mean().reset_index()