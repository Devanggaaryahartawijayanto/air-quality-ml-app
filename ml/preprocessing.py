"""
Preprocessing module for air quality data.
Handles missing values, type conversion, and date parsing.
"""

import pandas as pd
import numpy as np


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names by stripping whitespace and converting to lowercase."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def handle_missing_value_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Replace text representations of missing values with NaN."""
    df = df.copy()
    df.replace('N/A', np.nan, inplace=True)
    df.replace('null', np.nan, inplace=True)
    df.replace('TIDAK ADA DATA', np.nan, inplace=True)
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pollution columns to numeric type."""
    df = df.copy()
    numeric_cols = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida',
                   'karbon_monoksida', 'ozon', 'nitrogen_dioksida', 'max']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and create date column from periode_data and tanggal."""
    df = df.copy()
    
    df['tahun'] = df['periode_data'].astype(str).str[:4]
    df['bulan_num'] = df['periode_data'].astype(str).str[4:6]
    
    df['tanggal'] = pd.to_datetime(
        df['tahun'] + '-' + df['bulan_num'] + '-' + df['tanggal'].astype(str),
        errors='coerce'
    )
    
    return df


def drop_invalid_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with invalid dates."""
    df = df.copy()
    initial_rows = len(df)
    df = df.dropna(subset=['tanggal'])
    rows_dropped = initial_rows - len(df)
    print(f"🗑️  Rows dropped due to invalid dates: {rows_dropped}")
    return df


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame by date."""
    df = df.copy()
    df = df.sort_values('tanggal')
    return df


def forward_fill_pollution_data(df: pd.DataFrame) -> pd.DataFrame:
    """Forward fill pollution data per station."""
    df = df.copy()
    pollution_cols = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida',
                     'karbon_monoksida', 'ozon', 'nitrogen_dioksida']
    
    for col in pollution_cols:
        if col in df.columns:
            df[col] = df.groupby('stasiun')[col].transform(lambda x: x.ffill())
    
    return df


def preprocess_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Raw DataFrame
        verbose: Print processing steps
    
    Returns:
        Preprocessed DataFrame
    """
    if verbose:
        print("AIR QUALITY FORECASTING - DKI JAKARTA")
        print("=" * 60)
        print("📊 STEP 1: LOADING AND PREPROCESSING DATA")
        print("=" * 60)
    
    df_before = df.copy()
    
    # Execute preprocessing steps
    df = clean_column_names(df)
    if verbose:
        print("✅ Kolom yang tersedia:", df.columns.tolist())
    
    # Track missing values before
    missing_before = df.isnull().sum()
    if verbose:
        print("\n📋 MISSING VALUES SEBELUM PREPROCESSING:")
        missing_before_pct = (df.isnull().sum() / len(df)) * 100
        missing_df_before = pd.DataFrame({
            'Column': missing_before.index,
            'Missing_Count': missing_before.values,
            'Missing_Percentage': missing_before_pct.values
        }).sort_values('Missing_Count', ascending=False)
        print(missing_df_before[missing_df_before['Missing_Count'] > 0].to_string(index=False))
    
    if verbose:
        print("\n🔄 PROCESSING MISSING VALUES & DATA CLEANING...")
    
    df = handle_missing_value_strings(df)
    
    if verbose:
        print("\n🔢 CONVERTING NUMERIC COLUMNS...")
    df = convert_numeric_columns(df)
    
    if verbose:
        print("\n📅 PROCESSING DATE COLUMNS...")
    df = parse_dates(df)
    
    df = drop_invalid_dates(df)
    df = sort_by_date(df)
    
    if verbose:
        print("\n🧹 ADDITIONAL DATA CLEANING...")
    df = forward_fill_pollution_data(df)
    
    # Summary
    missing_after = df.isnull().sum()
    
    if verbose:
        print("\n🎯 PREPROCESSING SUMMARY:")
        print("=" * 50)
        print(f"📊 Original dataset: {len(df_before)} rows, {len(df_before.columns)} columns")
        print(f"📊 Cleaned dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"🗑️  Total rows dropped: {len(df_before) - len(df)} ({(len(df_before)-len(df))/len(df_before)*100:.2f}%)")
        print(f"📅 Date range: {df['tanggal'].min()} to {df['tanggal'].max()}")
        print(f"🏢 Stations: {df['stasiun'].nunique()}")
        
        total_cells_before = len(df_before) * len(df_before.columns)
        missing_cells_before = missing_before.sum()
        total_cells_after = len(df) * len(df.columns)
        missing_cells_after = missing_after.sum()
        
        data_quality_before = ((total_cells_before - missing_cells_before) / total_cells_before) * 100
        data_quality_after = ((total_cells_after - missing_cells_after) / total_cells_after) * 100
        
        print(f"📈 Data Quality BEFORE: {data_quality_before:.2f}%")
        print(f"📈 Data Quality AFTER: {data_quality_after:.2f}%")
        print(f"📈 Quality Improvement: {data_quality_after - data_quality_before:+.2f}%")
    
    return df


if __name__ == "__main__":
    from data_loader import load_air_quality_data
    
    df_raw = load_air_quality_data()
    df_clean = preprocess_data(df_raw)
    print(f"\n✅ Preprocessing completed: {df_clean.shape}")