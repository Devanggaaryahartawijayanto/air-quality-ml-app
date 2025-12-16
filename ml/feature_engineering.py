"""
Feature engineering module for air quality forecasting.
Creates time-based, lag, rolling, and rate-of-change features.
"""

import pandas as pd
import numpy as np


def select_station_with_most_data(df: pd.DataFrame, target_station: str = None) -> tuple:
    """
    Select station with most complete PM2.5 data.
    
    Args:
        df: Preprocessed DataFrame
        target_station: Specific station name, or None to auto-select
    
    Returns:
        (station_name, station_data)
    """
    if target_station is None:
        station_completeness = df.groupby('stasiun')['pm_duakomalima'].count()
        target_station = station_completeness.idxmax()
        print(f"🎯 Using station with most data: {target_station}")
    
    station_data = df[df['stasiun'] == target_station].copy()
    station_data = station_data.sort_values('tanggal')
    
    return target_station, station_data


def handle_pm25_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing PM2.5 values using forward fill and interpolation."""
    df = df.copy()
    
    pm25_missing_before = df['pm_duakomalima'].isna().sum()
    print(f"❌ PM2.5 missing values before: {pm25_missing_before} ({pm25_missing_before/len(df)*100:.2f}%)")
    
    df['pm_duakomalima_original'] = df['pm_duakomalima'].copy()
    
    # Forward fill
    df['pm_duakomalima'] = df['pm_duakomalima'].fillna(method='ffill')
    pm25_after_ffill = df['pm_duakomalima'].isna().sum()
    print(f"🔄 After forward fill: {pm25_after_ffill} missing")
    
    # Linear interpolation
    df['pm_duakomalima'] = df['pm_duakomalima'].interpolate(method='linear')
    pm25_after_interpolate = df['pm_duakomalima'].isna().sum()
    print(f"📈 After interpolation: {pm25_after_interpolate} missing")
    
    # Drop remaining missing
    if pm25_after_interpolate > 0:
        df = df.dropna(subset=['pm_duakomalima'])
        print(f"🗑️  Dropped {pm25_after_interpolate} rows with remaining missing PM2.5")
    
    print(f"✅ Final PM2.5 missing values: {df['pm_duakomalima'].isna().sum()}")
    
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic time-based features."""
    df = df.copy()
    
    print("\n⏰ CREATING TIME-BASED FEATURES...")
    df['day_of_week'] = df['tanggal'].dt.dayofweek
    df['month'] = df['tanggal'].dt.month
    df['year'] = df['tanggal'].dt.year
    df['day_of_year'] = df['tanggal'].dt.dayofyear
    df['quarter'] = df['tanggal'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df


def create_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create seasonal features specific to Indonesia."""
    df = df.copy()
    
    df['season'] = (df['month'] % 12 + 3) // 3
    df['is_rainy_season'] = ((df['month'] >= 10) | (df['month'] <= 3)).astype(int)
    
    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag features for PM2.5."""
    df = df.copy()
    
    print("\n📈 CREATING LAG FEATURES...")
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'pm25_lag_{lag}'] = df['pm_duakomalima'].shift(lag)
        missing_lag = df[f'pm25_lag_{lag}'].isna().sum()
        print(f"   Lag {lag:2d}: {missing_lag} missing values ({missing_lag/len(df)*100:.2f}%)")
    
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling statistics features."""
    df = df.copy()
    
    print("\n📊 CREATING ROLLING STATISTICS...")
    windows = [7, 14, 30]
    for window in windows:
        df[f'pm25_rolling_mean_{window}'] = df['pm_duakomalima'].rolling(window=window).mean()
        df[f'pm25_rolling_std_{window}'] = df['pm_duakomalima'].rolling(window=window).std()
        df[f'pm25_rolling_min_{window}'] = df['pm_duakomalima'].rolling(window=window).min()
        df[f'pm25_rolling_max_{window}'] = df['pm_duakomalima'].rolling(window=window).max()
        
        missing_rolling = df[f'pm25_rolling_mean_{window}'].isna().sum()
        print(f"   Window {window:2d}: {missing_rolling} missing values ({missing_rolling/len(df)*100:.2f}%)")
    
    return df


def create_rate_of_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rate of change features."""
    df = df.copy()
    
    print("\n📈 CREATING RATE OF CHANGE FEATURES...")
    df['pm25_day_change'] = df['pm_duakomalima'].diff()
    df['pm25_week_change'] = df['pm_duakomalima'].diff(7)
    
    missing_day_change = df['pm25_day_change'].isna().sum()
    missing_week_change = df['pm25_week_change'].isna().sum()
    print(f"   Day change: {missing_day_change} missing")
    print(f"   Week change: {missing_week_change} missing")
    
    return df


def balanced_feature_engineering(df: pd.DataFrame, target_station: str = None, verbose: bool = True) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Preprocessed DataFrame
        target_station: Specific station or None for auto-select
        verbose: Print processing steps
    
    Returns:
        DataFrame with engineered features
    """
    if verbose:
        print(f"\n🔧 FEATURE ENGINEERING")
        print("=" * 50)
    
    # Select station
    target_station, station_data = select_station_with_most_data(df, target_station)
    
    if verbose:
        print(f"\n🔍 FEATURE ENGINEERING FOR STATION: {target_station}")
        print("=" * 50)
        print(f"📊 Initial data points: {len(station_data)}")
    
    # Handle missing values
    if 'pm_duakomalima' in station_data.columns:
        station_data = handle_pm25_missing_values(station_data)
    
    # Create features
    station_data = create_time_features(station_data)
    station_data = create_seasonal_features(station_data)
    station_data = create_lag_features(station_data)
    station_data = create_rolling_features(station_data)
    station_data = create_rate_of_change_features(station_data)
    
    if verbose:
        print(f"\n✅ Features created: {len(station_data)} initial data points")
    
    # Drop rows with missing values from feature engineering
    initial_count = len(station_data)
    station_data = station_data.dropna()
    final_count = len(station_data)
    rows_dropped_fe = initial_count - final_count
    
    if verbose:
        print(f"✅ Final dataset after cleaning: {final_count} data points")
        print(f"🗑️  Rows dropped in feature engineering: {rows_dropped_fe} ({rows_dropped_fe/initial_count*100:.2f}%)")
        
        print(f"\n🎯 FEATURE ENGINEERING SUMMARY:")
        print("=" * 50)
        print(f"📊 Final dataset: {len(station_data)} samples")
        print(f"🔢 Total features created: {len(station_data.columns)}")
        print(f"📅 Date range: {station_data['tanggal'].min()} to {station_data['tanggal'].max()}")
    
    return station_data


if __name__ == "__main__":
    from data_loader import load_air_quality_data
    from preprocessing import preprocess_data
    
    df_raw = load_air_quality_data()
    df_clean = preprocess_data(df_raw, verbose=False)
    df_features = balanced_feature_engineering(df_clean)
    print(f"\n✅ Feature engineering completed: {df_features.shape}")