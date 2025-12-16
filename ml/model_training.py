"""
Model training module for air quality forecasting.
Handles data preparation, leakage checks, and training multiple models.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.impute import SimpleImputer
import xgboost as xgb
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import calculate_all_metrics, sanity_check_predictions


def prepare_train_test_split(station_data: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """
    Prepare chronological train-test split.
    
    Args:
        station_data: DataFrame with features
        test_size: Proportion of data for testing
    
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    print("\n📋 PREPARING TRAINING DATA")
    print("=" * 60)
    
    # Identify non-numeric columns
    non_numeric_cols = station_data.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Non-numeric columns to exclude: {non_numeric_cols}")
    
    # Columns to exclude
    excluded_cols = ['tanggal', 'stasiun', 'kategori'] + non_numeric_cols
    
    # Ensure pm_duakomalima not excluded
    if 'pm_duakomalima' in excluded_cols:
        excluded_cols.remove('pm_duakomalima')
    
    # Select numeric feature columns
    numeric_cols = station_data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in excluded_cols]
    
    print(f"📋 Features used: {len(feature_cols)}")
    
    # Chronological split
    split_idx = int(len(station_data) * (1 - test_size))
    X_train = station_data[feature_cols].iloc[:split_idx]
    X_test = station_data[feature_cols].iloc[split_idx:]
    y_train = station_data['pm_duakomalima'].iloc[:split_idx]
    y_test = station_data['pm_duakomalima'].iloc[split_idx:]
    
    print(f"\n📊 DATASET SPLIT SUMMARY:")
    print("=" * 40)
    print(f"🏋️  Training Data: {len(X_train)} samples ({len(X_train)/len(station_data)*100:.1f}%)")
    print(f"🧪 Testing Data:  {len(X_test)} samples ({len(X_test)/len(station_data)*100:.1f}%)")
    print(f"📅 Training Period: {station_data['tanggal'].iloc[0]} to {station_data['tanggal'].iloc[split_idx-1]}")
    print(f"📅 Testing Period:  {station_data['tanggal'].iloc[split_idx]} to {station_data['tanggal'].iloc[-1]}")
    
    return X_train, X_test, y_train, y_test


def check_data_leakage(X_train, X_test, y_train, y_test):
    """Check and fix potential data leakage issues."""
    print("🔍 PERFORMING DATA LEAKAGE CHECK...")
    
    # Remove target variable from features
    if 'pm_duakomalima' in X_train.columns:
        print("❌ Removing target variable from features...")
        X_train = X_train.drop('pm_duakomalima', axis=1)
        X_test = X_test.drop('pm_duakomalima', axis=1)
    
    # Remove duplicate features
    duplicate_features = X_train.columns[X_train.columns.duplicated()]
    if len(duplicate_features) > 0:
        print(f"❌ Removing duplicate features: {duplicate_features}")
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        X_test = X_test.loc[:, ~X_test.columns.duplicated()]
    
    # Remove suspicious columns
    suspicious_cols = [col for col in X_train.columns if 'pm25' in col.lower() or 'duakomalima' in col.lower()]
    if suspicious_cols:
        print(f"🚨 Suspicious columns found: {suspicious_cols}")
        X_train = X_train.drop(columns=suspicious_cols)
        X_test = X_test.drop(columns=suspicious_cols)
        print("✅ Removed suspicious columns")
    
    print("✅ Data leakage check completed")
    return X_train, X_test, y_train, y_test


def ensure_numeric_data(X_train, X_test, y_train, y_test):
    """Ensure all data is numeric and handle missing values."""
    print("🔢 Ensuring numeric data types...")
    
    # Convert to numeric
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    y_train = pd.to_numeric(y_train, errors='coerce')
    y_test = pd.to_numeric(y_test, errors='coerce')
    
    # Drop all-NaN columns
    X_train = X_train.dropna(axis=1, how='all')
    X_test = X_test[X_train.columns]
    
    # Handle missing values with imputer
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Convert back to DataFrame
    X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index)
    
    # Drop rows with target NaN
    train_mask = ~y_train.isna()
    test_mask = ~y_test.isna()
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    print(f"✅ Numeric data ensured - X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_model_configurations() -> dict:
    """Get dictionary of model configurations."""
    return {
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            max_features='sqrt',
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            enable_categorical=False
        ),
        'LinearRegression': LinearRegression()
    }


def train_single_model(model, X_train, X_test, y_train, y_test, model_name: str) -> dict:
    """
    Train and evaluate a single model.
    
    Returns:
        Dictionary with model, predictions, and metrics
    """
    print(f"\n🔧 Training {model_name}...")
    
    try:
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Sanity checks
        if not sanity_check_predictions(y_train, train_pred, f"{model_name} Train"):
            return None
        if not sanity_check_predictions(y_test, test_pred, f"{model_name} Test"):
            return None
        
        # Calculate metrics
        train_metrics = calculate_all_metrics(y_train, train_pred)
        test_metrics = calculate_all_metrics(y_test, test_pred)
        
        # Check for perfect scores (suspicious)
        if test_metrics['mae'] < 0.1 or test_metrics['r2'] > 0.999:
            print(f"   ❌ ❌ ❌ SUSPICIOUS: {model_name} has near-perfect scores!")
            print(f"      Test MAE: {test_metrics['mae']:.6f}, Test R²: {test_metrics['r2']:.6f}")
            print(f"      This indicates potential data leakage!")
            return None
        
        # Cross-validation
        try:
            tscv = TimeSeriesSplit(n_splits=min(5, len(X_train)//10))
            cv_scores = cross_val_score(model, X_train, y_train, cv=tscv,
                                      scoring='neg_mean_absolute_error')
            cv_mean_mae = -cv_scores.mean()
            cv_std_mae = cv_scores.std()
        except Exception as cv_error:
            print(f"   ⚠️  Cross-validation failed for {model_name}: {str(cv_error)}")
            cv_mean_mae = np.nan
            cv_std_mae = np.nan
        
        result = {
            'model': model,
            'train_mae': train_metrics['mae'],
            'test_mae': test_metrics['mae'],
            'train_rmse': train_metrics['rmse'],
            'test_rmse': test_metrics['rmse'],
            'train_r2': train_metrics['r2'],
            'test_r2': test_metrics['r2'],
            'train_mape': train_metrics['mape'],
            'test_mape': test_metrics['mape'],
            'cv_mean_mae': cv_mean_mae,
            'cv_std_mae': cv_std_mae,
            'train_pred': train_pred,
            'test_pred': test_pred
        }
        
        print(f"✅ {model_name} - Test MAE: {test_metrics['mae']:.2f}, R²: {test_metrics['r2']:.3f}")
        return result
        
    except Exception as e:
        print(f"❌ Error training {model_name}: {str(e)}")
        return None


def unbiased_model_training(X_train, X_test, y_train, y_test) -> dict:
    """
    Train multiple models with proper validation.
    
    Returns:
        Dictionary of model results
    """
    print("🚀 STARTING MODEL TRAINING PROCESS...")
    
    # Data validation
    X_train, X_test, y_train, y_test = check_data_leakage(X_train, X_test, y_train, y_test)
    X_train, X_test, y_train, y_test = ensure_numeric_data(X_train, X_test, y_train, y_test)
    
    # Debug info
    print(f"\n📊 Final data shapes:")
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("❌ ERROR: Data kosong setelah preprocessing!")
        return {}
    
    # Get models
    models = get_model_configurations()
    results = {}
    
    # Train each model
    for name, model in models.items():
        result = train_single_model(model, X_train, X_test, y_train, y_test, name)
        if result is not None:
            results[name] = result
    
    return results


if __name__ == "__main__":
    from data_loader import load_air_quality_data
    from preprocessing import preprocess_data
    from feature_engineering import balanced_feature_engineering
    
    # Load and prepare data
    df_raw = load_air_quality_data()
    df_clean = preprocess_data(df_raw, verbose=False)
    df_features = balanced_feature_engineering(df_clean, verbose=False)
    
    # Train-test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(df_features)
    
    # Train models
    results = unbiased_model_training(X_train, X_test, y_train, y_test)
    
    print(f"\n✅ Training completed: {len(results)} models trained")