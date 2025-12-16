"""
Feature pipeline wrapper combining preprocessing and feature engineering.
"""

import pandas as pd
import numpy as np
import pickle
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.preprocessing import preprocess_data
from ml.feature_engineering import balanced_feature_engineering


class FeaturePipeline:
    """
    Wrapper class for preprocessing and feature engineering pipeline.
    Designed for both training and inference.
    """
    
    def __init__(self, target_station: str = None):
        """
        Initialize feature pipeline.
        
        Args:
            target_station: Specific station name or None for auto-select
        """
        self.target_station = target_station
        self.fitted = False
        self.feature_columns = None
        self.target_column = 'pm_duakomalima'
    
    def fit(self, df: pd.DataFrame, verbose: bool = True) -> 'FeaturePipeline':
        """
        Fit the pipeline on training data.
        
        Args:
            df: Raw DataFrame
            verbose: Print processing steps
        
        Returns:
            Self for method chaining
        """
        # Preprocess
        df_clean = preprocess_data(df, verbose=verbose)
        
        # Feature engineering
        df_features = balanced_feature_engineering(df_clean, self.target_station, verbose=verbose)
        
        # Store feature columns (exclude non-numeric and target-related)
        non_numeric_cols = df_features.select_dtypes(include=['object', 'category']).columns.tolist()
        excluded_cols = ['tanggal', 'stasiun', 'kategori'] + non_numeric_cols
        
        if self.target_column in excluded_cols:
            excluded_cols.remove(self.target_column)
        
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [col for col in numeric_cols if col not in excluded_cols]
        
        self.fitted = True
        
        if verbose:
            print(f"\n✅ Pipeline fitted with {len(self.feature_columns)} features")
        
        return self
    
    def transform(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.
        
        Args:
            df: Raw DataFrame
            verbose: Print processing steps
        
        Returns:
            DataFrame with engineered features (numeric only)
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform. Call fit() first.")
        
        # Preprocess
        df_clean = preprocess_data(df, verbose=verbose)
        
        # Feature engineering
        df_features = balanced_feature_engineering(df_clean, self.target_station, verbose=verbose)
        
        return df_features
    
    def fit_transform(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Raw DataFrame
            verbose: Print processing steps
        
        Returns:
            DataFrame with engineered features
        """
        self.fit(df, verbose=verbose)
        return self.transform(df, verbose=verbose)
    
    def get_feature_columns(self) -> list:
        """Get list of feature column names."""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted first")
        return self.feature_columns.copy()
    
    def save(self, filepath: str):
        """Save pipeline to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Pipeline saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'FeaturePipeline':
        """Load pipeline from pickle file."""
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"✅ Pipeline loaded from {filepath}")
        return pipeline


if __name__ == "__main__":
    from ml.data_loader import load_air_quality_data
    
    # Test pipeline
    df_raw = load_air_quality_data()
    
    pipeline = FeaturePipeline()
    df_features = pipeline.fit_transform(df_raw)
    
    print(f"\n✅ Pipeline test completed")
    print(f"Features shape: {df_features.shape}")
    print(f"Feature columns: {pipeline.get_feature_columns()[:10]}...")  # Show first 10