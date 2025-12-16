"""
Inference module for production predictions.
"""

import pickle
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AirQualityPredictor:
    """
    Production predictor for air quality forecasting.
    Loads trained model and pipeline for inference.
    """
    
    def __init__(self, model_path: str, pipeline_path: str):
        """
        Initialize predictor with saved model and pipeline.
        
        Args:
            model_path: Path to saved model pickle file
            pipeline_path: Path to saved pipeline pickle file
        """
        self.model = None
        self.pipeline = None
        self.residual_std = None
        self.model_path = model_path
        self.pipeline_path = pipeline_path
        
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load model and pipeline from disk."""
        print(f"📦 Loading model from {self.model_path}...")
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"📦 Loading pipeline from {self.pipeline_path}...")
        with open(self.pipeline_path, 'rb') as f:
            self.pipeline = pickle.load(f)
        
        # Try to load residual std if available
        residual_std_path = os.path.join(os.path.dirname(self.model_path), 'residual_std.pkl')
        if os.path.exists(residual_std_path):
            with open(residual_std_path, 'rb') as f:
                self.residual_std = pickle.load(f)
            print(f"📦 Loaded residual std: {self.residual_std:.2f}")
        
        print("✅ Artifacts loaded successfully")
    
    def predict(self, df_raw: pd.DataFrame) -> dict:
        """
        Make prediction on raw data.
        
        Args:
            df_raw: Raw DataFrame (same format as training data)
        
        Returns:
            Dictionary with predictions and confidence intervals
        """
        # Transform data using pipeline
        df_features = self.pipeline.transform(df_raw, verbose=False)
        
        # Get feature columns
        feature_cols = self.pipeline.get_feature_columns()
        X = df_features[feature_cols]
        
        # Make predictions
        predictions = self.model.predict(X)
        
        result = {
            'predictions': predictions,
            'dates': df_features['tanggal'].values if 'tanggal' in df_features.columns else None
        }
        
        # Add confidence intervals if residual_std available
        if self.residual_std is not None:
            result['lower_bound'] = predictions - self.residual_std
            result['upper_bound'] = predictions + self.residual_std
        
        return result
    
    def predict_single(self, date_str: str, station: str = None) -> dict:
        """
        Predict for a single date (requires historical data context).
        
        Args:
            date_str: Date string in format 'YYYY-MM-DD'
            station: Station name (optional)
        
        Returns:
            Dictionary with prediction and metadata
        """
        # This would require loading recent historical data
        # For production, implement data fetching logic here
        raise NotImplementedError("Single date prediction requires historical context - implement data fetching")
    
    def batch_predict(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Make batch predictions and return as DataFrame.
        
        Args:
            df_raw: Raw DataFrame with multiple records
        
        Returns:
            DataFrame with predictions and dates
        """
        result = self.predict(df_raw)
        
        output_df = pd.DataFrame({
            'prediction': result['predictions']
        })
        
        if result['dates'] is not None:
            output_df['date'] = result['dates']
        
        if 'lower_bound' in result:
            output_df['lower_bound'] = result['lower_bound']
            output_df['upper_bound'] = result['upper_bound']
        
        return output_df


def load_predictor(model_dir: str = 'models') -> AirQualityPredictor:
    """
    Convenience function to load predictor from standard paths.
    
    Args:
        model_dir: Directory containing saved model artifacts
    
    Returns:
        Initialized AirQualityPredictor
    """
    model_path = os.path.join(model_dir, 'best_model.pkl')
    pipeline_path = os.path.join(model_dir, 'feature_pipeline.pkl')
    
    return AirQualityPredictor(model_path, pipeline_path)


if __name__ == "__main__":
    # Example usage
    print("Predictor module - use after training")
    print("\nExample usage:")
    print("predictor = load_predictor('models')")
    print("predictions = predictor.predict(df_raw)")