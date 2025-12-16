"""
Utility functions for model evaluation metrics.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


def calculate_mae(y_true, y_pred) -> float:
    """Calculate Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_r2(y_true, y_pred) -> float:
    """Calculate R-squared score."""
    return r2_score(y_true, y_pred)


def calculate_mape(y_true, y_pred) -> float:
    """Calculate Mean Absolute Percentage Error (as percentage)."""
    return mean_absolute_percentage_error(y_true, y_pred) * 100


def calculate_all_metrics(y_true, y_pred) -> dict:
    """
    Calculate all regression metrics.
    
    Returns:
        Dictionary with MAE, RMSE, R2, and MAPE
    """
    return {
        'mae': calculate_mae(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred)
    }


def sanity_check_predictions(y_true, y_pred, model_name: str) -> bool:
    """
    Perform sanity checks on model predictions.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of model for logging
    
    Returns:
        True if predictions look reasonable, False otherwise
    """
    print(f"   🔎 Sanity check for {model_name}...")
    
    # Check if predictions exactly match actual
    exact_matches = np.sum(np.isclose(y_pred, y_true))
    if exact_matches == len(y_true):
        print(f"   ❌ ❌ ❌ CRITICAL: {model_name} predictions are EXACTLY same as actual values!")
        return False
    
    # Check if model predicts only one value
    if np.std(y_pred) == 0:
        print(f"   ❌ ❌ ❌ CRITICAL: {model_name} predicts only one value!")
        return False
    
    # Check prediction range
    pred_range = np.max(y_pred) - np.min(y_pred)
    true_range = np.max(y_true) - np.min(y_true)
    
    if pred_range < true_range * 0.1:
        print(f"   ⚠️  WARNING: {model_name} prediction range very small")
        return True
    
    print(f"   ✅ {model_name} predictions look reasonable")
    return True


if __name__ == "__main__":
    # Test metrics
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([12, 19, 32, 38, 51])
    
    metrics = calculate_all_metrics(y_true, y_pred)
    print("Test metrics:", metrics)
    
    is_valid = sanity_check_predictions(y_true, y_pred, "TestModel")
    print(f"Predictions valid: {is_valid}")