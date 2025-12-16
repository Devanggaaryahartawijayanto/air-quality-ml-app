"""
Bias detection and analysis module.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import calculate_mae


def detect_and_mitigate_bias(data: pd.DataFrame, y_test, test_pred, X_test: pd.DataFrame) -> dict:
    """
    Detect and analyze bias in predictions.
    
    Args:
        data: Full dataset with dates
        y_test: Actual test values
        test_pred: Predicted test values
        X_test: Test features
    
    Returns:
        Dictionary with bias analysis results
    """
    print("\n" + "="*50)
    print("🔍 BIAS DETECTION ANALYSIS")
    print("="*50)
    
    bias_results = {}
    
    # 1. Overall bias
    overall_bias = np.mean(test_pred - y_test)
    bias_results['overall_bias'] = overall_bias
    print(f"📈 Overall Prediction Bias: {overall_bias:.2f} ({'over' if overall_bias > 0 else 'under'}estimation)")
    
    # 2. Temporal bias (by month)
    if 'month' in X_test.columns:
        temporal_bias = pd.DataFrame({
            'actual': y_test,
            'predicted': test_pred,
            'month': X_test['month'],
            'error': test_pred - y_test
        })
        
        monthly_bias = temporal_bias.groupby('month')['error'].mean()
        monthly_mae = temporal_bias.groupby('month').apply(
            lambda x: calculate_mae(x['actual'], x['predicted'])
        )
        
        bias_results['monthly_bias'] = monthly_bias
        bias_results['monthly_mae'] = monthly_mae
        
        print("\n📅 Monthly Bias Analysis:")
        for month in sorted(monthly_bias.index):
            print(f"  Month {month:2d}: Bias = {monthly_bias[month]:6.2f}, MAE = {monthly_mae[month]:6.2f}")
    
    # 3. Performance across PM2.5 value ranges
    value_ranges = [
        (0, 50, "Good"),
        (51, 100, "Moderate"),
        (101, 150, "Unhealthy"),
        (151, float('inf'), "Very Unhealthy")
    ]
    
    print("\n📊 Performance by Air Quality Level:")
    range_performance = {}
    
    for low, high, label in value_ranges:
        if high == float('inf'):
            mask = (y_test >= low)
        else:
            mask = (y_test >= low) & (y_test <= high)
        
        if mask.sum() > 0:
            range_mae = calculate_mae(y_test[mask], test_pred[mask])
            range_bias = np.mean(test_pred[mask] - y_test[mask])
            range_count = mask.sum()
            
            range_performance[label] = {
                'mae': range_mae,
                'bias': range_bias,
                'count': range_count
            }
            
            print(f"  {label:15s}: MAE = {range_mae:6.2f}, Bias = {range_bias:6.2f}, Samples = {range_count:3d}")
    
    bias_results['range_performance'] = range_performance
    
    # 4. Residual analysis
    residuals = test_pred - y_test
    bias_results['residuals'] = residuals
    bias_results['residual_std'] = np.std(residuals)
    
    print(f"\n📋 Residual Analysis:")
    print(f"  Standard Deviation: {np.std(residuals):.2f}")
    print(f"  Mean Absolute Residual: {np.mean(np.abs(residuals)):.2f}")
    print(f"  Residual Range: [{residuals.min():.2f}, {residuals.max():.2f}]")
    
    return bias_results


if __name__ == "__main__":
    print("Bias analysis module - use via train.py")