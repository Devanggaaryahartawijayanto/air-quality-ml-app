"""
Visualization module for model evaluation and bias analysis.
"""

import matplotlib.pyplot as plt
import numpy as np


def comprehensive_model_visualization(data, y_train, y_test, train_pred, test_pred,
                                    eval_results: dict, bias_results: dict, model_name: str):
    """
    Create comprehensive visualization for model evaluation.
    
    Args:
        data: Full dataset with dates
        y_train: Training actual values
        y_test: Testing actual values
        train_pred: Training predictions
        test_pred: Testing predictions
        eval_results: Dictionary with evaluation metrics
        bias_results: Dictionary with bias analysis
        model_name: Name of the model
    """
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Training predictions
    plt.subplot(3, 3, 1)
    train_dates = data['tanggal'].iloc[:len(y_train)]
    test_dates = data['tanggal'].iloc[-len(y_test):]
    
    plt.plot(train_dates, y_train.values, label='Actual Train', alpha=0.7, color='blue', linewidth=1)
    plt.plot(train_dates, train_pred, label='Predicted Train', alpha=0.7, color='orange', linewidth=1)
    plt.title(f'Training: Actual vs Predicted\n{model_name}\nR² = {eval_results["train_r2"]:.3f}')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot 2: Test predictions
    plt.subplot(3, 3, 2)
    plt.plot(test_dates, y_test.values, label='Actual Test', alpha=0.7, color='blue', linewidth=1)
    plt.plot(test_dates, test_pred, label='Predicted Test', alpha=0.7, color='red', linewidth=1)
    plt.title(f'Test: Actual vs Predicted\nMAE = {eval_results["test_mae"]:.2f}, R² = {eval_results["test_r2"]:.3f}')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot 3: Scatter plot
    plt.subplot(3, 3, 3)
    plt.scatter(y_train, train_pred, alpha=0.5, label='Train', s=20)
    plt.scatter(y_test, test_pred, alpha=0.5, label='Test', s=20)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', alpha=0.8)
    plt.xlabel('Actual PM2.5')
    plt.ylabel('Predicted PM2.5')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    
    # Plot 4: Residuals
    plt.subplot(3, 3, 4)
    residuals_train = y_train - train_pred
    residuals_test = y_test - test_pred
    plt.scatter(train_pred, residuals_train, alpha=0.5, label='Train', s=20)
    plt.scatter(test_pred, residuals_test, alpha=0.5, label='Test', s=20)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    plt.xlabel('Predicted PM2.5')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot\nBias: {bias_results["overall_bias"]:.2f}')
    plt.legend()
    
    # Plot 5: Error distribution
    plt.subplot(3, 3, 5)
    errors = np.abs(np.concatenate([y_train - train_pred, y_test - test_pred]))
    plt.hist(errors, bins=30, alpha=0.7, color='purple', density=True)
    plt.axvline(x=np.mean(errors), color='r', linestyle='--', label=f'Mean Error: {np.mean(errors):.2f}')
    plt.xlabel('Absolute Error')
    plt.ylabel('Density')
    plt.title('Error Distribution')
    plt.legend()
    
    # Plot 6: Monthly bias analysis
    plt.subplot(3, 3, 6)
    if 'monthly_bias' in bias_results:
        monthly_bias = bias_results['monthly_bias']
        months = range(1, 13)
        bias_values = [monthly_bias.get(month, 0) for month in months]
        
        plt.bar(months, bias_values, alpha=0.7, color='coral')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Month')
        plt.ylabel('Bias')
        plt.title('Monthly Prediction Bias')
        plt.xticks(months)
    
    # Plot 7: Performance by air quality level
    plt.subplot(3, 3, 7)
    if 'range_performance' in bias_results:
        range_perf = bias_results['range_performance']
        labels = list(range_perf.keys())
        mae_values = [range_perf[label]['mae'] for label in labels]
        
        plt.bar(labels, mae_values, alpha=0.7, color='lightgreen')
        plt.xlabel('Air Quality Level')
        plt.ylabel('MAE')
        plt.title('MAE by Air Quality Level')
        plt.xticks(rotation=45)
    
    # Plot 8: Model comparison
    plt.subplot(3, 3, 8)
    metrics = ['MAE', 'RMSE', 'R²']
    train_values = [eval_results['train_mae'], eval_results['train_rmse'], eval_results['train_r2']]
    test_values = [eval_results['test_mae'], eval_results['test_rmse'], eval_results['test_r2']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, train_values[:3], width, label='Train', alpha=0.7)
    plt.bar(x + width/2, test_values[:3], width, label='Test', alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    
    # Plot 9: Prediction intervals
    plt.subplot(3, 3, 9)
    sorted_indices = np.argsort(y_test.values)
    y_test_sorted = y_test.values[sorted_indices]
    test_pred_sorted = test_pred[sorted_indices]
    
    plt.plot(y_test_sorted, label='Actual', alpha=0.8)
    plt.plot(test_pred_sorted, label='Predicted', alpha=0.8)
    plt.fill_between(range(len(y_test_sorted)),
                    test_pred_sorted - bias_results['residual_std'],
                    test_pred_sorted + bias_results['residual_std'],
                    alpha=0.3, label='±1 STD')
    plt.xlabel('Sorted Test Samples')
    plt.ylabel('PM2.5')
    plt.title('Prediction with Uncertainty Intervals')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_train_test_split(station_data, split_idx: int):
    """Visualize train-test split."""
    plt.figure(figsize=(12, 6))
    
    # Timeline plot
    plt.subplot(1, 2, 1)
    dates = station_data['tanggal']
    pm25 = station_data['pm_duakomalima']
    
    plt.plot(dates[:split_idx], pm25[:split_idx],
             label='Training Data (80%)', color='blue', alpha=0.7, linewidth=1)
    plt.plot(dates[split_idx:], pm25[split_idx:],
             label='Testing Data (20%)', color='red', alpha=0.7, linewidth=1)
    
    split_date = dates.iloc[split_idx]
    plt.axvline(x=split_date, color='black', linestyle='--', alpha=0.8,
                label=f'Split Point: {split_date.strftime("%Y-%m-%d")}')
    
    plt.title('Chronological Train-Test Split')
    plt.ylabel('PM2.5 Concentration')
    plt.xlabel('Date')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Distribution comparison
    plt.subplot(1, 2, 2)
    y_train = pm25[:split_idx]
    y_test = pm25[split_idx:]
    
    train_test_data = [y_train, y_test]
    labels = [f'Training\n(n={len(y_train)})', f'Testing\n(n={len(y_test)})']
    
    box_plot = plt.boxplot(train_test_data, labels=labels, patch_artist=True)
    
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('PM2.5 Distribution: Training vs Testing')
    plt.ylabel('PM2.5 Concentration')
    
    train_mean = y_train.mean()
    test_mean = y_test.mean()
    plt.text(1, train_mean + 5, f'Mean: {train_mean:.1f}', ha='center', va='bottom', fontweight='bold')
    plt.text(2, test_mean + 5, f'Mean: {test_mean:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_data_distribution(df):
    """Plot data distribution analysis."""
    df['year'] = df['tanggal'].dt.year
    
    plt.figure(figsize=(15, 5))
    
    # Yearly distribution
    plt.subplot(1, 3, 1)
    yearly_counts = df['year'].value_counts().sort_index()
    yearly_counts.plot(kind='bar')
    plt.title('Data Distribution by Year')
    plt.xticks(rotation=45)
    
    # Monthly distribution
    plt.subplot(1, 3, 2)
    monthly_counts = df['tanggal'].dt.month.value_counts().sort_index()
    monthly_counts.plot(kind='bar')
    plt.title('Data Distribution by Month')
    
    # Station distribution
    plt.subplot(1, 3, 3)
    station_counts = df['stasiun'].value_counts()
    station_counts.plot(kind='bar')
    plt.title('Data Distribution by Station')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_feature_engineering_results(station_data):
    """Visualize data after feature engineering."""
    plt.figure(figsize=(10, 6))
    
    # Timeline plot
    plt.subplot(1, 2, 1)
    station_data.set_index('tanggal')['pm_duakomalima'].plot(alpha=0.7, color='blue')
    plt.title('PM2.5 Timeline After Preprocessing')
    plt.ylabel('PM2.5 Concentration')
    plt.xticks(rotation=45)
    
    # Distribution plot
    plt.subplot(1, 2, 2)
    station_data['pm_duakomalima'].hist(bins=30, alpha=0.7, color='green')
    plt.title('PM2.5 Distribution After Preprocessing')
    plt.xlabel('PM2.5 Concentration')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Visualization module - use via train.py")