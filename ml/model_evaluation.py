"""
Model evaluation and comparison module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def display_model_matrices(model_results: dict, y_train, y_test) -> pd.DataFrame:
    """
    Display comprehensive evaluation matrices for all models.
    
    Args:
        model_results: Dictionary of model results from training
        y_train: Training target values
        y_test: Testing target values
    
    Returns:
        DataFrame with comparison metrics
    """
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE MODEL EVALUATION MATRICES")
    print("="*80)
    
    if not model_results:
        print("❌ model_results is empty! No models to display.")
        return None
    
    print(f"📈 Number of models trained: {len(model_results)}")
    print(f"📊 Models available: {list(model_results.keys())}")
    
    # Create comparison DataFrame
    comparison_data = []
    
    for model_name, results in model_results.items():
        comparison_data.append({
            'Model': model_name,
            'Train_MAE': results.get('train_mae', np.nan),
            'Test_MAE': results.get('test_mae', np.nan),
            'Train_RMSE': results.get('train_rmse', np.nan),
            'Test_RMSE': results.get('test_rmse', np.nan),
            'Train_R2': results.get('train_r2', np.nan),
            'Test_R2': results.get('test_r2', np.nan),
            'Train_MAPE': results.get('train_mape', np.nan),
            'Test_MAPE': results.get('test_mape', np.nan),
            'CV_MAE': results.get('cv_mean_mae', np.nan),
            'CV_Std': results.get('cv_std_mae', np.nan)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display comparison matrix
    print("\n📈 MODEL COMPARISON MATRIX:")
    print("-" * 100)
    print(comparison_df.round(4).to_string(index=False))
    print("-" * 100)
    
    # Highlight best models
    best_models = {}
    
    if 'Test_MAE' in comparison_df.columns and not comparison_df['Test_MAE'].isna().all():
        best_models['Best Test MAE'] = comparison_df.loc[comparison_df['Test_MAE'].idxmin(), 'Model']
    
    if 'Test_R2' in comparison_df.columns and not comparison_df['Test_R2'].isna().all():
        best_models['Best Test R2'] = comparison_df.loc[comparison_df['Test_R2'].idxmax(), 'Model']
    
    if 'Test_MAPE' in comparison_df.columns and not comparison_df['Test_MAPE'].isna().all():
        best_models['Best Test MAPE'] = comparison_df.loc[comparison_df['Test_MAPE'].idxmin(), 'Model']
    
    if 'CV_Std' in comparison_df.columns and not comparison_df['CV_Std'].isna().all():
        best_models['Most Consistent (Lowest CV Std)'] = comparison_df.loc[comparison_df['CV_Std'].idxmin(), 'Model']
    
    if best_models:
        print("\n🏆 BEST MODELS BY METRIC:")
        for metric, model in best_models.items():
            print(f"   {metric}: {model}")
    
    # Visualizations
    visualize_model_comparison(comparison_df, model_results)
    
    # Detailed performance matrix for each model
    display_detailed_performance(model_results)
    
    # Clean comparison for slides
    print("\n🎯 CLEAN MODEL COMPARISON FOR SLIDE:")
    print("=" * 60)
    slide_df = comparison_df[['Model', 'Test_MAE', 'Test_R2', 'Test_MAPE']].copy()
    slide_df.columns = ['Model', 'MAE', 'R²', 'MAPE%']
    slide_df['MAE'] = slide_df['MAE'].round(2)
    slide_df['R²'] = slide_df['R²'].round(3)
    slide_df['MAPE%'] = slide_df['MAPE%'].round(1)
    print(slide_df.to_string(index=False))
    
    best_model = comparison_df.loc[comparison_df['Test_MAE'].idxmin(), 'Model']
    best_r2 = comparison_df.loc[comparison_df['Test_R2'].idxmax(), 'Test_R2']
    print(f"\n🏆 BEST MODEL: {best_model} (R² = {best_r2:.3f})")
    
    return comparison_df


def visualize_model_comparison(comparison_df: pd.DataFrame, model_results: dict):
    """Create visualization comparing model performance."""
    try:
        plt.figure(figsize=(16, 12))
        
        # Metrics to plot
        metrics_to_plot = ['Test_MAE', 'Test_RMSE', 'Test_R2', 'Test_MAPE']
        metrics_names = ['MAE (Lower Better)', 'RMSE (Lower Better)', 'R² (Higher Better)', 'MAPE % (Lower Better)']
        
        valid_plots = 0
        for i, (metric, name) in enumerate(zip(metrics_to_plot, metrics_names)):
            if metric in comparison_df.columns and not comparison_df[metric].isna().all():
                valid_plots += 1
                plt.subplot(2, 3, valid_plots)
                
                valid_data = comparison_df[['Model', metric]].dropna()
                if len(valid_data) > 0:
                    sorted_df = valid_data.sort_values(metric, ascending=metric != 'Test_R2')
                    colors = ['lightgreen' if x == sorted_df[metric].iloc[0] else 'lightcoral' for x in sorted_df[metric]]
                    bars = plt.barh(sorted_df['Model'], sorted_df[metric], color=colors, alpha=0.7)
                    plt.title(f'Model Comparison - {name}')
                    plt.xlabel(metric.replace('_', ' '))
                    
                    for bar, value in zip(bars, sorted_df[metric]):
                        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                                f'{value:.3f}', ha='left', va='center', fontsize=9)
        
        # Correlation heatmap
        if valid_plots < 4:
            plt.subplot(2, 3, 5)
        
        numeric_cols_for_corr = comparison_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols_for_corr) > 1:
            correlation_matrix = comparison_df[numeric_cols_for_corr].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                        square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            plt.title('Metrics Correlation Matrix')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Error in visualization: {str(e)}")


def display_detailed_performance(model_results: dict):
    """Display detailed performance matrix for each model."""
    print("\n" + "="*80)
    print("🔍 DETAILED PERFORMANCE MATRIX FOR EACH MODEL")
    print("="*80)
    
    for model_name, results in model_results.items():
        print(f"\n📋 {model_name} - Detailed Performance Matrix:")
        print("-" * 60)
        
        metrics_data = []
        
        # Training data
        if 'train_mae' in results and not pd.isna(results['train_mae']):
            metrics_data.append({
                'Dataset': 'Training',
                'MAE': f"{results['train_mae']:.3f}",
                'RMSE': f"{results.get('train_rmse', 'N/A')}",
                'R²': f"{results.get('train_r2', 'N/A')}",
                'MAPE%': f"{results.get('train_mape', 'N/A')}"
            })
        
        # Testing data
        if 'test_mae' in results and not pd.isna(results['test_mae']):
            metrics_data.append({
                'Dataset': 'Testing',
                'MAE': f"{results['test_mae']:.3f}",
                'RMSE': f"{results.get('test_rmse', 'N/A')}",
                'R²': f"{results.get('test_r2', 'N/A')}",
                'MAPE%': f"{results.get('test_mape', 'N/A')}"
            })
        
        # Cross-validation data
        if 'cv_mean_mae' in results and not pd.isna(results['cv_mean_mae']):
            metrics_data.append({
                'Dataset': 'Cross-Validation',
                'MAE': f"{results['cv_mean_mae']:.3f} ± {results.get('cv_std_mae', 0):.3f}",
                'RMSE': 'N/A',
                'R²': 'N/A',
                'MAPE%': 'N/A'
            })
        
        if metrics_data:
            metrics_matrix = pd.DataFrame(metrics_data)
            print(metrics_matrix.to_string(index=False))
            
            # Interpretation
            print(f"\n💡 Performance Interpretation:")
            test_r2 = results.get('test_r2', 0)
            if test_r2 > 0.8:
                r2_interpretation = "Excellent (Model explains most variance)"
            elif test_r2 > 0.6:
                r2_interpretation = "Good (Model captures main patterns)"
            elif test_r2 > 0.4:
                r2_interpretation = "Moderate (Consider feature engineering)"
            else:
                r2_interpretation = "Poor (Significant improvements needed)"
            
            train_mae = results.get('train_mae', 0)
            test_mae = results.get('test_mae', 0)
            overfitting_gap = train_mae - test_mae
            
            if overfitting_gap > 2:
                overfitting_interpretation = "⚠️ Potential overfitting (Train much better than test)"
            elif overfitting_gap < -2:
                overfitting_interpretation = "⚠️ Potential underfitting (Test better than train)"
            else:
                overfitting_interpretation = "✅ Balanced performance"
            
            print(f"   - R² Score: {r2_interpretation}")
            print(f"   - Overfitting Analysis: {overfitting_interpretation}")
            
            cv_mae = results.get('cv_mean_mae', np.nan)
            cv_std = results.get('cv_std_mae', np.nan)
            if not pd.isna(cv_mae):
                print(f"   - Consistency: CV MAE = {cv_mae:.3f} ± {cv_std:.3f}")
        else:
            print("No performance data available for this model")


def select_best_model(results: dict) -> tuple:
    """
    Select best model based on multiple metrics.
    
    Args:
        results: Dictionary of model results
    
    Returns:
        (best_model_name, best_result)
    """
    if not results:
        print("❌ No models successfully trained")
        return None, None
    
    best_model_name = None
    best_score = float('inf')
    
    for name, result in results.items():
        # Use test MAE + CV std as selection criterion
        if not np.isnan(result['cv_mean_mae']):
            score = result['test_mae'] + result['cv_std_mae']
        else:
            score = result['test_mae']
        
        if score < best_score:
            best_score = score
            best_model_name = name
    
    print(f"\n🎯 Best Model Selected: {best_model_name}")
    print(f"📊 Selection Score: {best_score:.3f}")
    
    return best_model_name, results[best_model_name]


if __name__ == "__main__":
    # This would be called from training pipeline
    print("Model evaluation module - use via train.py")