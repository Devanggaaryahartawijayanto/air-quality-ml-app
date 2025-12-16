"""
Main training orchestration script.
Coordinates full pipeline from data loading to model saving.
"""

import os
import sys
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data_loader import load_air_quality_data
from ml.preprocessing import preprocess_data
from ml.feature_engineering import balanced_feature_engineering
from ml.model_training import prepare_train_test_split, unbiased_model_training
from ml.model_evaluation import display_model_matrices, select_best_model
from ml.bias_analysis import detect_and_mitigate_bias
from ml.visualization import (comprehensive_model_visualization, 
                              plot_train_test_split, 
                              plot_data_distribution,
                              plot_feature_engineering_results)
from pipeline.feature_pipeline import FeaturePipeline


def main():
    """Main training pipeline execution."""
    
    print("="*80)
    print("AIR QUALITY FORECASTING - COMPLETE TRAINING PIPELINE")
    print("="*80)
    
    # ==================== STEP 1: LOAD DATA ====================
    print("\n" + "="*60)
    print("📥 STEP 1: LOADING DATA")
    print("="*60)
    
    df_raw = load_air_quality_data()
    print(f"✅ Data loaded: {df_raw.shape}")
    
    # ==================== STEP 2: PREPROCESSING ====================
    print("\n" + "="*60)
    print("🔄 STEP 2: PREPROCESSING")
    print("="*60)
    
    df_clean = preprocess_data(df_raw, verbose=True)
    
    # ==================== STEP 3: DATA DISTRIBUTION ANALYSIS ====================
    print("\n" + "="*60)
    print("📊 STEP 3: DATA DISTRIBUTION ANALYSIS")
    print("="*60)
    
    plot_data_distribution(df_clean)
    
    # ==================== STEP 4: FEATURE ENGINEERING ====================
    print("\n" + "="*60)
    print("🔧 STEP 4: FEATURE ENGINEERING")
    print("="*60)
    
    station_data = balanced_feature_engineering(df_clean, verbose=True)
    plot_feature_engineering_results(station_data)
    
    # ==================== STEP 5: CREATE FEATURE PIPELINE ====================
    print("\n" + "="*60)
    print("⚙️ STEP 5: CREATING FEATURE PIPELINE")
    print("="*60)
    
    # Create and fit pipeline for inference
    pipeline = FeaturePipeline()
    pipeline.fit(df_raw, verbose=False)
    
    print(f"✅ Pipeline created with {len(pipeline.get_feature_columns())} features")
    
    # ==================== STEP 6: PREPARE TRAIN-TEST SPLIT ====================
    print("\n" + "="*60)
    print("✂️ STEP 6: TRAIN-TEST SPLIT")
    print("="*60)
    
    X_train, X_test, y_train, y_test = prepare_train_test_split(station_data, test_size=0.2)
    
    split_idx = int(len(station_data) * 0.8)
    plot_train_test_split(station_data, split_idx)
    
    # ==================== STEP 7: MODEL TRAINING ====================
    print("\n" + "="*60)
    print("🤖 STEP 7: MODEL TRAINING")
    print("="*60)
    
    model_results = unbiased_model_training(X_train, X_test, y_train, y_test)
    
    if not model_results:
        print("❌ Training failed - no models trained successfully")
        return
    
    print(f"\n✅ Successfully trained {len(model_results)} models")
    
    # ==================== STEP 8: MODEL EVALUATION ====================
    print("\n" + "="*60)
    print("📊 STEP 8: MODEL EVALUATION")
    print("="*60)
    
    comparison_df = display_model_matrices(model_results, y_train, y_test)
    
    # ==================== STEP 9: SELECT BEST MODEL ====================
    print("\n" + "="*60)
    print("🎯 STEP 9: SELECTING BEST MODEL")
    print("="*60)
    
    best_model_name, best_result = select_best_model(model_results)
    
    if best_result is None:
        print("❌ Could not select best model")
        return
    
    # ==================== STEP 10: BIAS ANALYSIS ====================
    print("\n" + "="*60)
    print("🔍 STEP 10: BIAS ANALYSIS")
    print("="*60)
    
    bias_analysis = detect_and_mitigate_bias(
        station_data, 
        y_test, 
        best_result['test_pred'], 
        X_test
    )
    
    # ==================== STEP 11: COMPREHENSIVE VISUALIZATION ====================
    print("\n" + "="*60)
    print("📈 STEP 11: COMPREHENSIVE VISUALIZATION")
    print("="*60)
    
    comprehensive_model_visualization(
        station_data,
        y_train,
        y_test,
        best_result['train_pred'],
        best_result['test_pred'],
        best_result,
        bias_analysis,
        best_model_name
    )
    
    # ==================== STEP 12: SAVE ARTIFACTS ====================
    print("\n" + "="*60)
    print("💾 STEP 12: SAVING ARTIFACTS")
    print("="*60)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save best model
    model_path = 'models/best_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_result['model'], f)
    print(f"✅ Model saved to {model_path}")
    
    # Save feature pipeline
    pipeline_path = 'models/feature_pipeline.pkl'
    pipeline.save(pipeline_path)
    
    # Save residual std for confidence intervals
    residual_std_path = 'models/residual_std.pkl'
    with open(residual_std_path, 'wb') as f:
        pickle.dump(bias_analysis['residual_std'], f)
    print(f"✅ Residual std saved to {residual_std_path}")
    
    # Save model metadata
    metadata = {
        'model_name': best_model_name,
        'test_mae': best_result['test_mae'],
        'test_rmse': best_result['test_rmse'],
        'test_r2': best_result['test_r2'],
        'test_mape': best_result['test_mape'],
        'overall_bias': bias_analysis['overall_bias'],
        'residual_std': bias_analysis['residual_std'],
        'n_features': len(pipeline.get_feature_columns())
    }
    
    metadata_path = 'models/metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✅ Metadata saved to {metadata_path}")
    
    # ==================== FINAL SUMMARY ====================
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE - FINAL SUMMARY")
    print("="*80)
    
    print(f"\n🎯 Best Model: {best_model_name}")
    print(f"\n📊 Test Performance:")
    print(f"   - MAE:  {best_result['test_mae']:.2f}")
    print(f"   - RMSE: {best_result['test_rmse']:.2f}")
    print(f"   - R²:   {best_result['test_r2']:.3f}")
    print(f"   - MAPE: {best_result['test_mape']:.1f}%")
    
    print(f"\n🔍 Bias Analysis:")
    print(f"   - Overall Bias: {bias_analysis['overall_bias']:.2f}")
    print(f"   - Residual STD: {bias_analysis['residual_std']:.2f}")
    
    print(f"\n💡 MODEL INTERPRETATION:")
    if best_result['test_r2'] > 0.8:
        print("   ✅ Excellent predictive power! Model explains most variance in PM2.5")
    elif best_result['test_r2'] > 0.6:
        print("   ✅ Good predictive power. Model captures main patterns in data")
    elif best_result['test_r2'] > 0.4:
        print("   ⚠️  Moderate predictive power. Consider additional feature engineering")
    else:
        print("   ❌ Poor predictive power. Significant improvements needed")
    
    if abs(bias_analysis['overall_bias']) > 5:
        print("   ⚠️  Significant bias detected. Consider model calibration")
    else:
        print("   ✅ Minimal bias detected. Model is well-calibrated")
    
    print(f"\n💾 Saved Artifacts:")
    print(f"   - models/best_model.pkl")
    print(f"   - models/feature_pipeline.pkl")
    print(f"   - models/residual_std.pkl")
    print(f"   - models/metadata.pkl")
    
    print(f"\n🎯 Ready for production inference!")
    print("="*80)


if __name__ == "__main__":
    main()