import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION by deepseek
# ==========================================
CSV_PATH = r"C:\ml ppa\final_maximal_dataset.csv"
OUTPUT_DIR = r"C:\ml ppa\results"
SAVE_MODELS = True

# ==========================================
# MAIN FUNCTION - FIXED VERSION
# ==========================================

def train_ppa_xgboost():
    print("=" * 80)
    print("XGBOOST PPA PREDICTION MODEL - FIXED VERSION")
    print("=" * 80)
    
    # 1. LOAD AND INSPECT DATA
    print("\n1. Loading dataset...")
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"✓ Loaded {len(df):,} designs with {len(df.columns)} features")
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return
    
    # 2. DEFINE FEATURE SETS
    print("\n2. Defining feature sets...")
    
    # RTL features
    rtl_features = [
        'num_lines', 'max_bitwidth', 'min_bitwidth', 'avg_bitwidth',
        'total_bits', 'num_bitwidths', 'num_32bit', 'num_16bit', 'num_8bit', 'num_1bit',
        'num_add', 'num_sub', 'num_mul', 'num_div',
        'num_logic_and', 'num_logic_or', 'num_logic_xor',
        'num_shifts', 'num_comparisons',
        'num_always', 'num_assign', 'num_if', 'num_case', 'num_for',
        'num_ternary', 'num_branches',
        'num_wire', 'num_reg', 'num_input', 'num_output', 'num_modules'
    ]
    
    # Netlist features
    netlist_features = [
        'netlist_num_gates', 'netlist_num_nets', 'netlist_num_instances',
        'netlist_inv_count', 'netlist_and_count', 'netlist_or_count',
        'netlist_nand_count', 'netlist_nor_count', 'netlist_xor_count',
        'netlist_mux_count', 'netlist_buf_count', 'netlist_maj3_count'
    ]
    
    # Targets
    targets = ['total_cell_area', 'Power', 'critical_path_length']
    target_names = ['Area', 'Power', 'Delay']
    
    # 3. DATA PREPROCESSING
    print("\n3. Preprocessing data...")
    
    # Check what features we actually have
    available_rtl = [f for f in rtl_features if f in df.columns]
    available_netlist = [f for f in netlist_features if f in df.columns]
    
    print(f"✓ Available RTL features: {len(available_rtl)}/{len(rtl_features)}")
    print(f"✓ Available Netlist features: {len(available_netlist)}/{len(netlist_features)}")
    
    # Clean data
    df_clean = df.dropna(subset=targets).copy()
    print(f"✓ Clean designs: {len(df_clean)}")
    
    # 4. CREATE FEATURE SETS
    print("\n4. Creating feature sets...")
    
    # Fill missing values with 0
    X_rtl = df_clean[available_rtl].fillna(0)
    X_hybrid = df_clean[available_rtl + available_netlist].fillna(0)
    
    # Targets
    y_area = df_clean['total_cell_area'].values
    y_power = df_clean['Power'].values
    y_delay = df_clean['critical_path_length'].values
    
    # 5. HANDLE SKEWED DISTRIBUTIONS
    print("\n5. Transforming targets...")
    
    # Log transform to handle skewness
    y_area_log = np.log1p(y_area)  # log(1 + x)
    y_power_log = np.log1p(y_power)
    y_delay_log = np.log1p(y_delay)
    
    # 6. SPLIT DATA
    print("\n6. Splitting data...")
    
    # Common test indices
    test_size = 0.2
    indices = np.arange(len(X_rtl))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
    
    # RTL features splits
    X_rtl_train = X_rtl.iloc[train_idx]
    X_rtl_test = X_rtl.iloc[test_idx]
    
    # Hybrid features splits
    X_hybrid_train = X_hybrid.iloc[train_idx]
    X_hybrid_test = X_hybrid.iloc[test_idx]
    
    # Target splits
    y_area_train, y_area_test = y_area_log[train_idx], y_area_log[test_idx]
    y_power_train, y_power_test = y_power_log[train_idx], y_power_log[test_idx]
    y_delay_train, y_delay_test = y_delay_log[train_idx], y_delay_log[test_idx]
    
    # Original targets (for evaluation)
    y_area_orig_test = y_area[test_idx]
    y_power_orig_test = y_power[test_idx]
    y_delay_orig_test = y_delay[test_idx]
    
    # 7. XGBOOST PARAMETERS - SIMPLIFIED
    print("\n7. Setting up XGBoost...")
    
    xgb_params = {
        'n_estimators': 500,          # Reduced for faster training
        'learning_rate': 0.05,
        'max_depth': 6,               # Reduced to prevent overfitting
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    # 8. TRAIN RTL-ONLY MODELS
    print("\n" + "=" * 80)
    print("TRAINING RTL-ONLY MODELS")
    print("=" * 80)
    
    models_rtl = {}
    
    print("\nTraining Area model (RTL)...")
    model_area_rtl = xgb.XGBRegressor(**xgb_params)
    model_area_rtl.fit(X_rtl_train, y_area_train)
    models_rtl['area'] = model_area_rtl
    
    print("Training Power model (RTL)...")
    model_power_rtl = xgb.XGBRegressor(**xgb_params)
    model_power_rtl.fit(X_rtl_train, y_power_train)
    models_rtl['power'] = model_power_rtl
    
    print("Training Delay model (RTL)...")
    model_delay_rtl = xgb.XGBRegressor(**xgb_params)
    model_delay_rtl.fit(X_rtl_train, y_delay_train)
    models_rtl['delay'] = model_delay_rtl
    
    # 9. TRAIN HYBRID MODELS (if netlist features available)
    models_hybrid = {}
    if len(available_netlist) > 0:
        print("\n" + "=" * 80)
        print("TRAINING HYBRID MODELS (RTL + Netlist)")
        print("=" * 80)
        
        print("\nTraining Area model (Hybrid)...")
        model_area_hybrid = xgb.XGBRegressor(**xgb_params)
        model_area_hybrid.fit(X_hybrid_train, y_area_train)
        models_hybrid['area'] = model_area_hybrid
        
        print("Training Power model (Hybrid)...")
        model_power_hybrid = xgb.XGBRegressor(**xgb_params)
        model_power_hybrid.fit(X_hybrid_train, y_power_train)
        models_hybrid['power'] = model_power_hybrid
        
        print("Training Delay model (Hybrid)...")
        model_delay_hybrid = xgb.XGBRegressor(**xgb_params)
        model_delay_hybrid.fit(X_hybrid_train, y_delay_train)
        models_hybrid['delay'] = model_delay_hybrid
    
    # 10. EVALUATION FUNCTION
    def evaluate_model(model, X_test, y_test_log, y_test_orig, name, feature_type="", save_plots=True):
        """Evaluate model performance"""
        
        # Predict
        y_pred_log = model.predict(X_test)
        
        # Convert from log space
        y_pred = np.expm1(y_pred_log)
        y_true = y_test_orig
        
        # Metrics
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        
        # Print results
        print(f"\n{name} ({feature_type}):")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE:      {mae:.4f}")
        print(f"  RMSE:     {rmse:.4f}")
        print(f"  MAPE:     {mape:.1f}%")
        
        # Plot if requested
        if save_plots:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Scatter plot
            axes[0].scatter(y_true, y_pred, alpha=0.5, s=10, color='blue')
            axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                        'r--', lw=2)
            axes[0].set_xlabel('Actual')
            axes[0].set_ylabel('Predicted')
            axes[0].set_title(f'{name} - {feature_type}\nR² = {r2:.3f}')
            axes[0].grid(True, alpha=0.3)
            
            # Residual plot
            residuals = y_pred - y_true
            axes[1].hist(residuals, bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes[1].set_xlabel('Prediction Error')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'Error Distribution\nMAE = {mae:.2f}, MAPE = {mape:.1f}%')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/{feature_type}_{name}_results.png", dpi=150)
            plt.close()
        
        return {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'feature_type': feature_type,
            'target': name
        }
    
    # 11. EVALUATE ALL MODELS
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    results = []
    
    # Evaluate RTL models
    print("\n--- RTL-ONLY MODELS ---")
    
    for target_name, model in models_rtl.items():
        y_test_log = {
            'area': y_area_test,
            'power': y_power_test,
            'delay': y_delay_test
        }[target_name]
        
        y_test_orig = {
            'area': y_area_orig_test,
            'power': y_power_orig_test,
            'delay': y_delay_orig_test
        }[target_name]
        
        result = evaluate_model(
            model, X_rtl_test, y_test_log, y_test_orig,
            target_name.capitalize(), "RTL"
        )
        results.append(result)
    
    # Evaluate Hybrid models
    if models_hybrid:
        print("\n--- HYBRID MODELS ---")
        
        for target_name, model in models_hybrid.items():
            y_test_log = {
                'area': y_area_test,
                'power': y_power_test,
                'delay': y_delay_test
            }[target_name]
            
            y_test_orig = {
                'area': y_area_orig_test,
                'power': y_power_orig_test,
                'delay': y_delay_orig_test
            }[target_name]
            
            result = evaluate_model(
                model, X_hybrid_test, y_test_log, y_test_orig,
                target_name.capitalize(), "Hybrid"
            )
            results.append(result)
    
    # 12. FEATURE IMPORTANCE
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    
    def plot_feature_importance(model, features, title, filename):
        """Plot feature importance"""
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        
        # Top 15 features
        top_n = min(15, len(features))
        plt.barh(range(top_n), importance[indices[:top_n]][::-1], color='skyblue')
        plt.yticks(range(top_n), [features[i] for i in indices[:top_n]][::-1])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Features - {title}')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{filename}_importance.png", dpi=150)
        plt.close()
        
        # Print top 10
        print(f"\nTop 10 features for {title}:")
        for i in range(min(10, len(features))):
            print(f"  {i+1:2d}. {features[indices[i]]:25s} - {importance[indices[i]]:.4f}")
    
    # Plot for Area model
    plot_feature_importance(
        models_rtl['area'], 
        available_rtl,
        "Area Prediction",
        "rtl_area"
    )
    
    # 13. SAVE MODELS
    if SAVE_MODELS:
        print("\n" + "=" * 80)
        print("SAVING MODELS")
        print("=" * 80)
        
        import joblib
        import os
        
        os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)
        
        # Save RTL models
        for target, model in models_rtl.items():
            joblib.dump(model, f"{OUTPUT_DIR}/models/xgb_rtl_{target}.pkl")
            print(f"✓ Saved RTL {target} model")
        
        # Save Hybrid models
        if models_hybrid:
            for target, model in models_hybrid.items():
                joblib.dump(model, f"{OUTPUT_DIR}/models/xgb_hybrid_{target}.pkl")
                print(f"✓ Saved Hybrid {target} model")
        
        # Save feature lists
        with open(f"{OUTPUT_DIR}/models/features.txt", 'w') as f:
            f.write("RTL Features:\n")
            f.write("\n".join(available_rtl))
            
            if available_netlist:
                f.write("\n\nNetlist Features:\n")
                f.write("\n".join(available_netlist))
        
        print("✓ Saved feature lists")
    
    # 14. COMPARE MODEL PERFORMANCE
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    print("\n", results_df[['target', 'feature_type', 'r2', 'mae', 'mape']].to_string())
    
    # Plot comparison
    if models_hybrid:
        plt.figure(figsize=(10, 6))
        
        rtl_results = results_df[results_df['feature_type'] == 'RTL']
        hybrid_results = results_df[results_df['feature_type'] == 'Hybrid']
        
        x = np.arange(3)
        width = 0.35
        
        plt.bar(x - width/2, rtl_results['r2'].values, width, label='RTL-only', alpha=0.8)
        plt.bar(x + width/2, hybrid_results['r2'].values, width, label='Hybrid', alpha=0.8)
        
        plt.xlabel('Target')
        plt.ylabel('R² Score')
        plt.title('Model Comparison')
        plt.xticks(x, ['Area', 'Power', 'Delay'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150)
        plt.close()
    
    # 15. PREDICTION FUNCTION
    print("\n" + "=" * 80)
    print("PREDICTION FUNCTION")
    print("=" * 80)
    
    def predict_ppa(features_dict, model_type='rtl'):
        """
        Predict PPA for new design
        
        Args:
            features_dict: Dictionary of feature values
            model_type: 'rtl' or 'hybrid'
        """
        if model_type == 'rtl':
            features_list = available_rtl
            model_dict = models_rtl
        elif model_type == 'hybrid' and models_hybrid:
            features_list = available_rtl + available_netlist
            model_dict = models_hybrid
        else:
            print(f"Model type '{model_type}' not available, using RTL")
            features_list = available_rtl
            model_dict = models_rtl
        
        # Create feature vector
        feature_vec = []
        for feat in features_list:
            feature_vec.append(features_dict.get(feat, 0))
        
        # Predict
        predictions = {}
        for target in ['area', 'power', 'delay']:
            if target in model_dict:
                y_pred_log = model_dict[target].predict([feature_vec])[0]
                predictions[target] = float(np.expm1(y_pred_log))
        
        return predictions
    
    # Example
    print("\nExample prediction (using mean values from dataset):")
    example_features = {col: X_rtl[col].mean() for col in available_rtl[:5]}
    example_features.update({col: 0 for col in available_rtl[5:]})
    
    for model_type in (['rtl'] + (['hybrid'] if models_hybrid else [])):
        preds = predict_ppa(example_features, model_type=model_type)
        print(f"\n{model_type.upper()} model predictions:")
        for target, value in preds.items():
            print(f"  {target}: {value:.2f}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    
    return models_rtl, models_hybrid, results_df

# ==========================================
# RUN THE MODEL
# ==========================================

if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run training
    try:
        models_rtl, models_hybrid, results = train_ppa_xgboost()
        print(f"\nResults saved to: {OUTPUT_DIR}")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()