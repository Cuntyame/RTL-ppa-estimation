"""#!/usr/bin/env python3

Improved Multi-Model Ensemble for PPA Prediction
Fixes: MLP stability, Power/Delay modeling, better preprocessing
"""

import os
import joblib
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
CSV_PATH = r"C:\ml ppa\final_maximal_dataset.csv"
MODEL_DIR = r"C:\ml ppa\models_ensemble_v2"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
USE_NETLIST = True
REMOVE_OUTLIERS = True
OUTLIER_STD = 3.5  # Slightly more aggressive

# Target-specific transformations
TARGET_TRANSFORMS = {
    'area': 'log',      # Log works well for area
    'power': 'yeo-johnson',  # Better for power (handles zeros, negatives)
    'delay': 'log'      # Log for delay
}

# -------------- LOAD DATA ---------------
print("="*70)
print("IMPROVED ENSEMBLE PPA PREDICTION v2")
print("="*70)
print(f"\nAvailable models:")
print(f"  XGBoost: {HAS_XGB}")
print(f"  LightGBM: {HAS_LGBM}")
print(f"  CatBoost: {HAS_CATBOOST}")
print(f"  Random Forest: True")
print(f"  Gradient Boosting: True (sklearn fallback)")

print(f"\nLoading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Shape: {df.shape}")

# -------------- FEATURE SELECTION --------------
id_cols = ['Design_Name', 'RTL_Code']
area_col = 'total_cell_area' if 'total_cell_area' in df.columns else 'comb_area'
power_col = 'Power'
delay_col = 'critical_path_length'

rtl_features = [
    'num_lines', 'max_bitwidth', 'min_bitwidth', 'avg_bitwidth',
    'total_bits', 'num_bitwidths', 'num_32bit', 'num_16bit', 'num_8bit', 'num_1bit',
    'num_add', 'num_sub', 'num_mul', 'num_div',
    'num_logic_and', 'num_logic_or', 'num_logic_xor', 'num_shifts', 'num_comparisons',
    'num_always', 'num_assign', 'num_if', 'num_case', 'num_for', 'num_ternary', 'num_branches',
    'num_wire', 'num_reg', 'num_input', 'num_output', 'num_modules',
]

netlist_features = [
    'netlist_num_gates', 'netlist_num_nets', 'netlist_num_instances',
    'netlist_inv_count', 'netlist_and_count', 'netlist_or_count',
    'netlist_nand_count', 'netlist_nor_count', 'netlist_xor_count',
    'netlist_mux_count', 'netlist_buf_count', 'netlist_maj3_count'
]

available_rtl = [f for f in rtl_features if f in df.columns]
available_netlist = [f for f in netlist_features if f in df.columns]

if USE_NETLIST and available_netlist:
    base_features = available_rtl + available_netlist
    print(f"\nUsing HYBRID: {len(available_rtl)} RTL + {len(available_netlist)} netlist")
else:
    base_features = available_rtl
    print(f"Using PURE RTL: {len(available_rtl)} features")

# -------------- ENHANCED FEATURE ENGINEERING --------------
def engineer_features(df, base_features):
    """Enhanced feature engineering with interaction terms"""
    df_feat = df[base_features].copy()
    eps = 1e-8
    
    # Basic ratios
    if all(c in df.columns for c in ['num_mul', 'num_add']):
        df_feat['mul_add_ratio'] = df['num_mul'] / (df['num_add'] + eps)
    
    # Aggregates
    arith = [c for c in ['num_add', 'num_sub', 'num_mul', 'num_div'] if c in df.columns]
    if arith:
        df_feat['total_arithmetic'] = df[arith].sum(axis=1)
        if 'num_lines' in df.columns:
            df_feat['arith_density'] = df_feat['total_arithmetic'] / (df['num_lines'] + eps)
    
    logic = [c for c in ['num_logic_and', 'num_logic_or', 'num_logic_xor', 'num_shifts'] if c in df.columns]
    if logic:
        df_feat['total_logic'] = df[logic].sum(axis=1)
    
    control = [c for c in ['num_if', 'num_case', 'num_for', 'num_branches'] if c in df.columns]
    if control:
        df_feat['total_control'] = df[control].sum(axis=1)
    
    # Densities
    if all(c in df.columns for c in ['num_reg', 'num_lines']):
        df_feat['reg_density'] = df['num_reg'] / (df['num_lines'] + eps)
    
    # Bitwidth features
    if all(c in df.columns for c in ['max_bitwidth', 'min_bitwidth', 'avg_bitwidth']):
        df_feat['bitwidth_range'] = df['max_bitwidth'] - df['min_bitwidth']
        df_feat['bitwidth_std'] = (df['max_bitwidth'] - df['avg_bitwidth']) / (df['avg_bitwidth'] + eps)
    
    # Netlist features
    if 'netlist_num_gates' in df.columns:
        if 'num_lines' in df.columns:
            df_feat['gates_per_line'] = df['netlist_num_gates'] / (df['num_lines'] + eps)
        
        # Gate ratios
        for gate in ['inv', 'buf', 'mux', 'and', 'or']:
            col = f'netlist_{gate}_count'
            if col in df.columns:
                df_feat[f'{gate}_ratio'] = df[col] / (df['netlist_num_gates'] + eps)
        
        # Complexity indicator
        df_feat['gates_log'] = np.log1p(df['netlist_num_gates'])
        
        # Gate diversity (how many gate types used)
        gate_cols = [c for c in df.columns if c.startswith('netlist_') and c.endswith('_count')]
        if gate_cols:
            df_feat['gate_diversity'] = (df[gate_cols] > 0).sum(axis=1)
    
    # Power-specific features (switching activity proxies)
    if all(c in df.columns for c in ['num_reg', 'num_logic_xor']):
        df_feat['switching_proxy'] = df['num_reg'] + df['num_logic_xor'] * 2  # XOR has high switching
    
    if 'netlist_mux_count' in df.columns and 'netlist_num_gates' in df.columns:
        df_feat['mux_density'] = df['netlist_mux_count'] / (df['netlist_num_gates'] + eps)
    
    # Delay-specific features (critical path proxies)
    if all(c in df.columns for c in ['num_mul', 'num_div']):
        df_feat['slow_ops'] = df['num_mul'] * 2 + df['num_div'] * 3  # Weighted by typical delay
    
    if 'netlist_maj3_count' in df.columns:
        df_feat['maj3_normalized'] = np.log1p(df['netlist_maj3_count'])
    
    # Interaction terms for critical features
    if 'total_bits' in df_feat.columns:
        df_feat['bits_sqrt'] = np.sqrt(df_feat['total_bits'])
        df_feat['bits_log'] = np.log1p(df_feat['total_bits'])
    
    if 'total_arithmetic' in df_feat.columns and 'max_bitwidth' in df.columns:
        df_feat['arith_bitwidth_product'] = df_feat['total_arithmetic'] * df['max_bitwidth']
    
    return df_feat

print("\nEngineering features...")
df_features = engineer_features(df, base_features)
df_model = pd.concat([df_features, df[[area_col, power_col, delay_col]]], axis=1)

# -------------- DATA CLEANING --------------
print("\nCleaning data...")
before = len(df_model)
df_model = df_model.dropna()
print(f"After dropna: {len(df_model)}/{before}")

# Remove non-positive targets
before = len(df_model)
df_model = df_model[(df_model[area_col] > 0) & (df_model[power_col] > 0) & (df_model[delay_col] > 0)]
print(f"After removing ≤0: {len(df_model)}/{before}")

# Remove outliers per target
if REMOVE_OUTLIERS:
    before = len(df_model)
    for target in [area_col, power_col, delay_col]:
        q1, q3 = df_model[target].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - OUTLIER_STD * iqr
        upper = q3 + OUTLIER_STD * iqr
        df_model = df_model[(df_model[target] >= lower) & (df_model[target] <= upper)]
    print(f"After outliers (IQR method): {len(df_model)}/{before}")

# -------------- PREPARE DATA --------------
final_features = df_features.columns.tolist()
X = df_model[final_features].values
y_area_orig = df_model[area_col].values
y_power_orig = df_model[power_col].values
y_delay_orig = df_model[delay_col].values

print(f"\nFinal: {X.shape[0]} samples, {X.shape[1]} features")

# -------------- TARGET TRANSFORMATIONS --------------
print("\nApplying target-specific transformations...")

# Area: log transform
y_area = np.log1p(y_area_orig)
print(f"  Area: log transform")

# Power: Yeo-Johnson (better for skewed distributions with outliers)
power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
y_power = power_transformer.fit_transform(y_power_orig.reshape(-1, 1)).ravel()
print(f"  Power: Yeo-Johnson transform")

# Delay: log transform
y_delay = np.log1p(y_delay_orig)
print(f"  Delay: log transform")

# Save transformers
joblib.dump(power_transformer, os.path.join(MODEL_DIR, "power_transformer.joblib"))

# -------------- SPLIT DATA --------------
X_temp, X_test, idx_temp, idx_test = train_test_split(
    X, np.arange(len(X)), test_size=TEST_SIZE, random_state=RANDOM_STATE
)

val_size_adj = VAL_SIZE / (1 - TEST_SIZE)
X_train, X_val, idx_train, idx_val = train_test_split(
    X_temp, idx_temp, test_size=val_size_adj, random_state=RANDOM_STATE
)

# Split all targets using same indices
y_area_train, y_area_val, y_area_test = y_area[idx_train], y_area[idx_val], y_area[idx_test]
y_power_train, y_power_val, y_power_test = y_power[idx_train], y_power[idx_val], y_power[idx_test]
y_delay_train, y_delay_val, y_delay_test = y_delay[idx_train], y_delay[idx_val], y_delay[idx_test]

# Original values for evaluation
y_area_test_orig = y_area_orig[idx_test]
y_power_test_orig = y_power_orig[idx_test]
y_delay_test_orig = y_delay_orig[idx_test]

print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# -------------- SCALING --------------
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

# -------------- MODEL DEFINITIONS --------------
def get_models_for_target(target_name):
    """Get models with target-specific hyperparameters"""
    models = {}
    
    # Common params
    if target_name == 'power':
        # Power needs more regularization
        n_est, lr, depth, reg = 600, 0.03, 6, 3.0
    elif target_name == 'delay':
        # Delay also needs careful tuning
        n_est, lr, depth, reg = 600, 0.03, 6, 3.0
    else:  # area
        # Area is easier
        n_est, lr, depth, reg = 500, 0.05, 7, 2.0
    
    if HAS_XGB:
        models['xgb'] = XGBRegressor(
            n_estimators=n_est, learning_rate=lr, max_depth=depth,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            gamma=0.5, reg_alpha=reg*0.5, reg_lambda=reg,
            random_state=RANDOM_STATE, n_jobs=-1, tree_method='hist'
        )
    
    if HAS_LGBM:
        models['lgbm'] = LGBMRegressor(
            n_estimators=n_est, learning_rate=lr, max_depth=depth,
            num_leaves=min(31, 2**depth - 1), subsample=0.8, colsample_bytree=0.8,
            reg_alpha=reg*0.5, reg_lambda=reg, min_child_samples=20,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
    
    if HAS_CATBOOST:
        models['catboost'] = CatBoostRegressor(
            iterations=n_est, learning_rate=lr, depth=depth,
            l2_leaf_reg=reg, random_seed=RANDOM_STATE,
            verbose=False, thread_count=-1
        )
    
    # Random Forest - always available
    models['rf'] = RandomForestRegressor(
        n_estimators=300, max_depth=depth+5, min_samples_split=10,
        min_samples_leaf=5, max_features='sqrt',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    
    # Gradient Boosting (sklearn fallback)
    models['gbm'] = GradientBoostingRegressor(
        n_estimators=min(300, n_est), learning_rate=lr,
        max_depth=depth-1, subsample=0.8,
        random_state=RANDOM_STATE
    )
    
    return models

# -------------- TRAINING --------------
def train_ensemble(X_tr, y_tr, X_val, y_val, X_te, y_te, y_te_orig, name, inverse_transform_fn):
    """Train ensemble with proper inverse transform"""
    print(f"\n{'='*70}")
    print(f"Training Ensemble for {name.upper()}")
    print(f"{'='*70}")
    
    models = get_models_for_target(name.lower())
    trained = {}
    val_preds = {}
    test_preds = {}
    
    for model_name, model in models.items():
        try:
            print(f"\nTraining {model_name.upper()}...", end=' ')
            model.fit(X_tr, y_tr)
            
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_te)
            
            val_r2 = r2_score(y_val, val_pred)
            print(f"Val R² = {val_r2:.4f}")
            
            if val_r2 > -1.0:  # Only use models with reasonable performance
                trained[model_name] = model
                val_preds[model_name] = val_pred
                test_preds[model_name] = test_pred
            else:
                print(f"  (Excluded: R² too low)")
                
        except Exception as e:
            print(f"Failed: {str(e)[:50]}")
    
    if not trained:
        raise ValueError(f"No models trained successfully for {name}!")
    
    # Compute weights
    print(f"\n{'='*70}")
    print("Computing Ensemble Weights...")
    print(f"{'='*70}")
    
    weights = {}
    for mn in trained.keys():
        r2 = max(0, r2_score(y_val, val_preds[mn]))
        weights[mn] = r2 ** 2
    
    total = sum(weights.values())
    if total > 0:
        weights = {k: v/total for k, v in weights.items()}
    else:
        weights = {k: 1/len(weights) for k in weights.keys()}
    
    print("\nWeights:")
    for mn, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {mn.upper():<12} {w:.4f}")
    
    # Ensemble prediction
    test_pred = sum(test_preds[mn] * weights[mn] for mn in trained.keys())
    test_pred_orig = inverse_transform_fn(test_pred)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_te_orig, test_pred_orig))
    r2 = r2_score(y_te_orig, test_pred_orig)
    mae = mean_absolute_error(y_te_orig, test_pred_orig)
    mape = np.mean(np.abs((y_te_orig - test_pred_orig) / np.clip(y_te_orig, 1, None))) * 100
    
    print(f"\n{'='*70}")
    print(f"ENSEMBLE RESULTS: {name.upper()}")
    print(f"{'='*70}")
    print(f"Test R²:   {r2:.4f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE:  {mae:.2f}")
    print(f"Test MAPE: {mape:.2f}%")
    
    return {
        'models': trained,
        'weights': weights,
        'metrics': {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape}
    }

# -------------- TRAIN ALL ENSEMBLES --------------
print("\n" + "="*70)
print("TRAINING ENSEMBLES")
print("="*70)

# Area
area_ensemble = train_ensemble(
    X_train_scaled, y_area_train, X_val_scaled, y_area_val,
    X_test_scaled, y_area_test, y_area_test_orig,
    "AREA", lambda x: np.expm1(x)
)

# Power
power_ensemble = train_ensemble(
    X_train_scaled, y_power_train, X_val_scaled, y_power_val,
    X_test_scaled, y_power_test, y_power_test_orig,
    "POWER", lambda x: power_transformer.inverse_transform(x.reshape(-1, 1)).ravel()
)

# Delay
delay_ensemble = train_ensemble(
    X_train_scaled, y_delay_train, X_val_scaled, y_delay_val,
    X_test_scaled, y_delay_test, y_delay_test_orig,
    "DELAY", lambda x: np.expm1(x)
)

# -------------- SAVE --------------
print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

joblib.dump(area_ensemble, os.path.join(MODEL_DIR, "area_ensemble.joblib"))
joblib.dump(power_ensemble, os.path.join(MODEL_DIR, "power_ensemble.joblib"))
joblib.dump(delay_ensemble, os.path.join(MODEL_DIR, "delay_ensemble.joblib"))
joblib.dump(final_features, os.path.join(MODEL_DIR, "features.joblib"))
joblib.dump({'transforms': TARGET_TRANSFORMS}, os.path.join(MODEL_DIR, "config.joblib"))

print(f"Saved to: {MODEL_DIR}")

# -------------- SUMMARY --------------
print("\n" + "="*70)
print("FINAL ENSEMBLE RESULTS")
print("="*70)
print(f"{'Target':<10} {'R²':<10} {'RMSE':<12} {'MAE':<12} {'MAPE':<10}")
print("-" * 60)

for name, ens in [('Area', area_ensemble), ('Power', power_ensemble), ('Delay', delay_ensemble)]:
    m = ens['metrics']
    print(f"{name:<10} {m['r2']:<10.4f} {m['rmse']:<12.2f} {m['mae']:<12.2f} {m['mape']:<10.2f}%")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)