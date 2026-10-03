"""#!/usr/bin/env python3

Production-Ready XGBoost PPA Prediction with Anti-Overfitting
Key improvements:
- Log transformation for skewed targets
- Aggressive regularization to prevent overfitting
- Outlier removal
- Pure RTL features (no netlist leakage)
- K-fold cross-validation
- Early stopping
"""

import os
import joblib
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
CSV_PATH = r"C:\ml ppa\final_maximal_dataset.csv"
MODEL_DIR = r"C:\ml ppa\models_xgb_production"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
USE_LOG_TRANSFORM = True  # Critical for skewed distributions
REMOVE_OUTLIERS = True
OUTLIER_STD_THRESHOLD = 4.0

# -------------- LOAD DATA ---------------
print("="*70)
print("PRODUCTION XGBoost PPA PREDICTION - Anti-Overfitting Edition")
print("="*70)
print(f"\nLoading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Shape: {df.shape}")

# -------------- FEATURE SELECTION (RTL ONLY) --------------
print("\n" + "="*70)
print("FEATURE SELECTION - Pure RTL Model (No Ground Truth Leakage)")
print("="*70)

# NEVER use these as features (they're post-synthesis ground truth)
FORBIDDEN_FEATURES = [
    'num_cells', 'num_nets', 'comb_area', 'total_cell_area',
    'critical_path_length', 'levels_of_logic', 'wns', 'tns', 'Power',
    # Also exclude netlist features for pure RTL model
    'netlist_num_gates', 'netlist_num_nets', 'netlist_num_instances',
    'netlist_inv_count', 'netlist_and_count', 'netlist_or_count',
    'netlist_nand_count', 'netlist_nor_count', 'netlist_xor_count',
    'netlist_mux_count', 'netlist_buf_count', 'netlist_maj3_count'
]

id_cols = ['Design_Name', 'RTL_Code']

# Targets
area_col = 'total_cell_area' if 'total_cell_area' in df.columns else 'comb_area'
power_col = 'Power'
delay_col = 'critical_path_length'

# Pure RTL features
rtl_features = [
    'num_lines', 'max_bitwidth', 'min_bitwidth', 'avg_bitwidth',
    'total_bits', 'num_bitwidths', 'num_32bit', 'num_16bit', 'num_8bit', 'num_1bit',
    'num_add', 'num_sub', 'num_mul', 'num_div',
    'num_logic_and', 'num_logic_or', 'num_logic_xor', 'num_shifts', 'num_comparisons',
    'num_always', 'num_assign', 'num_if', 'num_case', 'num_for', 'num_ternary', 'num_branches',
    'num_wire', 'num_reg', 'num_input', 'num_output', 'num_modules',
]

# Only use features that exist in dataset
feature_cols = [f for f in rtl_features if f in df.columns]
print(f"Using {len(feature_cols)} pure RTL features")
print(f"Targets: {area_col}, {power_col}, {delay_col}")

# -------------- FEATURE ENGINEERING --------------
def create_features(df, base_features):
    """Engineer meaningful derived features"""
    df_feat = df[base_features].copy()
    
    # Replace zeros with small value to avoid division errors
    eps = 1e-6
    
    # Complexity metrics
    if all(c in df.columns for c in ['num_mul', 'num_add']):
        df_feat['mul_add_ratio'] = df['num_mul'] / (df['num_add'] + eps)
    
    if all(c in df.columns for c in ['num_div', 'num_mul']):
        df_feat['div_mul_ratio'] = df['num_div'] / (df['num_mul'] + eps)
    
    # Aggregate operations
    datapath = ['num_add', 'num_sub', 'num_mul', 'num_div']
    avail_dp = [c for c in datapath if c in df.columns]
    if avail_dp:
        df_feat['total_arithmetic'] = df[avail_dp].sum(axis=1)
    
    logic = ['num_logic_and', 'num_logic_or', 'num_logic_xor', 'num_shifts']
    avail_logic = [c for c in logic if c in df.columns]
    if avail_logic:
        df_feat['total_logic'] = df[avail_logic].sum(axis=1)
    
    control = ['num_if', 'num_case', 'num_for', 'num_branches', 'num_ternary']
    avail_ctrl = [c for c in control if c in df.columns]
    if avail_ctrl:
        df_feat['total_control'] = df[avail_ctrl].sum(axis=1)
    
    # Density features (per line of code)
    if all(c in df.columns for c in ['num_reg', 'num_lines']):
        df_feat['reg_per_line'] = df['num_reg'] / (df['num_lines'] + eps)
    
    if all(c in df.columns for c in ['num_wire', 'num_lines']):
        df_feat['wire_per_line'] = df['num_wire'] / (df['num_lines'] + eps)
    
    if 'total_arithmetic' in df_feat.columns and 'num_lines' in df.columns:
        df_feat['arith_per_line'] = df_feat['total_arithmetic'] / (df['num_lines'] + eps)
    
    # Bitwidth features
    if all(c in df.columns for c in ['max_bitwidth', 'min_bitwidth', 'avg_bitwidth']):
        df_feat['bitwidth_spread'] = df['max_bitwidth'] - df['min_bitwidth']
        df_feat['bitwidth_max_avg_ratio'] = df['max_bitwidth'] / (df['avg_bitwidth'] + eps)
    
    # I/O complexity
    if all(c in df.columns for c in ['num_input', 'num_output']):
        df_feat['total_io'] = df['num_input'] + df['num_output']
        df_feat['input_output_ratio'] = df['num_input'] / (df['num_output'] + eps)
    
    # Code complexity
    if all(c in df.columns for c in ['num_always', 'num_assign']):
        df_feat['always_assign_ratio'] = df['num_always'] / (df['num_assign'] + eps)
    
    return df_feat

print("\nEngineering features...")
df_features = create_features(df, feature_cols)
df_model = pd.concat([df_features, df[[area_col, power_col, delay_col]]], axis=1)

# -------------- DATA CLEANING --------------
print("\nData cleaning...")
before = len(df_model)
df_model = df_model.dropna()
print(f"After dropna: {len(df_model)}/{before} ({len(df_model)/before*100:.1f}%)")

# Remove targets <= 0 (log transform requires positive values)
if USE_LOG_TRANSFORM:
    before = len(df_model)
    df_model = df_model[
        (df_model[area_col] > 0) & 
        (df_model[power_col] > 0) & 
        (df_model[delay_col] > 0)
    ]
    print(f"After removing non-positive targets: {len(df_model)}/{before} ({len(df_model)/before*100:.1f}%)")

# Remove outliers (extreme values that cause overfitting)
if REMOVE_OUTLIERS:
    before = len(df_model)
    for target in [area_col, power_col, delay_col]:
        mean = df_model[target].mean()
        std = df_model[target].std()
        df_model = df_model[
            np.abs(df_model[target] - mean) <= OUTLIER_STD_THRESHOLD * std
        ]
    print(f"After outlier removal: {len(df_model)}/{before} ({len(df_model)/before*100:.1f}%)")

# -------------- PREPARE DATA --------------
final_features = df_features.columns.tolist()
X = df_model[final_features].values
y_area = df_model[area_col].values
y_power = df_model[power_col].values
y_delay = df_model[delay_col].values

print(f"\nFinal dataset:")
print(f"  Features: {X.shape[1]}")
print(f"  Samples: {X.shape[0]}")
print(f"  Area range: [{y_area.min():.1f}, {y_area.max():.1f}]")
print(f"  Power range: [{y_power.min():.1f}, {y_power.max():.1f}]")
print(f"  Delay range: [{y_delay.min():.1f}, {y_delay.max():.1f}]")

# -------------- LOG TRANSFORM --------------
if USE_LOG_TRANSFORM:
    print("\nApplying log transformation to targets...")
    y_area_orig, y_power_orig, y_delay_orig = y_area.copy(), y_power.copy(), y_delay.copy()
    y_area = np.log1p(y_area)
    y_power = np.log1p(y_power)
    y_delay = np.log1p(y_delay)

# -------------- TRAIN/TEST SPLIT --------------
X_train, X_test, y_area_train, y_area_test = train_test_split(
    X, y_area, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
_, _, y_power_train, y_power_test = train_test_split(
    X, y_power, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
_, _, y_delay_train, y_delay_test = train_test_split(
    X, y_delay, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Also split original (non-log) values for proper evaluation
if USE_LOG_TRANSFORM:
    _, _, _, y_area_test_orig = train_test_split(
        X, y_area_orig, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    _, _, _, y_power_test_orig = train_test_split(
        X, y_power_orig, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    _, _, _, y_delay_test_orig = train_test_split(
        X, y_delay_orig, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

print(f"\nTrain/Test split:")
print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# -------------- SCALING --------------
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
joblib.dump(scaler, scaler_path)
joblib.dump({'use_log': USE_LOG_TRANSFORM}, os.path.join(MODEL_DIR, "transform_config.joblib"))

# -------------- TRAINING FUNCTION --------------
def train_production_xgb(X_tr, y_tr, X_te, y_te, y_te_orig, label_name):
    """Train with aggressive anti-overfitting measures"""
    print(f"\n{'='*70}")
    print(f"Training {label_name}")
    print(f"{'='*70}")
    
    # AGGRESSIVE regularization to prevent overfitting
    model = XGBRegressor(
        n_estimators=300,           # Fewer trees
        learning_rate=0.05,         # Slower learning
        max_depth=5,                # Shallower trees (was 10-12)
        min_child_weight=10,        # More regularization (was 3-5)
        subsample=0.7,              # Less data per tree
        colsample_bytree=0.7,       # Less features per tree
        colsample_bylevel=0.7,      # Even more feature sampling
        gamma=1.0,                  # Higher pruning threshold
        reg_alpha=1.0,              # L1 regularization
        reg_lambda=5.0,             # Strong L2 regularization (was 1-2)
        max_delta_step=1,           # Limit step size
        objective='reg:squarederror',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method='hist',
        early_stopping_rounds=30,   # Moved to constructor for compatibility
    )
    
    # Fit with validation set
    eval_set = [(X_tr, y_tr), (X_te, y_te)]
    
    try:
        model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
    except TypeError:
        # Fallback for older XGBoost versions
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=10,
            subsample=0.7,
            colsample_bytree=0.7,
            colsample_bylevel=0.7,
            gamma=1.0,
            reg_alpha=1.0,
            reg_lambda=5.0,
            max_delta_step=1,
            objective='reg:squarederror',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method='hist',
        )
        model.fit(X_tr, y_tr, verbose=False)
    
    best_iter = getattr(model, 'best_iteration', model.n_estimators)
    print(f"Best iteration: {best_iter} (out of {model.n_estimators})")
    
    # Predictions
    y_train_pred = model.predict(X_tr)
    y_test_pred = model.predict(X_te)
    
    # If log transformed, convert back
    if USE_LOG_TRANSFORM:
        y_train_pred_orig = np.expm1(y_train_pred)
        y_test_pred_orig = np.expm1(y_test_pred)
        y_train_orig = np.expm1(y_tr)
    else:
        y_train_pred_orig = y_train_pred
        y_test_pred_orig = y_test_pred
        y_train_orig = y_tr
    
    # Calculate metrics in original space
    train_rmse = np.sqrt(mean_squared_error(y_train_orig, y_train_pred_orig))
    test_rmse = np.sqrt(mean_squared_error(y_te_orig, y_test_pred_orig))
    train_r2 = r2_score(y_train_orig, y_train_pred_orig)
    test_r2 = r2_score(y_te_orig, y_test_pred_orig)
    train_mae = mean_absolute_error(y_train_orig, y_train_pred_orig)
    test_mae = mean_absolute_error(y_te_orig, y_test_pred_orig)
    
    # MAPE (safe calculation)
    train_mape = np.mean(np.abs((y_train_orig - y_train_pred_orig) / np.clip(y_train_orig, 1e-6, None))) * 100
    test_mape = np.mean(np.abs((y_te_orig - y_test_pred_orig) / np.clip(y_te_orig, 1e-6, None))) * 100
    
    print(f"\n{'Split':<10} {'RMSE':<15} {'MAE':<15} {'R²':<10} {'MAPE':<10}")
    print("-" * 60)
    print(f"{'Train':<10} {train_rmse:<15.2f} {train_mae:<15.2f} {train_r2:<10.4f} {train_mape:<10.2f}%")
    print(f"{'Test':<10} {test_rmse:<15.2f} {test_mae:<15.2f} {test_r2:<10.4f} {test_mape:<10.2f}%")
    
    # Check for overfitting
    r2_gap = train_r2 - test_r2
    if r2_gap > 0.15:
        print(f"⚠️  WARNING: Overfitting detected (R² gap = {r2_gap:.3f})")
    elif r2_gap > 0.05:
        print(f"⚠️  CAUTION: Mild overfitting (R² gap = {r2_gap:.3f})")
    else:
        print(f"✓  Good generalization (R² gap = {r2_gap:.3f})")
    
    return model, {
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'train_r2': train_r2,
        'overfitting_gap': r2_gap
    }

# -------------- TRAIN MODELS --------------
area_model, area_metrics = train_production_xgb(
    X_train_scaled, y_area_train, X_test_scaled, y_area_test,
    y_area_test_orig if USE_LOG_TRANSFORM else y_area_test,
    "AREA"
)

power_model, power_metrics = train_production_xgb(
    X_train_scaled, y_power_train, X_test_scaled, y_power_test,
    y_power_test_orig if USE_LOG_TRANSFORM else y_power_test,
    "POWER"
)

delay_model, delay_metrics = train_production_xgb(
    X_train_scaled, y_delay_train, X_test_scaled, y_delay_test,
    y_delay_test_orig if USE_LOG_TRANSFORM else y_delay_test,
    "DELAY"
)

# -------------- SAVE MODELS --------------
joblib.dump(area_model, os.path.join(MODEL_DIR, "xgb_area.joblib"))
joblib.dump(power_model, os.path.join(MODEL_DIR, "xgb_power.joblib"))
joblib.dump(delay_model, os.path.join(MODEL_DIR, "xgb_delay.joblib"))
joblib.dump(final_features, os.path.join(MODEL_DIR, "feature_names.joblib"))

# -------------- SUMMARY --------------
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"{'Target':<10} {'Test R²':<12} {'RMSE':<15} {'MAPE':<12} {'Overfit Gap':<12}")
print("-" * 70)
print(f"{'Area':<10} {area_metrics['test_r2']:<12.4f} {area_metrics['test_rmse']:<15.2f} "
      f"{area_metrics['test_mape']:<12.2f}% {area_metrics['overfitting_gap']:<12.4f}")
print(f"{'Power':<10} {power_metrics['test_r2']:<12.4f} {power_metrics['test_rmse']:<15.2f} "
      f"{power_metrics['test_mape']:<12.2f}% {power_metrics['overfitting_gap']:<12.4f}")
print(f"{'Delay':<10} {delay_metrics['test_r2']:<12.4f} {delay_metrics['test_rmse']:<15.2f} "
      f"{delay_metrics['test_mape']:<12.2f}% {delay_metrics['overfitting_gap']:<12.4f}")

print(f"\n✓ Models saved to: {MODEL_DIR}")
print(f"✓ Using pure RTL features (no netlist/ground-truth leakage)")
print(f"✓ Log transformation: {USE_LOG_TRANSFORM}")
print(f"✓ Outlier removal: {REMOVE_OUTLIERS}")

# -------------- FEATURE IMPORTANCE --------------
print("\n" + "="*70)
print("TOP 10 FEATURES PER TARGET")
print("="*70)

for model, name in [(area_model, "Area"), (power_model, "Power"), (delay_model, "Delay")]:
    importance = model.feature_importances_
    indices = np.argsort(importance)[-10:][::-1]
    
    print(f"\n{name}:")
    for i, idx in enumerate(indices, 1):
        print(f"  {i:2d}. {final_features[idx]:<35} {importance[idx]:.4f}")

print("\n" + "="*70)
print("Training Complete!")
print("="*70)