"""#!/usr/bin/env python3
Hybrid (RTL + Netlist) Power Prediction Model
Focus: Proving the necessity of structural netlist features for Power
"""

import os
import joblib
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PowerTransformer
# from sklearn.preprocessing import RobustScaler
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
MODEL_DIR = r"C:\ml ppa\models_power_hybrid"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
REMOVE_OUTLIERS = True
OUTLIER_STD = 3.5  

# -------------- LOAD DATA ---------------
print("="*70)
print("HYBRID (RTL + NETLIST) POWER PREDICTION EXPERIMENT")
print("="*70)

df = pd.read_csv(CSV_PATH)
power_col = 'Power'

rtl_features = [
    'num_lines', 'max_bitwidth', 'min_bitwidth', 'avg_bitwidth',
    'total_bits', 'num_bitwidths', 'num_32bit', 'num_16bit', 'num_8bit', 'num_1bit',
    'num_add', 'num_sub', 'num_mul', 'num_div',
    'num_logic_and', 'num_logic_or', 'num_logic_xor', 'num_shifts', 'num_comparisons',
    'num_always', 'num_assign', 'num_if', 'num_case', 'num_for', 'num_ternary', 'num_branches',
    'num_wire', 'num_reg', 'num_input', 'num_output', 'num_modules',
]

# BRINGING BACK THE NETLIST FEATURES
netlist_features = [
    'netlist_num_gates', 'netlist_num_nets', 'netlist_num_instances',
    'netlist_inv_count', 'netlist_and_count', 'netlist_or_count',
    'netlist_nand_count', 'netlist_nor_count', 'netlist_xor_count',
    'netlist_mux_count', 'netlist_buf_count', 'netlist_maj3_count'
]

available_rtl = [f for f in rtl_features if f in df.columns]
available_netlist = [f for f in netlist_features if f in df.columns]

# Combine them for the Hybrid approach
base_features = available_rtl + available_netlist
print(f"Using HYBRID: {len(available_rtl)} RTL + {len(available_netlist)} Netlist features")

# -------------- POWER FEATURE ENGINEERING --------------
def engineer_power_features(df, base_features):
    """Specific feature engineering heavily focused on switching proxies and gate logic"""
    df_feat = df[base_features].copy()
    eps = 1e-8
    
    # 1. Base RTL Aggregates
    arith = [c for c in ['num_add', 'num_sub', 'num_mul', 'num_div'] if c in df.columns]
    if arith:
        df_feat['total_arithmetic'] = df[arith].sum(axis=1)
        
    logic = [c for c in ['num_logic_and', 'num_logic_or', 'num_logic_xor', 'num_shifts'] if c in df.columns]
    if logic:
        df_feat['total_logic'] = df[logic].sum(axis=1)

    # 2. Advanced Power Proxies (Switching & Capacitance Estimates)
    if all(c in df.columns for c in ['num_reg', 'num_logic_xor']):
        df_feat['high_toggle_proxy'] = df['num_reg'] + (df['num_logic_xor'] * 2)

    if all(c in df.columns for c in ['num_assign', 'avg_bitwidth', 'num_always', 'num_reg']):
        df_feat['comb_switching_proxy'] = df['num_assign'] * df['avg_bitwidth']
        df_feat['seq_switching_proxy'] = df['num_always'] * df['num_reg']
        
    if all(c in df.columns for c in ['num_input', 'num_output', 'avg_bitwidth']):
        df_feat['io_bandwidth_proxy'] = (df['num_input'] + df['num_output']) * df['avg_bitwidth']
        
    # 3. Netlist Proxies (The Missing Link for Power!)
    if 'netlist_num_gates' in df.columns:
        df_feat['gates_log'] = np.log1p(df['netlist_num_gates'])
        if 'num_lines' in df.columns:
            df_feat['gates_per_line'] = df['netlist_num_gates'] / (df['num_lines'] + eps)
            
    if 'netlist_mux_count' in df.columns and 'netlist_num_gates' in df.columns:
        df_feat['mux_density'] = df['netlist_mux_count'] / (df['netlist_num_gates'] + eps)
        
    # Gate diversity (how many gate types used - correlates with synthesis complexity)
    gate_cols = [c for c in df.columns if c.startswith('netlist_') and c.endswith('_count')]
    if gate_cols:
        df_feat['gate_diversity'] = (df[gate_cols] > 0).sum(axis=1)

    return df_feat

print("\nEngineering hybrid power-specific features...")
df_features = engineer_power_features(df, base_features)
df_model = pd.concat([df_features, df[[power_col]]], axis=1)

# -------------- DATA CLEANING --------------
before = len(df_model)
df_model = df_model.dropna()
df_model = df_model[df_model[power_col] > 0] 

if REMOVE_OUTLIERS:
    q1, q3 = df_model[power_col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - OUTLIER_STD * iqr
    upper = q3 + OUTLIER_STD * iqr
    df_model = df_model[(df_model[power_col] >= lower) & (df_model[power_col] <= upper)]
    
print(f"Valid Samples After Cleaning: {len(df_model)}/{before}")

# -------------- PREPARE DATA --------------
final_features = df_features.columns.tolist()
X = df_model[final_features].values
y_power_orig = df_model[power_col].values


# Power: Yeo-Johnson Transform (Better for extreme outliers in noisy datasets)
power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
y_power = power_transformer.fit_transform(y_power_orig.reshape(-1, 1)).ravel()
# Power: Simple log1p transform (stable)
# y_power = np.log1p(y_power_orig)

# Split
X_temp, X_test, idx_temp, idx_test = train_test_split(
    X, np.arange(len(X)), test_size=TEST_SIZE, random_state=RANDOM_STATE
)
val_size_adj = VAL_SIZE / (1 - TEST_SIZE)
X_train, X_val, idx_train, idx_val = train_test_split(
    X_temp, idx_temp, test_size=val_size_adj, random_state=RANDOM_STATE
)

y_train, y_val, y_test = y_power[idx_train], y_power[idx_val], y_power[idx_test]
y_test_orig = y_power_orig[idx_test]

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# -------------- EVALUATION METRICS --------------
def calculate_smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error to handle near-zero targets"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(numerator / (denominator + 1e-8)) * 100

# -------------- MODEL DEFINITIONS --------------
def get_power_models():
    """Using shallower depths to prevent overfitting, but letting the models learn from netlist data"""
    models = {}
    
    n_est, lr, depth, reg = 400, 0.02, 5, 5.0
    
    if HAS_XGB:
        models['xgb'] = XGBRegressor(
            n_estimators=n_est, learning_rate=lr, max_depth=depth,
            min_child_weight=7, subsample=0.75, colsample_bytree=0.75,
            reg_alpha=reg, reg_lambda=reg*2, random_state=RANDOM_STATE, n_jobs=-1
        )
    if HAS_LGBM:
        models['lgbm'] = LGBMRegressor(
            n_estimators=n_est, learning_rate=lr, max_depth=depth,
            num_leaves=31, subsample=0.75, colsample_bytree=0.75,
            reg_alpha=reg, reg_lambda=reg*2, min_child_samples=30,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
    if HAS_CATBOOST:
        models['catboost'] = CatBoostRegressor(
            iterations=n_est, learning_rate=lr, depth=depth,
            l2_leaf_reg=reg*2, random_seed=RANDOM_STATE, verbose=False
        )
    
    models['rf'] = RandomForestRegressor(
        n_estimators=300, max_depth=depth+3, min_samples_split=15,
        min_samples_leaf=7, max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1
    )
    
    return models

# -------------- TRAINING --------------
print(f"\nTraining Power Ensemble (Hybrid)...")
models = get_power_models()
trained, val_preds, test_preds = {}, {}, {}

for model_name, model in models.items():
    print(f"  Fitting {model_name.upper()}...", end=' ')
    model.fit(X_train_scaled, y_train)
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)
    
    val_r2 = r2_score(y_val, val_pred)
    print(f"Val R² = {val_r2:.4f}")
    
    if val_r2 > -1.0: # Capture all to see if netlist pushes it positive
        trained[model_name] = model
        val_preds[model_name] = val_pred
        test_preds[model_name] = test_pred

# Compute Weights
weights = {mn: max(0, r2_score(y_val, val_preds[mn])) ** 2 for mn in trained.keys()}
total_w = sum(weights.values())
if total_w > 0:
    weights = {k: v/total_w for k, v in weights.items()}
else:
    weights = {k: 1/len(weights) for k in weights.keys()}

# Predict
# test_pred_trans = sum(test_preds[mn] * weights[mn] for mn in trained.keys())

# Predict
test_pred_trans = sum(test_preds[mn] * weights[mn] for mn in trained.keys())

# Reverse the Yeo-Johnson transform
test_pred_orig = power_transformer.inverse_transform(test_pred_trans.reshape(-1, 1)).ravel()

# Reverse the log1p transform
test_pred_orig = np.expm1(test_pred_trans)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred_orig))
r2 = r2_score(y_test_orig, test_pred_orig)
mae = mean_absolute_error(y_test_orig, test_pred_orig)
smape = calculate_smape(y_test_orig, test_pred_orig)

print(f"\n{'='*40}")
print(f"HYBRID POWER ENSEMBLE RESULTS")
print(f"{'='*40}")
print(f"Test R²:    {r2:.4f}")
print(f"Test RMSE:  {rmse:.4f}")
print(f"Test MAE:   {mae:.4f}")
print(f"Test SMAPE: {smape:.2f}%")
print(f"{'='*40}")