"""#!/usr/bin/env python3
RTL-Only Delay Prediction Model
Focus: Pushing the limits of static RTL logic depth estimation
"""

import os
import joblib
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
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
MODEL_DIR = r"C:\ml ppa\models_delay_rtl_only"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
REMOVE_OUTLIERS = True
OUTLIER_STD = 3.5  

# -------------- LOAD DATA ---------------
print("="*70)
print("RTL-ONLY DELAY PREDICTION EXPERIMENT")
print("="*70)

df = pd.read_csv(CSV_PATH)
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

# STRICTLY RTL FEATURES
base_features = [f for f in rtl_features if f in df.columns]
print(f"Using PURE RTL: {len(base_features)} base features")

# -------------- DELAY FEATURE ENGINEERING --------------
def engineer_delay_features(df, base_features):
    """Aggressive logic depth proxies using ONLY RTL features"""
    df_feat = df[base_features].copy()
    eps = 1e-8
    
    # 1. Advanced Arithmetic Depth Proxy (Weighted by hardware delay characteristics)
    if all(c in df.columns for c in ["max_bitwidth", "num_mul", "num_div", "num_add", "num_sub"]):
        # Dividers and Multipliers are the slowest. Addition is faster. 
        # Everything scales with max_bitwidth.
        df_feat["arithmetic_depth_proxy"] = (
            (df["max_bitwidth"] * df["num_div"] * 5.0) + 
            (df["max_bitwidth"] * df["num_mul"] * 3.0) + 
            (df["max_bitwidth"] * df["num_add"] * 1.0) +
            (df["max_bitwidth"] * df["num_sub"] * 1.0)
        )
                                         
    # 2. Control Structure Depth (Mux Trees)
    if all(c in df.columns for c in ["num_case", "num_if", "num_comparisons", "num_ternary"]):
        df_feat["control_depth_proxy"] = (
            df["num_case"] * 3.0 + 
            df["num_if"] * 2.0 + 
            df["num_ternary"] * 1.5 + 
            df["num_comparisons"] * 1.0
        )
        
    # 3. Mux Width Proxy (Wide bitwidths going through branches = slow)
    if all(c in df.columns for c in ["num_branches", "avg_bitwidth"]):
        df_feat["mux_tree_proxy"] = df["num_branches"] * df["avg_bitwidth"]

    # 4. Pipeline Depth Estimation (Crucial for RTL-only)
    # How much logic is crammed between the registers?
    if "num_reg" in df.columns and "arithmetic_depth_proxy" in df_feat.columns and "control_depth_proxy" in df_feat.columns:
        total_complexity = df_feat["arithmetic_depth_proxy"] + df_feat["control_depth_proxy"]
        df_feat["pipeline_depth_proxy"] = total_complexity / (df["num_reg"] + eps)
        
    # 5. Combinational Density
    if all(c in df.columns for c in ["num_assign", "num_always", "num_lines"]):
        df_feat["comb_density"] = (df["num_assign"] + df["num_always"]) / (df["num_lines"] + eps)

    return df_feat

print("\nEngineering RTL-specific delay features...")
df_features = engineer_delay_features(df, base_features)

# -------------- GLOBAL DATA CLEANING --------------
# We include ALL targets to ensure we filter out the noisy outliers
df_model = pd.concat([df_features, df[[area_col, power_col, delay_col]]], axis=1)

before = len(df_model)
df_model = df_model.dropna()
df_model = df_model[(df_model[area_col] > 0) & (df_model[power_col] > 0) & (df_model[delay_col] > 0)]

if REMOVE_OUTLIERS:
    for target in [area_col, power_col, delay_col]:
        q1, q3 = df_model[target].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - OUTLIER_STD * iqr
        upper = q3 + OUTLIER_STD * iqr
        df_model = df_model[(df_model[target] >= lower) & (df_model[target] <= upper)]
    
print(f"Valid Samples After Global Cleaning: {len(df_model)}/{before}")

# -------------- PREPARE DATA --------------
final_features = df_features.columns.tolist()
X = df_model[final_features].values
y_delay_orig = df_model[delay_col].values

# Delay: log1p transform
y_delay = np.log1p(y_delay_orig)

# Split
X_temp, X_test, idx_temp, idx_test = train_test_split(
    X, np.arange(len(X)), test_size=TEST_SIZE, random_state=RANDOM_STATE
)
val_size_adj = VAL_SIZE / (1 - TEST_SIZE)
X_train, X_val, idx_train, idx_val = train_test_split(
    X_temp, idx_temp, test_size=val_size_adj, random_state=RANDOM_STATE
)

y_train, y_val, y_test = y_delay[idx_train], y_delay[idx_val], y_delay[idx_test]
y_test_orig = y_delay_orig[idx_test]

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# -------------- EVALUATION METRICS --------------
def calculate_smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error to stabilize evaluation"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(numerator / (denominator + 1e-8)) * 100

# -------------- MODEL DEFINITIONS --------------
def get_delay_models():
    """Giving the models a bit more depth to find non-linear RTL relationships"""
    models = {}
    
    n_est, lr, depth, reg = 600, 0.02, 6, 4.0 
    
    if HAS_XGB:
        models['xgb'] = XGBRegressor(
            n_estimators=n_est, learning_rate=lr, max_depth=depth,
            min_child_weight=6, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=reg, reg_lambda=reg*2, random_state=RANDOM_STATE, n_jobs=-1
        )
    if HAS_LGBM:
        models['lgbm'] = LGBMRegressor(
            n_estimators=n_est, learning_rate=lr, max_depth=depth,
            num_leaves=min(63, 2**depth - 1), subsample=0.8, colsample_bytree=0.8,
            reg_alpha=reg, reg_lambda=reg*2, min_child_samples=20,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
    if HAS_CATBOOST:
        models['catboost'] = CatBoostRegressor(
            iterations=n_est, learning_rate=lr, depth=depth,
            l2_leaf_reg=reg, random_seed=RANDOM_STATE, verbose=False
        )
    
    models['rf'] = RandomForestRegressor(
        n_estimators=400, max_depth=depth+4, min_samples_split=12,
        min_samples_leaf=6, max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1
    )
    
    return models

# -------------- TRAINING --------------
print(f"\nTraining Delay Ensemble (RTL-Only)...")
models = get_delay_models()
trained, val_preds, test_preds = {}, {}, {}

for model_name, model in models.items():
    print(f"  Fitting {model_name.upper()}...", end=' ')
    model.fit(X_train_scaled, y_train)
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)
    
    val_r2 = r2_score(y_val, val_pred)
    print(f"Val R² = {val_r2:.4f}")
    
    if val_r2 > -1.0: 
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
test_pred_trans = sum(test_preds[mn] * weights[mn] for mn in trained.keys())
test_pred_orig = np.expm1(test_pred_trans)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred_orig))
r2 = r2_score(y_test_orig, test_pred_orig)
mae = mean_absolute_error(y_test_orig, test_pred_orig)
smape = calculate_smape(y_test_orig, test_pred_orig)

print(f"\n{'='*40}")
print(f"RTL-ONLY DELAY ENSEMBLE RESULTS")
print(f"{'='*40}")
print(f"Test R²:    {r2:.4f}")
print(f"Test RMSE:  {rmse:.4f}")
print(f"Test MAE:   {mae:.4f}")
print(f"Test SMAPE: {smape:.2f}%")
print(f"{'='*40}")