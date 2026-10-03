"""#!/usr/bin/env python3
"
Train multiple regression models (XGBoost, LightGBM, RandomForest, MLP, optional CatBoost)
for PPA prediction from final_maximal_dataset.csv. Keeps your original robust_fit wrapper
for XGBoost and adds a unified loop to train/evaluate/save multiple models reproducibly.

Features:
- Single master index split (train/val/test) reused for all models
- Validation set used for early stopping where supported
- Per-model evaluation (RMSE, R2) and saving of metadata
- Tries to import LightGBM/CatBoost but continues if missing
"""

import os
import joblib
import numpy as np
import pandas as pd
import traceback
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Try optional libraries
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False

# ---------------- CONFIG ----------------
CSV_PATH = r"C:\ml ppa\final_maximal_dataset.csv"
MODEL_DIR = r"C:\ml ppa\models_multi"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE_WITHIN_TRAIN = 0.125  # fraction of TRAIN to hold out for early stopping validation

# -------------- LOAD DATA ---------------
print(f"Loading dataset from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Shape: {df.shape}")

# -------------- TARGET & FEATURES --------------
id_cols = ["Design_Name", "RTL_Code"]

# Targets
if "total_cell_area" in df.columns:
    area_col = "total_cell_area"
elif "comb_area" in df.columns:
    area_col = "comb_area"
else:
    raise ValueError("No area column found.")

if "Power" not in df.columns:
    raise ValueError("No Power column found.")
power_col = "Power"

if "critical_path_length" not in df.columns:
    raise ValueError("No critical_path_length column found.")
delay_col = "critical_path_length"

target_cols = [area_col, power_col, delay_col]

# Feature selection: all numeric except ids + targets
feature_cols = [
    c for c in df.columns
    if c not in id_cols and c not in target_cols and pd.api.types.is_numeric_dtype(df[c])
    and df[c].dtype != bool
]

print(f"Selected {len(feature_cols)} numeric features.")

# Prepare dataset
df_model = df[feature_cols + target_cols].copy()
before = len(df_model)
df_model = df_model.dropna()
after = len(df_model)
print(f"Rows before dropna: {before}  after: {after}")

X_all = df_model[feature_cols].values
y_area_all  = df_model[area_col].values
y_power_all = df_model[power_col].values
y_delay_all = df_model[delay_col].values

# consistent index-based split for all targets
indices = np.arange(X_all.shape[0])
idx_train, idx_test = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE)
idx_tr, idx_val = train_test_split(idx_train, test_size=VAL_SIZE_WITHIN_TRAIN, random_state=RANDOM_STATE)

# arrays
X_train = X_all[idx_tr]
X_val   = X_all[idx_val]
X_test  = X_all[idx_test]

y_area_train = y_area_all[idx_tr]
y_area_val   = y_area_all[idx_val]
y_area_test  = y_area_all[idx_test]

y_power_train = y_power_all[idx_tr]
y_power_val   = y_power_all[idx_val]
y_power_test  = y_power_all[idx_test]

y_delay_train = y_delay_all[idx_tr]
y_delay_val   = y_delay_all[idx_val]
y_delay_test  = y_delay_all[idx_test]

print(f"Train size: {X_train.shape[0]}  Val size: {X_val.shape[0]}  Test size: {X_test.shape[0]}")

# -------------- ROBUST FIT for XGBoost ----------------
import traceback

def robust_fit_xgb(model, X_tr, y_tr, X_val=None, y_val=None, early_stopping_rounds=30):
    """
    Try modern fit() call with eval_metric; if xgboost version doesn't accept eval_metric
    retry with fewer kwargs. Return fitted model and string describing method used.
    """
    attempts = []
    if X_val is not None and y_val is not None:
        attempts.append({"kwargs": {"eval_set": [(X_val, y_val)], "eval_metric": "rmse",
                                     "early_stopping_rounds": early_stopping_rounds, "verbose": 0}})
        attempts.append({"kwargs": {"eval_set": [(X_val, y_val)], "early_stopping_rounds": early_stopping_rounds, "verbose": 0}})
    attempts.append({"kwargs": {"verbose": 0}})
    attempts.append({"kwargs": {}})

    last_exc = None
    for i, attempt in enumerate(attempts, 1):
        try:
            model.fit(X_tr, y_tr, **attempt["kwargs"])
            return model, f"attempt_{i}_ok"
        except (TypeError, ValueError) as e:
            last_exc = e
        except Exception as e:
            last_exc = e
    tb = traceback.format_exc()
    raise RuntimeError(f"All fit() attempts failed. Last exception: {last_exc}\nTrace:\n{tb}")

# -------------- TRAIN / EVAL WRAPPERS ----------------

def eval_preds(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def train_and_eval_model(name, model, X_tr, y_tr, X_val, y_val, X_test, y_test, model_path, use_scaler_for_nn=False):
    """Train model, evaluate on test, save model+metadata. Returns metrics dict."""
    print(f"\n--- Training {name} ---")
    start = time.time()

    pipeline = None
    if use_scaler_for_nn:
        # wrap with scaler only for MLP
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
        fitted = pipeline.fit(X_tr, y_tr)
    else:
        # try specialized XGBoost robust fit
        if name == 'XGBoost' and HAS_XGB:
            fitted_model, method = robust_fit_xgb(model, X_tr, y_tr, X_val, y_val, early_stopping_rounds=30)
            model = fitted_model
            print(f"XGBoost fit completed using: {method}")
        else:
            # for LightGBM, CatBoost we can try passing eval_set if available
            try:
                if name == 'LightGBM' and HAS_LGB and (X_val is not None):
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)
                elif name == 'CatBoost' and HAS_CAT and (X_val is not None):
                    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False, use_best_model=True)
                else:
                    model.fit(X_tr, y_tr)
            except TypeError:
                # fallback to simpler fit signature
                model.fit(X_tr, y_tr)
        fitted = model

    train_time = time.time() - start

    # predict
    if pipeline is not None:
        y_pred = pipeline.predict(X_test)
    else:
        y_pred = model.predict(X_test)

    rmse, r2 = eval_preds(y_test, y_pred)
    print(f"{name} RMSE: {rmse:.6f}  R2: {r2:.6f}  (train_time={train_time:.1f}s)")

    # save model + metadata
    meta = {
        'model_name': name,
        'feature_cols': feature_cols,
        'target': None,  # fill when saving per-target
        'random_state': RANDOM_STATE,
        'train_size': X_tr.shape[0],
        'val_size': X_val.shape[0] if X_val is not None else 0,
        'test_size': X_test.shape[0],
    }

    # joblib dump
    joblib.dump({'model': pipeline if pipeline is not None else model, 'meta': meta}, model_path)
    print(f"Saved model -> {model_path}")

    return {'rmse': rmse, 'r2': r2, 'train_time_s': train_time}

# -------------- MODEL SUITE & TRAINING ----------------
model_suite = []
if HAS_XGB:
    model_suite.append(('XGBoost', XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        objective='reg:squarederror',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method='hist',
    )))
else:
    print("XGBoost not available; skipping XGBoost model.")

if HAS_LGB:
    model_suite.append(('LightGBM', lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )))
else:
    print("LightGBM not available; skipping LightGBM model.")

# Random Forest
model_suite.append(('RandomForest', RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)))

# MLP (needs scaling)
model_suite.append(('MLP', MLPRegressor(
    hidden_layer_sizes=(256,128),
    activation='relu',
    solver='adam',
    learning_rate_init=1e-3,
    max_iter=500,
    random_state=RANDOM_STATE,
)))

if HAS_CAT:
    model_suite.append(('CatBoost', CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_state=RANDOM_STATE,
        verbose=0,
    )))
else:
    print("CatBoost not available; skipping CatBoost.")

# Targets loop
results = {}
for target_name, (y_tr, y_val, y_test) in {
    'Area': (y_area_train, y_area_val, y_area_test),
    'Power': (y_power_train, y_power_val, y_power_test),
    'Delay': (y_delay_train, y_delay_val, y_delay_test),
}.items():
    print(f"\n======== TARGET: {target_name} ========")
    results[target_name] = {}
    for model_name, model in model_suite:
        safe_model_name = model_name.replace(' ', '_')
        model_path = os.path.join(MODEL_DIR, f"{safe_model_name}_{target_name}.joblib")
        use_scaler = (model_name == 'MLP')
        try:
            res = train_and_eval_model(model_name, model, X_train, y_tr, X_val, y_val, X_test, y_test, model_path, use_scaler_for_nn=use_scaler)
            # update saved metadata with the specific target
            obj = joblib.load(model_path)
            obj['meta']['target'] = target_name
            joblib.dump(obj, model_path)
            results[target_name][model_name] = res
        except Exception as e:
            print(f"Error training {model_name} for {target_name}: {e}")
            traceback.print_exc()

# -------------- SUMMARY ----------------
print("\nFINAL SUMMARY")
for t, d in results.items():
    print(f"\n-- {t} --")
    for m, metrics in d.items():
        print(f"{m}: RMSE={metrics['rmse']:.6f} R2={metrics['r2']:.6f} time={metrics['train_time_s']:.1f}s")

print(f"Models saved to: {MODEL_DIR}")