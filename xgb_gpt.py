"""#!/usr/bin/env python3
by gpt
Train XGBoost models for PPA prediction from final_maximal_dataset.csv
Robust fit() wrapper to handle xgboost version differences that may
reject eval_metric as a fit() kwarg.
"""

import os
import joblib
import numpy as np
import pandas as pd
import traceback

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------------- CONFIG ----------------
CSV_PATH = r"C:\ml ppa\final_maximal_dataset.csv"
MODEL_DIR = r"C:\ml ppa\models_xgb"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

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
]

print(f"Selected {len(feature_cols)} numeric features.")

# Prepare dataset
df_model = df[feature_cols + target_cols].copy()
before = len(df_model)
df_model = df_model.dropna()
after = len(df_model)
print(f"Rows before dropna: {before}  after: {after}")

X = df_model[feature_cols].values
y_area  = df_model[area_col].values
y_power = df_model[power_col].values
y_delay = df_model[delay_col].values

# consistent split indices for all targets
X_train, X_test, y_area_train, y_area_test = train_test_split(
    X, y_area, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
_, _, y_power_train, y_power_test = train_test_split(
    X, y_power, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
_, _, y_delay_train, y_delay_test = train_test_split(
    X, y_delay, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"Train size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")

# -------------- TRAIN/HELPER ----------------
def robust_fit(model, X_tr, y_tr, X_val=None, y_val=None, early_stopping_rounds=30):
    """
    Try modern fit() call with eval_metric; if xgboost version doesn't accept eval_metric,
    retry with fewer kwargs. Return fitted model and a string describing which method used.
    """
    # Candidate call signatures in order of preference
    attempts = [
        # modern: eval_metric + eval_set + early_stopping_rounds + verbose
        {"kwargs": {"eval_set": [(X_val, y_val)], "eval_metric": "rmse",
                    "early_stopping_rounds": early_stopping_rounds, "verbose": False}},
        # without eval_metric (some xgboost versions reject eval_metric in fit())
        {"kwargs": {"eval_set": [(X_val, y_val)], "early_stopping_rounds": early_stopping_rounds, "verbose": False}},
        # simplest: only X, y
        {"kwargs": {}},
    ]

    last_exc = None
    for i, attempt in enumerate(attempts, 1):
        try:
            model.fit(X_tr, y_tr, **attempt["kwargs"])
            return model, f"attempt_{i}_ok"
        except TypeError as e:
            # often indicates unexpected kwarg in this xgboost version
            last_exc = e
        except Exception as e:
            # for any other exception, keep trying but log
            last_exc = e
    # if all attempts fail, re-raise last exception with traceback
    tb = traceback.format_exc()
    raise RuntimeError(f"All fit() attempts failed. Last exception: {last_exc}\nTrace:\n{tb}")

def train_one_xgb(X_tr, y_tr, X_te, y_te, label_name, model_path):
    print(f"\n--- Training {label_name} ---")
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",  # fast; if unsupported, XGBoost falls back internally
    )

    model, method = robust_fit(model, X_tr, y_tr, X_te, y_te, early_stopping_rounds=30)
    print(f"fit() completed using: {method}")

    y_pred = model.predict(X_te)
    rmse = mean_squared_error(y_te, y_pred, squared=False)
    r2 = r2_score(y_te, y_pred)

    print(f"{label_name} RMSE: {rmse:.6f}  R2: {r2:.6f}")

    joblib.dump(model, model_path)
    print(f"Saved {label_name} model -> {model_path}")

    return model, rmse, r2

# --------------- TRAIN MODELS ----------------
area_model_path  = os.path.join(MODEL_DIR, "xgb_area.joblib")
power_model_path = os.path.join(MODEL_DIR, "xgb_power.joblib")
delay_model_path = os.path.join(MODEL_DIR, "xgb_delay.joblib")

area_model, area_rmse, area_r2 = train_one_xgb(
    X_train, y_area_train, X_test, y_area_test, "Area", area_model_path
)
power_model, power_rmse, power_r2 = train_one_xgb(
    X_train, y_power_train, X_test, y_power_test, "Power", power_model_path
)
delay_model, delay_rmse, delay_r2 = train_one_xgb(
    X_train, y_delay_train, X_test, y_delay_test, "Delay", delay_model_path
)

print("\nSUMMARY")
print(f"Area  RMSE={area_rmse:.6f} R2={area_r2:.6f}")
print(f"Power RMSE={power_rmse:.6f} R2={power_r2:.6f}")
print(f"Delay RMSE={delay_rmse:.6f} R2={delay_r2:.6f}")
print(f"Models saved to: {MODEL_DIR}")
