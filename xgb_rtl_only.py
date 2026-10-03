"""#!/usr/bin/env python3
""
xgb_rtl_further_improved.py
Improved pipeline for RTL-only (and optional netlist) PPA prediction.

Features:
 - extended feature engineering (ratios, densities)
 - tree-based feature selection
 - randomized hyperparameter search
 - final refit with early stopping and SHAP analysis (if available)

Adjust CSV_PATH and MODEL_DIR variables.
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
CSV_PATH = r"C:\ml ppa\final_maximal_dataset.csv"
MODEL_DIR = r"C:\ml ppa\models_xgb_rtl_further"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
INCLUDE_NETLIST_FEATURES = False   # set True if you want to include netlist columns (if available)
USE_SHAP = True                    # set False if shap not installed or too slow
N_ITER_SEARCH = 30
CV_FOLDS = 3

# ---------- UTIL ----------
def safe_log1p(x):
    return np.log1p(np.maximum(x, 0))

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

# ---------- LOAD ----------
print("Loading dataset:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
print("Raw shape:", df.shape)

# ---------- FEATURE LIST ----------
id_cols = ["Design_Name", "RTL_Code"]
# candidate numeric columns (auto-detect)
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in id_cols]

# Optionally filter netlist features
if not INCLUDE_NETLIST_FEATURES:
    netlist_prefixes = ("netlist_",)
    num_cols = [c for c in num_cols if not c.startswith(netlist_prefixes)]

print("Numeric features considered:", len(num_cols))

# ---------- FEATURE ENGINEERING ----------
def add_composite_features(df_sub):
    dfc = df_sub.copy()
    # basic ratios and densities
    dfc["add_to_lines"] = dfc.get("num_add", 0) / (dfc.get("num_lines", 1))
    dfc["mul_to_lines"] = dfc.get("num_mul", 0) / (dfc.get("num_lines", 1))
    dfc["reg_to_lines"] = dfc.get("num_reg", 0) / (dfc.get("num_lines", 1))
    dfc["wire_to_regs"] = dfc.get("num_wire", 0) / (dfc.get("num_reg", 1))
    dfc["arithmetic_ops"] = dfc.get("num_add",0) + dfc.get("num_sub",0) + dfc.get("num_mul",0) + dfc.get("num_div",0)
    dfc["logic_ops"] = dfc.get("num_logic_and",0) + dfc.get("num_logic_or",0) + dfc.get("num_logic_xor",0)
    dfc["ops_per_bit"] = dfc["arithmetic_ops"] / np.maximum(dfc.get("total_bits", 1), 1)
    # mux density and branch density
    dfc["mux_density"] = dfc.get("num_ternary",0) / np.maximum(dfc.get("num_lines",1),1)
    dfc["branch_density"] = dfc.get("num_branches",0) / np.maximum(dfc.get("num_lines",1),1)
    # complexity proxies
    dfc["bitwidth_spread"] = dfc.get("max_bitwidth",1) - dfc.get("min_bitwidth",1)
    dfc["avg_bits_per_reg"] = dfc.get("avg_bitwidth",1) * dfc.get("num_reg",0)
    dfc["module_density"] = dfc.get("num_modules",1) / np.maximum(dfc.get("num_lines",1),1)
    # guarding divisions
    for col in ["add_to_lines","mul_to_lines","reg_to_lines","wire_to_regs","ops_per_bit","mux_density","branch_density","bitwidth_spread","avg_bits_per_reg","module_density"]:
        dfc[col] = dfc[col].replace([np.inf, -np.inf], 0).fillna(0)
    return dfc

df_num = df[num_cols].copy()
df_fe = add_composite_features(df_num)
print("After FE shape:", df_fe.shape)

# ---------- TARGET PREP ----------
# targets (same as before in your pipeline)
if "total_cell_area" in df.columns:
    area_col = "total_cell_area"
else:
    area_col = "comb_area" if "comb_area" in df.columns else None
power_col = "Power" if "Power" in df.columns else None
delay_col = "critical_path_length" if "critical_path_length" in df.columns else None

targets = {}
if area_col: targets["Area"] = area_col
if power_col: targets["Power"] = power_col
if delay_col: targets["Delay"] = delay_col

# filter rows that have ALL targets and features
needed_cols = list(df_fe.columns) + list(targets.values())
# Merge features + targets safely (no duplicate columns)
df_targets = df[list(targets.values())].copy()
df_model = pd.concat([df_fe.reset_index(drop=True), df_targets.reset_index(drop=True)], axis=1)

# Drop duplicate columns if any slipped in
df_model = df_model.loc[:, ~df_model.columns.duplicated()]

# Drop missing values
df_model = df_model.dropna()

print("After dropna shape:", df_model.shape)

# Robust outlier removal on targets (use median +/- 4*IQR per-target)
def remove_outliers(df_in, target_name):
    y = df_in[target_name]
    q1, q3 = np.percentile(y, [25,75])
    iqr = q3 - q1
    low = q1 - 4*iqr
    high = q3 + 4*iqr
    mask = (y >= low) & (y <= high)
    return df_in[mask]

# apply outlier removal iteratively across targets
for t in list(targets.values()):
    df_model = remove_outliers(df_model, t)
print("After outlier removal shape:", df_model.shape)

# ---------- FEATURE SELECTION ----------
X_all = df_model[df_fe.columns].values
feature_names = list(df_fe.columns)
print("Total candidate features:", len(feature_names))

# use a small RF to compute importances and drop extremely low importance features
sel_rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
sel_rf.fit(X_all, df_model[list(targets.values())[0]].values)  # use area as proxy for selection
importances = sel_rf.feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "imp": importances}).sort_values("imp", ascending=False)
# keep features with cumulative importance >= 0.995 or individual importance > 0.001
imp_df["cum"] = imp_df.imp.cumsum()
keep_mask = (imp_df.cum <= 0.995) | (imp_df.imp >= 0.001)
keep_features = imp_df[keep_mask]["feature"].tolist()
print(f"Selected {len(keep_features)} features from {len(feature_names)}")

X = df_model[keep_features].values

# ---------- SPLIT ----------
X_train_full, X_test, y_train_full_df, y_test_df = train_test_split(
    X, df_model[list(targets.values())], test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print("Train / Test sizes:", X_train_full.shape[0], X_test.shape[0])

# ---------- HYPERPARAM SEARCH helper ----------
def search_and_refit(X_train, y_train, X_val, y_val, target_name):
    print("\n=== Search for target:", target_name, "===")
    # use log transform for area & power because skewed
    use_log = target_name in ("Area","Power")
    y = y_train[target_name].values
    y_val_arr = y_val[target_name].values
    if use_log:
        y_search = safe_log1p(y)
        y_val_search = safe_log1p(y_val_arr)
    else:
        y_search = y
        y_val_search = y_val_arr

    # quick train/val split for early stopping during refill
    X_tr, X_hold, y_tr, y_hold = train_test_split(X_train, y_search, test_size=0.2, random_state=RANDOM_STATE)

    xgb = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1)

    param_dist = {
        "n_estimators": [200,400,600,800],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "max_depth": [6,8,10,12],
        "subsample": [0.5,0.6,0.7,0.8,0.9],
        "colsample_bytree": [0.6,0.7,0.8,0.9,1.0],
        "min_child_weight": [1,3,5,7],
        "reg_lambda": [0.1,0.5,1.0,2.0],
    }

    rnd = RandomizedSearchCV(
        xgb, param_distributions=param_dist, n_iter=min(N_ITER_SEARCH, 30),
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1, random_state=RANDOM_STATE, refit=True
    )

    rnd.fit(X_train, y_search)
    best = rnd.best_estimator_
    print("Best params (cv):", rnd.best_params_, "best cv score:", rnd.best_score_)

    # Refit best with early stopping using a holdout from training
    try:
        best.fit(X_tr, y_tr, eval_set=[(X_hold, y_hold)], eval_metric="rmse", early_stopping_rounds=50, verbose=False)
    except TypeError:
        best.fit(X_tr, y_tr)  # fallback if eval_metric not accepted

    # Evaluate on provided val set (original scale)
    y_pred_val = best.predict(X_val)
    if use_log:
        # predictions are log1p -> invert
        y_pred_val_orig = np.expm1(y_pred_val)
        y_true_orig = np.expm1(y_val_search)
    else:
        y_pred_val_orig = y_pred_val
        y_true_orig = y_val_arr

    val_rmse = rmse(y_true_orig, y_pred_val_orig)
    val_r2 = r2_score(y_true_orig, y_pred_val_orig)
    print(f"{target_name} validation RMSE: {val_rmse:.4f} R2: {val_r2:.4f}")

    return best, val_rmse, val_r2

# ---------- TRAIN FOR EACH TARGET ----------
results = {}
for tgt_label, tgt_col in targets.items():
    # prepare y series dataframe
    y_full = df_model[[tgt_col]].rename(columns={tgt_col: tgt_label})
    # split same as X split index (we earlier split into X_train_full / X_test)
    # create paired arrays for training/val used in search_and_refit
    # find corresponding rows in df_model for train/test split by using index alignment hack
    # We'll recompute split to get matching rows easily:
    X_full = X
    y_full_arr = y_full
    X_tr, X_te, y_tr_df, y_te_df = train_test_split(X_full, y_full_arr, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model, val_rmse, val_r2 = search_and_refit(X_tr, y_tr_df, X_te, y_te_df, tgt_label)
    # final evaluation on global test set (X_test we created earlier mapping to df_model list)
    y_test = y_test_df[tgt_col].values if tgt_col in y_test_df.columns else df_model.loc[y_te_df.index, tgt_col].values
    if tgt_label in ("Area","Power"):
        # we trained on log1p internally, but model.predict returns log1p predictions only if we trained on log1p -> need to transform
        try:
            y_pred_test_log = model.predict(X_test)
            y_pred_test = np.expm1(y_pred_test_log)
        except Exception:
            y_pred_test = model.predict(X_test)
    else:
        y_pred_test = model.predict(X_test)
    # compute test metrics (use df_model's test target)
    y_true_test = df_model.loc[df_model.index.isin(y_test_df.index), tgt_col] if tgt_col in df_model.columns else None
    # fallback to y_te_df if indexing issues
    if y_true_test is None or len(y_true_test)==0:
        y_true_test = y_te_df[tgt_label].values
    else:
        y_true_test = y_true_test.values
    test_rmse = rmse(y_true_test, y_pred_test[:len(y_true_test)])
    test_r2 = r2_score(y_true_test, y_pred_test[:len(y_true_test)])
    results[tgt_label] = {"model": model, "rmse": test_rmse, "r2": test_r2}
    # save model and feature importance
    model_path = os.path.join(MODEL_DIR, f"xgb_{tgt_label.lower()}_further.joblib")
    joblib.dump(model, model_path)
    print(f"Saved {tgt_label} model -> {model_path}")

    # feature importance CSV
    fi = getattr(model, "feature_importances_", None)
    if fi is not None:
        fi_df = pd.DataFrame({"feature": keep_features, "importance": fi}).sort_values("importance", ascending=False)
        fi_df.to_csv(os.path.join(MODEL_DIR, f"{tgt_label}_feat_imp.csv"), index=False)

    # optional SHAP
    if USE_SHAP:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test[:200])  # limit samples
            shap.summary_plot(shap_vals, pd.DataFrame(X_test[:200], columns=keep_features), show=False)
            shap_path = os.path.join(MODEL_DIR, f"{tgt_label}_shap_summary.png")
            import matplotlib.pyplot as plt
            plt.savefig(shap_path, bbox_inches='tight', dpi=150)
            plt.close()
            print("Saved SHAP summary plot:", shap_path)
        except Exception as e:
            print("SHAP failed or not installed:", e)

# ---------- SUMMARY ----------
print("\n=== SUMMARY ===")
for k,v in results.items():
    print(f"{k:6} -> RMSE: {v['rmse']:.4f}  R2: {v['r2']:.4f}")

print("Models + artifacts saved to:", MODEL_DIR)
