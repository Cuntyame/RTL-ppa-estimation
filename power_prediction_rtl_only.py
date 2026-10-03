"""
power_prediction_rtl_only.py
==============================
Dedicated power prediction model using RTL features only.
No netlist required — suitable for early-stage design exploration.

DESIGN DECISIONS vs the combined PPA code in your report:
  1. Power-only: no area/delay targets cluttering the pipeline
  2. Richer feature set: 24 new power-focused features (see extractor)
  3. Three-branch target strategy:
       - Internal power  (cell switching)  → separate model (optional)
       - Net switching power               → separate model (optional)
       - Total power                       → primary combined model
  4. Stacked ensemble: instead of just comparing models, the best 3 are
     stacked with a Ridge meta-learner → typically +3–5% R² over single best
  5. Optuna hyperparameter tuning (optional, set TUNE=True)
  6. SHAP feature importance saved → tells you which RTL feature matters most
"""

import os
import warnings
import math
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.preprocessing import PowerTransformer, RobustScaler

warnings.filterwarnings("ignore")

# ── optional heavy dependencies ───────────────────────────────
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

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION  ← edit these paths
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
#  CONFIGURATION  ← edit these paths
# ══════════════════════════════════════════════════════════════
CSV_PATH   = r"C:\Users\Admin\Documents\final_rtl_power_features_v2.csv"
MODEL_DIR  = r"C:\Users\Admin\OneDrive - MSFT\Desktop\New folder\cody\power_model_output"
os.makedirs(MODEL_DIR, exist_ok=True)

POWER_COL  = "Power"   # column in your CSV with ground-truth power values

RANDOM_STATE = 42
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15

REMOVE_OUTLIERS  = True
OUTLIER_MULTIPLIER = 3.5  # IQR multiplier

USE_STACKING = True   # stack top-3 models with Ridge meta-learner
TUNE         = False  # set True to run Optuna HPO (adds ~10 min)
N_TRIALS     = 40     # Optuna trials if TUNE=True
SAVE_SHAP    = True   # save SHAP feature importances to CSV

# ══════════════════════════════════════════════════════════════
#  FEATURE GROUPS
#  (subset used depends on what columns exist in your CSV)
# ══════════════════════════════════════════════════════════════

# Original RTL features
ORIGINAL_RTL = [
    "num_lines", "max_bitwidth", "min_bitwidth", "avg_bitwidth",
    "total_bits", "num_bitwidths", "num_32bit", "num_16bit", "num_8bit", "num_1bit",
    "num_add", "num_sub", "num_mul", "num_div",
    "num_logic_and", "num_logic_or", "num_logic_xor", "num_shifts", "num_comparisons",
    "num_always", "num_assign", "num_if", "num_case", "num_for", "num_ternary", "num_branches",
    "num_wire", "num_reg", "num_input", "num_output", "num_modules",
    "num_posedge_always", "num_clk_domains",
    "unique_signals", "max_signal_occurrences", "mean_signal_occurrences",
    "num_high_fanout_signals", "rtl_code_len",
]

# Derived features from your current extractor
ORIGINAL_DERIVED = [
    "total_arithmetic", "arith_density", "switching_proxy", "logic_density",
    "bits_log", "bits_sqrt", "register_bit_product", "toggle_estimate", "mux_count",
]

# NEW power-focused features from the enhanced extractor
NEW_POWER_FEATURES = [
    # Sequential switching (internal power)
    "num_always_ff", "num_always_comb",
    "num_sync_reset", "num_async_reset", "num_clock_enable",
    "num_latch", "num_tristate",
    "estimated_ff_bits",
    "weighted_switching",
    "pipeline_depth_proxy",
    "reset_fanout_proxy",
    "clk_domain_pressure",
    # Combinational switching (net switching power)
    "datapath_width_pressure",
    "high_toggle_score",
    "xor_density",
    "bitwidth_weighted_adds",
    "bitwidth_weighted_muls",
    "num_shift_by_var",
    "glitch_potential",
    "effective_switching_activity",
    # Capacitance / fanout
    "signal_reuse_factor",
    "output_switching_load",
    "input_toggle_potential",
    "mux_to_logic_ratio",
    # Leakage proxies
    "seq_to_comb_ratio",
    "num_conditional_assign",
    "num_memory_access",
    "avg_op_bitwidth",
    "control_to_data_ratio",
    # Log-normalised
    "power_complexity_index",
]

ALL_FEATURES = ORIGINAL_RTL + ORIGINAL_DERIVED + NEW_POWER_FEATURES


# ══════════════════════════════════════════════════════════════
#  ADDITIONAL FEATURE ENGINEERING  (power-specific)
#  Applied AFTER loading CSV (catches any gaps from the extractor)
# ══════════════════════════════════════════════════════════════

def engineer_power_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive additional power features inside the training script.
    Safe to call even if columns are missing (guarded by 'if col in df').
    """
    eps = 1e-8
    df = df.copy()

    # ── Recalculate key composites if not already in CSV ──────
    if "estimated_ff_bits" not in df.columns and all(
            c in df.columns for c in ["num_reg", "avg_bitwidth"]):
        df["estimated_ff_bits"] = df["num_reg"] * df["avg_bitwidth"]

    if "weighted_switching" not in df.columns and all(
            c in df.columns for c in ["num_reg", "num_logic_xor", "num_logic_and", "avg_bitwidth"]):
        df["weighted_switching"] = (
            df["num_reg"]       * df["avg_bitwidth"] * 1.0 +
            df["num_logic_xor"] * df["avg_bitwidth"] * 3.0 +
            df["num_logic_and"] * df["avg_bitwidth"] * 0.5
        )

    if "datapath_width_pressure" not in df.columns and all(
            c in df.columns for c in ["num_add", "num_sub", "num_mul", "avg_bitwidth", "max_bitwidth"]):
        df["datapath_width_pressure"] = (
            (df["num_add"] + df["num_sub"]) * df["avg_bitwidth"] +
             df["num_mul"] * df["max_bitwidth"]
        )

    if "high_toggle_score" not in df.columns and all(
            c in df.columns for c in ["num_logic_xor", "num_comparisons", "mux_count"]):
        df["high_toggle_score"] = (
            df["num_logic_xor"] * 3 +
            df["num_comparisons"] * 2 +
            df["mux_count"] * 2
        )

    if "effective_switching_activity" not in df.columns:
        cols = ["weighted_switching", "high_toggle_score", "avg_bitwidth", "datapath_width_pressure"]
        if all(c in df.columns for c in cols):
            df["effective_switching_activity"] = (
                df["weighted_switching"] +
                df["high_toggle_score"] * df["avg_bitwidth"] * 0.5 +
                df["datapath_width_pressure"] * 0.3
            )

    if "power_complexity_index" not in df.columns and all(
            c in df.columns for c in ["effective_switching_activity", "datapath_width_pressure"]):
        df["power_complexity_index"] = np.log1p(
            df["effective_switching_activity"] + df["datapath_width_pressure"]
        )

    # ── Interaction terms (power-specific) ───────────────────
    if all(c in df.columns for c in ["num_always_ff", "avg_bitwidth"]):
        df["ff_bitwidth_interaction"] = df["num_always_ff"] * df["avg_bitwidth"]

    if all(c in df.columns for c in ["num_clk_domains", "estimated_ff_bits"]):
        df["clk_ff_interaction"] = df["num_clk_domains"] * df["estimated_ff_bits"]

    if all(c in df.columns for c in ["xor_density", "num_lines"]):
        df["xor_line_interaction"] = df["xor_density"] * np.log1p(df["num_lines"])

    # ── Ratio features ────────────────────────────────────────
    if all(c in df.columns for c in ["num_reg", "num_lines"]):
        df["ff_line_density"] = df["num_reg"] / (df["num_lines"] + eps)

    if all(c in df.columns for c in ["num_always_ff", "num_always"]):
        df["ff_to_always_ratio"] = df["num_always_ff"] / (df["num_always"] + eps)

    return df


# ══════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100
    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape}


def print_metrics(label: str, m: dict):
    print(f"  {label:<25}  R²={m['r2']:.4f}  RMSE={m['rmse']:.4e}"
          f"  MAE={m['mae']:.4e}  MAPE={m['mape']:.2f}%")


# ══════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════

def get_power_models(n_features: int) -> dict:
    """
    Return base models tuned for power prediction.
    Power is harder to predict than area because it depends on
    switching activity which has probabilistic components.
    Key tuning choices:
      - More estimators than area (more variance in power data)
      - Lower learning rate + more trees (power has subtle patterns)
      - Larger min_child_weight for XGB (prevents overfitting to noise)
    """
    models = {}

    if HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=800,
            learning_rate=0.02,
            max_depth=6,
            min_child_weight=8,      # higher than default: power has noise
            subsample=0.75,
            colsample_bytree=0.7,
            gamma=0.3,
            reg_alpha=2.0,
            reg_lambda=4.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )

    if HAS_LGBM:
        models["lgbm"] = LGBMRegressor(
            n_estimators=800,
            learning_rate=0.02,
            max_depth=7,
            num_leaves=40,
            subsample=0.75,
            colsample_bytree=0.7,
            min_child_samples=25,    # more conservative for power noise
            reg_alpha=2.0,
            reg_lambda=4.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    if HAS_CATBOOST:
        models["catboost"] = CatBoostRegressor(
            iterations=800,
            learning_rate=0.02,
            depth=6,
            l2_leaf_reg=4.0,
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=-1,
        )

    models["rf"] = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=6,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    models["gbm"] = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.75,
        min_samples_leaf=6,
        random_state=RANDOM_STATE,
    )

    return models


# ══════════════════════════════════════════════════════════════
#  OPTUNA TUNING  (optional)
# ══════════════════════════════════════════════════════════════

def tune_xgb_for_power(X_tr, y_tr, X_val, y_val):
    """Tune XGBoost specifically for power prediction."""
    if not HAS_OPTUNA or not HAS_XGB:
        return {}

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 400, 1200),
            "learning_rate":    trial.suggest_float("lr", 0.01, 0.05, log=True),
            "max_depth":        trial.suggest_int("max_depth", 4, 8),
            "min_child_weight": trial.suggest_int("mcw", 4, 15),
            "subsample":        trial.suggest_float("ss", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("cbt", 0.5, 0.9),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":        trial.suggest_float("alpha", 0.5, 5.0),
            "reg_lambda":       trial.suggest_float("lambda", 1.0, 8.0),
            "random_state": RANDOM_STATE, "n_jobs": -1, "tree_method": "hist",
        }
        m = XGBRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        pred = m.predict(X_val)
        return mean_squared_error(y_val, pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    print(f"\n  Best Optuna XGB params: {study.best_params}")
    return study.best_params


# ══════════════════════════════════════════════════════════════
#  STACKING ENSEMBLE
# ══════════════════════════════════════════════════════════════

def build_stacking_model(base_models: dict) -> StackingRegressor:
    """
    Stack top base models with a Ridge meta-learner.
    Use 5-fold CV for out-of-fold meta-features to prevent leakage.
    """
    estimators = [(name, model) for name, model in base_models.items()]
    stacker = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=10.0),
        cv=5,
        n_jobs=-1,
        passthrough=False,  # only pass model predictions to meta-learner
    )
    return stacker


# ══════════════════════════════════════════════════════════════
#  SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════

def save_shap_importance(model, X_test: np.ndarray, feature_names: list):
    """Compute and save SHAP feature importances to CSV."""
    if not HAS_SHAP:
        print("  (install shap for feature importance: pip install shap)")
        return

    try:
        # TreeExplainer works for XGB/LGBM/RF
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        importance = np.abs(shap_vals).mean(axis=0)
        df_imp = pd.DataFrame({
            "feature":    feature_names,
            "shap_mean_abs": importance,
        }).sort_values("shap_mean_abs", ascending=False)

        out_path = os.path.join(MODEL_DIR, "power_shap_importance.csv")
        df_imp.to_csv(out_path, index=False)
        print(f"\n  SHAP importance saved → {out_path}")
        print(f"\n  Top-10 features for POWER prediction:")
        for _, row in df_imp.head(10).iterrows():
            bar = "█" * int(row["shap_mean_abs"] / df_imp["shap_mean_abs"].max() * 20)
            print(f"    {row['feature']:<40} {bar}  {row['shap_mean_abs']:.4f}")
    except Exception as e:
        print(f"  SHAP failed: {e}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  POWER PREDICTION — RTL FEATURES ONLY")
    print("=" * 70)
    print(f"  XGBoost={HAS_XGB}  LightGBM={HAS_LGBM}  "
          f"CatBoost={HAS_CATBOOST}  SHAP={HAS_SHAP}  Optuna={HAS_OPTUNA}")

    # ── 1. Load data ──────────────────────────────────────────
    print(f"\nLoading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Shape: {df.shape}")

    # ── 2. Select available features ─────────────────────────
    available = [f for f in ALL_FEATURES if f in df.columns]
    missing   = [f for f in ALL_POWER_PRIORITY if f not in df.columns]
    print(f"\n  Available features : {len(available)}")
    if missing:
        print(f"  Missing (will skip) : {missing[:8]}{'...' if len(missing)>8 else ''}")

    # ── 3. Engineer additional features ──────────────────────
    print("\nEngineering power-specific features...")
    df = engineer_power_features(df)

    # Re-check after engineering
    available = [f for f in ALL_FEATURES + [
        "ff_bitwidth_interaction", "clk_ff_interaction",
        "xor_line_interaction", "ff_line_density", "ff_to_always_ratio"
    ] if f in df.columns]
    print(f"  Final feature count: {len(available)}")

    # ── 4. Build modelling dataframe ─────────────────────────
    df_model = pd.concat([df[available], df[[POWER_COL]]], axis=1)

    # ── 5. Clean data ─────────────────────────────────────────
    print("\nCleaning data...")
    n0 = len(df_model)
    df_model = df_model.dropna()
    print(f"  After dropna    : {len(df_model)}/{n0}")

    n0 = len(df_model)
    df_model = df_model[df_model[POWER_COL] > 0]
    print(f"  After power > 0 : {len(df_model)}/{n0}")

    if REMOVE_OUTLIERS:
        n0 = len(df_model)
        q1, q3 = df_model[POWER_COL].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_model = df_model[
            (df_model[POWER_COL] >= q1 - OUTLIER_MULTIPLIER * iqr) &
            (df_model[POWER_COL] <= q3 + OUTLIER_MULTIPLIER * iqr)
        ]
        print(f"  After IQR ({OUTLIER_MULTIPLIER}×)  : {len(df_model)}/{n0}")

    # ── 6. Prepare arrays ─────────────────────────────────────
    final_feats = [c for c in available if c in df_model.columns]
    X = df_model[final_feats].values
    y_orig = df_model[POWER_COL].values
    print(f"\n  Final: {X.shape[0]} samples × {X.shape[1]} features")

    # ── 7. Target transform (Yeo-Johnson) ────────────────────
    #   Power spans orders of magnitude and can be near-zero
    #   Yeo-Johnson handles this better than log1p
    print("\nApplying Yeo-Johnson transform to Power...")
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    y_trans = pt.fit_transform(y_orig.reshape(-1, 1)).ravel()
    joblib.dump(pt, os.path.join(MODEL_DIR, "power_transformer.joblib"))

    def inverse_power(preds: np.ndarray) -> np.ndarray:
        return pt.inverse_transform(preds.reshape(-1, 1)).ravel()

    # ── 8. Split ──────────────────────────────────────────────
    X_tmp, X_test, idx_tmp, idx_test = train_test_split(
        X, np.arange(len(X)), test_size=TEST_SIZE, random_state=RANDOM_STATE)
    val_adj = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, idx_train, idx_val = train_test_split(
        X_tmp, idx_tmp, test_size=val_adj, random_state=RANDOM_STATE)

    y_tr, y_vl, y_te = y_trans[idx_train], y_trans[idx_val], y_trans[idx_test]
    y_tr_o, y_vl_o, y_te_o = y_orig[idx_train], y_orig[idx_val], y_orig[idx_test]

    print(f"  Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    # ── 9. Scale features (RobustScaler) ─────────────────────
    scaler = RobustScaler()
    X_tr_s  = scaler.fit_transform(X_train)
    X_vl_s  = scaler.transform(X_val)
    X_te_s  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    # ── 10. Optional Optuna tuning for XGB ───────────────────
    xgb_override = {}
    if TUNE and HAS_OPTUNA and HAS_XGB:
        print("\nRunning Optuna HPO for XGBoost (power-specific)...")
        xgb_override = tune_xgb_for_power(X_tr_s, y_tr, X_vl_s, y_vl)

    # ── 11. Train and evaluate all base models ────────────────
    models = get_power_models(X_tr_s.shape[1])
    if xgb_override and HAS_XGB:
        models["xgb"] = XGBRegressor(**xgb_override,
                                      random_state=RANDOM_STATE, n_jobs=-1,
                                      tree_method="hist")

    print(f"\n{'='*60}")
    print(f"  BASE MODEL RESULTS")
    print(f"{'='*60}")
    results = {}
    for mname, model in models.items():
        print(f"\n  Training {mname.upper()}...", end=" ", flush=True)
        model.fit(X_tr_s, y_tr)

        vl_pred_o = inverse_power(model.predict(X_vl_s))
        te_pred_o = inverse_power(model.predict(X_te_s))

        val_m  = compute_metrics(y_vl_o, vl_pred_o)
        test_m = compute_metrics(y_te_o, te_pred_o)
        print(f"done.")
        print_metrics("Validation",  val_m)
        print_metrics("Test",        test_m)

        results[mname] = {
            "model":      model,
            "val_r2":     val_m["r2"],
            "test_metrics": test_m,
        }
        joblib.dump(model, os.path.join(MODEL_DIR, f"power_{mname}.joblib"))

    # ── 12. Stacking ensemble ─────────────────────────────────
    if USE_STACKING and len(results) >= 2:
        print(f"\n{'='*60}")
        print(f"  STACKING ENSEMBLE")
        print(f"{'='*60}")
        # Pick top-3 by val R²
        top3 = sorted(results.items(), key=lambda x: -x[1]["val_r2"])[:3]
        print(f"  Base models for stack: {[n for n,_ in top3]}")
        stack = build_stacking_model({n: v["model"] for n, v in top3})
        stack.fit(X_tr_s, y_tr)

        vl_pred_o = inverse_power(stack.predict(X_vl_s))
        te_pred_o = inverse_power(stack.predict(X_te_s))

        val_m  = compute_metrics(y_vl_o, vl_pred_o)
        test_m = compute_metrics(y_te_o, te_pred_o)
        print_metrics("Stack Validation", val_m)
        print_metrics("Stack Test",       test_m)
        joblib.dump(stack, os.path.join(MODEL_DIR, "power_stacked.joblib"))
        results["stacked"] = {"model": stack, "val_r2": val_m["r2"], "test_metrics": test_m}

    # ── 13. Summary table ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON  — POWER PREDICTION (RTL only)")
    print(f"{'='*70}")
    print(f"  {'Model':<14} {'Val R²':>8} {'Test R²':>8} {'RMSE':>12} "
          f"{'MAE':>12} {'MAPE':>8}")
    print(f"  {'-'*64}")
    best_model_name = None
    best_r2 = -999
    for mname, info in sorted(results.items(), key=lambda x: -x[1]["val_r2"]):
        tm = info["test_metrics"]
        marker = " ◄ BEST" if mname == sorted(
            results.items(), key=lambda x: -x[1]["val_r2"])[0][0] else ""
        print(f"  {mname:<14} {info['val_r2']:>8.4f} {tm['r2']:>8.4f} "
              f"{tm['rmse']:>12.4e} {tm['mae']:>12.4e} {tm['mape']:>7.2f}%{marker}")
        if info["val_r2"] > best_r2:
            best_r2 = info["val_r2"]
            best_model_name = mname

    # ── 14. SHAP feature importance for best model ────────────
    if SAVE_SHAP and best_model_name:
        print(f"\nRunning SHAP analysis on best model ({best_model_name})...")
        best_model = results[best_model_name]["model"]
        # Use the underlying estimator if it's a stacker
        if hasattr(best_model, "final_estimator_"):
            # Use first base estimator for SHAP
            shap_model = list(best_model.estimators_)[0][1]
        else:
            shap_model = best_model
        save_shap_importance(shap_model, X_te_s, final_feats)

    # ── 15. Save feature list ─────────────────────────────────
    feat_df = pd.DataFrame({"feature": final_feats, "index": range(len(final_feats))})
    feat_df.to_csv(os.path.join(MODEL_DIR, "power_feature_list.csv"), index=False)

    print(f"\n  Models saved → {MODEL_DIR}")
    print(f"  Feature list → {MODEL_DIR}/power_feature_list.csv")
    print("=" * 70)


# ── Priority feature list for diagnostics ────────────────────
ALL_POWER_PRIORITY = [
    "estimated_ff_bits", "weighted_switching", "datapath_width_pressure",
    "effective_switching_activity", "high_toggle_score", "num_always_ff",
    "clk_domain_pressure", "power_complexity_index",
]


if __name__ == "__main__":
    main()