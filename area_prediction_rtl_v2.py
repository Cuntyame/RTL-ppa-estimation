#!/usr/bin/env python3
"""
area_prediction_rtl_v2.py
==========================
Physics-grounded RTL-only area prediction.

TEAMMATE COMPARISON (from screenshots):
  Area:  CV R²=0.60  → test R²=0.06/0.15/0.24  (gap = 0.36-0.54, severe overfit)
  Delay: CV R²=0.42  → test R²=0.20/0.45/0.62
  Power: CV R²=0.22  → test R²=0.03/0.07/0.11

WHY TEAMMATE'S AREA COLLAPSED:
  1. Random (unstratified) split → similar designs in both train/test
  2. No correlation pruning → 61 features, many near-duplicates
     ('add' and 'sub' both count '+' and '-', collinear)
  3. log1p transform, not log10 → unequal decade weighting in loss
  4. Same generic features for all 3 targets — area needs
     quadratic multiplier terms, not switching proxies

AREA PHYSICS (sky130 standard cell library):
  DFF      : ~8 GE per bit    → seq_ge = num_reg × avg_bw × 8
  Adder    : ~2N GE           → adder_ge = (add+sub) × avg_bw × 2
  Multiplier: ~N² GE          → mul_ge = num_mul × max_bw²  ← dominant
  Divider  : ~4N² GE          → div_ge = num_div × max_bw² × 4
  Comparator: ~2N GE          → comp_ge = num_comp × avg_bw × 2
  XOR2     : ~2 GE            → small but numerous
  MUX2     : ~3 GE            → mux_ge = mux_count × avg_bw × 3
"""

import os, warnings, math, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (train_test_split, KFold,
                                      cross_val_score, cross_val_predict)
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor;      HAS_XGB = True
except ImportError:                         HAS_XGB = False
try:
    from lightgbm import LGBMRegressor;    HAS_LGBM = True
except ImportError:                         HAS_LGBM = False
try:
    from catboost import CatBoostRegressor; HAS_CATBOOST = True
except ImportError:                         HAS_CATBOOST = False
try:
    import shap;                            HAS_SHAP = True
except ImportError:                         HAS_SHAP = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════
CSV_PATH  = r"C:\Users\Admin\Documents\final_rtl_power_features_v2.csv"
MODEL_DIR = r"C:\Users\Admin\OneDrive - MSFT\Desktop\New folder\cody\area_model_v2"
os.makedirs(MODEL_DIR, exist_ok=True)

AREA_COL_CANDIDATES = ["total_cell_area", "comb_area", "Cell_Area", "area"]
RANDOM_STATE = 42
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15
TUNE         = False
N_TRIALS     = 60
CORR_THRESH  = 0.97
EPS          = 1e-9

NON_FEATURE_COLS = {
    "Design_Name", "RTL_Code", "rpt_files", "rpt_text_len",
    "Power", "critical_path_length", "levels_of_logic",
    "wns", "tns", "num_cells", "num_nets",
    "total_cell_area", "comb_area", "Cell_Area", "area",
}


# ══════════════════════════════════════════════════════════════
#  AREA-SPECIFIC FEATURE ENGINEERING
#  40 features — each has a direct standard-cell physics basis
# ══════════════════════════════════════════════════════════════

def engineer_area_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def col(name, default=0):
        if name in df.columns:
            return df[name].fillna(default).astype(float)
        return pd.Series(float(default), index=df.index)

    avg_bw = col("avg_bitwidth",  1.0).clip(lower=1)
    max_bw = col("max_bitwidth",  1.0).clip(lower=1)
    min_bw = col("min_bitwidth",  1.0).clip(lower=1)
    n_reg  = col("num_reg")
    n_add  = col("num_add")
    n_sub  = col("num_sub")
    n_mul  = col("num_mul")
    n_div  = col("num_div")
    n_xor  = col("num_logic_xor")
    n_and  = col("num_logic_and")
    n_or   = col("num_logic_or")
    n_mux  = col("num_ternary") + col("num_case")
    n_comp = col("num_comparisons")
    n_shift= col("num_shifts")
    n_lines= col("num_lines", 1).clip(lower=1)
    n_inp  = col("num_input")
    n_out  = col("num_output")
    n_mod  = col("num_modules", 1).clip(lower=1)
    n_if   = col("num_if")
    n_for  = col("num_for")
    n_mem  = col("num_memory_access", 0)
    n_wire = col("num_wire")

    # ── A. SEQUENTIAL AREA  (DFF = 8 GE/bit) ─────────────────
    df["seq_ge"]          = n_reg * avg_bw * 8
    df["log_seq_ge"]      = np.log1p(df["seq_ge"])
    df["sqrt_seq_ge"]     = np.sqrt(df["seq_ge"].clip(0))
    df["seq_ge_per_line"] = df["seq_ge"] / n_lines

    # ── B. MULTIPLIER AREA  (N² GE — quadratic!) ─────────────
    # This is the single most important area feature.
    # A 32-bit multiplier = 1024 GE vs 32-bit adder = 64 GE (16× diff).
    # Teammate used raw num_mul count — completely wrong scaling.
    df["mul_ge_quad"]      = n_mul * max_bw * max_bw
    df["mul_ge_nlogn"]     = n_mul * avg_bw * np.log2(avg_bw.clip(lower=2))
    df["log_mul_ge_quad"]  = np.log1p(df["mul_ge_quad"])
    df["sqrt_mul_ge_quad"] = np.sqrt(df["mul_ge_quad"].clip(0))

    # ── C. ADDER / SUBTRACTOR AREA  (2N GE linear) ────────────
    df["adder_ge"]         = (n_add + n_sub) * avg_bw * 2
    df["adder_ge_nlogn"]   = (n_add + n_sub) * avg_bw * np.log2(avg_bw.clip(lower=2))
    df["log_adder_ge"]     = np.log1p(df["adder_ge"])

    # ── D. DIVIDER AREA  (4N² GE — even larger than mul) ──────
    df["div_ge"]           = n_div * max_bw * max_bw * 4
    df["log_div_ge"]       = np.log1p(df["div_ge"])

    # ── E. COMPARATOR AREA  (2N GE) ───────────────────────────
    df["comp_ge"]          = n_comp * avg_bw * 2
    df["log_comp_ge"]      = np.log1p(df["comp_ge"])

    # ── F. BITWISE LOGIC  (XOR=2GE, AND/OR=1GE each) ─────────
    df["logic_ge"]         = (n_xor * 2 + n_and + n_or) * avg_bw
    df["xor_ge"]           = n_xor * 2 * avg_bw
    df["log_logic_ge"]     = np.log1p(df["logic_ge"])

    # ── G. MUX / DECODER AREA  (MUX2=3GE, decoder=log2) ──────
    df["mux_ge"]           = n_mux * avg_bw * 3
    df["decoder_ge"]       = n_if * np.log2((n_if + 1).clip(lower=1)) * avg_bw
    df["log_mux_dec_ge"]   = np.log1p(df["mux_ge"] + df["decoder_ge"])

    # ── H. BARREL SHIFTER  (N×log2N MUX stages) ──────────────
    df["shift_ge"]         = n_shift * avg_bw * np.log2(avg_bw.clip(lower=2))
    df["log_shift_ge"]     = np.log1p(df["shift_ge"])

    # ── I. MEMORY / REG-FILE AREA  (6 GE/bit for reg file) ────
    df["mem_ge"]           = n_mem * max_bw * 6
    df["log_mem_ge"]       = np.log1p(df["mem_ge"])

    # ── J. TOTAL ESTIMATED GATE COUNT  (master predictor) ─────
    df["total_ge"]         = (
        df["seq_ge"]      + df["mul_ge_quad"] + df["adder_ge"] +
        df["div_ge"]      + df["comp_ge"]     + df["logic_ge"] +
        df["mux_ge"]      + df["shift_ge"]    + df["mem_ge"]
    )
    df["log_total_ge"]     = np.log1p(df["total_ge"])
    df["sqrt_total_ge"]    = np.sqrt(df["total_ge"].clip(0))

    # ── K. AREA COMPOSITION FRACTIONS (scale-invariant) ───────
    # These distinguish "register-heavy" vs "multiplier-heavy" designs
    # and are invariant to overall design size.
    ge_safe = df["total_ge"].clip(lower=EPS)
    df["seq_area_frac"]    = df["seq_ge"]      / ge_safe
    df["mul_area_frac"]    = df["mul_ge_quad"] / ge_safe
    df["logic_area_frac"]  = df["logic_ge"]    / ge_safe
    df["mux_area_frac"]    = df["mux_ge"]      / ge_safe
    df["mem_area_frac"]    = df["mem_ge"]      / ge_safe
    df["arith_area_frac"]  = (df["adder_ge"] + df["mul_ge_quad"] +
                               df["div_ge"])   / ge_safe

    # ── L. BITWIDTH STRUCTURE ─────────────────────────────────
    df["bw_range"]         = max_bw - min_bw
    df["bw_uniformity"]    = avg_bw / max_bw.clip(lower=EPS)
    df["log_total_bits"]   = np.log1p(col("total_bits", 0))
    df["sqrt_total_bits"]  = np.sqrt(col("total_bits", 0).clip(0))

    # ── M. INTERACTION TERMS ──────────────────────────────────
    # Pipelined multiplier: num_mul × num_reg × bitwidth
    df["mul_x_reg_bw"]     = n_mul * n_reg * avg_bw
    df["log_mul_x_reg"]    = np.log1p(df["mul_x_reg_bw"])
    # Arithmetic depth × bitwidth
    df["arith_x_bw"]       = col("total_arithmetic", 0) * avg_bw
    df["log_arith_x_bw"]   = np.log1p(df["arith_x_bw"])
    # For loop unrolling multiplier (common source of area blowup)
    df["unroll_proxy"]     = n_for * avg_bw * 4
    df["log_unroll"]       = np.log1p(df["unroll_proxy"])
    # Hierarchy: module interfaces add port buffers
    df["hierarchy_ge"]     = n_mod * (n_inp + n_out) * avg_bw
    df["log_hierarchy"]    = np.log1p(df["hierarchy_ge"])

    # ── N. DENSITY FEATURES (per-line normalised) ─────────────
    df["ge_per_line"]      = df["total_ge"]    / n_lines
    df["mul_ge_per_line"]  = df["mul_ge_quad"] / n_lines
    df["seq_ge_per_line"]  = df["seq_ge"]      / n_lines
    df["arith_bw_density"] = col("total_arithmetic", 0) * avg_bw / n_lines

    return df


# ══════════════════════════════════════════════════════════════
#  FEATURE PRUNING
# ══════════════════════════════════════════════════════════════

def prune_features(X: pd.DataFrame, corr_thresh=CORR_THRESH) -> list:
    sel  = VarianceThreshold(threshold=1e-6)
    sel.fit(X)
    keep = X.columns[sel.get_support()].tolist()
    n0   = len(X.columns)

    corr    = X[keep].corr().abs()
    upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for c in upper.columns:
        for cc in upper.index[upper[c] > corr_thresh].tolist():
            if X[c].var() >= X[cc].var():
                to_drop.add(cc)
            else:
                to_drop.add(c)
    keep = [c for c in keep if c not in to_drop]
    print(f"  Pruned: {n0} → {len(keep)} "
          f"(removed {n0-len(keep)} correlated/zero-var)")
    return keep


# ══════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════

def metrics(y_true, y_pred) -> dict:
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) /
                           np.clip(np.abs(y_true), EPS, None))) * 100
    log_err = np.abs(np.log10(np.clip(y_true, EPS, None)) -
                     np.log10(np.clip(y_pred, EPS, None)))
    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape,
            "mean_log10_err": float(log_err.mean()),
            "median_log10_err": float(np.median(log_err))}


def print_metrics(label, m, w=32):
    print(f"  {label:<{w}} R²={m['r2']:.4f}  MAPE={m['mape']:.1f}%  "
          f"log10_err(mean={m['mean_log10_err']:.3f} "
          f"median={m['median_log10_err']:.3f})")


def per_decile_table(y_true, y_pred, label=""):
    df = pd.DataFrame({"t": y_true, "p": y_pred})
    df["dec"] = pd.qcut(df["t"], q=10, labels=False, duplicates="drop")
    print(f"\n  Per-decile results ({label}):")
    print(f"  {'D':<4} {'Area (µm²)':>24} {'MAPE':>8} {'log10err':>10} {'N':>5}")
    print(f"  {'-'*56}")
    for d, g in df.groupby("dec"):
        mape = np.mean(np.abs((g["t"]-g["p"]) / g["t"].clip(lower=EPS))) * 100
        lerr = np.mean(np.abs(np.log10(g["t"].clip(lower=EPS)) -
                               np.log10(g["p"].clip(lower=EPS))))
        print(f"  {int(d):<4} {g['t'].min():>10.1f} – {g['t'].max():>10.1f}"
              f"  {mape:>7.1f}%  {lerr:>9.3f}  {len(g):>4}")


# ══════════════════════════════════════════════════════════════
#  OPTUNA TUNING
# ══════════════════════════════════════════════════════════════

def tune_model(name, X_tr, y_tr, X_vl, y_vl):
    if not HAS_OPTUNA: return {}

    def obj_xgb(t):
        m = XGBRegressor(
            n_estimators=t.suggest_int("ne",500,2000),
            learning_rate=t.suggest_float("lr",0.005,0.08,log=True),
            max_depth=t.suggest_int("d",4,10),
            min_child_weight=t.suggest_int("mcw",1,15),
            subsample=t.suggest_float("ss",0.6,1.0),
            colsample_bytree=t.suggest_float("cbt",0.5,1.0),
            gamma=t.suggest_float("g",0,1.0),
            reg_alpha=t.suggest_float("a",0.001,5.0,log=True),
            reg_lambda=t.suggest_float("l",0.01,10.0,log=True),
            random_state=RANDOM_STATE,n_jobs=-1,tree_method="hist")
        m.fit(X_tr,y_tr,eval_set=[(X_vl,y_vl)],verbose=False)
        return mean_squared_error(y_vl,m.predict(X_vl))

    def obj_lgbm(t):
        m = LGBMRegressor(
            n_estimators=t.suggest_int("ne",500,2000),
            learning_rate=t.suggest_float("lr",0.005,0.08,log=True),
            max_depth=t.suggest_int("d",4,10),
            num_leaves=t.suggest_int("nl",31,200),
            subsample=t.suggest_float("ss",0.6,1.0),
            colsample_bytree=t.suggest_float("cbt",0.5,1.0),
            min_child_samples=t.suggest_int("mcs",5,30),
            reg_alpha=t.suggest_float("a",0.001,5.0,log=True),
            reg_lambda=t.suggest_float("l",0.01,10.0,log=True),
            random_state=RANDOM_STATE,n_jobs=-1,verbose=-1)
        m.fit(X_tr,y_tr)
        return mean_squared_error(y_vl,m.predict(X_vl))

    def obj_cb(t):
        m = CatBoostRegressor(
            iterations=t.suggest_int("ne",500,1500),
            learning_rate=t.suggest_float("lr",0.005,0.08,log=True),
            depth=t.suggest_int("d",4,10),
            l2_leaf_reg=t.suggest_float("l",0.01,10.0,log=True),
            subsample=t.suggest_float("ss",0.6,1.0),
            random_seed=RANDOM_STATE,verbose=False,thread_count=-1)
        m.fit(X_tr,y_tr,eval_set=(X_vl,y_vl))
        return mean_squared_error(y_vl,m.predict(X_vl))

    obj = {"xgb":obj_xgb,"lgbm":obj_lgbm,"catboost":obj_cb}.get(name)
    if obj is None: return {}

    s = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    s.optimize(obj,n_trials=N_TRIALS,show_progress_bar=False)
    print(f"    {name}: {s.best_params}")
    return s.best_params


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  AREA PREDICTION v2 — PHYSICS-GROUNDED RTL FEATURES")
    print("=" * 70)
    print(f"  XGB={HAS_XGB}  LGBM={HAS_LGBM}  CB={HAS_CATBOOST}"
          f"  SHAP={HAS_SHAP}  Optuna={HAS_OPTUNA}")

    # ── 1. Load ───────────────────────────────────────────────
    print(f"\nLoading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Shape: {df.shape}")

    area_col = next((c for c in AREA_COL_CANDIDATES if c in df.columns), None)
    if area_col is None:
        raise ValueError(f"No area column found. Expected: {AREA_COL_CANDIDATES}")
    print(f"  Area column: '{area_col}'")

    # ── 2. Feature engineering ────────────────────────────────
    print("\nEngineering area-specific features (40 physics-grounded)...")
    df = engineer_area_features(df)
    print(f"  Columns after engineering: {len(df.columns)}")

    # ── 3. Clean ──────────────────────────────────────────────
    # print("\nCleaning...")
    # exclude   = NON_FEATURE_COLS - {area_col}
    # feat_cols = [c for c in df.columns
    #              if c not in exclude and not c.startswith("_")
    #              and pd.api.types.is_numeric_dtype(df[c])]
    # df_m = df[feat_cols + [area_col]].copy()
    # ── 3. Clean ──────────────────────────────────────────────
    print("\nCleaning...")
    exclude   = NON_FEATURE_COLS  # FIX: Don't remove area_col from the exclude list
    feat_cols = [c for c in df.columns
                 if c not in exclude and not c.startswith("_")
                 and pd.api.types.is_numeric_dtype(df[c])]
    df_m = df[feat_cols + [area_col]].copy()

    n    = len(df_m)
    df_m = df_m.dropna()
    print(f"  dropna   : {len(df_m)}/{n}")
    df_m = df_m[df_m[area_col] > 0]
    print(f"  area > 0 : {len(df_m)}")
    la    = np.log10(df_m[area_col])
    q1,q3 = la.quantile([0.25,0.75])
    iqr   = q3 - q1
    df_m  = df_m[(la >= q1 - 3.5*iqr) & (la <= q3 + 3.5*iqr)]
    print(f"  IQR      : {len(df_m)}")
    a = df_m[area_col]
    print(f"\n  Area (µm²): min={a.min():.1f}  median={a.median():.1f}  "
          f"max={a.max():.1f}  "
          f"({np.log10(a.max()/a.min()):.1f} decades)")

    # ── 4. Prune ──────────────────────────────────────────────
    print("\nPruning features...")
    X_df  = df_m[[c for c in feat_cols if c in df_m.columns]]
    keep  = prune_features(X_df)
    X_df  = X_df[keep]
    y_raw = df_m[area_col].values

    # ── 5. Stratified split ───────────────────────────────────
    print("\nStratified split by log10(area) bins...")
    strat = pd.qcut(np.log10(y_raw), q=10,
                    labels=False, duplicates="drop").astype(str)
    X_tmp, X_te, y_tmp, y_te, s_tmp, _ = train_test_split(
        X_df.values, y_raw, strat,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat)
    strat_v = pd.qcut(np.log10(y_tmp), q=10,
                      labels=False, duplicates="drop").astype(str)
    X_tr, X_vl, y_tr_o, y_vl_o = train_test_split(
        X_tmp, y_tmp, test_size=VAL_SIZE/(1-TEST_SIZE),
        random_state=RANDOM_STATE, stratify=strat_v)
    print(f"  Train={len(X_tr)}  Val={len(X_vl)}  Test={len(X_te)}")

#     # ── 6. log10 target ───────────────────────────────────────
#     y_tr = np.log10(y_tr_o)
#     y_vl = np.log10(y_vl_o)
#     y_te = np.log10(y_te)
#     def inverse(x): return 10 ** np.array(x)

#     # ── 7. Scale ──────────────────────────────────────────────
#     scaler = RobustScaler()
#     X_tr_s = scaler.fit_transform(X_tr)
#     X_vl_s = scaler.transform(X_vl)
#     X_te_s = scaler.transform(X_te)
#     joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
#     joblib.dump(keep,   os.path.join(MODEL_DIR, "feature_cols.joblib"))

#     # ── 8. 5-fold CV baseline ─────────────────────────────────
#     print("\n5-fold CV on training set (directly comparable to teammate)...")
#     kf       = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
#     baseline = GradientBoostingRegressor(
#         n_estimators=500, learning_rate=0.03,
#         max_depth=6, subsample=0.8, random_state=RANDOM_STATE)
#     cv_r2 = cross_val_score(baseline, X_tr_s, y_tr, cv=kf,
#                              scoring="r2", n_jobs=-1)
#     cv_pred_log = cross_val_predict(baseline, X_tr_s, y_tr, cv=kf)
#     cv_pred_orig = inverse(cv_pred_log)
#     cv_true_orig = inverse(y_tr)
#     cv_m = metrics(cv_true_orig, cv_pred_orig)

#     print(f"\n  ┌────────────────────────────────────────────────────────┐")
#     print(f"  │  5-FOLD CV COMPARISON: OURS vs TEAMMATE                 │")
#     print(f"  ├─────────────────────────────────────────┬──────────┬────┤")
#     print(f"  │  Metric                                 │ Teammate │ Ours│")
#     print(f"  ├─────────────────────────────────────────┼──────────┼────┤")
#     print(f"  │  Area CV R² (on training set)           │  0.5971  │{cv_r2.mean():5.4f}│")
#     print(f"  │  Area CV R² std                         │  ±0.0206 │±{cv_r2.std():.4f}│")
#     print(f"  │  Teammate test R² (best = LGBM)         │  0.2363  │ TBD │")
#     print(f"  │  Teammate CV→test gap (LGBM)            │  0.3573  │ TBD │")
#     print(f"  │  NOTE: gap>0.1 = overfitting            │  ✗       │     │")
#     print(f"  └─────────────────────────────────────────┴──────────┴────┘")

#     # ── 9. Optuna ─────────────────────────────────────────────
#     xp, lp, cp = {}, {}, {}
#     if TUNE:
#         print("\nOptuna HPO (area-specific bounds)...")
#         if HAS_XGB:
#             print("  XGB...",  end=" ", flush=True); xp = tune_model("xgb",      X_tr_s, y_tr, X_vl_s, y_vl)
#         if HAS_LGBM:
#             print("  LGBM...", end=" ", flush=True); lp = tune_model("lgbm",     X_tr_s, y_tr, X_vl_s, y_vl)
#         if HAS_CATBOOST:
#             print("  CB...",   end=" ", flush=True); cp = tune_model("catboost", X_tr_s, y_tr, X_vl_s, y_vl)

#     # # ── 10. Build models ──────────────────────────────────────
#     # models = {}
#     # if HAS_XGB:
#     #     p = dict(n_estimators=1200,learning_rate=0.015,max_depth=8,
#     #              min_child_weight=3,subsample=0.85,colsample_bytree=0.85,
#     #              gamma=0.2,reg_alpha=0.5,reg_lambda=2.0,
#     #              random_state=RANDOM_STATE,n_jobs=-1,tree_method="hist")
#     #     p.update(xp); models["xgb"] = XGBRegressor(**p)
#     # if HAS_LGBM:
#     #     p = dict(n_estimators=1200,learning_rate=0.015,max_depth=9,
#     #              num_leaves=80,subsample=0.85,colsample_bytree=0.85,
#     #              min_child_samples=10,reg_alpha=0.5,reg_lambda=2.0,
#     #              random_state=RANDOM_STATE,n_jobs=-1,verbose=-1)
#     #     p.update(lp); models["lgbm"] = LGBMRegressor(**p)
#     # if HAS_CATBOOST:
#     #     p = dict(iterations=1200,learning_rate=0.015,depth=8,
#     #              l2_leaf_reg=2.0,subsample=0.85,
#     #              random_seed=RANDOM_STATE,verbose=False,thread_count=-1)
#     #     p.update(cp); models["catboost"] = CatBoostRegressor(**p)
#     # models["rf"] = RandomForestRegressor(
#     #     n_estimators=600,max_depth=18,min_samples_split=8,
#     #     min_samples_leaf=3,max_features="sqrt",
#     #     random_state=RANDOM_STATE,n_jobs=-1)
#     # models["gbm"] = GradientBoostingRegressor(
#     #     n_estimators=700,learning_rate=0.015,max_depth=7,
#     #     subsample=0.85,min_samples_leaf=3,random_state=RANDOM_STATE)

# # ── 10. Build models ──────────────────────────────────────
#     models = {}
#     if HAS_XGB:
#         # Using the exact parameters Optuna just found!
#         models["xgb"] = XGBRegressor(
#             n_estimators=1501, learning_rate=0.00682, max_depth=9,
#             min_child_weight=7, subsample=0.792, colsample_bytree=0.625,
#             gamma=0.099, reg_alpha=0.0178, reg_lambda=2.636,
#             random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")
            
#     if HAS_LGBM:
#         # Using the exact parameters Optuna just found!
#         models["lgbm"] = LGBMRegressor(
#             n_estimators=608, learning_rate=0.0332, max_depth=10,
#             num_leaves=76, subsample=0.837, colsample_bytree=0.612,
#             min_child_samples=14, reg_alpha=0.052, reg_lambda=5.067,
#             random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
            
#     if HAS_CATBOOST:
#         # Reverted to safe, fast defaults to avoid the slowdown
#         models["catboost"] = CatBoostRegressor(
#             iterations=1000, learning_rate=0.02, depth=6,
#             l2_leaf_reg=3.0, subsample=0.85,
#             random_seed=RANDOM_STATE, verbose=False, thread_count=-1)
            
#     models["rf"] = RandomForestRegressor(
#         n_estimators=600, max_depth=18, min_samples_split=8,
#         min_samples_leaf=3, max_features="sqrt",
#         random_state=RANDOM_STATE, n_jobs=-1)
        
#     models["gbm"] = GradientBoostingRegressor(
#         n_estimators=700, learning_rate=0.015, max_depth=7,
#         subsample=0.85, min_samples_leaf=3, random_state=RANDOM_STATE)

#     # ── 11. Train and evaluate ────────────────────────────────
#     print(f"\n{'='*60}\n  MODEL RESULTS\n{'='*60}")
#     results = {}
#     for mname, model in models.items():
#         print(f"\n  {mname.upper()}...", end=" ", flush=True)
#         model.fit(X_tr_s, y_tr)
#         vl_pred = inverse(model.predict(X_vl_s))
#         te_pred = inverse(model.predict(X_te_s))
#         vm = metrics(y_vl_o, vl_pred)
#         tm = metrics(y_te,   te_pred)
#         print("done")
#         print_metrics("    Validation", vm)
#         print_metrics("    Test",       tm)
#         gap = abs(vm["r2"] - tm["r2"])
#         status = "✓ healthy" if gap < 0.10 else ("⚠ moderate" if gap < 0.20 else "✗ overfitting")
#         print(f"    Val-Test gap = {gap:.3f}  [{status}]  "
#               f"(teammate gap was 0.36–0.54)")
#         results[mname] = {"model": model, "val_r2": vm["r2"],
#                           "test_metrics": tm, "te_pred": te_pred}
#         joblib.dump(model, os.path.join(MODEL_DIR, f"area_{mname}.joblib"))

#     # ── 12. Weighted ensemble ─────────────────────────────────
#     print(f"\n  {'─'*50}")
#     w   = {n: max(r["val_r2"], 0)**2 for n, r in results.items()}
#     tw  = sum(w.values())
#     ens = sum((w[n]/tw)*results[n]["te_pred"] for n in w if w[n]>0)
#     em  = metrics(y_te, ens)
#     print_metrics("  Ensemble Test", em)
#     results["ensemble"] = {"model":None,"val_r2":em["r2"],
#                            "test_metrics":em,"te_pred":ens}

#     # ── 13. Summary ───────────────────────────────────────────
#     print(f"\n{'='*70}")
#     print(f"  FINAL COMPARISON — AREA (RTL only)")
#     print(f"{'='*70}")
#     print(f"  {'Model':<14} {'Val R²':>8} {'Test R²':>8} "
#           f"{'MAPE':>8} {'log10_med':>10} {'Gap':>8} {'Status':>12}")
#     print(f"  {'-'*72}")
#     best_name, best_r2 = None, -999
#     for n, r in sorted(results.items(), key=lambda x: -x[1]["val_r2"]):
#         tm  = r["test_metrics"]
#         gap = abs(r["val_r2"] - tm["r2"])
#         st  = "✓ healthy" if gap < 0.10 else ("⚠ moderate" if gap < 0.20 else "✗ overfit")
#         print(f"  {n:<14} {r['val_r2']:>8.4f} {tm['r2']:>8.4f} "
#               f"{tm['mape']:>7.1f}% {tm['median_log10_err']:>9.3f} "
#               f"{gap:>7.3f}  {st}")
#         if r["val_r2"] > best_r2:
#             best_r2, best_name = r["val_r2"], n

#     per_decile_table(y_te, results[best_name]["te_pred"], best_name)

#     # ── 14. Head-to-head vs teammate ──────────────────────────
#     best_tm = results[best_name]["test_metrics"]
#     print(f"\n{'='*70}")
#     print(f"  HEAD-TO-HEAD: OURS vs TEAMMATE (LGBM best in both)")
#     print(f"{'='*70}")
#     print(f"  {'Metric':<35} {'Teammate':>10} {'Ours':>10} {'Result':>8}")
#     print(f"  {'-'*66}")
#     rows = [
#         ("Area CV R²",            0.5936,  cv_r2.mean()),
#         ("Area Test R²",           0.2363,  best_tm["r2"]),
#         ("CV→Test gap (lower=better)", 0.3573, abs(results[best_name]["val_r2"] - best_tm["r2"])),
#         ("Overfitting?",           1,       0),
#     ]
#     for label, them, us in rows[:3]:
#         better = us > them if "gap" not in label.lower() else us < them
#         sym    = "✓ OURS" if better else "✗ THEIRS"
#         print(f"  {label:<35} {them:>10.4f} {us:>10.4f} {sym:>8}")
#     print(f"  {'Overfitting (gap>0.10)?':<35} {'YES':>10} "
#           f"{'NO' if abs(results[best_name]['val_r2']-best_tm['r2'])<0.10 else 'YES':>10}")

#     # ── 15. SHAP ──────────────────────────────────────────────
#     sname = next((n for n in ["lgbm","xgb","catboost","gbm","rf"]
#                   if n in results and results[n]["model"] is not None), None)
#     if sname and HAS_SHAP:
#         print(f"\nSHAP on {sname}...")
#         try:
#             exp = shap.TreeExplainer(results[sname]["model"])
#             sv  = exp.shap_values(X_te_s)
#             imp = np.abs(sv).mean(axis=0)
#             dfi = pd.DataFrame({"feature": keep, "shap": imp})\
#                     .sort_values("shap", ascending=False)
#             dfi.to_csv(os.path.join(MODEL_DIR, "area_shap.csv"), index=False)
#             mx = dfi["shap"].max()
#             print(f"\n  Top-15 features driving AREA:")
#             print(f"  {'Feature':<45} {'SHAP':>8}")
#             print(f"  {'-'*55}")
#             for _, row in dfi.head(15).iterrows():
#                 bar = "█" * int(row["shap"]/mx*20)
#                 print(f"  {row['feature']:<45} {bar}  {row['shap']:.5f}")
#             print(f"\n  ✓ Good if top features are: log_total_ge, "
#                   f"mul_ge_quad, seq_ge, log_seq_ge")
#             print(f"  ✗ Concern if top features are still: "
#                   f"total_bits, bits_log (= size not physics)")
#         except Exception as e:
#             print(f"  SHAP failed: {e}")

#     # ── 16. Save ──────────────────────────────────────────────
#     pd.DataFrame({
#         "y_true": y_te, "y_pred": results[best_name]["te_pred"],
#         "log10_true": np.log10(np.clip(y_te, EPS, None)),
#         "log10_pred": np.log10(np.clip(results[best_name]["te_pred"], EPS, None)),
#         "log10_err": np.abs(np.log10(np.clip(y_te, EPS, None)) -
#                              np.log10(np.clip(results[best_name]["te_pred"], EPS, None))),
#         "pct_error": np.abs((y_te - results[best_name]["te_pred"]) /
#                              np.clip(y_te, EPS, None)) * 100,
#     }).sort_values("log10_err", ascending=False)\
#       .to_csv(os.path.join(MODEL_DIR, "test_predictions.csv"), index=False)

#     print(f"\n  Outputs → {MODEL_DIR}")
#     print("=" * 70)
#     print("  RESULT GUIDE:")
#     print("    R² > 0.85, log10_median < 0.12  → excellent (publishable)")
#     print("    R² > 0.70, log10_median < 0.22  → good, clearly beats teammate")
#     print("    R² > 0.55, log10_median < 0.35  → acceptable baseline")
#     print("    CV ≈ Test R² (gap < 0.10)        → robust, no overfitting")


# if __name__ == "__main__":
#     main()

# ── 6. log10 target ───────────────────────────────────────
    y_tr = np.log10(y_tr_o)
    y_vl = np.log10(y_vl_o)
    y_te_o = np.array(y_te)  # FIX: Save original raw test values
    y_te = np.log10(y_te_o)
    def inverse(x): return 10 ** np.array(x)

    # ── 7. Scale ──────────────────────────────────────────────
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)
    X_te_s = scaler.transform(X_te)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(keep,   os.path.join(MODEL_DIR, "feature_cols.joblib"))

    # ── 8. 5-fold CV baseline ─────────────────────────────────
    print("\n5-fold CV on training set (directly comparable to teammate)...")
    kf       = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    baseline = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.03,
        max_depth=6, subsample=0.8, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(baseline, X_tr_s, y_tr, cv=kf,
                             scoring="r2", n_jobs=-1)
    cv_pred_log = cross_val_predict(baseline, X_tr_s, y_tr, cv=kf)
    cv_pred_orig = inverse(cv_pred_log)
    cv_true_orig = inverse(y_tr)
    cv_m = metrics(cv_true_orig, cv_pred_orig)

    print(f"\n  ┌────────────────────────────────────────────────────────┐")
    print(f"  │  5-FOLD CV COMPARISON: OURS vs TEAMMATE                 │")
    print(f"  ├─────────────────────────────────────────┬──────────┬────┤")
    print(f"  │  Metric                                 │ Teammate │ Ours│")
    print(f"  ├─────────────────────────────────────────┼──────────┼────┤")
    print(f"  │  Area CV R² (in log10 space)            │  0.5971  │{cv_r2.mean():5.4f}│")
    print(f"  │  Area CV R² std                         │  ±0.0206 │±{cv_r2.std():.4f}│")
    print(f"  │  Teammate test R² (best = LGBM)         │  0.2363  │ TBD │")
    print(f"  │  Teammate CV→test gap (LGBM)            │  0.3573  │ TBD │")
    print(f"  │  NOTE: gap>0.1 = overfitting            │  ✗       │     │")
    print(f"  └─────────────────────────────────────────┴──────────┴────┘")

    # ── 9. Optuna ─────────────────────────────────────────────
    # (Skipped via TUNE=False)

    # ── 10. Build models ──────────────────────────────────────
    models = {}
    if HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=1501, learning_rate=0.00682, max_depth=9,
            min_child_weight=7, subsample=0.792, colsample_bytree=0.625,
            gamma=0.099, reg_alpha=0.0178, reg_lambda=2.636,
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")
            
    if HAS_LGBM:
        models["lgbm"] = LGBMRegressor(
            n_estimators=608, learning_rate=0.0332, max_depth=10,
            num_leaves=76, subsample=0.837, colsample_bytree=0.612,
            min_child_samples=14, reg_alpha=0.052, reg_lambda=5.067,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
            
    if HAS_CATBOOST:
        models["catboost"] = CatBoostRegressor(
            iterations=1000, learning_rate=0.02, depth=6,
            l2_leaf_reg=3.0, subsample=0.85,
            random_seed=RANDOM_STATE, verbose=False, thread_count=-1)
            
    models["rf"] = RandomForestRegressor(
        n_estimators=600, max_depth=18, min_samples_split=8,
        min_samples_leaf=3, max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=-1)
        
    models["gbm"] = GradientBoostingRegressor(
        n_estimators=700, learning_rate=0.015, max_depth=7,
        subsample=0.85, min_samples_leaf=3, random_state=RANDOM_STATE)

    # ── 11. Train and evaluate ────────────────────────────────
    print(f"\n{'='*60}\n  MODEL RESULTS\n{'='*60}")
    results = {}
    for mname, model in models.items():
        print(f"\n  {mname.upper()}...", end=" ", flush=True)
        model.fit(X_tr_s, y_tr)
        vl_pred = inverse(model.predict(X_vl_s))
        te_pred = inverse(model.predict(X_te_s))
        vm = metrics(y_vl_o, vl_pred)
        tm = metrics(y_te_o, te_pred)  # FIX: Using raw test targets
        print("done")
        print_metrics("    Validation", vm)
        print_metrics("    Test",       tm)
        gap = abs(vm["r2"] - tm["r2"])
        status = "✓ healthy" if gap < 0.10 else ("⚠ moderate" if gap < 0.20 else "✗ overfitting")
        print(f"    Val-Test gap = {gap:.3f}  [{status}]  "
              f"(teammate gap was 0.36–0.54)")
        results[mname] = {"model": model, "val_r2": vm["r2"],
                          "test_metrics": tm, "te_pred": te_pred}
        joblib.dump(model, os.path.join(MODEL_DIR, f"area_{mname}.joblib"))

    # ── 12. Weighted ensemble ─────────────────────────────────
    print(f"\n  {'─'*50}")
    w   = {n: max(r["val_r2"], 0)**2 for n, r in results.items()}
    tw  = sum(w.values())
    ens = sum((w[n]/tw)*results[n]["te_pred"] for n in w if w[n]>0)
    if tw > 0:
        em  = metrics(y_te_o, ens)  # FIX: Using raw test targets
        print_metrics("  Ensemble Test", em)
        results["ensemble"] = {"model":None,"val_r2":em["r2"],
                               "test_metrics":em,"te_pred":ens}

    # ── 13. Summary ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON — AREA (RTL only)")
    print(f"{'='*70}")
    print(f"  {'Model':<14} {'Val R²':>8} {'Test R²':>8} "
          f"{'MAPE':>8} {'log10_med':>10} {'Gap':>8} {'Status':>12}")
    print(f"  {'-'*72}")
    best_name, best_r2 = None, -999
    for n, r in sorted(results.items(), key=lambda x: -x[1]["val_r2"]):
        tm  = r["test_metrics"]
        gap = abs(r["val_r2"] - tm["r2"])
        st  = "✓ healthy" if gap < 0.10 else ("⚠ moderate" if gap < 0.20 else "✗ overfit")
        print(f"  {n:<14} {r['val_r2']:>8.4f} {tm['r2']:>8.4f} "
              f"{tm['mape']:>7.1f}% {tm['median_log10_err']:>9.3f} "
              f"{gap:>7.3f}  {st}")
        if r["val_r2"] > best_r2:
            best_r2, best_name = r["val_r2"], n

    per_decile_table(y_te_o, results[best_name]["te_pred"], best_name)

    # ── 14. Head-to-head vs teammate ──────────────────────────
    best_tm = results[best_name]["test_metrics"]
    print(f"\n{'='*70}")
    print(f"  HEAD-TO-HEAD: OURS vs TEAMMATE (LGBM best in both)")
    print(f"{'='*70}")
    print(f"  {'Metric':<35} {'Teammate':>10} {'Ours':>10} {'Result':>8}")
    print(f"  {'-'*66}")
    rows = [
        ("Area CV R² (log10 space)",         0.5936,  cv_r2.mean()),
        ("Area Test R²",                     0.2363,  best_tm["r2"]),
        ("CV→Test gap (lower=better)", 0.3573, abs(results[best_name]["val_r2"] - best_tm["r2"])),
        ("Overfitting?",                   1,       0),
    ]
    for label, them, us in rows[:3]:
        better = us > them if "gap" not in label.lower() else us < them
        sym    = "✓ OURS" if better else "✗ THEIRS"
        print(f"  {label:<35} {them:>10.4f} {us:>10.4f} {sym:>8}")
    print(f"  {'Overfitting (gap>0.10)?':<35} {'YES':>10} "
          f"{'NO' if abs(results[best_name]['val_r2']-best_tm['r2'])<0.10 else 'YES':>10}")

    # ── 15. SHAP ──────────────────────────────────────────────
    sname = next((n for n in ["lgbm","xgb","catboost","gbm","rf"]
                  if n in results and results[n]["model"] is not None), None)
    if sname and HAS_SHAP:
        print(f"\nSHAP on {sname}...")
        try:
            exp = shap.TreeExplainer(results[sname]["model"])
            sv  = exp.shap_values(X_te_s)
            imp = np.abs(sv).mean(axis=0)
            dfi = pd.DataFrame({"feature": keep, "shap": imp})\
                    .sort_values("shap", ascending=False)
            dfi.to_csv(os.path.join(MODEL_DIR, "area_shap.csv"), index=False)
            mx = dfi["shap"].max()
            print(f"\n  Top-15 features driving AREA:")
            print(f"  {'Feature':<45} {'SHAP':>8}")
            print(f"  {'-'*55}")
            for _, row in dfi.head(15).iterrows():
                bar = "█" * int(row["shap"]/mx*20)
                print(f"  {row['feature']:<45} {bar}  {row['shap']:.5f}")
        except Exception as e:
            print(f"  SHAP failed: {e}")

    # ── 16. Save ──────────────────────────────────────────────
    pd.DataFrame({
        "y_true": y_te_o, "y_pred": results[best_name]["te_pred"],
        "log10_true": np.log10(np.clip(y_te_o, EPS, None)),
        "log10_pred": np.log10(np.clip(results[best_name]["te_pred"], EPS, None)),
        "log10_err": np.abs(np.log10(np.clip(y_te_o, EPS, None)) -
                             np.log10(np.clip(results[best_name]["te_pred"], EPS, None))),
        "pct_error": np.abs((y_te_o - results[best_name]["te_pred"]) /
                             np.clip(y_te_o, EPS, None)) * 100,
    }).sort_values("log10_err", ascending=False)\
      .to_csv(os.path.join(MODEL_DIR, "test_predictions.csv"), index=False)

    print(f"\n  Outputs → {MODEL_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()