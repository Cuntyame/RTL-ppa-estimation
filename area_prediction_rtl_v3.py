#!/usr/bin/env python3
r"""
area_prediction_rtl_v3.py
==========================
Improved area prediction using the dedicated area feature CSV.

FIXES OVER V2 (based on output analysis):
  1. METRICS CONSISTENCY: v2 used Optuna to minimise log10-space MSE
     but reported R² in original scale → val R²≈0.07 looked terrible
     even though log10_median=0.21 was actually reasonable.
     V3 evaluates R² in log10 space consistently for model selection,
     then also reports original-scale R² for comparison.

  2. PHYSICS FEATURE PROTECTION: v2 let correlation pruner drop
     mul_ge_quad (corr > 0.97 with log_total_bits).
     V3 has a PROTECTED feature list that is never pruned, regardless
     of correlation.

  3. WRONG FEATURES IN CSV: v2 used the power-focused CSV which had
     num_clk_domains as #2 SHAP feature (irrelevant for area).
     V3 expects the dedicated area CSV from extract_rtl_area_features.py.

  4. VAL/TEST ASYMMETRY: v2 val R² << test R² for all models.
     Root cause: Optuna tuning minimised val MSE (log10 space) but
     evaluation was original-space R². Fixed by consistent log10 R².

RUN EXTRACTOR FIRST:
  python extract_rtl_area_features.py \
      --dataset_root "C:\ml ppa\Final_Clean_Dataset" \
      --out_csv "C:\Users\Admin\Documents\rtl_area_features.csv"

THEN:
  python area_prediction_rtl_v3.py
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

# ← Change to your area-specific CSV after running the extractor
# CSV_PATH  = r"C:\Users\Admin\Documents\rtl_area_features.csv"
# ← Change to your area-specific CSV after running the extractor
CSV_PATH  = r"C:\Users\Admin\Documents\final_rtl_area_features.csv"

# Fallback: use the power CSV if area CSV not yet generated
CSV_FALLBACK = r"C:\Users\Admin\Documents\final_rtl_power_features_v2.csv"

MODEL_DIR = r"C:\Users\Admin\OneDrive - MSFT\Desktop\New folder\cody\area_model_v3"
os.makedirs(MODEL_DIR, exist_ok=True)

AREA_COL_CANDIDATES = ["total_cell_area", "comb_area", "Cell_Area", "area"]

RANDOM_STATE = 42
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15
TUNE         = True
N_TRIALS     = 60
CORR_THRESH  = 0.97
EPS          = 1e-9

# Features that are NEVER pruned regardless of correlation.
# These are the core physics features — their correlations with size
# proxies are real (bigger designs DO have more multipliers), but
# keeping them separately lets the model learn the NONLINEAR quadratic
# relationship (mul_ge_quad scales as N², not N).
PROTECTED_FEATURES = [
    "mul_ge_quad",      # N² scaling — cannot be replaced by linear proxies
    "log_mul_ge_quad",  # log-compressed version
    "sqrt_mul_ge_quad",
    "seq_ge",           # DFF area = 8 × reg × bitwidth
    "log_seq_ge",
    "adder_ge",         # 2N adder area
    "log_adder_ge",
    "div_ge",           # 4N² divider
    "total_ge",         # sum of all estimates
    "log_total_ge",     # log-compressed master predictor
    "sqrt_total_ge",
    "seq_area_frac",    # scale-invariant composition
    "mul_area_frac",
    "arith_area_frac",
    "generate_ge",      # for-generate unrolling
    "unroll_ge",        # for-loop unrolling
    "mul_x_reg",        # pipelined multiplier interaction
]

# NON_FEATURE_COLS = {
#     "Design_Name", "RTL_Code", "rpt_files", "rpt_text_len",
#     "Power", "critical_path_length", "levels_of_logic",
#     "wns", "tns",
#     "total_cell_area", "comb_area", "Cell_Area", "area",
# }
NON_FEATURE_COLS = {
    "Design_Name", "RTL_Code", "rpt_files", "rpt_text_len",
    "Power", "critical_path_length", "levels_of_logic",
    "wns", "tns", "num_cells", "num_nets", # <-- LEAK PLUGGED HERE
    "total_cell_area", "comb_area", "Cell_Area", "area",
}

# ══════════════════════════════════════════════════════════════
#  INLINE FEATURE ENGINEERING
#  (run even on the power CSV as fallback — fills missing columns)
# ══════════════════════════════════════════════════════════════

def engineer_area_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute physics features from base RTL counts if not already present."""
    df = df.copy()

    def col(name, default=0.0):
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
    n_mod  = col("num_modules", 1).clip(lower=1)
    n_inp  = col("num_input")
    n_out  = col("num_output")
    n_if   = col("num_if")
    n_for  = col("num_for")
    n_mem  = col("num_memory_access", 0)

    # Only compute if not already in the CSV (extractor may have done it)
    def add_if_missing(name, series):
        if name not in df.columns:
            df[name] = series

    # Sequential (DFF = 8 GE/bit)
    add_if_missing("seq_ge",          n_reg * avg_bw * 8)
    add_if_missing("log_seq_ge",      np.log1p(df.get("seq_ge", n_reg * avg_bw * 8)))
    add_if_missing("sqrt_seq_ge",     np.sqrt((df.get("seq_ge", n_reg * avg_bw * 8)).clip(0)))

    # Multiplier (N² — CRITICAL: quadratic not linear)
    add_if_missing("mul_ge_quad",     n_mul * max_bw * max_bw)
    add_if_missing("mul_ge_linear",   n_mul * avg_bw)
    add_if_missing("log_mul_ge_quad", np.log1p(df.get("mul_ge_quad", n_mul * max_bw * max_bw)))
    add_if_missing("sqrt_mul_ge_quad",np.sqrt((df.get("mul_ge_quad", n_mul * max_bw * max_bw)).clip(0)))

    # Adder (2N)
    add_if_missing("adder_ge",        (n_add + n_sub) * avg_bw * 2)
    add_if_missing("adder_ge_nlogn",  (n_add + n_sub) * avg_bw * np.log2(avg_bw.clip(lower=2)))
    add_if_missing("log_adder_ge",    np.log1p(df.get("adder_ge", (n_add + n_sub) * avg_bw * 2)))

    # Divider (4N²)
    add_if_missing("div_ge",          n_div * max_bw * max_bw * 4)
    add_if_missing("log_div_ge",      np.log1p(df.get("div_ge", n_div * max_bw * max_bw * 4)))

    # Comparator (2N)
    add_if_missing("comp_ge",         n_comp * avg_bw * 2)

    # Bitwise (XOR=2, AND/OR=1 per bit)
    add_if_missing("logic_ge",        (n_xor * 2 + n_and + n_or) * avg_bw)
    add_if_missing("xor_ge",          n_xor * 2 * avg_bw)

    # MUX / decoder
    add_if_missing("mux_ge",          n_mux * avg_bw * 3)
    add_if_missing("decoder_ge",      n_if * np.log2((n_if + 1).clip(lower=1)) * avg_bw)

    # Shift (barrel shifter N×log2N)
    add_if_missing("shift_ge",        n_shift * avg_bw * np.log2(avg_bw.clip(lower=2)))

    # Memory (6 GE/bit)
    add_if_missing("mem_ge",          n_mem * max_bw * 6)

    # For loop (unroll estimate)
    add_if_missing("unroll_ge",       n_for * avg_bw * 4)

    # Generate
    gen_cnt = col("generate_count", 0)
    add_if_missing("generate_ge",     gen_cnt * avg_bw * 4)

    # Total gate estimate
    cols_for_total = ["seq_ge","mul_ge_quad","adder_ge","div_ge",
                      "comp_ge","logic_ge","mux_ge","shift_ge","mem_ge",
                      "unroll_ge","generate_ge"]
    total_ge = sum(df[c] for c in cols_for_total if c in df.columns)
    add_if_missing("total_ge",        total_ge)
    add_if_missing("log_total_ge",    np.log1p(df.get("total_ge", total_ge)))
    add_if_missing("sqrt_total_ge",   np.sqrt(df.get("total_ge", total_ge).clip(0)))

    # Fractions
    ge_safe = df["total_ge"].clip(lower=EPS) if "total_ge" in df.columns else total_ge.clip(lower=EPS)
    add_if_missing("seq_area_frac",   df.get("seq_ge",   n_reg*avg_bw*8)  / ge_safe)
    add_if_missing("mul_area_frac",   df.get("mul_ge_quad", n_mul*max_bw**2) / ge_safe)
    add_if_missing("logic_area_frac", df.get("logic_ge", (n_xor*2+n_and+n_or)*avg_bw) / ge_safe)
    add_if_missing("arith_area_frac", (df.get("adder_ge", (n_add+n_sub)*avg_bw*2) +
                                        df.get("mul_ge_quad", n_mul*max_bw**2) +
                                        df.get("div_ge", n_div*max_bw**2*4)) / ge_safe)

    # Bitwidth
    add_if_missing("bw_range",        max_bw - min_bw)
    add_if_missing("bw_uniformity",   avg_bw / max_bw.clip(lower=EPS))
    add_if_missing("log_total_bits",  np.log1p(col("total_bits", 0)))
    add_if_missing("sqrt_total_bits", np.sqrt(col("total_bits", 0).clip(0)))
    add_if_missing("bits_x_arith",    col("total_bits", 0) * (n_add+n_sub+n_mul+n_div+1))

    # Interactions
    add_if_missing("mul_x_reg",       n_mul * n_reg * avg_bw)
    add_if_missing("log_mul_x_reg",   np.log1p(df.get("mul_x_reg", n_mul*n_reg*avg_bw)))
    total_arith = n_add + n_sub + n_mul + n_div
    add_if_missing("total_arithmetic",total_arith)
    add_if_missing("arith_x_bw",      total_arith * avg_bw)
    add_if_missing("log_arith_x_bw",  np.log1p(df.get("arith_x_bw", total_arith*avg_bw)))
    add_if_missing("hierarchy_ge",    n_mod * (n_inp + n_out) * avg_bw)
    add_if_missing("log_hierarchy_ge",np.log1p(df.get("hierarchy_ge", n_mod*(n_inp+n_out)*avg_bw)))
    add_if_missing("unroll_factor",   n_for * 4)

    # Densities
    add_if_missing("ge_per_line",     df.get("total_ge", total_ge) / n_lines)
    add_if_missing("mul_ge_per_line", df.get("mul_ge_quad", n_mul*max_bw**2) / n_lines)
    add_if_missing("seq_ge_per_line", df.get("seq_ge", n_reg*avg_bw*8) / n_lines)
    add_if_missing("arith_bw_density",total_arith * avg_bw / n_lines)

    return df


# ══════════════════════════════════════════════════════════════
#  FEATURE PRUNING  (with protected list)
# ══════════════════════════════════════════════════════════════

def prune_features(X: pd.DataFrame, corr_thresh=CORR_THRESH,
                   protected: list = None) -> list:
    protected = protected or []
    protected_present = [f for f in protected if f in X.columns]

    # Variance threshold — never drop protected
    sel  = VarianceThreshold(threshold=1e-6)
    sel.fit(X)
    keep = X.columns[sel.get_support()].tolist()
    keep = list(set(keep) | set(protected_present))
    n0   = len(X.columns)

    # Correlation filter — never drop protected
    corr    = X[keep].corr().abs()
    upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for c in upper.columns:
        if c in protected_present:
            continue
        for cc in upper.index[upper[c] > corr_thresh].tolist():
            if cc in protected_present:
                continue
            if X[c].var() >= X[cc].var():
                to_drop.add(cc)
            else:
                to_drop.add(c)
    keep = [c for c in keep if c not in to_drop]
    print(f"  Pruned: {n0} → {len(keep)} "
          f"(removed {n0-len(keep)}, protected {len(protected_present)})")
    return keep


# ══════════════════════════════════════════════════════════════
#  METRICS  — evaluated consistently in log10 space
# ══════════════════════════════════════════════════════════════

def metrics_log10(y_true_orig, y_pred_orig) -> dict:
    """Primary metrics computed in log10 space — consistent with Optuna."""
    lt = np.log10(np.clip(y_true_orig, EPS, None))
    lp = np.log10(np.clip(y_pred_orig, EPS, None))
    r2_log    = r2_score(lt, lp)
    rmse_log  = np.sqrt(mean_squared_error(lt, lp))
    log_err   = np.abs(lt - lp)

    # Also compute original-scale for comparison with teammate
    r2_orig   = r2_score(y_true_orig, y_pred_orig)
    mape_orig = np.mean(np.abs((y_true_orig - y_pred_orig) /
                                np.clip(y_true_orig, EPS, None))) * 100
    return {
        "r2_log10":         r2_log,
        "rmse_log10":       rmse_log,
        "mean_log10_err":   float(log_err.mean()),
        "median_log10_err": float(np.median(log_err)),
        "r2_original":      r2_orig,
        "mape_original":    mape_orig,
    }


def print_m(label, m, w=30):
    print(f"  {label:<{w}}  "
          f"R²(log10)={m['r2_log10']:.4f}  "
          f"R²(orig)={m['r2_original']:.4f}  "
          f"log10_median={m['median_log10_err']:.3f}  "
          f"MAPE={m['mape_original']:.1f}%")


def per_decile_table(y_true, y_pred, label=""):
    df = pd.DataFrame({"t": y_true, "p": y_pred})
    df["dec"] = pd.qcut(df["t"], q=10, labels=False, duplicates="drop")
    print(f"\n  Per-decile ({label}):")
    print(f"  {'D':<4} {'Area (µm²)':>24} {'MAPE':>8} {'log10err':>10} {'N':>5}")
    print(f"  {'-'*56}")
    for d, g in df.groupby("dec"):
        mape = np.mean(np.abs((g["t"]-g["p"])/g["t"].clip(lower=EPS)))*100
        lerr = np.mean(np.abs(np.log10(g["t"].clip(lower=EPS)) -
                               np.log10(g["p"].clip(lower=EPS))))
        print(f"  {int(d):<4} {g['t'].min():>10.1f} – {g['t'].max():>10.1f}"
              f"  {mape:>7.1f}%  {lerr:>9.3f}  {len(g):>4}")


# ══════════════════════════════════════════════════════════════
#  OPTUNA — tuning target is log10 R² (consistent with evaluation)
# ══════════════════════════════════════════════════════════════

def tune_model(name, X_tr, y_tr_log10, X_vl, y_vl_log10):
    if not HAS_OPTUNA: return {}

    # Objective: maximise R² in log10 space (= minimise -R²)
    def score(m, X, y):
        return r2_score(y, m.predict(X))

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
        m.fit(X_tr, y_tr_log10, eval_set=[(X_vl,y_vl_log10)], verbose=False)
        return -score(m, X_vl, y_vl_log10)   # minimise -R²

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
        m.fit(X_tr, y_tr_log10)
        return -score(m, X_vl, y_vl_log10)

    def obj_cb(t):
        m = CatBoostRegressor(
            iterations=t.suggest_int("ne",500,1000),
            learning_rate=t.suggest_float("lr",0.005,0.08,log=True),
            depth=t.suggest_int("d",4,6),
            l2_leaf_reg=t.suggest_float("l",0.01,10.0,log=True),
            subsample=t.suggest_float("ss",0.6,1.0),
            random_seed=RANDOM_STATE,verbose=False,thread_count=-1)
        m.fit(X_tr, y_tr_log10, eval_set=(X_vl, y_vl_log10))
        return -score(m, X_vl, y_vl_log10)

    obj = {"xgb":obj_xgb,"lgbm":obj_lgbm,"catboost":obj_cb}.get(name)
    if obj is None: return {}

    s = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    s.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"    {name}: val log10-R²={-s.best_value:.4f}")
    return s.best_params


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  AREA PREDICTION v3 — PHYSICS FEATURES + CONSISTENT LOG10 METRICS")
    print("=" * 70)
    print(f"  XGB={HAS_XGB}  LGBM={HAS_LGBM}  CB={HAS_CATBOOST}"
          f"  SHAP={HAS_SHAP}  Optuna={HAS_OPTUNA}")

    # ── 1. Load (area CSV preferred, power CSV as fallback) ───
    csv_to_use = CSV_PATH if os.path.exists(CSV_PATH) else CSV_FALLBACK
    if not os.path.exists(CSV_PATH):
        print(f"\n[INFO] Area CSV not found. Using fallback: {CSV_FALLBACK}")
        print(f"       Run extract_rtl_area_features.py first for best results.")
    print(f"\nLoading: {csv_to_use}")
    df = pd.read_csv(csv_to_use)
    print(f"  Shape: {df.shape}")

    area_col = next((c for c in AREA_COL_CANDIDATES if c in df.columns), None)
    if area_col is None:
        raise ValueError(f"No area column found. Expected: {AREA_COL_CANDIDATES}")
    print(f"  Area column: '{area_col}'")

    # ── 2. Feature engineering ────────────────────────────────
    print("\nEngineering area-specific physics features...")
    df = engineer_area_features(df)
    print(f"  Columns after engineering: {len(df.columns)}")

    # ── 3. Clean ──────────────────────────────────────────────
    print("\nCleaning...")
    exclude   = NON_FEATURE_COLS #- {area_col}
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
    df_m  = df_m[(la >= q1-3.5*iqr) & (la <= q3+3.5*iqr)]
    print(f"  IQR      : {len(df_m)}")
    a = df_m[area_col]
    print(f"\n  Area (µm²): min={a.min():.1f}  median={a.median():.1f}  "
          f"max={a.max():.1f}  "
          f"({np.log10(a.max()/a.min()):.1f} decades)")

    # ── 4. Prune (with protected physics features) ────────────
    print("\nPruning features (protecting physics features)...")
    X_df  = df_m[[c for c in feat_cols if c in df_m.columns]]
    keep  = prune_features(X_df, corr_thresh=CORR_THRESH,
                           protected=PROTECTED_FEATURES)
    X_df  = X_df[keep]
    y_raw = df_m[area_col].values
    print(f"  Physics features in final set: "
          f"{sum(1 for f in PROTECTED_FEATURES if f in keep)}/{len(PROTECTED_FEATURES)}")

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
        X_tmp, y_tmp,
        test_size=VAL_SIZE/(1-TEST_SIZE),
        random_state=RANDOM_STATE, stratify=strat_v)
    print(f"  Train={len(X_tr)}  Val={len(X_vl)}  Test={len(X_te)}")

    # ── 6. log10 target ───────────────────────────────────────
    y_tr = np.log10(y_tr_o)
    y_vl = np.log10(y_vl_o)
    y_te_log = np.log10(y_te)

    # ── 7. Scale ──────────────────────────────────────────────
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)
    X_te_s = scaler.transform(X_te)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(keep,   os.path.join(MODEL_DIR, "feature_cols.joblib"))

    # ── 8. 5-fold CV baseline (log10 R²) ─────────────────────
    print("\n5-fold CV on training set...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base_gbm = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.03,
        max_depth=6, subsample=0.8, random_state=RANDOM_STATE)
    cv_r2_log = cross_val_score(base_gbm, X_tr_s, y_tr,
                                 cv=kf, scoring="r2", n_jobs=-1)
    cv_pred   = cross_val_predict(base_gbm, X_tr_s, y_tr, cv=kf)
    cv_pred_o = 10 ** cv_pred
    cv_r2_orig = r2_score(y_tr_o, cv_pred_o)

    print(f"\n  ┌──────────────────────────────────────────────────────┐")
    print(f"  │  CV RESULTS                                            │")
    print(f"  │  5-fold CV R² (log10 space)  = {cv_r2_log.mean():.4f} ± {cv_r2_log.std():.4f}  │")
    print(f"  │  5-fold CV R² (original)     = {cv_r2_orig:.4f}                 │")
    print(f"  │                                                        │")
    print(f"  │  Teammate GBM CV R² (orig)   = 0.5971                 │")
    print(f"  │  Teammate GBM test R² (orig) = 0.1515 (gap=0.446)    │")
    print(f"  │  Our target: CV R² ≈ test R² (gap < 0.10)             │")
    print(f"  └──────────────────────────────────────────────────────┘")

    # ── 9. Optuna (maximising log10 R²) ──────────────────────
    xp, lp, cp = {}, {}, {}
    if TUNE:
        print("\nOptuna HPO (maximising log10-space R²)...")
        if HAS_XGB:
            print("  XGB...",  end=" ", flush=True)
            xp = tune_model("xgb",      X_tr_s, y_tr, X_vl_s, y_vl)
        if HAS_LGBM:
            print("  LGBM...", end=" ", flush=True)
            lp = tune_model("lgbm",     X_tr_s, y_tr, X_vl_s, y_vl)
        if HAS_CATBOOST:
            print("  CB...",   end=" ", flush=True)
            cp = tune_model("catboost", X_tr_s, y_tr, X_vl_s, y_vl)

    # ── 10. Build and train models ────────────────────────────
    # models = {}
    # if HAS_XGB:
    #     p = dict(n_estimators=1200,learning_rate=0.015,max_depth=8,
    #              min_child_weight=3,subsample=0.85,colsample_bytree=0.85,
    #              gamma=0.2,reg_alpha=0.5,reg_lambda=2.0,
    #              random_state=RANDOM_STATE,n_jobs=-1,tree_method="hist")
    #     p.update(xp); models["xgb"] = XGBRegressor(**p)
    # if HAS_LGBM:
    #     p = dict(n_estimators=1200,learning_rate=0.015,max_depth=9,
    #              num_leaves=80,subsample=0.85,colsample_bytree=0.85,
    #              min_child_samples=10,reg_alpha=0.5,reg_lambda=2.0,
    #              random_state=RANDOM_STATE,n_jobs=-1,verbose=-1)
    #     p.update(lp); models["lgbm"] = LGBMRegressor(**p)
    # if HAS_CATBOOST:
    #     p = dict(iterations=1200,learning_rate=0.015,depth=8,
    #              l2_leaf_reg=2.0,subsample=0.85,
    #              random_seed=RANDOM_STATE,verbose=False,thread_count=-1)
    #     p.update(cp); models["catboost"] = CatBoostRegressor(**p)
    # models["rf"] = RandomForestRegressor(
    #     n_estimators=600,max_depth=18,min_samples_split=8,
    #     min_samples_leaf=3,max_features="sqrt",
    #     random_state=RANDOM_STATE,n_jobs=-1)
    # models["gbm"] = GradientBoostingRegressor(
    #     n_estimators=700,learning_rate=0.015,max_depth=7,
    #     subsample=0.85,min_samples_leaf=3,random_state=RANDOM_STATE)

    # ── 10. Build and train models ────────────────────────────
    models = {}
    if HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=xp.get("ne", 1200),
            learning_rate=xp.get("lr", 0.015),
            max_depth=xp.get("d", 8),
            min_child_weight=xp.get("mcw", 3),
            subsample=xp.get("ss", 0.85),
            colsample_bytree=xp.get("cbt", 0.85),
            gamma=xp.get("g", 0.2),
            reg_alpha=xp.get("a", 0.5),
            reg_lambda=xp.get("l", 2.0),
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"
        )
        
    if HAS_LGBM:
        models["lgbm"] = LGBMRegressor(
            n_estimators=lp.get("ne", 1200),
            learning_rate=lp.get("lr", 0.015),
            max_depth=lp.get("d", 9),
            num_leaves=lp.get("nl", 80),
            subsample=lp.get("ss", 0.85),
            colsample_bytree=lp.get("cbt", 0.85),
            min_child_samples=lp.get("mcs", 10),
            reg_alpha=lp.get("a", 0.5),
            reg_lambda=lp.get("l", 2.0),
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        
    if HAS_CATBOOST:
        models["catboost"] = CatBoostRegressor(
            iterations=cp.get("ne", 1000),
            learning_rate=cp.get("lr", 0.015),
            depth=cp.get("d", 6),
            l2_leaf_reg=cp.get("l", 2.0),
            subsample=cp.get("ss", 0.85),
            random_seed=RANDOM_STATE, verbose=False, thread_count=-1
        )
        
    models["rf"] = RandomForestRegressor(
        n_estimators=600,max_depth=18,min_samples_split=8,
        min_samples_leaf=3,max_features="sqrt",
        random_state=RANDOM_STATE,n_jobs=-1)
        
    models["gbm"] = GradientBoostingRegressor(
        n_estimators=700,learning_rate=0.015,max_depth=7,
        subsample=0.85,min_samples_leaf=3,random_state=RANDOM_STATE)

    print(f"\n{'='*60}\n  MODEL RESULTS (primary: log10-space R²)\n{'='*60}")
    results = {}
    for mname, model in models.items():
        print(f"\n  {mname.upper()}...", end=" ", flush=True)
        model.fit(X_tr_s, y_tr)
        vl_pred = 10 ** model.predict(X_vl_s)
        te_pred = 10 ** model.predict(X_te_s)
        vm = metrics_log10(y_vl_o, vl_pred)
        tm = metrics_log10(y_te,   te_pred)
        print("done")
        print_m("    Validation", vm)
        print_m("    Test",       tm)
        gap = abs(vm["r2_log10"] - tm["r2_log10"])
        status = "✓ healthy" if gap < 0.10 else ("⚠ moderate" if gap < 0.20 else "✗ overfitting")
        print(f"    log10 Val-Test gap={gap:.3f}  [{status}]")
        results[mname] = {"model":model,"val_r2":vm["r2_log10"],
                          "test_metrics":tm,"te_pred":te_pred}
        joblib.dump(model, os.path.join(MODEL_DIR, f"area_{mname}.joblib"))

    # ── 11. Weighted ensemble ─────────────────────────────────
    print(f"\n  {'─'*50}")
    w   = {n: max(r["val_r2"], 0)**2 for n, r in results.items()}
    tw  = sum(w.values())
    if tw > 0:
        ens = sum((w[n]/tw)*results[n]["te_pred"] for n in w if w[n]>0)
        em  = metrics_log10(y_te, ens)
        print_m("  Ensemble Test", em)
        results["ensemble"] = {"model":None,"val_r2":em["r2_log10"],
                               "test_metrics":em,"te_pred":ens}

    # ── 12. Summary ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY — AREA (RTL only, v3)")
    print(f"{'='*70}")
    print(f"  {'Model':<14} {'ValR²(log)':>11} {'TstR²(log)':>11} "
          f"{'TstR²(orig)':>12} {'log10_med':>10} {'Gap':>7}")
    print(f"  {'-'*70}")
    best_name, best_r2 = None, -999
    for n, r in sorted(results.items(), key=lambda x: -x[1]["val_r2"]):
        tm  = r["test_metrics"]
        gap = abs(r["val_r2"] - tm["r2_log10"])
        st  = "✓" if gap < 0.10 else ("⚠" if gap < 0.20 else "✗")
        print(f"  {n:<14} {r['val_r2']:>11.4f} {tm['r2_log10']:>11.4f} "
              f"{tm['r2_original']:>12.4f} {tm['median_log10_err']:>9.3f} "
              f"{gap:>6.3f}{st}")
        if r["val_r2"] > best_r2:
            best_r2, best_name = r["val_r2"], n

    per_decile_table(y_te, results[best_name]["te_pred"], best_name)

    # ── 13. Vs teammate ───────────────────────────────────────
    best_tm = results[best_name]["test_metrics"]
    print(f"\n{'='*70}")
    print(f"  HEAD-TO-HEAD vs TEAMMATE")
    print(f"{'='*70}")
    print(f"  {'Metric':<40} {'Teammate':>10} {'Ours':>10} {'Result':>8}")
    print(f"  {'-'*70}")
    cmp = [
        ("Area CV R² (original scale)",    0.5971, cv_r2_orig),
        ("Area best test R² (original)",   0.2363, best_tm["r2_original"]),
        ("Area best test log10-median",     None,   best_tm["median_log10_err"]),
        ("CV→Test gap (log10 R²)",          0.3573, abs(results[best_name]["val_r2"]
                                                        - best_tm["r2_log10"])),
    ]
    for label, them, us in cmp:
        if them is None:
            print(f"  {label:<40} {'N/A':>10} {us:>10.4f}")
            continue
        if "gap" in label.lower():
            better = us < them
        else:
            better = us > them
        sym = "✓ OURS" if better else "✗ THEIRS"
        print(f"  {label:<40} {them:>10.4f} {us:>10.4f} {sym:>8}")

    # ── 14. SHAP ──────────────────────────────────────────────
    sname = next((n for n in ["lgbm","xgb","catboost","gbm","rf"]
                  if n in results and results[n]["model"] is not None), None)
    if sname and HAS_SHAP:
        print(f"\nSHAP on {sname}...")
        try:
            exp = shap.TreeExplainer(results[sname]["model"])
            sv  = exp.shap_values(X_te_s)
            imp = np.abs(sv).mean(axis=0)
            dfi = pd.DataFrame({"feature":keep,"shap":imp})\
                    .sort_values("shap",ascending=False)
            dfi.to_csv(os.path.join(MODEL_DIR,"area_shap_v3.csv"),index=False)
            mx = dfi["shap"].max()
            phy_in_top15 = [r["feature"] for _,r in dfi.head(15).iterrows()
                            if r["feature"] in PROTECTED_FEATURES]
            print(f"\n  Top-15 features (physics features marked ★):")
            print(f"  {'Feature':<45} {'SHAP':>8}")
            print(f"  {'-'*55}")
            for _,row in dfi.head(15).iterrows():
                bar = "█"*int(row["shap"]/mx*20)
                tag = " ★" if row["feature"] in PROTECTED_FEATURES else ""
                print(f"  {row['feature']:<45} {bar}  {row['shap']:.5f}{tag}")
            print(f"\n  Physics features in top-15: {len(phy_in_top15)}/15")
            print(f"  {phy_in_top15}")
            if len(phy_in_top15) >= 3:
                print(f"  ✓ Physics features are working!")
            else:
                print(f"  → Run with dedicated area CSV for better physics signal")
        except Exception as e:
            print(f"  SHAP failed: {e}")

    # ── 15. Save ──────────────────────────────────────────────
    pd.DataFrame({
        "y_true": y_te, "y_pred": results[best_name]["te_pred"],
        "log10_true": y_te_log,
        "log10_pred": np.log10(np.clip(results[best_name]["te_pred"],EPS,None)),
        "log10_err": np.abs(y_te_log -
                             np.log10(np.clip(results[best_name]["te_pred"],EPS,None))),
        "pct_err": np.abs((y_te - results[best_name]["te_pred"]) /
                           np.clip(y_te,EPS,None))*100,
    }).sort_values("log10_err",ascending=False)\
      .to_csv(os.path.join(MODEL_DIR,"test_predictions_v3.csv"),index=False)

    print(f"\n  Outputs → {MODEL_DIR}")
    print("=" * 70)
    print("  RESULT GUIDE (log10 R²):")
    print("    R²(log10) > 0.80, log10_median < 0.12 → excellent")
    print("    R²(log10) > 0.65, log10_median < 0.20 → good, beats teammate")
    print("    R²(log10) > 0.50, log10_median < 0.30 → baseline")
    print("\n  NEXT STEP:")
    print("    Run extract_rtl_area_features.py to get a dedicated area CSV.")
    print("    Point CSV_PATH at the new file and re-run for best results.")


if __name__ == "__main__":
    main()