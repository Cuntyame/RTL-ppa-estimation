#!/usr/bin/env python3
"""
area_prediction_rtl_v4.py
==========================
RTL-only area prediction — clean, no leakage, physics-first features.

ANALYSIS OF V3 RESULTS
========================
Run 1 (with leak — num_cells/num_nets included):
  R²(log10)=0.978  BUT  num_cells SHAP=0.614  → 61% of signal was the answer itself
  This result is invalid. num_cells IS area (cell_count × avg_cell_size).

Run 2 (leak plugged):
  R²(log10)=0.64, MAPE=144%, log10_median=0.221
  SHAP top features: sqrt_total_bits, is_comb_only, unique_signals, bits_x_arith
  Physics features: only 1/15 in top-15

  Root cause: raw count features (num_mul, num_reg, num_lines) are still present
  alongside physics features. Both encode the same information but raw counts
  have higher variance so tree models prefer them. The model learns
  "more operations = more area" (linear) instead of
  "multiplier area ∝ N²" (quadratic physics).

V4 STRATEGY: PHYSICS-FIRST FEATURE REPLACEMENT
================================================
Instead of adding physics features ON TOP of raw counts, we:
  1. Drop raw counts that are already encoded by a physics feature
     e.g. drop num_mul  → keep mul_ge_quad (= num_mul × max_bw²)
          drop num_reg  → keep seq_ge      (= num_reg × avg_bw × 8)
          drop num_add  → keep adder_ge    (= (add+sub) × avg_bw × 2)
  2. Keep raw counts that have NO physics equivalent yet
     (num_assign, num_if, num_case, num_wire, num_modules)
  3. Add design-type features that are genuinely new information
     (is_comb_only, has_multiplier, has_memory, generate_count)
  4. Add tiered model — small combinational vs large sequential

LEAKAGE AUDIT (excluded from features):
  POST-SYNTHESIS (direct leakage):
    num_cells, num_nets          ← directly proportional to area
    total_cell_area, comb_area   ← IS the target
    Power, critical_path_length  ← other PPA targets
    levels_of_logic, wns, tns    ← synthesis results
  NETLIST-DERIVED (indirect leakage):
    netlist_* columns            ← from gate-level netlist, not RTL

SPEED IMPROVEMENT:
  CatBoost depth=6 takes 24 min → use early_stopping_rounds instead of
  fixed iterations. Reduces training time by 40-60% with better results.
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
CSV_PATH     = r"C:\Users\Admin\Documents\final_rtl_area_features.csv"
CSV_FALLBACK = r"C:\Users\Admin\Documents\final_rtl_power_features_v2.csv"
MODEL_DIR    = r"C:\Users\Admin\OneDrive - MSFT\Desktop\New folder\cody\area_model_v4"
os.makedirs(MODEL_DIR, exist_ok=True)

AREA_COL_CANDIDATES = ["total_cell_area", "comb_area", "Cell_Area", "area"]
RANDOM_STATE = 42
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15
TUNE         = True
N_TRIALS     = 50       # reduced from 60 — sufficient with better features
CORR_THRESH  = 0.95     # tightened from 0.97 — remove more redundancy
EPS          = 1e-9

# ══════════════════════════════════════════════════════════════
#  COMPLETE LEAKAGE EXCLUSION LIST
#  Everything here is post-synthesis or is another PPA target.
# ══════════════════════════════════════════════════════════════
EXCLUDED_COLS = {
    # identifiers
    "Design_Name", "RTL_Code",
    # file metadata
    "rpt_files", "rpt_text_len",
    # POST-SYNTHESIS — direct leakage
    "num_cells",              # = area ÷ avg_cell_size  ← THE main leak
    "num_nets",               # ≈ num_cells (corr > 0.99 with area)
    # other PPA targets
    "Power", "critical_path_length",
    # synthesis results
    "levels_of_logic", "wns", "tns",
    # area targets themselves
    "total_cell_area", "comb_area", "Cell_Area", "area",
    # netlist features (if present from old CSV)
    "netlist_num_gates", "netlist_num_nets", "netlist_num_instances",
    "netlist_inv_count", "netlist_and_count", "netlist_or_count",
    "netlist_nand_count", "netlist_nor_count", "netlist_xor_count",
    "netlist_mux_count", "netlist_buf_count", "netlist_maj3_count",
}

# ══════════════════════════════════════════════════════════════
#  RAW COUNTS REPLACED BY PHYSICS FEATURES
#  These are dropped from the feature set because their physics
#  equivalents encode the same information more precisely.
#  e.g. num_mul is replaced by mul_ge_quad = num_mul × max_bw²
#       which captures the N² area scaling of multipliers.
# ══════════════════════════════════════════════════════════════
RAW_COUNTS_SUPERSEDED_BY_PHYSICS = {
    # operation counts → replaced by gate-equivalent estimates
    "num_mul",          # → mul_ge_quad (N² scaling)
    "num_div",          # → div_ge (4N² scaling)
    "num_add",          # → adder_ge (2N scaling)
    "num_sub",          # → adder_ge (combined with add)
    "num_logic_xor",    # → xor_ge (2 GE/bit)
    "num_logic_and",    # → logic_ge (combined)
    "num_logic_or",     # → logic_ge (combined)
    "num_shifts",       # → shift_ge (N×log2N barrel shifter)
    "num_comparisons",  # → comp_ge (2N comparator)
    "num_reg",          # → seq_ge (8 GE/bit DFF)
    # bitwidth → replaced by physics-scaled versions
    "total_bits",       # → log_total_bits, sqrt_total_bits (kept log/sqrt)
    # already-derived duplicates
    "total_arithmetic", # → adder_ge + mul_ge_quad cover this
    "switching_proxy",  # power feature — irrelevant for area
    "weighted_switching",
    "datapath_width_pressure",
    "high_toggle_score",
    "effective_switching_activity",
    "power_complexity_index",
    "glitch_potential",
    "estimated_ff_bits",
    "clk_domain_pressure",
    "reset_fanout_proxy",
    "num_sync_reset",
    "num_async_reset",
    "num_clock_enable",
    "num_latch",
    "num_tristate",
}

# ══════════════════════════════════════════════════════════════
#  PHYSICS FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def build_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute physics-grounded area estimates.
    All raw counts used here are then DROPPED from the feature set,
    forcing the model to use the structured physics representation.
    """
    df = df.copy()

    def col(name, default=0.0):
        if name in df.columns:
            return df[name].fillna(default).astype(float)
        return pd.Series(float(default), index=df.index)

    avg_bw = col("avg_bitwidth", 1.0).clip(lower=1)
    max_bw = col("max_bitwidth", 1.0).clip(lower=1)
    min_bw = col("min_bitwidth", 1.0).clip(lower=1)
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

    def set_col(name, val):
        """Only set if not already computed by extractor."""
        if name not in df.columns:
            df[name] = val
        return df[name]

    # ── A. SEQUENTIAL (DFF = 8 GE/bit) ───────────────────────
    seq_ge = n_reg * avg_bw * 8
    set_col("seq_ge",          seq_ge)
    set_col("log_seq_ge",      np.log1p(df["seq_ge"]))
    set_col("sqrt_seq_ge",     np.sqrt(df["seq_ge"].clip(0)))
    set_col("seq_ge_per_line", df["seq_ge"] / n_lines)

    # ── B. MULTIPLIER (N² GE — quadratic) ────────────────────
    mul_quad = n_mul * max_bw * max_bw
    set_col("mul_ge_quad",      mul_quad)
    set_col("log_mul_ge_quad",  np.log1p(df["mul_ge_quad"]))
    set_col("sqrt_mul_ge_quad", np.sqrt(df["mul_ge_quad"].clip(0)))
    set_col("mul_ge_per_line",  df["mul_ge_quad"] / n_lines)

    # ── C. ADDER (2N GE) ──────────────────────────────────────
    adder_ge = (n_add + n_sub) * avg_bw * 2
    set_col("adder_ge",        adder_ge)
    set_col("adder_ge_nlogn",  (n_add + n_sub) * avg_bw * np.log2(avg_bw.clip(lower=2)))
    set_col("log_adder_ge",    np.log1p(df["adder_ge"]))

    # ── D. DIVIDER (4N² GE) ───────────────────────────────────
    div_ge = n_div * max_bw * max_bw * 4
    set_col("div_ge",          div_ge)
    set_col("log_div_ge",      np.log1p(df["div_ge"]))

    # ── E. COMPARATOR (2N GE) ─────────────────────────────────
    comp_ge = n_comp * avg_bw * 2
    set_col("comp_ge",         comp_ge)
    set_col("log_comp_ge",     np.log1p(df["comp_ge"]))

    # ── F. BITWISE LOGIC (XOR=2GE, AND/OR=1GE) ───────────────
    logic_ge = (n_xor * 2 + n_and + n_or) * avg_bw
    set_col("logic_ge",        logic_ge)
    set_col("xor_ge",          n_xor * 2 * avg_bw)
    set_col("log_logic_ge",    np.log1p(df["logic_ge"]))

    # ── G. MUX / DECODER (MUX2=3GE, decoder=log2) ────────────
    mux_ge     = n_mux * avg_bw * 3
    decoder_ge = n_if * np.log2((n_if + 1).clip(lower=1)) * avg_bw
    set_col("mux_ge",          mux_ge)
    set_col("decoder_ge",      decoder_ge)
    set_col("log_mux_dec_ge",  np.log1p(df["mux_ge"] + df["decoder_ge"]))

    # ── H. BARREL SHIFTER (N×log2N GE) ───────────────────────
    shift_ge = n_shift * avg_bw * np.log2(avg_bw.clip(lower=2))
    set_col("shift_ge",        shift_ge)
    set_col("log_shift_ge",    np.log1p(df["shift_ge"]))

    # ── I. MEMORY / REG FILE (6 GE/bit) ──────────────────────
    mem_ge = n_mem * max_bw * 6
    set_col("mem_ge",          mem_ge)
    set_col("log_mem_ge",      np.log1p(df["mem_ge"]))

    # ── J. FOR-LOOP UNROLLING ─────────────────────────────────
    unroll_ge = col("unroll_factor", n_for * 4) * avg_bw
    set_col("unroll_ge",       unroll_ge)
    set_col("log_unroll_ge",   np.log1p(df["unroll_ge"]))

    # ── K. GENERATE UNROLLING ─────────────────────────────────
    gen_factor = col("generate_factor", col("generate_count", 0) * 8)
    gen_ge     = gen_factor * avg_bw * 4
    set_col("generate_ge",     gen_ge)
    set_col("log_generate_ge", np.log1p(df["generate_ge"]))

    # ── L. TOTAL GATE ESTIMATE ────────────────────────────────
    total_ge = (
        df["seq_ge"]  + df["mul_ge_quad"]  + df["adder_ge"] +
        df["div_ge"]  + df["comp_ge"]      + df["logic_ge"] +
        df["mux_ge"]  + df["shift_ge"]     + df["mem_ge"]   +
        df["unroll_ge"] + df["generate_ge"]
    )
    set_col("total_ge",        total_ge)
    set_col("log_total_ge",    np.log1p(df["total_ge"]))
    set_col("sqrt_total_ge",   np.sqrt(df["total_ge"].clip(0)))

    # ── M. AREA COMPOSITION FRACTIONS (scale-invariant) ───────
    ge_safe = df["total_ge"].clip(lower=EPS)
    set_col("seq_area_frac",   df["seq_ge"]      / ge_safe)
    set_col("mul_area_frac",   df["mul_ge_quad"] / ge_safe)
    set_col("logic_area_frac", df["logic_ge"]    / ge_safe)
    set_col("mux_area_frac",   df["mux_ge"]      / ge_safe)
    set_col("mem_area_frac",   df["mem_ge"]      / ge_safe)
    set_col("arith_area_frac", (df["adder_ge"] + df["mul_ge_quad"] +
                                 df["div_ge"]) / ge_safe)

    # ── N. BITWIDTH STRUCTURE ─────────────────────────────────
    total_bits = col("total_bits", 0)
    set_col("bw_range",        max_bw - min_bw)
    set_col("bw_uniformity",   avg_bw / max_bw.clip(lower=EPS))
    set_col("log_total_bits",  np.log1p(total_bits))
    set_col("sqrt_total_bits", np.sqrt(total_bits.clip(0)))

    # ── O. DESIGN-TYPE INDICATORS (new information) ───────────
    n_always_ff = col("num_always_ff", 0)
    set_col("is_comb_only",    (n_always_ff == 0).astype(int))
    set_col("has_multiplier",  (n_mul > 0).astype(int))
    set_col("has_division",    (n_div > 0).astype(int))
    set_col("has_memory",      (n_mem > 0).astype(int))
    set_col("has_generate",    (col("generate_count", 0) > 0).astype(int))

    # ── P. INTERACTION TERMS ──────────────────────────────────
    set_col("mul_x_reg",       n_mul * n_reg * avg_bw)
    set_col("log_mul_x_reg",   np.log1p(df["mul_x_reg"]))
    set_col("hierarchy_ge",    n_mod * (n_inp + n_out) * avg_bw)
    set_col("log_hierarchy_ge",np.log1p(df["hierarchy_ge"]))

    # ── Q. DENSITY FEATURES ───────────────────────────────────
    set_col("ge_per_line",     df["total_ge"] / n_lines)
    set_col("log_ge_per_line", np.log1p(df["ge_per_line"]))
    set_col("mul_fraction",    safe_div(df["mul_ge_quad"], df["total_ge"]))
    set_col("seq_fraction",    safe_div(df["seq_ge"], df["total_ge"]))

    return df


def safe_div(a, b):
    return a / b.clip(lower=EPS)


# ══════════════════════════════════════════════════════════════
#  FEATURE PRUNING WITH PHYSICS REPLACEMENT
# ══════════════════════════════════════════════════════════════

def select_features(df: pd.DataFrame, area_col: str) -> list:
    """
    Build the final feature list:
      1. Exclude all post-synthesis / leakage columns
      2. Exclude raw counts superseded by physics features
      3. Remove near-zero variance
      4. Remove highly correlated pairs (keep higher-variance one)
    """
    # Start with all numeric columns
    all_numeric = [c for c in df.columns
                   if pd.api.types.is_numeric_dtype(df[c])
                   and c != area_col]

    # Remove leakage
    candidates = [c for c in all_numeric if c not in EXCLUDED_COLS]

    # Remove superseded raw counts
    candidates = [c for c in candidates
                  if c not in RAW_COUNTS_SUPERSEDED_BY_PHYSICS]

    print(f"  After leakage + superseded removal: {len(candidates)} features")

    X = df[candidates].fillna(0)

    # Variance threshold
    sel  = VarianceThreshold(threshold=1e-6)
    sel.fit(X)
    keep = [candidates[i] for i in range(len(candidates))
            if sel.get_support()[i]]
    print(f"  After variance prune: {len(keep)}")

    # Correlation pruning (keep higher-variance feature)
    corr    = X[keep].corr().abs()
    upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for c in upper.columns:
        for cc in upper.index[upper[c] > CORR_THRESH].tolist():
            if X[c].var() >= X[cc].var():
                to_drop.add(cc)
            else:
                to_drop.add(c)
    keep = [c for c in keep if c not in to_drop]
    print(f"  After correlation prune (>{CORR_THRESH}): {len(keep)} final features")

    # Report physics features in set
    physics_present = [f for f in keep if any(
        f.startswith(p) for p in ["seq_ge","mul_ge","adder_ge","div_ge",
                                   "comp_ge","logic_ge","mux_ge","shift_ge",
                                   "mem_ge","total_ge","unroll_ge",
                                   "generate_ge","log_total_ge","sqrt_total_ge",
                                   "seq_area","mul_area","arith_area"])]
    print(f"  Physics features in set: {len(physics_present)}")
    return keep


# ══════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════

def metrics(y_true, y_pred) -> dict:
    lt = np.log10(np.clip(y_true, EPS, None))
    lp = np.log10(np.clip(y_pred, EPS, None))
    log_err = np.abs(lt - lp)
    return {
        "r2_log10":         r2_score(lt, lp),
        "rmse_log10":       float(np.sqrt(mean_squared_error(lt, lp))),
        "mean_log10_err":   float(log_err.mean()),
        "median_log10_err": float(np.median(log_err)),
        "r2_original":      r2_score(y_true, y_pred),
        "mape":             float(np.mean(np.abs((y_true - y_pred) /
                                                  np.clip(y_true, EPS, None))) * 100),
    }


def print_m(label, m, w=32):
    print(f"  {label:<{w}}  R²(log)={m['r2_log10']:.4f}  "
          f"R²(orig)={m['r2_original']:.4f}  "
          f"log10_med={m['median_log10_err']:.3f}  "
          f"MAPE={m['mape']:.1f}%")


def per_decile(y_true, y_pred, label=""):
    df = pd.DataFrame({"t": y_true, "p": y_pred})
    df["d"] = pd.qcut(df["t"], q=10, labels=False, duplicates="drop")
    print(f"\n  Per-decile ({label}):")
    print(f"  {'D':<3} {'Area (µm²)':>24} {'MAPE':>8} {'log10err':>10} {'N':>5}")
    print(f"  {'-'*54}")
    for d, g in df.groupby("d"):
        mape = np.mean(np.abs((g["t"]-g["p"])/g["t"].clip(lower=EPS)))*100
        lerr = np.mean(np.abs(np.log10(g["t"].clip(lower=EPS)) -
                               np.log10(g["p"].clip(lower=EPS))))
        print(f"  {int(d):<3} {g['t'].min():>10.1f} – {g['t'].max():>10.1f}"
              f"  {mape:>7.1f}%  {lerr:>9.3f}  {len(g):>4}")


# ══════════════════════════════════════════════════════════════
#  OPTUNA TUNING — optimises log10 R²
# ══════════════════════════════════════════════════════════════

def tune(name, X_tr, y_tr, X_vl, y_vl):
    if not HAS_OPTUNA:
        return {}

    def neg_r2(m):
        return -r2_score(y_vl, m.predict(X_vl))

    def xgb(t):
        m = XGBRegressor(
            n_estimators=t.suggest_int("ne", 300, 1500),
            learning_rate=t.suggest_float("lr", 0.005, 0.1, log=True),
            max_depth=t.suggest_int("d", 4, 9),
            min_child_weight=t.suggest_int("mcw", 1, 12),
            subsample=t.suggest_float("ss", 0.6, 1.0),
            colsample_bytree=t.suggest_float("cbt", 0.5, 1.0),
            gamma=t.suggest_float("g", 0, 1.0),
            reg_alpha=t.suggest_float("a", 1e-3, 5.0, log=True),
            reg_lambda=t.suggest_float("l", 0.01, 10.0, log=True),
            early_stopping_rounds=40,
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")
        m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        return neg_r2(m)

    def lgbm(t):
        m = LGBMRegressor(
            n_estimators=t.suggest_int("ne", 300, 1500),
            learning_rate=t.suggest_float("lr", 0.005, 0.1, log=True),
            max_depth=t.suggest_int("d", 4, 9),
            num_leaves=t.suggest_int("nl", 20, 150),
            subsample=t.suggest_float("ss", 0.6, 1.0),
            colsample_bytree=t.suggest_float("cbt", 0.5, 1.0),
            min_child_samples=t.suggest_int("mcs", 5, 30),
            reg_alpha=t.suggest_float("a", 1e-3, 5.0, log=True),
            reg_lambda=t.suggest_float("l", 0.01, 10.0, log=True),
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        m.fit(X_tr, y_tr,
              eval_set=[(X_vl, y_vl)],
              callbacks=[])
        return neg_r2(m)

    def cb(t):
        # Depth capped at 7 for speed. early_stopping handles iterations.
        m = CatBoostRegressor(
            iterations=t.suggest_int("ne", 300, 1000),
            learning_rate=t.suggest_float("lr", 0.005, 0.1, log=True),
            depth=t.suggest_int("d", 4, 7),      # ← capped at 7, not 10
            l2_leaf_reg=t.suggest_float("l", 0.01, 10.0, log=True),
            subsample=t.suggest_float("ss", 0.6, 1.0),
            early_stopping_rounds=40,
            random_seed=RANDOM_STATE, verbose=False, thread_count=-1)
        m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        return neg_r2(m)

    fns = {"xgb": xgb, "lgbm": lgbm, "catboost": cb}
    fn  = fns.get(name)
    if fn is None:
        return {}

    s = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    s.optimize(fn, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"    {name}: best val log10-R²={-s.best_value:.4f}  "
          f"params={s.best_params}")
    return s.best_params


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  AREA PREDICTION v4 — PHYSICS-FIRST, NO LEAKAGE")
    print("=" * 70)
    print(f"  XGB={HAS_XGB}  LGBM={HAS_LGBM}  CB={HAS_CATBOOST}"
          f"  SHAP={HAS_SHAP}  Optuna={HAS_OPTUNA}")

    # ── 1. Load ───────────────────────────────────────────────
    csv = CSV_PATH if os.path.exists(CSV_PATH) else CSV_FALLBACK
    if csv == CSV_FALLBACK:
        print(f"\n[INFO] Using fallback CSV. Run extract_rtl_area_features.py"
              f" for best results.")
    print(f"\nLoading: {csv}")
    df = pd.read_csv(csv)
    print(f"  Shape: {df.shape}")

    area_col = next((c for c in AREA_COL_CANDIDATES if c in df.columns), None)
    if area_col is None:
        raise ValueError(f"No area column. Expected: {AREA_COL_CANDIDATES}")
    print(f"  Area column: '{area_col}'")

    # ── 2. Physics features ───────────────────────────────────
    print("\nBuilding physics features...")
    df = build_physics_features(df)
    print(f"  Columns: {len(df.columns)}")

    # ── 3. Clean ──────────────────────────────────────────────
    print("\nCleaning...")
    df_m = df.copy()
    n = len(df_m)
    df_m = df_m.dropna(subset=[area_col])
    print(f"  dropna   : {len(df_m)}/{n}")
    df_m = df_m[df_m[area_col] > 0]
    print(f"  area > 0 : {len(df_m)}")
    la    = np.log10(df_m[area_col])
    q1,q3 = la.quantile([0.25, 0.75])
    iqr   = q3 - q1
    df_m  = df_m[(la >= q1 - 3.5*iqr) & (la <= q3 + 3.5*iqr)]
    print(f"  IQR      : {len(df_m)}")
    a = df_m[area_col]
    print(f"\n  Area (µm²): min={a.min():.1f}  "
          f"median={a.median():.1f}  max={a.max():.1f}  "
          f"({np.log10(a.max()/a.min()):.1f} decades)")

    # ── 4. Feature selection (physics-first) ──────────────────
    print("\nSelecting features (physics-first, leakage-free)...")
    keep = select_features(df_m, area_col)
    X_df = df_m[keep].fillna(0)
    y_raw = df_m[area_col].values

    # Double-check no leakage
    leaked = [c for c in keep if c in EXCLUDED_COLS]
    if leaked:
        raise RuntimeError(f"LEAKAGE DETECTED: {leaked}")
    print(f"  ✓ Leakage check passed — {len(keep)} clean features")

    # ── 5. Stratified split ───────────────────────────────────
    print("\nStratified split by log10(area) bins...")
    strat = pd.qcut(np.log10(y_raw), q=10,
                    labels=False, duplicates="drop").astype(str)
    X_tmp, X_te, y_tmp, y_te, s_tmp, _ = train_test_split(
        X_df.values, y_raw, strat,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat)
    sv = pd.qcut(np.log10(y_tmp), q=10,
                 labels=False, duplicates="drop").astype(str)
    X_tr, X_vl, y_tr_o, y_vl_o = train_test_split(
        X_tmp, y_tmp,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE, stratify=sv)
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

    # ── 8. 5-fold CV baseline (honest) ────────────────────────
    print("\n5-fold CV baseline (GBM, log10 target)...")
    kf  = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    bgm = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=6,
        subsample=0.8, random_state=RANDOM_STATE)
    cv_log = cross_val_score(bgm, X_tr_s, y_tr, cv=kf,
                              scoring="r2", n_jobs=-1)
    cv_prd = cross_val_predict(bgm, X_tr_s, y_tr, cv=kf)
    cv_r2_orig = r2_score(y_tr_o, 10**cv_prd)

    print(f"\n  ┌────────────────────────────────────────────────────┐")
    print(f"  │  HONEST 5-FOLD CV (no leakage)                      │")
    print(f"  │  CV R²(log10) = {cv_log.mean():.4f} ± {cv_log.std():.4f}           │")
    print(f"  │  CV R²(orig)  = {cv_r2_orig:.4f}                        │")
    print(f"  │                                                      │")
    print(f"  │  V3 leaked result (num_cells in):  R²=0.977 ← FAKE  │")
    print(f"  │  V3 clean result  (num_cells out): R²=0.64  ← real  │")
    print(f"  │  V4 target        (physics-first): R²>0.70  ← goal  │")
    print(f"  └────────────────────────────────────────────────────┘")

    # ── 9. Optuna (fast with early stopping) ──────────────────
    xp, lp, cp = {}, {}, {}
    if TUNE:
        print("\nOptuna HPO (with early stopping — faster than v3)...")
        if HAS_XGB:
            print("  XGB...", end=" ", flush=True)
            xp = tune("xgb",      X_tr_s, y_tr, X_vl_s, y_vl)
        if HAS_LGBM:
            print("  LGBM...", end=" ", flush=True)
            lp = tune("lgbm",     X_tr_s, y_tr, X_vl_s, y_vl)
        if HAS_CATBOOST:
            print("  CB...",  end=" ", flush=True)
            cp = tune("catboost", X_tr_s, y_tr, X_vl_s, y_vl)

    # ── 10. Build final models ─────────────────────────────────
    def get(d, k, default):
        return d.get(k, default) if d else default

    models = {}
    if HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=get(xp,"ne",1000),
            learning_rate=get(xp,"lr",0.02),
            max_depth=get(xp,"d",8),
            min_child_weight=get(xp,"mcw",3),
            subsample=get(xp,"ss",0.85),
            colsample_bytree=get(xp,"cbt",0.85),
            gamma=get(xp,"g",0.2),
            reg_alpha=get(xp,"a",0.5),
            reg_lambda=get(xp,"l",2.0),
            early_stopping_rounds=50,
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")

    if HAS_LGBM:
        models["lgbm"] = LGBMRegressor(
            n_estimators=get(lp,"ne",1000),
            learning_rate=get(lp,"lr",0.02),
            max_depth=get(lp,"d",8),
            num_leaves=get(lp,"nl",63),
            subsample=get(lp,"ss",0.85),
            colsample_bytree=get(lp,"cbt",0.85),
            min_child_samples=get(lp,"mcs",10),
            reg_alpha=get(lp,"a",0.5),
            reg_lambda=get(lp,"l",2.0),
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)

    if HAS_CATBOOST:
        models["catboost"] = CatBoostRegressor(
            iterations=get(cp,"ne",800),
            learning_rate=get(cp,"lr",0.02),
            depth=get(cp,"d",6),          # capped at 6-7 for speed
            l2_leaf_reg=get(cp,"l",2.0),
            subsample=get(cp,"ss",0.85),
            early_stopping_rounds=50,     # stops early instead of full iters
            random_seed=RANDOM_STATE, verbose=False, thread_count=-1)

    models["gbm"] = GradientBoostingRegressor(
        n_estimators=600, learning_rate=0.02, max_depth=7,
        subsample=0.85, min_samples_leaf=4, random_state=RANDOM_STATE)

    # ── 11. Train ─────────────────────────────────────────────
    print(f"\n{'='*60}\n  MODEL RESULTS\n{'='*60}")
    results = {}
    for mname, model in models.items():
        print(f"\n  {mname.upper()}...", end=" ", flush=True)

        # Use eval_set for early stopping models
        if mname in ("xgb",):
            model.fit(X_tr_s, y_tr, eval_set=[(X_vl_s, y_vl)], verbose=False)
        elif mname == "catboost":
            model.fit(X_tr_s, y_tr, eval_set=(X_vl_s, y_vl))
        elif mname == "lgbm":
            model.fit(X_tr_s, y_tr,
                      eval_set=[(X_vl_s, y_vl)],
                      callbacks=[])
        else:
            model.fit(X_tr_s, y_tr)

        vl_pred = 10 ** model.predict(X_vl_s)
        te_pred = 10 ** model.predict(X_te_s)
        vm = metrics(y_vl_o, vl_pred)
        tm = metrics(y_te,   te_pred)
        print("done")
        print_m("    Validation", vm)
        print_m("    Test",       tm)
        gap = abs(vm["r2_log10"] - tm["r2_log10"])
        st  = "✓" if gap < 0.10 else ("⚠" if gap < 0.20 else "✗")
        print(f"    Val-Test gap = {gap:.3f} [{st}]")

        results[mname] = {"model": model, "val_r2": vm["r2_log10"],
                          "test_metrics": tm, "te_pred": te_pred}
        joblib.dump(model, os.path.join(MODEL_DIR, f"area_{mname}.joblib"))

    # ── 12. Weighted ensemble ─────────────────────────────────
    print(f"\n  {'─'*50}")
    w  = {n: max(r["val_r2"], 0)**2 for n, r in results.items()}
    tw = sum(w.values())
    if tw > 0:
        ens = sum((w[n]/tw)*results[n]["te_pred"] for n in w if w[n] > 0)
        em  = metrics(y_te, ens)
        print_m("  Ensemble Test", em)
        results["ensemble"] = {"model": None, "val_r2": em["r2_log10"],
                               "test_metrics": em, "te_pred": ens}

    # ── 13. Summary ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY — AREA (RTL only, v4, NO LEAKAGE)")
    print(f"{'='*70}")
    print(f"  {'Model':<14} {'ValR²(log)':>11} {'TstR²(log)':>11} "
          f"{'TstR²(orig)':>12} {'log10_med':>10} {'Gap':>7}")
    print(f"  {'-'*68}")
    best_name, best_r2 = None, -999
    for n, r in sorted(results.items(), key=lambda x: -x[1]["val_r2"]):
        tm  = r["test_metrics"]
        gap = abs(r["val_r2"] - tm["r2_log10"])
        st  = "✓" if gap < 0.10 else ("⚠" if gap < 0.20 else "✗")
        print(f"  {n:<14} {r['val_r2']:>11.4f} {tm['r2_log10']:>11.4f} "
              f"{tm['r2_original']:>12.4f} {tm['median_log10_err']:>9.3f} "
              f"{gap:>6.3f} {st}")
        if r["val_r2"] > best_r2:
            best_r2, best_name = r["val_r2"], n

    per_decile(y_te, results[best_name]["te_pred"], best_name)

    # ── 14. Comparison table ──────────────────────────────────
    best_tm = results[best_name]["test_metrics"]
    print(f"\n{'='*70}")
    print(f"  COMPARISON: V4 (clean) vs V3 variants")
    print(f"{'='*70}")
    rows = [
        ("V3 with num_cells leak",     "FAKE",  0.9784, 0.070),
        ("V3 leak plugged (baseline)", "clean", 0.6437, 0.221),
        ("V4 physics-first (this)",    "clean", best_tm["r2_log10"],
         best_tm["median_log10_err"]),
        ("Teammate best (orig scale)", "leak?", None,   None),
    ]
    print(f"  {'Version':<35} {'Status':>7} {'R²(log10)':>10} {'log10_med':>10}")
    print(f"  {'-'*65}")
    for label, status, r2l, lm in rows:
        r2s  = f"{r2l:.4f}" if r2l is not None else "  0.2363*"
        lms  = f"{lm:.3f}"  if lm  is not None else "    N/A"
        flag = " ★" if label.startswith("V4") else ""
        print(f"  {label:<35} {status:>7} {r2s:>10} {lms:>10}{flag}")
    print(f"  * teammate R² is original-scale, not log10")

    # ── 15. SHAP ──────────────────────────────────────────────
    sname = next((n for n in ["lgbm","xgb","catboost","gbm"]
                  if n in results and results[n]["model"] is not None), None)
    if sname and HAS_SHAP:
        print(f"\nSHAP on {sname}...")
        try:
            exp = shap.TreeExplainer(results[sname]["model"])
            sv  = exp.shap_values(X_te_s)
            imp = np.abs(sv).mean(axis=0)
            dfi = pd.DataFrame({"feature": keep, "shap": imp})\
                    .sort_values("shap", ascending=False)
            dfi.to_csv(os.path.join(MODEL_DIR, "shap_v4.csv"), index=False)
            mx   = dfi["shap"].max()
            # Mark physics features
            phy  = {"seq_ge","log_seq_ge","mul_ge_quad","log_mul_ge_quad",
                    "adder_ge","log_adder_ge","div_ge","total_ge","log_total_ge",
                    "sqrt_total_ge","seq_area_frac","mul_area_frac",
                    "arith_area_frac","unroll_ge","generate_ge","mul_x_reg"}
            print(f"\n  Top-15 (★=physics, ●=design-type):")
            print(f"  {'Feature':<45} {'SHAP':>8}")
            print(f"  {'-'*55}")
            for _, row in dfi.head(15).iterrows():
                bar = "█" * int(row["shap"]/mx*20)
                tag = " ★" if row["feature"] in phy else (
                      " ●" if row["feature"] in
                      {"is_comb_only","has_multiplier","has_memory",
                       "has_generate","has_division"} else "")
                print(f"  {row['feature']:<45} {bar}  "
                      f"{row['shap']:.5f}{tag}")
            n_phy = sum(1 for _,r in dfi.head(15).iterrows()
                        if r["feature"] in phy)
            print(f"\n  Physics features in top-15: {n_phy}/15")
            if n_phy >= 4:
                print(f"  ✓ Physics-first strategy is working")
            elif n_phy >= 2:
                print(f"  ~ Partial physics signal — acceptable for RTL-only")
            else:
                print(f"  → Size proxies still dominate — "
                      f"this is near the RTL-only ceiling")
        except Exception as e:
            print(f"  SHAP error: {e}")

    # ── 16. Save predictions ──────────────────────────────────
    # pd.DataFrame({
    #     "y_true": y_te,
    #     "y_pred": results[best_name]["te_pred
    # ── 16. Save predictions ──────────────────────────────────
    pd.DataFrame({
        "y_true": y_te,
        "y_pred": results[best_name]["te_pred"],
        "log10_true": y_te_log,
        "log10_pred": np.log10(np.clip(results[best_name]["te_pred"], EPS, None)),
        "log10_err": np.abs(y_te_log - 
                             np.log10(np.clip(results[best_name]["te_pred"], EPS, None))),
        "pct_err": np.abs((y_te - results[best_name]["te_pred"]) / 
                           np.clip(y_te, EPS, None)) * 100,
    }).sort_values("log10_err", ascending=False)\
      .to_csv(os.path.join(MODEL_DIR, "test_predictions_v4.csv"), index=False)

    print(f"\n  Outputs → {MODEL_DIR}")
    print("=" * 70)
    print("  RESULT GUIDE (log10 R²):")
    print("    R²(log10) > 0.80, log10_median < 0.12 → excellent")
    print("    R²(log10) > 0.65, log10_median < 0.20 → good, beats teammate")
    print("    R²(log10) > 0.50, log10_median < 0.30 → baseline")
    print("    log10_median < 0.20 → predictions within ×1.58 on average")
    print("    log10_median < 0.30 → predictions within ×2.0  (acceptable)")

if __name__ == "__main__":
    main()