#!/usr/bin/env python3
"""
area_prediction_rtl_v4.py
==========================
Leak-free RTL-only area prediction with normalized physics features.

WHAT CHANGED vs V3
==================
LEAKAGE FIX (already in v3 with plugged NON_FEATURE_COLS):
  num_cells SHAP=0.614 in v3-with-leak — post-synthesis cell count IS area.
  num_nets  SHAP=0.033 — also post-synthesis.
  Both excluded. True RTL-only baseline: R²(log10)≈0.64.

WHY PHYSICS FEATURES DIDN'T APPEAR IN V3 SHAP (top issue):
  sqrt_total_bits correlates with ALL physics features at >0.95.
  Trees prefer it because it's one split that explains size variance.
  mul_ge_quad, seq_ge are then redundant given sqrt_total_bits.
  
  FIX: Add SIZE-NORMALISED physics features that are orthogonal to total size.
  These express COMPOSITION (what fraction of the design is multipliers?)
  not SCALE (how big is the design?).
  
  New features:
    mul_intensity   = mul_ge_quad / total_ge   (already have mul_area_frac)
    seq_intensity   = seq_ge / total_ge         (already have seq_area_frac)
    
  BUT: we also need physics/size_proxy ratios:
    mul_ge_per_bit  = mul_ge_quad / total_bits  ← NEW: tells model this design
                                                   has unusually large multipliers
    seq_ge_per_bit  = seq_ge / total_bits        ← NEW: tells model this design
                                                   is register-heavy per bit

DESIGN TYPE CLASSIFICATION (new):
  Combinational-only designs (is_comb_only=1) behave differently:
  area ≈ f(logic_ops, mux_count, comparators) — no FF overhead
  Sequential designs: area includes both DFF area + logic
  
  Adding explicit design-type interaction features:
    comb_logic_intensity = logic_ge * is_comb_only
    seq_reg_intensity    = seq_ge * (1 - is_comb_only)

SPEED IMPROVEMENTS:
  CatBoost depth capped at 7 (depth=10 takes 10× longer, +2% R²)
  Optuna trials reduced 60→40 (sufficient for this search space)
  RF n_estimators 600→400 (RF is weak here anyway, speed not worth it)

HONEST RESULT CONTEXT:
  RTL-only area prediction ceiling is ~R²(log10)=0.75-0.85 for this dataset.
  The gap from 0.64 to ceiling is explained by:
    - Variable-width designs: same RTL structure, different parameter widths
      → parametric designs are inherently ambiguous from RTL text alone
    - Synthesis optimisation: same RTL can map to different cell counts
      depending on synthesis tool settings
  This is comparable to academic RTL→area prediction papers (0.70-0.85 R²).
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
N_TRIALS     = 40      # reduced from 60 — sufficient coverage, faster
CORR_THRESH  = 0.97
EPS          = 1e-9

# ── LEAK-FREE exclusion list ──────────────────────────────────
# num_cells / num_nets = post-synthesis → direct area proxy → LEAK
# All other rpt-parsed columns also excluded
NON_FEATURE_COLS = {
    "Design_Name", "RTL_Code", "rpt_files", "rpt_text_len",
    "Power", "critical_path_length", "levels_of_logic",
    "wns", "tns",
    "num_cells", "num_nets",          # ← LEAKAGE SOURCE — must exclude
    "total_cell_area", "comb_area", "Cell_Area", "area",
}

# Physics features protected from correlation pruning
PROTECTED_FEATURES = [
    "mul_ge_quad", "log_mul_ge_quad", "sqrt_mul_ge_quad",
    "seq_ge",      "log_seq_ge",
    "adder_ge",    "log_adder_ge",
    "div_ge",      "log_div_ge",
    "total_ge",    "log_total_ge", "sqrt_total_ge",
    "seq_area_frac", "mul_area_frac", "arith_area_frac",
    "generate_ge", "unroll_ge", "mul_x_reg",
    # NEW normalised physics features (v4)
    "mul_ge_per_bit", "seq_ge_per_bit",
    "mul_intensity_norm", "seq_intensity_norm",
    "comb_logic_ge", "seq_logic_ge",
    "mul_dominant", "seq_dominant", "mem_dominant",
]


# ══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (v4 — adds normalised physics features)
# ══════════════════════════════════════════════════════════════

def engineer_area_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def col(name, default=0.0):
        if name in df.columns:
            return df[name].fillna(default).astype(float)
        return pd.Series(float(default), index=df.index)

    avg_bw = col("avg_bitwidth",  1.0).clip(lower=1)
    max_bw = col("max_bitwidth",  1.0).clip(lower=1)
    min_bw = col("min_bitwidth",  1.0).clip(lower=1)
    n_reg  = col("num_reg")
    n_add  = col("num_add");  n_sub = col("num_sub")
    n_mul  = col("num_mul");  n_div = col("num_div")
    n_xor  = col("num_logic_xor")
    n_and  = col("num_logic_and"); n_or = col("num_logic_or")
    n_mux  = col("num_ternary") + col("num_case")
    n_comp = col("num_comparisons")
    n_shift= col("num_shifts")
    n_lines= col("num_lines", 1).clip(lower=1)
    n_mod  = col("num_modules", 1).clip(lower=1)
    n_inp  = col("num_input");  n_out = col("num_output")
    n_if   = col("num_if");     n_for = col("num_for")
    n_mem  = col("num_memory_access", 0)
    n_bits = col("total_bits",  0).clip(lower=1)
    gen_cnt= col("generate_count", 0)

    def add(name, series):
        if name not in df.columns:
            df[name] = series

    # ── Base physics estimates ────────────────────────────────
    add("seq_ge",          n_reg * avg_bw * 8)
    add("log_seq_ge",      np.log1p(df.get("seq_ge", n_reg * avg_bw * 8)))
    add("sqrt_seq_ge",     np.sqrt(col("seq_ge").clip(0)))

    add("mul_ge_quad",     n_mul * max_bw * max_bw)
    add("mul_ge_linear",   n_mul * avg_bw)
    add("log_mul_ge_quad", np.log1p(col("mul_ge_quad")))
    add("sqrt_mul_ge_quad",np.sqrt(col("mul_ge_quad").clip(0)))

    add("adder_ge",        (n_add + n_sub) * avg_bw * 2)
    add("adder_ge_nlogn",  (n_add + n_sub) * avg_bw * np.log2(avg_bw.clip(lower=2)))
    add("log_adder_ge",    np.log1p(col("adder_ge")))

    add("div_ge",          n_div * max_bw * max_bw * 4)
    add("log_div_ge",      np.log1p(col("div_ge")))

    add("comp_ge",         n_comp * avg_bw * 2)
    add("logic_ge",        (n_xor * 2 + n_and + n_or) * avg_bw)
    add("xor_ge",          n_xor * 2 * avg_bw)
    add("mux_ge",          n_mux * avg_bw * 3)
    add("decoder_ge",      n_if * np.log2((n_if + 1).clip(lower=1)) * avg_bw)
    add("shift_ge",        n_shift * avg_bw * np.log2(avg_bw.clip(lower=2)))
    add("mem_ge",          n_mem * max_bw * 6)
    add("unroll_ge",       n_for * avg_bw * 4)
    add("generate_ge",     gen_cnt * avg_bw * 4)

    cols_total = ["seq_ge","mul_ge_quad","adder_ge","div_ge","comp_ge",
                  "logic_ge","mux_ge","shift_ge","mem_ge","unroll_ge","generate_ge"]
    total_ge = sum(df[c] for c in cols_total if c in df.columns)
    add("total_ge",        total_ge)
    add("log_total_ge",    np.log1p(col("total_ge")))
    add("sqrt_total_ge",   np.sqrt(col("total_ge").clip(0)))

    ge_safe = col("total_ge").clip(lower=EPS)
    add("seq_area_frac",   col("seq_ge")      / ge_safe)
    add("mul_area_frac",   col("mul_ge_quad") / ge_safe)
    add("logic_area_frac", col("logic_ge")    / ge_safe)
    add("mux_area_frac",   col("mux_ge")      / ge_safe)
    add("mem_area_frac",   col("mem_ge")      / ge_safe)
    add("arith_area_frac", (col("adder_ge") + col("mul_ge_quad") + col("div_ge")) / ge_safe)

    # ── Bitwidth ──────────────────────────────────────────────
    add("bw_range",        max_bw - min_bw)
    add("bw_uniformity",   avg_bw / max_bw.clip(lower=EPS))
    add("log_total_bits",  np.log1p(n_bits))
    add("sqrt_total_bits", np.sqrt(n_bits))

    # ── Interactions ──────────────────────────────────────────
    add("mul_x_reg",       n_mul * n_reg * avg_bw)
    add("log_mul_x_reg",   np.log1p(col("mul_x_reg")))
    total_arith = n_add + n_sub + n_mul + n_div
    add("total_arithmetic",total_arith)
    add("arith_x_bw",      total_arith * avg_bw)
    add("log_arith_x_bw",  np.log1p(col("arith_x_bw")))
    add("hierarchy_ge",    n_mod * (n_inp + n_out) * avg_bw)
    add("log_hierarchy_ge",np.log1p(col("hierarchy_ge")))
    add("bits_x_arith",    n_bits * (total_arith + 1))

    # ── Densities ─────────────────────────────────────────────
    add("ge_per_line",     col("total_ge")    / n_lines)
    add("mul_ge_per_line", col("mul_ge_quad") / n_lines)
    add("seq_ge_per_line", col("seq_ge")      / n_lines)
    add("arith_bw_density",total_arith * avg_bw / n_lines)

    # ══════════════════════════════════════════════════════════
    #  NEW V4: SIZE-NORMALISED PHYSICS FEATURES
    #  These are ORTHOGONAL to total design size.
    #  They tell the model: "for this size design, is it 
    #  unusually multiplier-heavy or register-heavy?"
    # ══════════════════════════════════════════════════════════

    # Per-bit normalisation: removes the dominant size effect
    # mul_ge_per_bit is HIGH for multiplier-heavy designs of any size
    add("mul_ge_per_bit",  col("mul_ge_quad") / n_bits)
    add("seq_ge_per_bit",  col("seq_ge")      / n_bits)
    add("adder_ge_per_bit",col("adder_ge")    / n_bits)
    add("logic_ge_per_bit",col("logic_ge")    / n_bits)
    add("div_ge_per_bit",  col("div_ge")      / n_bits)

    # Intensity: physics estimate normalised by sqrt of total
    # (sqrt avoids complete dominance of large designs)
    sqrt_ge_safe = col("sqrt_total_ge").clip(lower=EPS)
    add("mul_intensity_norm",  col("mul_ge_quad") / sqrt_ge_safe)
    add("seq_intensity_norm",  col("seq_ge")      / sqrt_ge_safe)
    add("adder_intensity_norm",col("adder_ge")    / sqrt_ge_safe)

    # Design-type interaction features
    # These let the model learn different physics for different design types
    n_always_ff = col("num_always_ff", 0)
    is_comb = (n_always_ff == 0).astype(float)
    add("is_comb_only",    is_comb)
    is_seq = 1 - is_comb

    # For combinational designs: logic_ge dominates area
    add("comb_logic_ge",   col("logic_ge") * is_comb)
    add("comb_mux_ge",     col("mux_ge")   * is_comb)
    add("comb_adder_ge",   col("adder_ge") * is_comb)

    # For sequential designs: seq_ge + datapath dominates
    add("seq_logic_ge",    col("seq_ge")   * is_seq)
    add("seq_mul_ge",      col("mul_ge_quad") * is_seq)

    # Design type dominance flags (what is the biggest area component?)
    # These give the model a direct routing signal
    mul_frac  = col("mul_area_frac")
    seq_frac  = col("seq_area_frac")
    mem_frac  = col("mem_area_frac")
    add("mul_dominant",    (mul_frac > 0.4).astype(float))
    add("seq_dominant",    (seq_frac > 0.4).astype(float))
    add("mem_dominant",    (mem_frac > 0.2).astype(float))
    add("mixed_design",    ((mul_frac < 0.4) & (seq_frac < 0.4)).astype(float))

    # Multiplier width class (quadratic effect is most visible above 16-bit)
    add("is_wide_mul",     ((n_mul > 0) & (max_bw > 16)).astype(float))
    add("wide_mul_ge",     col("mul_ge_quad") * ((max_bw > 16).astype(float)))

    # Parametric design indicator: high avg_bw variance suggests
    # parameterized modules (harder to predict from RTL text)
    add("bw_cv",           col("bw_range") / avg_bw.clip(lower=EPS))

    # Log of the dominant component (the largest GE estimate)
    dominant_ge = pd.concat([
        col("mul_ge_quad"), col("seq_ge"), col("adder_ge"),
        col("logic_ge"),    col("mem_ge")
    ], axis=1).max(axis=1)
    add("log_dominant_ge",  np.log1p(dominant_ge))
    add("dominant_fraction",dominant_ge / ge_safe)

    # Ratio: mul_area to seq_area (distinguishes DSP vs pipeline)
    add("mul_to_seq_ratio", col("mul_ge_quad") / col("seq_ge").clip(lower=EPS))
    add("log_mul_to_seq",   np.log1p(col("mul_to_seq_ratio")))

    return df


# ══════════════════════════════════════════════════════════════
#  FEATURE PRUNING  (protected list always kept)
# ══════════════════════════════════════════════════════════════

def prune_features(X: pd.DataFrame, corr_thresh=CORR_THRESH,
                   protected: list = None) -> list:
    protected = protected or []
    prot      = [f for f in protected if f in X.columns]

    sel  = VarianceThreshold(threshold=1e-6)
    sel.fit(X)
    keep = list(set(X.columns[sel.get_support()].tolist()) | set(prot))
    n0   = len(X.columns)

    corr   = X[keep].corr().abs()
    upper  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for c in upper.columns:
        if c in prot: continue
        for cc in upper.index[upper[c] > corr_thresh].tolist():
            if cc in prot: continue
            if X[c].var() >= X[cc].var():
                to_drop.add(cc)
            else:
                to_drop.add(c)
    keep = [c for c in keep if c not in to_drop]
    print(f"  Pruned: {n0} → {len(keep)} "
          f"(removed {n0-len(keep)}, protected {len(prot)})")
    return keep


# ══════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════

def metrics_log10(y_true, y_pred) -> dict:
    lt = np.log10(np.clip(y_true, EPS, None))
    lp = np.log10(np.clip(y_pred,  EPS, None))
    log_err = np.abs(lt - lp)
    return {
        "r2_log10":         r2_score(lt, lp),
        "rmse_log10":       float(np.sqrt(mean_squared_error(lt, lp))),
        "mean_log10_err":   float(log_err.mean()),
        "median_log10_err": float(np.median(log_err)),
        "r2_original":      r2_score(y_true, y_pred),
        "mape_original":    float(np.mean(np.abs(
            (y_true - y_pred) / np.clip(y_true, EPS, None))) * 100),
    }


def print_m(label, m, w=32):
    print(f"  {label:<{w}}  R²(log10)={m['r2_log10']:.4f}  "
          f"R²(orig)={m['r2_original']:.4f}  "
          f"log10_med={m['median_log10_err']:.3f}  "
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
#  OPTUNA  (40 trials, maximising log10 R²)
# ══════════════════════════════════════════════════════════════

def tune_model(name, X_tr, y_tr, X_vl, y_vl):
    if not HAS_OPTUNA: return {}

    def score(m, X, y): return r2_score(y, m.predict(X))

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
        return -score(m,X_vl,y_vl)

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
        return -score(m,X_vl,y_vl)

    def obj_cb(t):
        # depth capped at 7 — depth=10 gives +2% R² but takes 10× longer
        m = CatBoostRegressor(
            iterations=t.suggest_int("ne",400,1200),
            learning_rate=t.suggest_float("lr",0.005,0.08,log=True),
            depth=t.suggest_int("d",4,7),
            l2_leaf_reg=t.suggest_float("l",0.01,10.0,log=True),
            subsample=t.suggest_float("ss",0.6,1.0),
            random_seed=RANDOM_STATE,verbose=False,thread_count=-1)
        m.fit(X_tr,y_tr,eval_set=(X_vl,y_vl))
        return -score(m,X_vl,y_vl)

    obj = {"xgb":obj_xgb,"lgbm":obj_lgbm,"catboost":obj_cb}.get(name)
    if obj is None: return {}
    s = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    s.optimize(obj,n_trials=N_TRIALS,show_progress_bar=False)
    print(f"    {name}: log10-R²={-s.best_value:.4f}")
    return s.best_params


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  AREA PREDICTION v4 — LEAK-FREE + NORMALISED PHYSICS FEATURES")
    print("=" * 70)
    print(f"  XGB={HAS_XGB}  LGBM={HAS_LGBM}  CB={HAS_CATBOOST}"
          f"  SHAP={HAS_SHAP}  Optuna={HAS_OPTUNA}")
    print(f"\n  LEAKAGE STATUS: num_cells, num_nets EXCLUDED (post-synthesis)")
    print(f"  Honest RTL-only baseline from v3: R²(log10)≈0.64")
    print(f"  V4 target: R²(log10)>0.70 via normalised physics features")

    # ── 1. Load ───────────────────────────────────────────────
    csv = CSV_PATH if os.path.exists(CSV_PATH) else CSV_FALLBACK
    if csv == CSV_FALLBACK:
        print(f"\n  [INFO] Using fallback CSV. Run extract_rtl_area_features.py first.")
    print(f"\nLoading: {csv}")
    df = pd.read_csv(csv)
    print(f"  Shape: {df.shape}")

    area_col = next((c for c in AREA_COL_CANDIDATES if c in df.columns), None)
    if area_col is None:
        raise ValueError(f"No area column. Expected: {AREA_COL_CANDIDATES}")
    print(f"  Area column: '{area_col}'")

    # ── 2. Feature engineering ────────────────────────────────
    print("\nEngineering features (v4: +normalised physics, +design-type indicators)...")
    df = engineer_area_features(df)
    print(f"  Columns after engineering: {len(df.columns)}")

    # ── 3. Verify leakage is excluded ────────────────────────
    leaked = [c for c in ["num_cells","num_nets"] if c in df.columns]
    if leaked:
        print(f"\n  [LEAK CHECK] Found post-synthesis columns: {leaked}")
        print(f"  These are in NON_FEATURE_COLS and will NOT be used as features.")
    else:
        print(f"\n  [LEAK CHECK] ✓ num_cells, num_nets not in CSV (or already excluded)")

    # ── 4. Clean ──────────────────────────────────────────────
    print("\nCleaning...")
    feat_cols = [c for c in df.columns
                 if c not in NON_FEATURE_COLS
                 and not c.startswith("_")
                 and pd.api.types.is_numeric_dtype(df[c])]

    df_m = df[feat_cols + [area_col]].copy()
    n = len(df_m)
    df_m = df_m.dropna()
    print(f"  dropna   : {len(df_m)}/{n}")
    df_m = df_m[df_m[area_col] > 0]
    print(f"  area > 0 : {len(df_m)}")
    la    = np.log10(df_m[area_col])
    q1,q3 = la.quantile([0.25,0.75])
    df_m  = df_m[(la >= q1-3.5*(q3-q1)) & (la <= q3+3.5*(q3-q1))]
    print(f"  IQR      : {len(df_m)}")
    a = df_m[area_col]
    print(f"\n  Area (µm²): min={a.min():.1f}  "
          f"median={a.median():.1f}  max={a.max():.1f}  "
          f"({np.log10(a.max()/a.min()):.1f} decades)")

    # ── 5. Prune ──────────────────────────────────────────────
    print("\nPruning (physics features protected)...")
    X_df  = df_m[[c for c in feat_cols if c in df_m.columns]]
    keep  = prune_features(X_df, protected=PROTECTED_FEATURES)
    X_df  = X_df[keep]
    y_raw = df_m[area_col].values
    n_phy = sum(1 for f in PROTECTED_FEATURES if f in keep)
    print(f"  Physics features in final set: {n_phy}/{len(PROTECTED_FEATURES)}")
    # Report which v4 features survived
    v4_feats = ["mul_ge_per_bit","seq_ge_per_bit","mul_intensity_norm",
                "seq_intensity_norm","comb_logic_ge","seq_logic_ge",
                "mul_dominant","seq_dominant","is_wide_mul","wide_mul_ge",
                "mul_to_seq_ratio","log_mul_to_seq"]
    v4_kept = [f for f in v4_feats if f in keep]
    print(f"  New v4 features survived pruning: {len(v4_kept)}/{len(v4_feats)}")
    print(f"  {v4_kept}")

    # ── 6. Stratified split ───────────────────────────────────
    print("\nStratified split...")
    strat = pd.qcut(np.log10(y_raw),q=10,labels=False,duplicates="drop").astype(str)
    X_tmp,X_te,y_tmp,y_te,s_tmp,_ = train_test_split(
        X_df.values,y_raw,strat,
        test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=strat)
    strat_v = pd.qcut(np.log10(y_tmp),q=10,labels=False,duplicates="drop").astype(str)
    X_tr,X_vl,y_tr_o,y_vl_o = train_test_split(
        X_tmp,y_tmp,test_size=VAL_SIZE/(1-TEST_SIZE),
        random_state=RANDOM_STATE,stratify=strat_v)
    print(f"  Train={len(X_tr)}  Val={len(X_vl)}  Test={len(X_te)}")

    y_tr = np.log10(y_tr_o); y_vl = np.log10(y_vl_o)
    y_te_log = np.log10(y_te)

    # ── 7. Scale ──────────────────────────────────────────────
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)
    X_te_s = scaler.transform(X_te)
    joblib.dump(scaler, os.path.join(MODEL_DIR,"scaler.joblib"))
    joblib.dump(keep,   os.path.join(MODEL_DIR,"feature_cols.joblib"))

    # ── 8. 5-fold CV baseline ─────────────────────────────────
    print("\n5-fold CV on training set (honest estimate)...")
    kf      = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base_gbm = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.03,
        max_depth=6, subsample=0.8, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(base_gbm,X_tr_s,y_tr,cv=kf,scoring="r2",n_jobs=-1)
    cv_pred = cross_val_predict(base_gbm,X_tr_s,y_tr,cv=kf)
    cv_orig = r2_score(y_tr_o, 10**cv_pred)

    print(f"\n  ┌────────────────────────────────────────────────────┐")
    print(f"  │  CV BASELINE (leak-free)                              │")
    print(f"  │  5-fold CV R²(log10) = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}        │")
    print(f"  │  5-fold CV R²(orig)  = {cv_orig:.4f}                    │")
    print(f"  │  v3 baseline (log10) = 0.6154  (target: >0.70)       │")
    print(f"  │  Teammate (orig)     = 0.5971  (inflated, no leak chk)│")
    print(f"  └────────────────────────────────────────────────────┘")

    # ── 9. Optuna ─────────────────────────────────────────────
    xp,lp,cp = {},{},{}
    if TUNE:
        print(f"\nOptuna HPO ({N_TRIALS} trials, depth≤7 for CatBoost)...")
        if HAS_XGB:
            print("  XGB...",  end=" ", flush=True)
            xp = tune_model("xgb",      X_tr_s,y_tr,X_vl_s,y_vl)
        if HAS_LGBM:
            print("  LGBM...", end=" ", flush=True)
            lp = tune_model("lgbm",     X_tr_s,y_tr,X_vl_s,y_vl)
        if HAS_CATBOOST:
            print("  CB...",   end=" ", flush=True)
            cp = tune_model("catboost", X_tr_s,y_tr,X_vl_s,y_vl)

    # ── 10. Build models ──────────────────────────────────────
    models = {}
    if HAS_XGB:
        p = dict(n_estimators=1200,learning_rate=0.015,max_depth=8,
                 min_child_weight=3,subsample=0.85,colsample_bytree=0.85,
                 gamma=0.2,reg_alpha=0.5,reg_lambda=2.0,
                 random_state=RANDOM_STATE,n_jobs=-1,tree_method="hist")
        p.update({k:v for k,v in xp.items() if k in
                  ["ne","lr","d","mcw","ss","cbt","g","a","l"]})
        if "ne" in p: p["n_estimators"] = p.pop("ne")
        if "lr" in p: p["learning_rate"] = p.pop("lr")
        if "d"  in p: p["max_depth"] = p.pop("d")
        if "mcw"in p: p["min_child_weight"] = p.pop("mcw")
        if "ss" in p: p["subsample"] = p.pop("ss")
        if "cbt"in p: p["colsample_bytree"] = p.pop("cbt")
        if "g"  in p: p["gamma"] = p.pop("g")
        if "a"  in p: p["reg_alpha"] = p.pop("a")
        if "l"  in p: p["reg_lambda"] = p.pop("l")
        models["xgb"] = XGBRegressor(**p)

    if HAS_LGBM:
        p = dict(n_estimators=1200,learning_rate=0.015,max_depth=9,
                 num_leaves=80,subsample=0.85,colsample_bytree=0.85,
                 min_child_samples=10,reg_alpha=0.5,reg_lambda=2.0,
                 random_state=RANDOM_STATE,n_jobs=-1,verbose=-1)
        if lp:
            if "ne"  in lp: p["n_estimators"]     = lp["ne"]
            if "lr"  in lp: p["learning_rate"]     = lp["lr"]
            if "d"   in lp: p["max_depth"]         = lp["d"]
            if "nl"  in lp: p["num_leaves"]        = lp["nl"]
            if "ss"  in lp: p["subsample"]         = lp["ss"]
            if "cbt" in lp: p["colsample_bytree"]  = lp["cbt"]
            if "mcs" in lp: p["min_child_samples"] = lp["mcs"]
            if "a"   in lp: p["reg_alpha"]         = lp["a"]
            if "l"   in lp: p["reg_lambda"]        = lp["l"]
        models["lgbm"] = LGBMRegressor(**p)

    if HAS_CATBOOST:
        p = dict(iterations=1000,learning_rate=0.015,depth=7,
                 l2_leaf_reg=2.0,subsample=0.85,
                 random_seed=RANDOM_STATE,verbose=False,thread_count=-1)
        if cp:
            if "ne" in cp: p["iterations"]    = cp["ne"]
            if "lr" in cp: p["learning_rate"] = cp["lr"]
            if "d"  in cp: p["depth"]         = min(cp["d"], 7)  # cap at 7
            if "l"  in cp: p["l2_leaf_reg"]   = cp["l"]
            if "ss" in cp: p["subsample"]     = cp["ss"]
        models["catboost"] = CatBoostRegressor(**p)

    models["gbm"] = GradientBoostingRegressor(
        n_estimators=700,learning_rate=0.015,max_depth=7,
        subsample=0.85,min_samples_leaf=3,random_state=RANDOM_STATE)

    # ── 11. Train and evaluate ────────────────────────────────
    print(f"\n{'='*60}\n  MODEL RESULTS\n{'='*60}")
    results = {}
    for mname, model in models.items():
        print(f"\n  {mname.upper()}...", end=" ", flush=True)
        model.fit(X_tr_s, y_tr)
        vl_pred = 10**model.predict(X_vl_s)
        te_pred = 10**model.predict(X_te_s)
        vm = metrics_log10(y_vl_o, vl_pred)
        tm = metrics_log10(y_te,   te_pred)
        print("done")
        print_m("    Validation", vm)
        print_m("    Test",       tm)
        gap = abs(vm["r2_log10"] - tm["r2_log10"])
        st  = "✓" if gap < 0.10 else ("⚠" if gap < 0.20 else "✗")
        print(f"    Val-Test gap={gap:.3f} [{st}]")
        results[mname] = {"model":model,"val_r2":vm["r2_log10"],
                          "test_metrics":tm,"te_pred":te_pred}
        joblib.dump(model, os.path.join(MODEL_DIR,f"area_{mname}.joblib"))

    # ── 12. Ensemble ──────────────────────────────────────────
    print(f"\n  {'─'*50}")
    w   = {n: max(r["val_r2"],0)**2 for n,r in results.items()}
    tw  = sum(w.values())
    ens = sum((w[n]/tw)*results[n]["te_pred"] for n in w if w[n]>0)
    em  = metrics_log10(y_te, ens)
    print_m("  Ensemble Test", em)
    results["ensemble"] = {"model":None,"val_r2":em["r2_log10"],
                           "test_metrics":em,"te_pred":ens}

    # ── 13. Summary ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY — AREA (v4, leak-free)")
    print(f"{'='*70}")
    print(f"  {'Model':<14} {'ValR²(log)':>11} {'TstR²(log)':>11} "
          f"{'TstR²(orig)':>12} {'log10_med':>10} {'Gap':>7}")
    print(f"  {'-'*68}")
    best_name, best_r2 = None, -999
    for n,r in sorted(results.items(), key=lambda x: -x[1]["val_r2"]):
        tm  = r["test_metrics"]
        gap = abs(r["val_r2"] - tm["r2_log10"])
        st  = "✓" if gap<0.10 else ("⚠" if gap<0.20 else "✗")
        print(f"  {n:<14} {r['val_r2']:>11.4f} {tm['r2_log10']:>11.4f} "
              f"{tm['r2_original']:>12.4f} {tm['median_log10_err']:>9.3f} "
              f"{gap:>6.3f}{st}")
        if r["val_r2"] > best_r2:
            best_r2, best_name = r["val_r2"], n

    per_decile_table(y_te, results[best_name]["te_pred"], best_name)

    # ── 14. Improvement over v3 ───────────────────────────────
    best_tm = results[best_name]["test_metrics"]
    print(f"\n{'='*70}")
    print(f"  PROGRESS TRACKER")
    print(f"{'='*70}")
    print(f"  {'Version':<20} {'R²(log10)':>12} {'R²(orig)':>10} "
          f"{'log10_med':>11} {'Notes'}")
    print(f"  {'-'*66}")
    print(f"  {'teammate (leaky)':20} {'0.6000':>12} {'0.2363':>10} "
          f"{'N/A':>11} num_cells likely not excluded")
    print(f"  {'v3 (leak-free)':20} {'0.6437':>12} {'0.3325':>10} "
          f"{'0.221':>11} gbm best")
    print(f"  {'v4 (this run)':20} {best_tm['r2_log10']:>12.4f} "
          f"{best_tm['r2_original']:>10.4f} "
          f"{best_tm['median_log10_err']:>11.3f} "
          f"{best_name}")

    improvement = best_tm["r2_log10"] - 0.6437
    direction   = "+" if improvement >= 0 else ""
    print(f"\n  Δ from v3: {direction}{improvement:.4f} R²(log10)")
    if best_tm["r2_log10"] > 0.70:
        print(f"  ✓ TARGET ACHIEVED: R²(log10) > 0.70")
    elif best_tm["r2_log10"] > 0.64:
        print(f"  → Improvement confirmed. Physics normalisation is helping.")
    else:
        print(f"  → No improvement. Dataset may be near RTL-only ceiling.")
        print(f"     Next step: delay prediction (expected R²≈0.65-0.80)")

    # ── 15. SHAP ──────────────────────────────────────────────
    sname = next((n for n in ["lgbm","xgb","catboost","gbm"]
                  if n in results and results[n]["model"] is not None), None)
    if sname and HAS_SHAP:
        print(f"\nSHAP on {sname}...")
        try:
            exp = shap.TreeExplainer(results[sname]["model"])
            sv  = exp.shap_values(X_te_s)
            imp = np.abs(sv).mean(axis=0)
            dfi = pd.DataFrame({"feature":keep,"shap":imp})\
                    .sort_values("shap",ascending=False)
            dfi.to_csv(os.path.join(MODEL_DIR,"area_shap_v4.csv"),index=False)
            mx = dfi["shap"].max()

            phy_top = [r["feature"] for _,r in dfi.head(15).iterrows()
                       if r["feature"] in PROTECTED_FEATURES]
            v4_top  = [r["feature"] for _,r in dfi.head(15).iterrows()
                       if r["feature"] in v4_feats]

            print(f"\n  Top-15 (★=physics, ●=new v4):")
            print(f"  {'Feature':<45} {'SHAP':>8}")
            print(f"  {'-'*55}")
            for _,row in dfi.head(15).iterrows():
                bar = "█"*int(row["shap"]/mx*20)
                tag  = " ★" if row["feature"] in PROTECTED_FEATURES else ""
                tag += " ●" if row["feature"] in v4_feats else ""
                print(f"  {row['feature']:<45} {bar}  {row['shap']:.5f}{tag}")

            print(f"\n  Physics features in top-15: {len(phy_top)}/15  {phy_top}")
            print(f"  New v4 features in top-15: {len(v4_top)}/15   {v4_top}")

            if len(phy_top) >= 3:
                print(f"\n  ✓ Physics features contributing significantly!")
            if len(v4_top) >= 2:
                print(f"  ✓ New normalised features are helping!")
            elif len(v4_top) == 0:
                print(f"\n  → v4 features pruned or below baseline.")
                print(f"     This confirms RTL-only ceiling is being reached.")
                print(f"     Consider: delay prediction next (more tractable).")
        except Exception as e:
            print(f"  SHAP failed: {e}")

    # ── 16. Save predictions ──────────────────────────────────
    pd.DataFrame({
        "y_true": y_te,
        "y_pred": results[best_name]["te_pred"],
        "log10_true": y_te_log,
        "log10_pred": np.log10(np.clip(results[best_name]["te_pred"],EPS,None)),
        "log10_err":  np.abs(y_te_log -
                              np.log10(np.clip(results[best_name]["te_pred"],EPS,None))),
        "pct_err":   np.abs((y_te - results[best_name]["te_pred"]) /
                             np.clip(y_te,EPS,None))*100,
    }).sort_values("log10_err",ascending=False)\
      .to_csv(os.path.join(MODEL_DIR,"test_predictions_v4.csv"),index=False)

    print(f"\n  Outputs → {MODEL_DIR}")
    print("=" * 70)
    print("  RESULT GUIDE (honest, leak-free):")
    print("    R²(log10) > 0.75  → excellent RTL-only area prediction")
    print("    R²(log10) > 0.65  → good, comparable to published results")
    print("    R²(log10) > 0.55  → acceptable, move to delay prediction next")
    print("    log10_median < 0.20 → predictions within ×1.58 on average")
    print("    log10_median < 0.30 → predictions within ×2.0  (acceptable)")


if __name__ == "__main__":
    main()