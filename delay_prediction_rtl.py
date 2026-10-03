#!/usr/bin/env python3
r"""
delay_prediction_rtl.py
========================
RTL-only critical path delay prediction using the dedicated delay
feature CSV produced by extract_rtl_delay_features.py.

RUN ORDER:
  1. python extract_rtl_delay_features.py \
         --dataset_root "C:\ml ppa\Final_Clean_Dataset" \
         --out_csv "C:\Users\Admin\Documents\rtl_delay_features.csv"
  2. python delay_prediction_rtl.py

WHY DELAY IS DIFFERENT FROM AREA AND POWER:
  Area  → structural (gate count)  → features: N², 8×DFF, etc.
  Power → dynamic (switching)      → features: FF-bits, XOR density
  Delay → temporal (critical path) → features: PIPELINE STAGES + slow ops

  The critical path is determined by the LONGEST combinational chain
  between any two registers (or I/O ports). RTL tells us:
    1. How slow are the operations? (mul > add > logic)
    2. How wide are they? (32-bit chain >> 8-bit)
    3. Are there pipeline registers? (MOST IMPORTANT — breaks the path)
    4. How deep is the mux/control logic? (nested if/case adds stages)

  Key insight: delay does NOT grow with design size the way area does.
  A 4-stage pipelined 32-bit multiplier has LESS delay than a single
  stage 32-bit multiplier, even though it is 4× bigger.
  → the pipeline_reduced_delay feature encodes this correctly.

TARGET TRANSFORM:
  Delay (critical_path_length in ns) spans ~2-3 decades, less than
  area/power. We compare log1p vs log10 vs raw and pick the best.
  Unlike area (always log10), delay can sometimes be predicted well
  in raw space if the range is small.

PROTECTED FEATURES (never pruned):
  critical_path_estimate  — physics-based delay sum
  pipeline_reduced_delay  — delay ÷ pipeline_stages (dominant predictor)
  mul_delay_proxy         — num_mul × max_bw × 0.4 ns
  adder_delay_proxy       — (add+sub) × max_bw × 0.3 ns
  div_delay_proxy         — num_div × max_bw² × 0.5 ns
  slow_ops_x_bw           — weighted slow ops × max_bw
  carry_chain_depth       — ripple carry length
  pipeline_stages         — num_always_ff
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
    from xgboost import XGBRegressor;       HAS_XGB = True
except ImportError:                          HAS_XGB = False
try:
    from lightgbm import LGBMRegressor;     HAS_LGBM = True
except ImportError:                          HAS_LGBM = False
try:
    from catboost import CatBoostRegressor;  HAS_CATBOOST = True
except ImportError:                          HAS_CATBOOST = False
try:
    import shap;                             HAS_SHAP = True
except ImportError:                          HAS_SHAP = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════
CSV_PATH     = r"C:\Users\Admin\Documents\final_rtl_delay_features.csv"
CSV_FALLBACK = r"C:\Users\Admin\Documents\final_rtl_power_features_v2.csv"
MODEL_DIR    = r"C:\Users\Admin\OneDrive - MSFT\Desktop\New folder\cody\delay_model_v1"
os.makedirs(MODEL_DIR, exist_ok=True)

DELAY_COL_CANDIDATES = ["critical_path_length", "Critical_Path_Length",
                         "delay", "Delay"]

RANDOM_STATE = 42
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15
TUNE         = True
N_TRIALS     = 60
CORR_THRESH  = 0.97
EPS          = 1e-9

# Sky130 gate delays (ns) — same constants as extractor
DELAY_FA   = 0.30
DELAY_MUL  = 0.40
DELAY_DIV  = 0.50
DELAY_XOR2 = 0.20
DELAY_MUX2 = 0.15
DELAY_AND2 = 0.15
DELAY_INV  = 0.10
DELAY_MAJ3 = 0.30

# ── Protected physics features — never dropped by pruner ─────
# These encode the actual delay physics and must survive
# even when correlated with size proxies.
PROTECTED_FEATURES = [
    "critical_path_estimate",   # physics-based sum of all delay contributions
    "log_cp_estimate",          # log-compressed version
    "sqrt_cp_estimate",
    "pipeline_reduced_delay",   # delay ÷ pipeline_stages — dominant predictor
    "log_pipeline_reduced",
    "mul_delay_proxy",          # num_mul × max_bw × 0.4 ns
    "log_mul_delay",
    "adder_delay_proxy",        # (add+sub) × max_bw × 0.3 ns
    "log_adder_delay",
    "div_delay_proxy",          # num_div × max_bw² × 0.5 ns
    "log_div_delay",
    "carry_chain_depth",        # ripple carry length in bits
    "log_carry_chain",
    "slow_ops_x_bw",            # weighted slow ops × max_bw
    "log_slow_ops_x_bw",
    "slow_ops_weighted",
    "pipeline_stages",          # num_always_ff
    "comb_ops_per_stage",       # combinational ops between registers
    "xor_delay_proxy",
    "shift_delay_proxy",
]

NON_FEATURE_COLS = {
    "Design_Name", "RTL_Code", "rpt_files", "rpt_text_len",
    "Power", "total_cell_area", "comb_area", "levels_of_logic",
    # "wns", "tns", "num_cells",
    "wns", "tns", "num_cells", "num_nets",
    # exclude all delay target variants
    "critical_path_length", "Critical_Path_Length", "delay", "Delay",
}


# ══════════════════════════════════════════════════════════════
#  INLINE FEATURE ENGINEERING
#  Recomputes delay physics features from base counts.
#  Safe to call on both the dedicated delay CSV and the fallback
#  power CSV — any already-computed column is skipped.
# ══════════════════════════════════════════════════════════════

def engineer_delay_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # def col(name, default=0.0):
    #     if name in df.columns:
    #         return df[name].fillna(default).astype(float)
    #     return pd.Series(float(default), index=df.index)

    def col(name, default=0.0):
        if name in df.columns:
            return df[name].fillna(default).astype(float)
        if isinstance(default, pd.Series):
            return default.astype(float)  # FIX: Safely return the fallback series
        return pd.Series(float(default), index=df.index)

    def add_if_missing(name, series):
        if name not in df.columns:
            df[name] = series

    avg_bw  = col("avg_bw",      col("avg_bitwidth",  1.0)).clip(lower=1)
    max_bw  = col("max_bw",      col("max_bitwidth",  1.0)).clip(lower=1)
    min_bw  = col("min_bw",      col("min_bitwidth",  1.0)).clip(lower=1)
    n_reg   = col("num_reg")
    n_add   = col("num_add")
    n_sub   = col("num_sub")
    n_mul   = col("num_mul")
    n_div   = col("num_div")
    n_xor   = col("num_logic_xor")
    n_and   = col("num_logic_and")
    n_or    = col("num_logic_or")
    n_mux   = col("num_ternary") + col("num_case")
    n_comp  = col("num_comparisons")
    n_shift = col("num_shifts")
    n_if    = col("num_if")
    n_lines = col("num_lines", 1).clip(lower=1)
    n_always= col("num_always")

    # Pipeline stages — most important single feature for delay
    n_always_ff = col("num_always_ff",
                      col("num_posedge_always", 0))
    stages = (n_always_ff).clip(lower=1)

    # ── Core delay proxies ────────────────────────────────────
    adder_delay = (n_add + n_sub) * max_bw * DELAY_FA
    mul_delay   = n_mul * max_bw * DELAY_MUL
    div_delay   = n_div * max_bw * max_bw * DELAY_DIV
    xor_delay   = n_xor * avg_bw * DELAY_XOR2
    shift_delay = n_shift * np.log2(avg_bw.clip(lower=2)) * DELAY_MUX2
    comp_delay  = n_comp * avg_bw * DELAY_FA
    mux_delay   = n_mux  * avg_bw * DELAY_MUX2
    and_delay   = (n_and + n_or) * avg_bw * DELAY_AND2

    add_if_missing("adder_delay_proxy",  adder_delay)
    add_if_missing("log_adder_delay",    np.log1p(adder_delay))
    add_if_missing("mul_delay_proxy",    mul_delay)
    add_if_missing("log_mul_delay",      np.log1p(mul_delay))
    add_if_missing("div_delay_proxy",    div_delay)
    add_if_missing("log_div_delay",      np.log1p(div_delay))
    add_if_missing("xor_delay_proxy",    xor_delay)
    add_if_missing("log_xor_delay",      np.log1p(xor_delay))
    add_if_missing("shift_delay_proxy",  shift_delay)
    add_if_missing("log_shift_delay",    np.log1p(shift_delay))
    add_if_missing("comp_delay_proxy",   comp_delay)

    # Total critical path estimate (physics sum)
    cp_est = (mul_delay + adder_delay + div_delay + xor_delay +
              shift_delay + mux_delay + and_delay)
    add_if_missing("critical_path_estimate", cp_est)
    add_if_missing("log_cp_estimate",        np.log1p(cp_est))
    add_if_missing("sqrt_cp_estimate",       np.sqrt(cp_est.clip(0)))

    # Pipelining REDUCES delay — most important interaction
    add_if_missing("pipeline_stages",        n_always_ff)
    add_if_missing("pipeline_reduced_delay", cp_est / stages)
    add_if_missing("log_pipeline_reduced",   np.log1p(cp_est / stages))

    # Slow ops
    slow_w = 3*n_div + 2*n_mul + 1*(n_add+n_sub) + 0.5*n_xor + 0.3*n_comp
    add_if_missing("slow_ops",               2*n_mul + 3*n_div)
    add_if_missing("slow_ops_weighted",      slow_w)
    add_if_missing("slow_ops_x_bw",         slow_w * max_bw)
    add_if_missing("log_slow_ops_x_bw",     np.log1p(slow_w * max_bw))

    # Carry chain
    carry = (n_add + n_sub) * max_bw
    add_if_missing("carry_chain_depth",      carry)
    add_if_missing("log_carry_chain",        np.log1p(carry))
    add_if_missing("has_long_carry",         (carry > 32).astype(int))

    # Combinational depth
    total_comb = n_add+n_sub+n_mul+n_div+n_xor+n_and+n_or+n_comp
    add_if_missing("comb_ops_total",         total_comb)
    add_if_missing("comb_ops_per_stage",     total_comb / stages)
    add_if_missing("log_comb_per_stage",     np.log1p(total_comb / stages))

    # Bitwidth interactions
    add_if_missing("max_bw_x_slow_ops",     max_bw * slow_w)
    add_if_missing("log_bw_x_slow",         np.log1p(max_bw * slow_w))
    add_if_missing("bw_range",              max_bw - min_bw)

    # Design type indicators
    add_if_missing("is_pipelined",          (n_always_ff > 1).astype(int))
    add_if_missing("is_purely_comb",        (n_always_ff == 0).astype(int))
    add_if_missing("has_multiplier",        (n_mul > 0).astype(int))
    add_if_missing("has_division",          (n_div > 0).astype(int))

    # ── NEW: additional features not in extractor ─────────────

    # Slow op density (normalised by pipeline stages × bitwidth)
    # A 32-bit multiplier in a 4-stage pipeline has lower delay than
    # a 32-bit multiplier with no pipeline — this ratio captures it.
    df["effective_delay_density"] = (
        df["slow_ops_x_bw"] / (stages * max_bw).clip(lower=EPS))
    df["log_effective_density"]   = np.log1p(df["effective_delay_density"])

    # Fraction of ops that are delay-critical
    df["critical_op_fraction"]    = (n_mul + n_div) / total_comb.clip(lower=1)

    # Fanout-weighted delay (high-fanout signals need buffers → +delay)
    hf = col("num_high_fanout", col("num_high_fanout_signals", 0))
    df["fanout_delay_pressure"]   = hf * DELAY_INV * 2

    # MUX chain depth proxy (nested if/case → long MUX chain)
    begin_cnt = col("begin_count", 0)
    nesting   = begin_cnt / (n_if + n_if.clip(lower=1))
    df["mux_chain_depth"]         = n_mux * avg_bw * nesting.clip(upper=10)
    df["log_mux_chain"]           = np.log1p(df["mux_chain_depth"])

    # Pipeline efficiency: how well are slow ops distributed across stages?
    # Ideal: each stage has equal combinational delay
    # Proxy: if all slow ops in one always_comb block → high delay
    df["pipeline_balance_proxy"]  = (
        df["mul_delay_proxy"] /
        (stages * df["critical_path_estimate"].clip(lower=EPS)))

    # Accumulator pattern: a <= a + x → creates a feedback path
    # Feedback paths set minimum clock period regardless of pipeline depth
    df["has_feedback_proxy"]      = col("has_accumulator", 0)

    # Log bitwidth features
    df["log_max_bw"]              = np.log1p(max_bw)
    df["log_total_bits"]          = np.log1p(col("total_bits", 0))
    df["sqrt_max_bw"]             = np.sqrt(max_bw.clip(0))

    # Interaction: multiplier × bitwidth² (true delay = N × 0.4 ns, so
    # wider muls add delay super-linearly when chained)
    df["mul_bw_squared"]          = n_mul * max_bw * max_bw * DELAY_MUL
    df["log_mul_bw_squared"]      = np.log1p(df["mul_bw_squared"])

    # Signal complexity (high fanout = buffer insertion = added delay)
    df["signal_complexity"]       = (col("unique_signals", 0) *
                                     col("mean_signal_occurrences", 1))
    df["log_signal_complexity"]   = np.log1p(df["signal_complexity"])

    # Lines of code per pipeline stage (density of logic per stage)
    df["lines_per_stage"]         = n_lines / stages
    df["log_lines_per_stage"]     = np.log1p(df["lines_per_stage"])

    return df


# ══════════════════════════════════════════════════════════════
#  BEST TARGET TRANSFORM SELECTOR
#  Unlike area (always log10), delay can be better predicted
#  in log1p or even raw space depending on the range.
#  This function tests all three and picks the best on CV.
# ══════════════════════════════════════════════════════════════

def select_best_transform(X_tr_s, y_tr_raw, cv=5, random_state=42):
    """
    Compare log10, log1p, and sqrt transforms for delay prediction.
    Returns (transform_name, y_transformed, inverse_fn).
    """
    from sklearn.linear_model import Ridge

    transforms = {
        "log10": (np.log10(np.clip(y_tr_raw, EPS, None)),
                  lambda x: 10 ** np.array(x)),
        "log1p": (np.log1p(y_tr_raw),
                  lambda x: np.expm1(np.array(x))),
        "sqrt":  (np.sqrt(y_tr_raw),
                  lambda x: np.array(x) ** 2),
        "raw":   (y_tr_raw,
                  lambda x: np.array(x)),
    }

    kf      = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    probe   = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=5, random_state=random_state)
    best_name, best_r2 = None, -999

    print(f"\n  Transform selection (5-fold CV R² on training set):")
    for name, (y_t, inv_fn) in transforms.items():
        cv_scores = cross_val_score(probe, X_tr_s, y_t,
                                    cv=kf, scoring="r2")
        # evaluate in original scale
        cv_preds  = cross_val_predict(probe, X_tr_s, y_t, cv=kf)
        r2_orig   = r2_score(y_tr_raw, inv_fn(cv_preds))
        print(f"    {name:<8}  CV R²(transformed)={cv_scores.mean():.4f}±{cv_scores.std():.4f}"
              f"  R²(original)={r2_orig:.4f}")
        if cv_scores.mean() > best_r2:
            best_r2   = cv_scores.mean()
            best_name = name

    best_y, best_inv = transforms[best_name]
    print(f"  → Selected transform: {best_name}  (CV R²={best_r2:.4f})")
    return best_name, best_y, best_inv


# ══════════════════════════════════════════════════════════════
#  FEATURE PRUNING  (with protected list)
# ══════════════════════════════════════════════════════════════

def prune_features(X: pd.DataFrame, corr_thresh=CORR_THRESH,
                   protected: list = None) -> list:
    protected = set(protected or []) & set(X.columns)
    sel  = VarianceThreshold(threshold=1e-6)
    sel.fit(X)
    keep = set(X.columns[sel.get_support()].tolist()) | protected
    keep = list(keep)
    n0   = len(X.columns)

    corr    = X[keep].corr().abs()
    upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for c in upper.columns:
        if c in protected:
            continue
        for cc in upper.index[upper[c] > corr_thresh].tolist():
            if cc in protected:
                continue
            if X[c].var() >= X[cc].var():
                to_drop.add(cc)
            else:
                to_drop.add(c)
    keep = [c for c in keep if c not in to_drop]
    print(f"  Pruned: {n0} → {len(keep)} "
          f"(removed {n0-len(keep)}, "
          f"protected {len(protected & set(keep))}/{len(protected)})")
    return keep


# ══════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred) -> dict:
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) /
                           np.clip(np.abs(y_true), EPS, None))) * 100
    # delay-specific: absolute error in ns (interpretable)
    mae_ns = float(np.mean(np.abs(y_true - y_pred)))
    p90_ns = float(np.percentile(np.abs(y_true - y_pred), 90))
    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape,
            "mae_ns": mae_ns, "p90_err_ns": p90_ns}


def print_metrics(label, m, w=30):
    print(f"  {label:<{w}}  R²={m['r2']:.4f}  "
          f"MAPE={m['mape']:.1f}%  "
          f"MAE={m['mae_ns']:.3f}ns  "
          f"P90_err={m['p90_err_ns']:.3f}ns")


def per_decile_table(y_true, y_pred, label=""):
    df = pd.DataFrame({"t": y_true, "p": y_pred})
    df["dec"] = pd.qcut(df["t"], q=10, labels=False, duplicates="drop")
    print(f"\n  Per-decile results ({label}):")
    print(f"  {'D':<4} {'Delay (ns) range':>22} {'MAPE':>8} "
          f"{'MAE(ns)':>9} {'N':>5}")
    print(f"  {'-'*54}")
    for d, g in df.groupby("dec"):
        mape   = np.mean(np.abs((g["t"]-g["p"]) /
                                 g["t"].clip(lower=EPS))) * 100
        mae_ns = np.mean(np.abs(g["t"] - g["p"]))
        print(f"  {int(d):<4} {g['t'].min():>9.3f} – {g['t'].max():>9.3f} ns"
              f"  {mape:>7.1f}%  {mae_ns:>8.3f}  {len(g):>4}")


# ══════════════════════════════════════════════════════════════
#  OPTUNA TUNING
# ══════════════════════════════════════════════════════════════

def tune_model(name, X_tr, y_tr, X_vl, y_vl):
    """Maximise R² on validation set in transformed space."""
    if not HAS_OPTUNA: return {}

    def score(m, X, y):
        return r2_score(y, m.predict(X))

    def obj_xgb(t):
        m = XGBRegressor(
            n_estimators     = t.suggest_int("ne", 400, 2000),
            learning_rate    = t.suggest_float("lr", 0.005, 0.08, log=True),
            max_depth        = t.suggest_int("d",  3, 9),
            min_child_weight = t.suggest_int("mcw", 1, 20),
            subsample        = t.suggest_float("ss", 0.55, 1.0),
            colsample_bytree = t.suggest_float("cbt", 0.5, 1.0),
            gamma            = t.suggest_float("g", 0, 1.5),
            reg_alpha        = t.suggest_float("a", 1e-3, 8.0, log=True),
            reg_lambda       = t.suggest_float("l", 0.01, 15.0, log=True),
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")
        m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        return -score(m, X_vl, y_vl)

    def obj_lgbm(t):
        m = LGBMRegressor(
            n_estimators     = t.suggest_int("ne", 400, 2000),
            learning_rate    = t.suggest_float("lr", 0.005, 0.08, log=True),
            max_depth        = t.suggest_int("d",  3, 9),
            num_leaves       = t.suggest_int("nl", 20, 180),
            subsample        = t.suggest_float("ss", 0.55, 1.0),
            colsample_bytree = t.suggest_float("cbt", 0.5, 1.0),
            min_child_samples= t.suggest_int("mcs", 5, 40),
            reg_alpha        = t.suggest_float("a", 1e-3, 8.0, log=True),
            reg_lambda       = t.suggest_float("l", 0.01, 15.0, log=True),
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        m.fit(X_tr, y_tr)
        return -score(m, X_vl, y_vl)

    def obj_cb(t):
        m = CatBoostRegressor(
            iterations    = t.suggest_int("ne", 400, 1500),
            learning_rate = t.suggest_float("lr", 0.005, 0.08, log=True),
            depth         = t.suggest_int("d",   3, 7),
            l2_leaf_reg   = t.suggest_float("l", 0.01, 15.0, log=True),
            subsample     = t.suggest_float("ss", 0.55, 1.0),
            random_seed=RANDOM_STATE, verbose=False, thread_count=-1)
        m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        return -score(m, X_vl, y_vl)

    obj = {"xgb": obj_xgb, "lgbm": obj_lgbm, "catboost": obj_cb}.get(name)
    if obj is None: return {}

    s = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    s.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"    {name}: val R²={-s.best_value:.4f}  params={s.best_params}")
    return s.best_params


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  DELAY (CRITICAL PATH) PREDICTION — RTL FEATURES ONLY")
    print("=" * 70)
    print(f"  XGB={HAS_XGB}  LGBM={HAS_LGBM}  CB={HAS_CATBOOST}"
          f"  SHAP={HAS_SHAP}  Optuna={HAS_OPTUNA}")

    # ── 1. Load CSV ───────────────────────────────────────────
    csv_to_use = CSV_PATH if os.path.exists(CSV_PATH) else CSV_FALLBACK
    if not os.path.exists(CSV_PATH):
        print(f"\n[INFO] Delay CSV not found. Using fallback: {CSV_FALLBACK}")
        print(f"       Run extract_rtl_delay_features.py first for best results.")
    print(f"\nLoading: {csv_to_use}")
    df = pd.read_csv(csv_to_use)
    print(f"  Shape: {df.shape}")

    # detect delay column
    delay_col = next((c for c in DELAY_COL_CANDIDATES if c in df.columns), None)
    if delay_col is None:
        raise ValueError(
            f"No delay column found. Expected one of {DELAY_COL_CANDIDATES}.\n"
            f"Columns present: {list(df.columns)}")
    print(f"  Delay column: '{delay_col}'")

    # ── 2. Feature engineering ────────────────────────────────
    print("\nEngineering delay-specific features...")
    df = engineer_delay_features(df)
    print(f"  Columns after engineering: {len(df.columns)}")

    # ── 3. Clean ──────────────────────────────────────────────
    print("\nCleaning data...")
    exclude   = NON_FEATURE_COLS #- {delay_col}
    feat_cols = [c for c in df.columns
                 if c not in exclude
                 and not c.startswith("_")
                 and pd.api.types.is_numeric_dtype(df[c])]

    df_m = df[feat_cols + [delay_col]].copy()
    n    = len(df_m)
    df_m = df_m.dropna()
    print(f"  dropna    : {len(df_m)}/{n}")
    df_m = df_m[df_m[delay_col] > 0]
    print(f"  delay > 0 : {len(df_m)}")

    # IQR on log10(delay) — works even for 2-decade range
    ld    = np.log10(df_m[delay_col])
    q1,q3 = ld.quantile([0.25, 0.75])
    iqr   = q3 - q1
    df_m  = df_m[(ld >= q1 - 3.5*iqr) & (ld <= q3 + 3.5*iqr)]
    print(f"  IQR       : {len(df_m)}")

    d = df_m[delay_col]
    print(f"\n  Delay distribution (ns):")
    print(f"    min={d.min():.3f}  p25={d.quantile(.25):.3f}  "
          f"median={d.median():.3f}  p75={d.quantile(.75):.3f}  "
          f"max={d.max():.3f}")
    print(f"    log10 range: {np.log10(d.min()):.2f} to "
          f"{np.log10(d.max()):.2f}  "
          f"({np.log10(d.max()/d.min()):.1f} decades)")

    # ── 4. Feature pruning ────────────────────────────────────
    print("\nPruning features (protecting delay physics features)...")
    X_df  = df_m[[c for c in feat_cols if c in df_m.columns]]
    keep  = prune_features(X_df, corr_thresh=CORR_THRESH,
                           protected=PROTECTED_FEATURES)
    X_df  = X_df[keep]
    y_raw = df_m[delay_col].values
    print(f"  Final feature count : {len(keep)}")
    n_prot = sum(1 for f in PROTECTED_FEATURES if f in keep)
    print(f"  Physics features kept: {n_prot}/{len(PROTECTED_FEATURES)}")

    # ── 5. Stratified split ───────────────────────────────────
    # Stratify on log10(delay) bins so every magnitude is represented
    print("\nStratified split by log10(delay) bins...")
    strat = pd.qcut(np.log10(y_raw), q=10,
                    labels=False, duplicates="drop").astype(str)
    X_tmp, X_te, y_tmp, y_te, s_tmp, _ = train_test_split(
        X_df.values, y_raw, strat,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat)
    strat_v = pd.qcut(np.log10(y_tmp), q=10,
                      labels=False, duplicates="drop").astype(str)
    X_tr, X_vl, y_tr_o, y_vl_o = train_test_split(
        X_tmp, y_tmp,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE, stratify=strat_v)
    print(f"  Train={len(X_tr)}  Val={len(X_vl)}  Test={len(X_te)}")

    # ── 6. Scale ──────────────────────────────────────────────
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)
    X_te_s = scaler.transform(X_te)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(keep,   os.path.join(MODEL_DIR, "feature_cols.joblib"))

    # ── 7. Select best target transform ──────────────────────
    # Delay can be better in log1p vs log10 vs raw depending on range.
    # Test all three on training data and pick the best.
    t_name, y_tr_t, inverse_fn = select_best_transform(
        X_tr_s, y_tr_o, cv=5, random_state=RANDOM_STATE)
    y_vl_t = {
        "log10": np.log10(np.clip(y_vl_o, EPS, None)),
        "log1p": np.log1p(y_vl_o),
        "sqrt":  np.sqrt(y_vl_o),
        "raw":   y_vl_o,
    }[t_name]

    # Save transform name for inference
    joblib.dump({"transform": t_name}, os.path.join(MODEL_DIR, "transform.joblib"))

    # ── 8. 5-fold CV baseline ─────────────────────────────────
    print("\n5-fold CV on training set...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    baseline = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.03,
        max_depth=5, subsample=0.8, random_state=RANDOM_STATE)
    cv_r2   = cross_val_score(baseline, X_tr_s, y_tr_t, cv=kf,
                               scoring="r2", n_jobs=-1)
    cv_pred = cross_val_predict(baseline, X_tr_s, y_tr_t, cv=kf)
    cv_orig = inverse_fn(cv_pred)
    cv_m    = compute_metrics(y_tr_o, cv_orig)

    print(f"\n  ┌────────────────────────────────────────────────────┐")
    print(f"  │  5-FOLD CV BASELINE (transform={t_name})            │")
    print(f"  │  CV R² (transformed space) = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}  │")
    print(f"  │  CV R² (original ns scale) = {cv_m['r2']:.4f}                │")
    print(f"  │  CV MAE (ns)               = {cv_m['mae_ns']:.4f} ns            │")
    print(f"  │                                                    │")
    print(f"  │  Teammate LGBM delay test R² = 0.6179             │")
    print(f"  │  Teammate LGBM delay CV R²   = 0.4195 (gap=0.198) │")
    print(f"  │  Target: CV R² ≈ test R², gap < 0.10              │")
    print(f"  └────────────────────────────────────────────────────┘")

    # ── 9. Optuna HPO ─────────────────────────────────────────
    xp, lp, cp = {}, {}, {}
    if TUNE:
        print(f"\nOptuna HPO (target: val R² in {t_name} space)...")
        if HAS_XGB:
            print("  XGB...",  end=" ", flush=True)
            xp = tune_model("xgb",      X_tr_s, y_tr_t, X_vl_s, y_vl_t)
        if HAS_LGBM:
            print("  LGBM...", end=" ", flush=True)
            lp = tune_model("lgbm",     X_tr_s, y_tr_t, X_vl_s, y_vl_t)
        if HAS_CATBOOST:
            print("  CB...",   end=" ", flush=True)
            cp = tune_model("catboost", X_tr_s, y_tr_t, X_vl_s, y_vl_t)

    # ── 10. Build models ──────────────────────────────────────
    # Delay-specific defaults differ from area/power:
    #   - Shallower trees (delay has less complex feature interactions)
    #   - Fewer estimators (risk of overfit on structured patterns)
    #   - Higher subsampling (delay data has regional clusters)
    # models = {}
    # if HAS_XGB:
    #     p = dict(n_estimators=1000, learning_rate=0.02, max_depth=7,
    #              min_child_weight=4, subsample=0.8, colsample_bytree=0.8,
    #              gamma=0.3, reg_alpha=1.0, reg_lambda=3.0,
    #              random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")
    #     p.update(xp); models["xgb"] = XGBRegressor(**p)

    # if HAS_LGBM:
    #     p = dict(n_estimators=1000, learning_rate=0.02, max_depth=8,
    #              num_leaves=60, subsample=0.8, colsample_bytree=0.8,
    #              min_child_samples=12, reg_alpha=1.0, reg_lambda=3.0,
    #              random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    #     p.update(lp); models["lgbm"] = LGBMRegressor(**p)

    # if HAS_CATBOOST:
    #     p = dict(iterations=1000, learning_rate=0.02, depth=7,
    #              l2_leaf_reg=3.0, subsample=0.8,
    #              random_seed=RANDOM_STATE, verbose=False, thread_count=-1)
    #     p.update(cp); models["catboost"] = CatBoostRegressor(**p)

    # models["rf"] = RandomForestRegressor(
    #     n_estimators=500, max_depth=15, min_samples_split=10,
    #     min_samples_leaf=4, max_features="sqrt",
    #     random_state=RANDOM_STATE, n_jobs=-1)

    # models["gbm"] = GradientBoostingRegressor(
    #     n_estimators=600, learning_rate=0.02, max_depth=6,
    #     subsample=0.8, min_samples_leaf=4,
    #     random_state=RANDOM_STATE)

    # ── 10. Build models ──────────────────────────────────────
    def get_val(d, key, default):
        return d.get(key, default) if d else default

    models = {}
    if HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=get_val(xp, "ne", 1000),
            learning_rate=get_val(xp, "lr", 0.02),
            max_depth=get_val(xp, "d", 7),
            min_child_weight=get_val(xp, "mcw", 4),
            subsample=get_val(xp, "ss", 0.8),
            colsample_bytree=get_val(xp, "cbt", 0.8),
            gamma=get_val(xp, "g", 0.3),
            reg_alpha=get_val(xp, "a", 1.0),
            reg_lambda=get_val(xp, "l", 3.0),
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")

    if HAS_LGBM:
        models["lgbm"] = LGBMRegressor(
            n_estimators=get_val(lp, "ne", 1000),
            learning_rate=get_val(lp, "lr", 0.02),
            max_depth=get_val(lp, "d", 8),
            num_leaves=get_val(lp, "nl", 60),
            subsample=get_val(lp, "ss", 0.8),
            colsample_bytree=get_val(lp, "cbt", 0.8),
            min_child_samples=get_val(lp, "mcs", 12),
            reg_alpha=get_val(lp, "a", 1.0),
            reg_lambda=get_val(lp, "l", 3.0),
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)

    if HAS_CATBOOST:
        models["catboost"] = CatBoostRegressor(
            iterations=get_val(cp, "ne", 1000),
            learning_rate=get_val(cp, "lr", 0.02),
            depth=get_val(cp, "d", 7),
            l2_leaf_reg=get_val(cp, "l", 3.0),
            subsample=get_val(cp, "ss", 0.8),
            random_seed=RANDOM_STATE, verbose=False, thread_count=-1)

    models["rf"] = RandomForestRegressor(
        n_estimators=500, max_depth=15, min_samples_split=10,
        min_samples_leaf=4, max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=-1)

    models["gbm"] = GradientBoostingRegressor(
        n_estimators=600, learning_rate=0.02, max_depth=6,
        subsample=0.8, min_samples_leaf=4,
        random_state=RANDOM_STATE)

    # ── 11. Train and evaluate ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MODEL RESULTS  (transform={t_name})")
    print(f"{'='*60}")

    results = {}
    for mname, model in models.items():
        print(f"\n  {mname.upper()}...", end=" ", flush=True)
        model.fit(X_tr_s, y_tr_t)

        vl_pred = inverse_fn(model.predict(X_vl_s))
        te_pred = inverse_fn(model.predict(X_te_s))

        vm = compute_metrics(y_vl_o, vl_pred)
        tm = compute_metrics(y_te,   te_pred)
        print("done")
        print_metrics("    Validation", vm)
        print_metrics("    Test",       tm)

        gap    = abs(vm["r2"] - tm["r2"])
        status = ("✓ healthy"   if gap < 0.10 else
                  "⚠ moderate"  if gap < 0.20 else
                  "✗ overfitting")
        print(f"    Val-Test gap = {gap:.3f}  [{status}]")

        results[mname] = {
            "model": model, "val_r2": vm["r2"],
            "test_metrics": tm, "te_pred": te_pred,
        }
        joblib.dump(model, os.path.join(MODEL_DIR, f"delay_{mname}.joblib"))

    # ── 12. Weighted ensemble ─────────────────────────────────
    print(f"\n  {'─'*50}")
    w   = {n: max(r["val_r2"], 0) ** 2 for n, r in results.items()}
    tw  = sum(w.values())
    if tw > 0:
        ens = sum((w[n] / tw) * results[n]["te_pred"]
                  for n in w if w[n] > 0)
        em  = compute_metrics(y_te, ens)
        print_metrics("  Ensemble Test", em)
        results["ensemble"] = {
            "model": None, "val_r2": em["r2"],
            "test_metrics": em, "te_pred": ens,
        }

    # ── 13. Summary table ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON — DELAY (RTL only)")
    print(f"{'='*70}")
    print(f"  {'Model':<14} {'Val R²':>8} {'Test R²':>8} "
          f"{'MAPE':>8} {'MAE(ns)':>9} {'P90(ns)':>9} {'Gap':>7}")
    print(f"  {'-'*72}")
    best_name, best_r2 = None, -999
    for n, r in sorted(results.items(), key=lambda x: -x[1]["val_r2"]):
        tm  = r["test_metrics"]
        gap = abs(r["val_r2"] - tm["r2"])
        st  = "✓" if gap < 0.10 else ("⚠" if gap < 0.20 else "✗")
        print(f"  {n:<14} {r['val_r2']:>8.4f} {tm['r2']:>8.4f} "
              f"{tm['mape']:>7.1f}% {tm['mae_ns']:>8.3f} "
              f"{tm['p90_err_ns']:>8.3f} {gap:>6.3f}{st}")
        if r["val_r2"] > best_r2:
            best_r2, best_name = r["val_r2"], n

    per_decile_table(y_te, results[best_name]["te_pred"], best_name)

    # ── 14. vs teammate ───────────────────────────────────────
    best_tm = results[best_name]["test_metrics"]
    print(f"\n{'='*70}")
    print(f"  HEAD-TO-HEAD vs TEAMMATE (LGBM best for delay)")
    print(f"{'='*70}")
    print(f"  {'Metric':<38} {'Teammate':>10} {'Ours':>10} {'Result':>8}")
    print(f"  {'-'*68}")
    cmp = [
        ("Delay CV R²  (training)",      0.4195, cv_r2.mean()),
        ("Delay Test R² (best model)",   0.6179, best_tm["r2"]),
        ("CV→Test gap  (lower=better)",  0.1984,
         abs(results[best_name]["val_r2"] - best_tm["r2"])),
    ]
    for label, them, us in cmp:
        better = us > them if "gap" not in label.lower() else us < them
        sym    = "✓ OURS" if better else "✗ THEIRS"
        print(f"  {label:<38} {them:>10.4f} {us:>10.4f} {sym:>8}")

    # ── 15. SHAP ──────────────────────────────────────────────
    sname = next(
        (n for n in ["lgbm", "xgb", "catboost", "gbm", "rf"]
         if n in results and results[n]["model"] is not None), None)
    if sname and HAS_SHAP:
        print(f"\nSHAP analysis on {sname}...")
        try:
            exp  = shap.TreeExplainer(results[sname]["model"])
            sv   = exp.shap_values(X_te_s)
            imp  = np.abs(sv).mean(axis=0)
            dfi  = (pd.DataFrame({"feature": keep, "shap": imp})
                    .sort_values("shap", ascending=False))
            dfi.to_csv(os.path.join(MODEL_DIR, "delay_shap.csv"), index=False)
            mx   = dfi["shap"].max()
            phy_in_top15 = [r["feature"] for _, r in dfi.head(15).iterrows()
                            if r["feature"] in PROTECTED_FEATURES]
            print(f"\n  Top-15 features (★ = physics feature):")
            print(f"  {'Feature':<45} {'SHAP':>8}")
            print(f"  {'-'*55}")
            for _, row in dfi.head(15).iterrows():
                bar = "█" * int(row["shap"] / mx * 20)
                tag = " ★" if row["feature"] in PROTECTED_FEATURES else ""
                print(f"  {row['feature']:<45} {bar}  {row['shap']:.5f}{tag}")
            print(f"\n  Physics features in top-15 : {len(phy_in_top15)}/15")
            print(f"  ✓ Good if top features include:"
                  f" pipeline_reduced_delay, mul_delay_proxy,")
            print(f"             carry_chain_depth, slow_ops_x_bw")
            if "pipeline_reduced_delay" in phy_in_top15:
                print(f"  ✓ pipeline_reduced_delay is in top-15 — physics working!")
            else:
                print(f"  → pipeline_reduced_delay not in top-15 — "
                      f"check extractor ran correctly")
        except Exception as e:
            print(f"  SHAP failed: {e}")

    # ── 16. Save predictions ──────────────────────────────────
    pred_df = pd.DataFrame({
        "y_true_ns":   y_te,
        "y_pred_ns":   results[best_name]["te_pred"],
        "abs_err_ns":  np.abs(y_te - results[best_name]["te_pred"]),
        "pct_err":     np.abs((y_te - results[best_name]["te_pred"]) /
                               np.clip(y_te, EPS, None)) * 100,
        "log10_true":  np.log10(np.clip(y_te, EPS, None)),
        "log10_pred":  np.log10(np.clip(results[best_name]["te_pred"], EPS, None)),
    }).sort_values("abs_err_ns", ascending=False)
    pred_df.to_csv(os.path.join(MODEL_DIR, "test_predictions.csv"), index=False)

    print(f"\n  Outputs → {MODEL_DIR}")
    print("=" * 70)
    print("  RESULT GUIDE:")
    print("    R² > 0.75, MAPE < 20%  → excellent (delay is structurally predictable)")
    print("    R² > 0.60, MAPE < 35%  → good, clearly beats teammate CV score")
    print("    R² > 0.45, MAPE < 50%  → baseline")
    print("    pipeline_reduced_delay in SHAP top-5 → physics features working")


if __name__ == "__main__":
    main()