"""#!/usr/bin/env python3
""
power_predict_experiment.py

Run RTL-only vs RTL+Netlist experiments to predict Power.

Usage:
    python power_predict_experiment.py --csv_path /path/to/dataset.csv \
        --power_col Power --use_group_col design_family --out_dir ./results

Notes:
 - The script uses log1p transform for power by default (safer for relative errors).
 - If XGBoost/LightGBM/CatBoost are not installed, it falls back to RandomForest or
   GradientBoosting from scikit-learn.
 - The script does automatic feature engineering for RTL and netlist, runs a
   feature-selection pipeline, and reports robust metrics including SMAPE.
"""

import os
import argparse
import warnings
from typing import List, Tuple, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split, KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# Optional libraries
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
EPS = 1e-9

# -----------------------
# Metrics
# -----------------------
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) + EPS
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true), 1e-6, None)
    return 100.0 * np.mean(np.abs((y_true - y_pred) / denom))

def median_ape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true), 1e-6, None)
    return 100.0 * np.median(np.abs((y_true - y_pred) / denom))

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(safe_mape(y_true, y_pred))
    s = float(smape(y_true, y_pred))
    medape = float(median_ape(y_true, y_pred))
    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape, "smape": s, "median_ape": medape}

# -----------------------
# Feature engineering
# -----------------------
def engineer_rtl_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create an expanded, optimized RTL feature set (returns new DataFrame)."""
    df_r = pd.DataFrame(index=df.index)
    # Basic counters (if present)
    candidates = [
        "num_lines", "num_modules", "num_reg", "num_wire", "num_input", "num_output",
        "total_bits", "num_add", "num_sub", "num_mul", "num_div",
        "num_logic_and", "num_logic_or", "num_logic_xor", "num_shifts",
        "num_comparisons", "num_if", "num_case", "num_for", "num_ternary", "num_branches",
        "max_bitwidth", "min_bitwidth", "avg_bitwidth"
    ]
    for c in candidates:
        if c in df.columns:
            df_r[c] = df[c].astype(float)

    # Derived features
    if {"num_add", "num_sub", "num_mul", "num_div"}.intersection(df.columns):
        arith_cols = [c for c in ["num_add", "num_sub", "num_mul", "num_div"] if c in df.columns]
        df_r["total_arithmetic"] = df[arith_cols].sum(axis=1)
    if "num_lines" in df.columns and "total_arithmetic" in df_r.columns:
        df_r["arith_density"] = df_r["total_arithmetic"] / (df["num_lines"] + EPS)

    # operation ratios & interactions
    if "num_mul" in df.columns and "num_add" in df.columns:
        df_r["mul_add_ratio"] = df["num_mul"] / (df["num_add"] + EPS)
    for a, b in [("num_logic_xor", "num_reg"), ("total_arithmetic", "total_bits")]:
        if a in df.columns and b in df.columns:
            df_r[f"{a}_x_{b}"] = (df[a].astype(float) * df[b].astype(float))

    # bits transforms
    if "total_bits" in df.columns:
        df_r["bits_log"] = np.log1p(df["total_bits"].astype(float))
        df_r["bits_sqrt"] = np.sqrt(df["total_bits"].astype(float))
    if all(c in df.columns for c in ["max_bitwidth", "min_bitwidth"]):
        df_r["bitwidth_range"] = df["max_bitwidth"].astype(float) - df["min_bitwidth"].astype(float)
    if "avg_bitwidth" in df.columns and "total_bits" in df.columns:
        df_r["bits_per_line"] = df["total_bits"].astype(float) / (df.get("num_lines", 1) + EPS)

    # control/logic densities
    logic_cols = [c for c in ["num_logic_and", "num_logic_or", "num_logic_xor", "num_shifts"] if c in df.columns]
    if logic_cols:
        df_r["total_logic"] = df[logic_cols].sum(axis=1)
        if "num_lines" in df.columns:
            df_r["logic_density"] = df_r["total_logic"] / (df["num_lines"] + EPS)

    control_cols = [c for c in ["num_if", "num_case", "num_for", "num_branches"] if c in df.columns]
    if control_cols:
        df_r["total_control"] = df[control_cols].sum(axis=1)
        if "num_lines" in df.columns:
            df_r["control_density"] = df_r["total_control"] / (df["num_lines"] + EPS)

    # switching proxy (strongly correlated with power)
    if "num_reg" in df.columns and "num_logic_xor" in df.columns:
        df_r["switching_proxy_simple"] = df["num_reg"].astype(float) + 2.0 * df["num_logic_xor"].astype(float)

    # additional engineered features
    if "num_reg" in df.columns and "num_lines" in df.columns:
        df_r["reg_density"] = df["num_reg"].astype(float) / (df["num_lines"].astype(float) + EPS)

    # fill inf/nan
    df_r = df_r.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df_r


def engineer_netlist_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create netlist-derived features (returns new DataFrame)."""
    df_n = pd.DataFrame(index=df.index)
    netlist_candidates = [
        "netlist_num_gates", "netlist_num_nets", "netlist_num_instances",
        "netlist_inv_count", "netlist_and_count", "netlist_or_count",
        "netlist_nand_count", "netlist_nor_count", "netlist_xor_count",
        "netlist_mux_count", "netlist_buf_count", "netlist_maj3_count"
    ]
    for c in netlist_candidates:
        if c in df.columns:
            df_n[c] = df[c].astype(float)

    if "netlist_num_gates" in df.columns:
        df_n["gates_log"] = np.log1p(df["netlist_num_gates"].astype(float))
        if "num_lines" in df.columns:
            df_n["gates_per_line"] = df["netlist_num_gates"].astype(float) / (df["num_lines"].astype(float) + EPS)

    gate_cols = [c for c in df.columns if c.startswith("netlist_") and c.endswith("_count")]
    if gate_cols:
        total = df[gate_cols].sum(axis=1).replace(0, EPS)
        for col in gate_cols:
            df_n[f"{col}_ratio"] = df[col].astype(float) / total

        # gate diversity
        df_n["gate_diversity"] = (df[gate_cols] > 0).sum(axis=1)

    df_n = df_n.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df_n

# -----------------------
# Feature selection
# -----------------------
def select_top_features(X: pd.DataFrame, y: np.ndarray, k: int = 40) -> List[str]:
    """
    Two-stage selection:
      1) mutual_info_regression (fast, nonparametric) to short-list 2*k
      2) train a light trees (LGBM or RF) to get importance and pick top k
    """
    n_features = X.shape[1]
    k = min(k, n_features)
    # 1) mutual info
    try:
        mi = mutual_info_regression(X.values, y, random_state=RANDOM_STATE)
        mi_idx = np.argsort(mi)[::-1][: min(n_features, 2*k)]
        shortlist = X.columns[mi_idx].tolist()
    except Exception:
        shortlist = X.columns.tolist()[: min(n_features, 2*k)]

    X_short = X[shortlist]

    # 2) importance via tree
    if HAS_LGBM:
        model = LGBMRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)

    model.fit(X_short, y)
    try:
        imp = getattr(model, "feature_importances_", None)
        if imp is None:
            imp = np.zeros(X_short.shape[1])
    except Exception:
        imp = np.zeros(X_short.shape[1])
    idx = np.argsort(imp)[::-1][:k]
    selected = X_short.columns[idx].tolist()
    return selected

# -----------------------
# Model factory
# -----------------------
def get_default_models() -> Dict[str, BaseEstimator]:
    models = {}
    if HAS_XGB:
        models["xgb"] = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                     random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    if HAS_LGBM:
        models["lgbm"] = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=7,
                                       random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    if HAS_CAT:
        models["cat"] = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                                          random_seed=RANDOM_STATE, verbose=False)
    # Always include strong sklearn fallback
    models["rf"] = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1)
    models["gb"] = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=RANDOM_STATE)
    return models

# -----------------------
# Training / evaluation
# -----------------------
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, inverse_target_fn=None) -> Dict:
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    if inverse_target_fn is not None:
        # predictions currently in transformed space (if training on transformed target)
        val_pred_orig = inverse_target_fn(val_pred)
        y_val_orig = inverse_target_fn(y_val) if y_val.ndim==1 else inverse_target_fn(y_val)
        metrics = compute_metrics(y_val_orig, val_pred_orig)
    else:
        metrics = compute_metrics(y_val, val_pred)
    return {"model": model, "metrics": metrics, "val_pred": val_pred}

# -----------------------
# Experiment runner
# -----------------------
def run_experiment(df: pd.DataFrame,
                   power_col: str,
                   use_netlist: bool,
                   out_dir: str,
                   group_col: Optional[str] = None,
                   select_k: int = 40) -> Dict:
    """
    Run experiment and return summary dict with metrics & artifacts saved in out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    # 1) Create features
    df_rtl = engineer_rtl_features(df)
    if use_netlist:
        df_net = engineer_netlist_features(df)
        X_all = pd.concat([df_rtl, df_net], axis=1)
    else:
        X_all = df_rtl

    # Keep only finite
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 2) Targets (use log1p to stabilize relative errors; store original for eval)
    if power_col not in df.columns:
        raise ValueError(f"Power column '{power_col}' not found in CSV")
    y_orig = df[power_col].astype(float).values
    # mask non-positive? keep zeros; log1p handles zeros
    y_trans = np.log1p(y_orig)

    # 3) Split (GroupKFold if group provided and enough groups)
    if group_col and group_col in df.columns:
        groups = df[group_col].values
        # produce a single train/val/test split using GroupKFold (k=5) and pick one fold as test
        gkf = GroupKFold(n_splits=5)
        # use first split as test selection (deterministic)
        splits = list(gkf.split(X_all, y_trans, groups=groups))
        train_idx, test_idx = splits[0]
        # further split train into train/val
        X_train_full = X_all.iloc[train_idx]
        y_train_full = y_trans[train_idx]
        X_test = X_all.iloc[test_idx]
        y_test = y_trans[test_idx]
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full,
                                                          test_size=0.1765, random_state=RANDOM_STATE)  # ~15% overall val
    else:
        # standard random split
        X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_trans, test_size=0.15, random_state=RANDOM_STATE)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=RANDOM_STATE)

    # Save index counts
    print(f"[experiment] use_netlist={use_netlist} -> train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # 4) Feature selection on training set
    selected = select_top_features(X_train, y_train, k=select_k)
    print(f"[feature-selection] selected {len(selected)} features")
    X_train_sel = X_train[selected]
    X_val_sel = X_val[selected]
    X_test_sel = X_test[selected]

    # 5) Scaling
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train_sel)
    X_val_s = scaler.transform(X_val_sel)
    X_test_s = scaler.transform(X_test_sel)

    joblib.dump(scaler, os.path.join(out_dir, f"scaler_use_netlist_{use_netlist}.joblib"))

    # 6) Model training (we train multiple models and report best by R2 on validation)
    models = get_default_models()
    results = {}
    # inverse function for log1p
    inv_fn = lambda arr: np.expm1(np.array(arr))

    for name, model in models.items():
        try:
            m = model
            m.fit(X_train_s, y_train)
            # validation predictions (transformed space)
            val_pred_trans = m.predict(X_val_s)
            val_pred_orig = inv_fn(val_pred_trans)
            y_val_orig = inv_fn(y_val)
            metrics = compute_metrics(y_val_orig, val_pred_orig)
            print(f"  model={name} | val_r2={metrics['r2']:.4f} smape={metrics['smape']:.2f}% mape={metrics['mape']:.2f}%")
            # test eval
            test_pred_trans = m.predict(X_test_s)
            test_pred_orig = inv_fn(test_pred_trans)
            y_test_orig = inv_fn(y_test)
            test_metrics = compute_metrics(y_test_orig, test_pred_orig)
            # save model
            joblib.dump(m, os.path.join(out_dir, f"model_{name}_use_netlist_{use_netlist}.joblib"))
            results[name] = {"val_metrics": metrics, "test_metrics": test_metrics, "model_ref": os.path.join(out_dir, f"model_{name}_use_netlist_{use_netlist}.joblib")}
        except Exception as e:
            print(f"Model {name} failed: {e}")

    # 7) Baseline & permutation importance for top model
    # pick best by val r2
    best_name = max(results.keys(), key=lambda n: results[n]["val_metrics"]["r2"]) if results else None
    if best_name:
        best_model = joblib.load(results[best_name]["model_ref"])
        # permutation importance on validation set (sklearn's implementation)
        try:
            perm = permutation_importance(best_model, X_val_s, y_val, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
            perm_idx = np.argsort(perm.importances_mean)[::-1]
            top_perm = [(selected[i], float(perm.importances_mean[i])) for i in perm_idx[:20]]
            results["feature_importance_perm"] = top_perm
            # also save to CSV
            pd.DataFrame(top_perm, columns=["feature", "importance"]).to_csv(os.path.join(out_dir, f"perm_importance_{use_netlist}.csv"), index=False)
        except Exception as e:
            print("Permutation importance failed:", e)

    # 8) Save summary
    summary_rows = []
    for mname, info in results.items():
        if mname == "feature_importance_perm":
            continue
        tm = info["test_metrics"]
        vm = info["val_metrics"]
        summary_rows.append({
            "use_netlist": use_netlist,
            "model": mname,
            "val_r2": vm["r2"],
            "val_smape": vm["smape"],
            "test_r2": tm["r2"],
            "test_smape": tm["smape"],
            "test_rmse": tm["rmse"],
            "test_mae": tm["mae"],
            "test_mape": tm["mape"],
            "test_median_ape": tm["median_ape"],
            "model_path": info["model_ref"]
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(out_dir, f"summary_use_netlist_{use_netlist}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[saved] summary -> {summary_csv}")
    return {"summary_df": summary_df, "results": results, "selected_features": selected, "out_dir": out_dir}

# -----------------------
# CLI / main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Power prediction: RTL-only vs RTL+Netlist experiment")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--power_col", type=str, default="Power", help="Name of power column in CSV")
    parser.add_argument("--group_col", type=str, default=None, help="Optional grouping column (e.g., design_family)")
    parser.add_argument("--out_dir", type=str, default="./results_power", help="Output directory for models and summaries")
    parser.add_argument("--select_k", type=int, default=40, help="Number of features to select")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    # quick sanity: drop rows with missing power or with negative labels if that should be excluded
    before = len(df)
    df = df.dropna(subset=[args.power_col])
    print(f"[load] rows: {before} -> after dropna power: {len(df)}")

    # Run RTL-only
    out_rtl = os.path.join(args.out_dir, "rtl_only")
    res_rtl = run_experiment(df, power_col=args.power_col, use_netlist=False, out_dir=out_rtl, group_col=args.group_col, select_k=args.select_k)

    # Run RTL+Netlist
    out_hybrid = os.path.join(args.out_dir, "rtl_plus_netlist")
    res_hybrid = run_experiment(df, power_col=args.power_col, use_netlist=True, out_dir=out_hybrid, group_col=args.group_col, select_k=args.select_k)

    # Combine summaries and present side-by-side
    df_a = res_rtl["summary_df"]
    df_b = res_hybrid["summary_df"]
    combined = pd.concat([df_a, df_b], ignore_index=True)
    combined_csv = os.path.join(args.out_dir, "combined_summary.csv")
    combined.to_csv(combined_csv, index=False)
    print(f"[done] Combined summary saved to {combined_csv}")
    print(combined[["use_netlist", "model", "test_r2", "test_smape", "test_mape", "test_rmse"]].sort_values(["use_netlist", "test_r2"], ascending=[False, False]).to_string(index=False))

if __name__ == "__main__":
    main()