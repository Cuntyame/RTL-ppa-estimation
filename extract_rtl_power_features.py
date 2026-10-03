#!/usr/bin/env python3
r"""
extract_rtl_power_features_with_rpt.py

Extract canonical RTL features (31), engineered power proxies, and parse .rpt files (PPA, timing, stats).
Usage:
  python extract_rtl_power_features_with_rpt.py --dataset_root "C:\ml ppa\Final_Clean_Dataset" --out_csv final.csv
"""

import argparse
import re
from pathlib import Path
from collections import Counter
import math
import numpy as np
import pandas as pd

EPS = 1e-9

import os
import time
from pathlib import Path

def save_dataframe_safe(df, out_csv: str, max_attempts: int = 3) -> str:
    """
    Try to save DataFrame to out_csv. If PermissionError occurs, try safe alternatives:
      1) write to temporary alternate filename (pid, timestamp) in same folder
      2) attempt atomic replace to final path
      3) if all fail, raise PermissionError with diagnostic message

    Returns the path actually written (string).
    """
    out_path = Path(out_csv)
    parent = out_path.parent
    if not parent.exists():
        # attempt to create parent directories
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise PermissionError(f"Cannot create directory {parent}: {e}")

    # First straightforward attempt
    try:
        save_dataframe_safe(df, out_csv)
        # df.to_csv(str(out_path), index=False)
        print(f"[saved] {out_path}")
        return str(out_path)
    except PermissionError as e:
        print(f"[warn] PermissionError writing {out_path}: {e}")

    # Try safe alternates
    attempts = [
        parent / f"{out_path.stem}.tmp{out_path.suffix}",
        parent / f"{out_path.stem}.{os.getpid()}{out_path.suffix}",
        parent / f"{out_path.stem}.{int(time.time())}{out_path.suffix}"
    ]

    for alt in attempts:
        try:
            df.to_csv(str(alt), index=False)
            # attempt atomic replace (may still fail if target locked)
            try:
                os.replace(str(alt), str(out_path))
                print(f"[saved via alt -> replaced] {out_path}")
                return str(out_path)
            except PermissionError as e_replace:
                # cannot replace target (likely target locked); keep alt and report alt path
                print(f"[warn] Could not replace target file (locked). Alt file written at: {alt}")
                return str(alt)
            except Exception as e_replace:
                # If replace fails for other reasons, keep alt and continue trying other alts
                print(f"[warn] os.replace failed for {alt} -> {out_path}: {e_replace}")
                return str(alt)
        except PermissionError as pe_alt:
            print(f"[warn] PermissionError writing alt file {alt}: {pe_alt}")
        except Exception as ex_alt:
            print(f"[warn] Failed writing alt file {alt}: {ex_alt}")

    # If we get here, all attempts failed
    raise PermissionError(
        f"Failed to write CSV to {out_path} or alternatives. Possible reasons:\n"
        "- The file is open in another program (Excel, Notepad, Explorer preview) — close it.\n"
        "- You don't have write permission to the folder. Try writing to a different folder (e.g. Documents).\n"
        "- Antivirus / OneDrive is locking the file. Pause OneDrive or exclude the folder from antivirus.\n\n"
        f"Try running with --out_csv \"C:\\Users\\Admin\\Documents\\{out_path.name}\", or run the script as Administrator."
    )
# --------------------------
# Utility: read files safely
# --------------------------
def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""

def clean_rtl_text(text: str) -> str:
    # remove single-line comments, block comments, and string literals
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r'"(?:\\.|[^"\\])*"', "", text, flags=re.DOTALL)
    return text

# --------------------------
# Bitwidth features
# --------------------------
BITRANGE_RE = re.compile(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]")

def extract_bitwidth_features(rtl: str) -> dict:
    matches = BITRANGE_RE.findall(rtl)
    if not matches:
        return {
            "max_bitwidth": 1,
            "min_bitwidth": 1,
            "avg_bitwidth": 1.0,
            "total_bits": 1,
            "num_bitwidths": 0,
            "num_32bit": 0,
            "num_16bit": 0,
            "num_8bit": 0,
            "num_1bit": 1,
        }
    bitwidths = [abs(int(a) - int(b)) + 1 for a, b in matches]
    return {
        "max_bitwidth": int(max(bitwidths)),
        "min_bitwidth": int(min(bitwidths)),
        "avg_bitwidth": float(np.mean(bitwidths)),
        "total_bits": int(sum(bitwidths)),
        "num_bitwidths": int(len(bitwidths)),
        "num_32bit": int(bitwidths.count(32)),
        "num_16bit": int(bitwidths.count(16)),
        "num_8bit": int(bitwidths.count(8)),
        "num_1bit": int(bitwidths.count(1)),
    }

# --------------------------
# Operator counts
# --------------------------
def count_operators(rtl: str) -> dict:
    # Basic operator counts; keep simple and robust
    return {
        "num_add": rtl.count("+"),
        "num_sub": rtl.count("-"),
        "num_mul": rtl.count("*"),
        "num_div": rtl.count("/"),
        "num_logic_and": rtl.count("&"),
        "num_logic_or": rtl.count("|"),
        "num_logic_xor": rtl.count("^"),
        "num_shifts": rtl.count("<<") + rtl.count(">>"),
        "num_comparisons": len(re.findall(r"==|!=|<=|>=|(?<![<>=])<(?![<>=])|(?<![<>=])>(?![<>=])", rtl)),
    }

# --------------------------
# Structural / control features
# --------------------------
def extract_structural_features(rtl: str) -> dict:
    feats = {
        "num_lines": rtl.count("\n") + 1,
        "num_always": len(re.findall(r"\balways\b", rtl)),
        "num_assign": len(re.findall(r"\bassign\b", rtl)),
        "num_if": len(re.findall(r"\bif\b", rtl)),
        "num_case": len(re.findall(r"\bcase\b", rtl)),
        "num_for": len(re.findall(r"\bfor\b", rtl)),
        "num_ternary": rtl.count("?"),
        "num_wire": len(re.findall(r"\bwire\b", rtl)),
        "num_reg": len(re.findall(r"\breg\b", rtl)),
        "num_input": len(re.findall(r"\binput\b", rtl)),
        "num_output": len(re.findall(r"\boutput\b", rtl)),
        "num_modules": max(0, len(re.findall(r"\bmodule\b", rtl)) - len(re.findall(r"\bendmodule\b", rtl))),
    }
    feats["num_branches"] = feats["num_if"] + feats["num_case"] + feats["num_ternary"]
    return feats

# --------------------------
# Clock / posedge detection
# --------------------------
POS_EDGE_RE = re.compile(r"@(posedge|negedge)\s+([a-zA-Z_][\w]*)", re.IGNORECASE)
CLK_CANDIDATE_RE = re.compile(r"\b(clk|clock|clk_i|clk_)\w*\b", re.IGNORECASE)

def detect_clock_features(rtl: str) -> dict:
    matches = POS_EDGE_RE.findall(rtl)
    clocks = set([m[1] for m in matches])
    # heuristics: if no posedge found, still check for candidate identifiers
    candidate_clks = set(re.findall(CLK_CANDIDATE_RE, rtl))
    num_clk_domains = len(clocks) if clocks else (1 if candidate_clks else 0)
    return {"num_posedge_always": int(len(matches)), "num_clk_domains": int(num_clk_domains)}

# --------------------------
# Signal / fanout proxy
# --------------------------
IDENT_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")

def extract_signal_fanout_proxy(rtl: str) -> dict:
    ids = IDENT_RE.findall(rtl)
    if not ids:
        return {
            "unique_signals": 0,
            "max_signal_occurrences": 0,
            "mean_signal_occurrences": 0.0,
            "num_high_fanout_signals": 0,
        }
    freq = Counter(ids)
    vals = np.array(list(freq.values()), dtype=float)
    return {
        "unique_signals": int(len(freq)),
        "max_signal_occurrences": int(vals.max()),
        "mean_signal_occurrences": float(vals.mean()),
        "num_high_fanout_signals": int((vals > 10).sum()),
    }

# --------------------------
# Power-engineered features (including industry features)
# --------------------------
def compute_power_engineered_features(base: dict, bitwidth: dict, signal_proxy: dict) -> dict:
    total_arith = base.get("num_add", 0) + base.get("num_sub", 0) + base.get("num_mul", 0) + base.get("num_div", 0)
    total_logic = base.get("num_logic_and", 0) + base.get("num_logic_or", 0) + base.get("num_logic_xor", 0) + base.get("num_shifts", 0)
    feats = {}
    feats["total_arithmetic"] = int(total_arith)
    feats["arith_density"] = float(total_arith / max(base.get("num_lines", 1), 1))
    feats["switching_proxy"] = float(base.get("num_reg", 0) + 2.0 * base.get("num_logic_xor", 0))
    feats["logic_density"] = float(total_logic / max(base.get("num_lines", 1), 1))
    feats["mux_count"] = int(base.get("num_ternary", 0) + base.get("num_case", 0))
    feats["bits_log"] = float(math.log1p(bitwidth.get("total_bits", 0)))
    feats["bits_sqrt"] = float(math.sqrt(max(bitwidth.get("total_bits", 0), 0)))
    feats["register_bit_product"] = float(base.get("num_reg", 0) * bitwidth.get("avg_bitwidth", 1.0))
    feats["toggle_estimate"] = float(feats["switching_proxy"] * feats["logic_density"])
    # Industry-style features
    feats["bit_toggle_load"] = float(bitwidth.get("total_bits", 0) * feats["switching_proxy"])            # capacitance × toggle proxy
    feats["fanout_switching_pressure"] = float((base.get("num_logic_xor", 0) + base.get("num_logic_and", 0)) * bitwidth.get("avg_bitwidth", 1.0))
    feats["reg_to_logic_ratio"] = float(base.get("num_reg", 0) / max(total_logic, 1))
    # include signal proxy summary
    feats.update({
        "unique_signals": int(signal_proxy.get("unique_signals", 0)),
        "max_signal_occurrences": int(signal_proxy.get("max_signal_occurrences", 0)),
        "mean_signal_occurrences": float(signal_proxy.get("mean_signal_occurrences", 0.0)),
        "num_high_fanout_signals": int(signal_proxy.get("num_high_fanout_signals", 0)),
    })
    return feats

# --------------------------
# .rpt parsing
# --------------------------
# robust multi-pattern extraction of common PPA/timing numbers from .rpt text
RPT_PATTERNS = {
    "total_cell_area": [
        r"Total cell area[:=]\s*([\d,\.]+)",
        r"Cell Area[:=]\s*([\d,\.]+)",
        r"Combinational area[:=]\s*([\d,\.]+)"
    ],
    "comb_area": [
        r"Combinational area[:=]\s*([\d,\.]+)",
    ],
    "Power": [
        r"Total Dynamic Power\s*[:=]?\s*([\d\.]+)\s*([munp]?W)?",
        r"Total power\s*[:=]?\s*([\d\.]+)\s*([munp]?W)?",
        r"Dynamic Power\s*[:=]?\s*([\d\.]+)\s*([munp]?W)?"
    ],
    "critical_path_length": [
        r"Critical Path Length[:=]\s*([\d\.]+)",
        r"critical path[:=]\s*([\d\.]+)",
        r"data arrival time\s+([\d\.]+)"
    ],
    "levels_of_logic": [
        r"Levels of Logic[:=]\s*([\d\.]+)"
    ],
    "wns": [
        r"WNS[:=]\s*([-\d\.]+)"
    ],
    "tns": [
        r"TNS[:=]\s*([-\d\.]+)"
    ],
    "num_cells": [
        r"Number of cells[:=]?\s*([\d,]+)",
        r"cells:\s*([\d,]+)"
    ],
    "num_nets": [
        r"Number of nets[:=]?\s*([\d,]+)"
    ]
}

UNIT_MULTIPLIERS = {
    "pw": 1e-9,  # power: pW -> mW
    "nw": 1e-6,
    "uw": 1e-3,
    "mw": 1.0,
    "w": 1000.0
}

def parse_rpt_text(rpt_text: str) -> dict:
    out = {}
    txt = rpt_text.replace(",", "")  # strip comma thousands
    # search each pattern list until first match
    for key, pats in RPT_PATTERNS.items():
        found = None
        found_unit = None
        for p in pats:
            m = re.search(p, txt, flags=re.IGNORECASE)
            if m:
                # some patterns capture unit in group(2)
                try:
                    val = float(m.group(1))
                    # unit handling if present
                    unit = m.group(2).lower() if m.lastindex and m.lastindex >= 2 and m.group(2) else ""
                except Exception:
                    try:
                        # fallback: if group(1) contains multiple numbers, take first
                        val = float(re.findall(r"[\d\.]+", m.group(1))[0])
                        unit = m.group(2).lower() if m.lastindex and m.lastindex >= 2 and m.group(2) else ""
                    except Exception:
                        continue
                # normalize units for Power to mW
                if key == "Power":
                    mult = 1.0
                    if unit:
                        unit = unit.lower()
                        mult = UNIT_MULTIPLIERS.get(unit, 1.0)
                        # UNIT_MULTIPLIERS maps to mW already for known units
                    val_mw = val * mult
                    found = float(val_mw)
                elif key in ("num_cells", "num_nets"):
                    try:
                        found = int(float(val))
                    except Exception:
                        found = None
                else:
                    found = float(val)
                found_unit = unit
                break
        out[key] = found if found is not None else None
    return out

# --------------------------
# Build per-design feature row
# --------------------------
def extract_features_for_design(design_dir: Path) -> dict:
    # find primary .v file (prefer <design>.v else first .v)
    vfiles = sorted(design_dir.glob("*.v"))
    if not vfiles:
        return None
    vpath = vfiles[0]
    rtl_raw = read_text_safe(vpath)
    rtl = clean_rtl_text(rtl_raw)

    # canonical RTL features (31)
    bitwidth_feats = extract_bitwidth_features(rtl)
    op_feats = count_operators(rtl)
    struct_feats = extract_structural_features(rtl)

    canonical_names = [
        "num_lines", "max_bitwidth", "min_bitwidth", "avg_bitwidth",
        "total_bits", "num_bitwidths", "num_32bit", "num_16bit", "num_8bit", "num_1bit",
        "num_add", "num_sub", "num_mul", "num_div",
        "num_logic_and", "num_logic_or", "num_logic_xor", "num_shifts",
        "num_comparisons",
        "num_always", "num_assign", "num_if", "num_case", "num_for", "num_ternary",
        "num_branches",
        "num_wire", "num_reg", "num_input", "num_output", "num_modules"
    ]

    # merge base dict to lookup values
    base = {}
    base.update(bitwidth_feats)
    base.update(op_feats)
    base.update(struct_feats)

    # assemble canonical in order
    canonical = {k: base.get(k, 0) for k in canonical_names}

    # additional engineered features
    signal_proxy = extract_signal_fanout_proxy(rtl)
    clock_feats = detect_clock_features(rtl)
    power_feats = compute_power_engineered_features(base, bitwidth_feats, signal_proxy)

    # parse .rpt files in folder
    rpt_files = sorted(list(design_dir.glob("*.rpt")))
    rpt_texts = [read_text_safe(p) for p in rpt_files]
    rpt_concat = "\n\n".join(rpt_texts)
    rpt_metrics = parse_rpt_text(rpt_concat) if rpt_concat.strip() else {k: None for k in RPT_PATTERNS.keys()}

    # add minimal rpt metadata
    rpt_meta = {
        "rpt_files": ";".join([p.name for p in rpt_files]) if rpt_files else "",
        "rpt_text_len": len(rpt_concat),
    }

    # final merged row
    row = {"Design_Name": design_dir.name}
    row.update(canonical)                 # 31 fields
    row.update(power_feats)               # engineered power features
    row.update(signal_proxy)              # fanout proxies
    row.update(clock_feats)               # clock features
    row.update({"rtl_code_len": len(rtl)})
    # rpt parsed numeric metrics
    row.update(rpt_metrics)
    row.update(rpt_meta)

    return row

# --------------------------
# Dataset builder
# --------------------------
def build_dataset(dataset_root: str, out_csv: str) -> pd.DataFrame:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"{dataset_root} not found")
    design_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    rows = []
    print(f"[scan] found {len(design_dirs)} design folders under {dataset_root}")
    for d in design_dirs:
        try:
            row = extract_features_for_design(d)
            if row:
                rows.append(row)
        except Exception as e:
            print(f"[warn] failed to process {d.name}: {e}")
    df = pd.DataFrame(rows)
    # Ensure consistent column order: Design_Name, canonical 31, engineered, rpt metrics, rpt files
    df.to_csv(out_csv, index=False)
    print(f"[done] wrote {len(df)} rows to {out_csv}")
    return df

# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract RTL & .rpt features for power prediction")
    parser.add_argument("--dataset_root", required=True, help="Root folder containing per-design subfolders")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    args = parser.parse_args()
    build_dataset(args.dataset_root, args.out_csv)

if __name__ == "__main__":
    main()