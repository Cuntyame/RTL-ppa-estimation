#!/usr/bin/env python3
r"""
extract_rtl_area_features.py
==============================
Dedicated RTL feature extractor for AREA prediction.
Extracts area-physics features directly from Verilog .v files.

USAGE:
  python extract_rtl_area_features.py \
      --dataset_root "C:\ml ppa\Final_Clean_Dataset" \
      --out_csv "C:\Users\Admin\Documents\rtl_area_features.csv"

WHY A SEPARATE EXTRACTOR FROM POWER:
  Power features: switching activity proxies (FF bits, XOR density,
                  clock enables, glitch potential) — all DYNAMIC
  Area features:  structural cell count estimates (multiplier width²,
                  register bits, decoder logic) — all STATIC
  The power extractor's features like num_clock_enable, clk_domain_pressure,
  glitch_potential are irrelevant noise for area prediction and get picked
  up as spurious correlates (clock domains ≠ area driver).

AREA PHYSICS (sky130_fd_sc_hd standard cell library):
  Cell type        GE estimate   Scales with
  ─────────────────────────────────────────────────────────
  DFF              8 GE/bit      num_reg × avg_bitwidth
  Full adder       2 GE/bit      (add+sub) × avg_bitwidth
  Multiplier       N² GE         num_mul × max_bitwidth²
  Divider          4N² GE        num_div × max_bitwidth²
  Comparator       2N GE         num_comp × avg_bitwidth
  XOR2             2 GE/bit      num_xor × avg_bitwidth
  AND2/OR2         1 GE/bit      num_and/or × avg_bitwidth
  MUX2             3 GE/bit      mux_count × avg_bitwidth
  Decoder          log2(N) GE    case_items × log2(num_case)
  Barrel shifter   N×log2N GE    num_shifts × avg_bw × log2(avg_bw)
  Memory (reg file) 6 GE/bit    array_depth × width

NEW FEATURES VS POWER EXTRACTOR:
  - generate_count      : generate blocks unroll to N copies of logic
  - generate_factor     : estimated unroll multiplier
  - instance_count      : submodule instantiations (hierarchy area)
  - param_count         : parameterized widths (variable-width datapaths)
  - concat_width        : concatenation widths ({a,b,c} → wide buses)
  - case_items          : case statement items → decoder size
  - mem_depth_proxy     : indexed array depth estimate
  - wide_bus_count      : buses wider than 16 bits (routing area)
  - always_star_count   : always@(*) = combinational → area, not power
  - task_function_count : inlined tasks multiply cell count
"""

import re
import os
import csv
import math
import time
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, Any

import numpy as np
import pandas as pd

EPS = 1e-9

# ══════════════════════════════════════════════════════════════
#  FILE I/O
# ══════════════════════════════════════════════════════════════

def read_safe(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def clean_verilog(code: str) -> str:
    """Remove comments and string literals."""
    code = re.sub(r"//.*?$",            "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/",         "", code, flags=re.DOTALL)
    code = re.sub(r'"(?:\\.|[^"\\])*"', "", code, flags=re.DOTALL)
    return code


# ══════════════════════════════════════════════════════════════
#  BITWIDTH EXTRACTION
# ══════════════════════════════════════════════════════════════

BITRANGE_RE = re.compile(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]")


def extract_bitwidths(rtl: str) -> dict:
    matches = BITRANGE_RE.findall(rtl)
    if not matches:
        return dict(max_bw=1, min_bw=1, avg_bw=1.0, total_bits=1,
                    num_bw_decls=0, n32bit=0, n16bit=0, n8bit=0, n1bit=0,
                    wide_bus_count=0)
    bw = [abs(int(a) - int(b)) + 1 for a, b in matches]
    return dict(
        max_bw         = int(max(bw)),
        min_bw         = int(min(bw)),
        avg_bw         = float(np.mean(bw)),
        total_bits     = int(sum(bw)),
        num_bw_decls   = len(bw),
        n32bit         = bw.count(32),
        n16bit         = bw.count(16),
        n8bit          = bw.count(8),
        n1bit          = bw.count(1),
        wide_bus_count = sum(1 for b in bw if b > 16),
    )


# ══════════════════════════════════════════════════════════════
#  OPERATOR COUNTS
# ══════════════════════════════════════════════════════════════

def count_operators(rtl: str) -> dict:
    return dict(
        num_add         = rtl.count("+"),
        num_sub         = rtl.count("-"),
        num_mul         = rtl.count("*"),
        num_div         = rtl.count("/"),
        num_logic_and   = rtl.count("&"),
        num_logic_or    = rtl.count("|"),
        num_logic_xor   = rtl.count("^"),
        num_shifts      = rtl.count("<<") + rtl.count(">>"),
        num_mod         = rtl.count("%"),
        num_comparisons = len(re.findall(
            r"==|!=|<=|>=|(?<![<>=])<(?![<>=])|(?<![<>=])>(?![<>=])", rtl)),
        num_concat      = rtl.count("{"),   # {a,b} concatenation
        num_ternary     = rtl.count("?"),
    )


# ══════════════════════════════════════════════════════════════
#  STRUCTURAL FEATURES
# ══════════════════════════════════════════════════════════════

def extract_structural(rtl: str) -> dict:
    f = dict(
        num_lines      = rtl.count("\n") + 1,
        rtl_code_len   = len(rtl),
        num_always     = len(re.findall(r"\balways\b",   rtl)),
        num_assign     = len(re.findall(r"\bassign\b",   rtl)),
        num_if         = len(re.findall(r"\bif\b",       rtl)),
        num_case       = len(re.findall(r"\bcase\b",     rtl)),
        num_for        = len(re.findall(r"\bfor\b",      rtl)),
        num_while      = len(re.findall(r"\bwhile\b",    rtl)),
        num_wire       = len(re.findall(r"\bwire\b",     rtl)),
        num_reg        = len(re.findall(r"\breg\b",      rtl)),
        num_input      = len(re.findall(r"\binput\b",    rtl)),
        num_output     = len(re.findall(r"\boutput\b",   rtl)),
        num_inout      = len(re.findall(r"\binout\b",    rtl)),
        num_parameter  = len(re.findall(r"\bparameter\b",rtl)),
        num_localparam = len(re.findall(r"\blocalparam\b",rtl)),
        num_modules    = max(0,
            len(re.findall(r"\bmodule\b", rtl)) -
            len(re.findall(r"\bendmodule\b", rtl))),

        # Area-specific structural features
        # generate blocks: for generate = N copies of logic → N× area
        generate_count = len(re.findall(r"\bgenerate\b", rtl)),

        # Submodule instantiations: each adds the submodule's area
        instance_count = len(re.findall(r"\.\w+\s*\(", rtl)),

        # Task/function inlining: each call duplicates logic
        task_count     = len(re.findall(r"\btask\b",    rtl)),
        function_count = len(re.findall(r"\bfunction\b",rtl)),

        # always@(*) = purely combinational blocks
        always_star    = len(re.findall(
            r"always\s*@\s*\(\s*\*\s*\)|always\s*@\s*\*", rtl)),

        # Case item count (more items → larger decoder)
        case_items     = len(re.findall(
            r"^\s*\d+\s*(?:'[bBhHoOdD])?\s*\w*\s*:", rtl,
            re.MULTILINE)),

        # Non-blocking vs blocking (NB = sequential = registers)
        nb_assign      = len(re.findall(r"<=(?!=)", rtl)),
        bl_assign      = len(re.findall(r"=(?!>|=)", rtl)),

        # begin/end depth proxy (nested logic = more area)
        begin_count    = len(re.findall(r"\bbegin\b", rtl)),
    )
    f["num_branches"] = f["num_if"] + f["num_case"] + f["num_ternary"] \
                        if "num_ternary" in f else f["num_if"] + f["num_case"]
    return f


# ══════════════════════════════════════════════════════════════
#  AREA-SPECIFIC ADVANCED FEATURES
# ══════════════════════════════════════════════════════════════

def extract_area_advanced(rtl: str, base: dict, bw: dict) -> dict:
    """
    Features that are unique to area prediction.
    These would be NOISE for a power extractor.
    """
    f = {}
    avg_bw = max(bw["avg_bw"], 1.0)
    max_bw = max(bw["max_bw"], 1.0)
    n_lines = max(base["num_lines"], 1)
    n_reg   = base["num_reg"]
    n_mul   = base["num_mul"]
    n_div   = base["num_div"]
    n_add   = base["num_add"]
    n_sub   = base["num_sub"]
    n_xor   = base["num_logic_xor"]
    n_and   = base["num_logic_and"]
    n_or    = base["num_logic_or"]
    n_mux   = base["num_ternary"] + base["num_case"]
    n_comp  = base["num_comparisons"]
    n_shift = base["num_shifts"]
    n_for   = base["num_for"]
    n_mod   = max(base["num_modules"], 1)
    n_inp   = base["num_input"]
    n_out   = base["num_output"]

    # ── PHYSICS-BASED GATE EQUIVALENT ESTIMATES ───────────────

    # Sequential: DFF = 8 GE/bit (measured from sky130 library)
    f["seq_ge"]           = n_reg * avg_bw * 8
    f["log_seq_ge"]       = math.log1p(f["seq_ge"])
    f["sqrt_seq_ge"]      = math.sqrt(max(f["seq_ge"], 0))

    # Multiplier: array multiplier = N² GE (dominant area term)
    # A 32-bit mul = 1024 GE; a 32-bit adder = 64 GE
    f["mul_ge_quad"]      = n_mul * max_bw * max_bw
    f["mul_ge_linear"]    = n_mul * avg_bw          # underestimate (for comparison)
    f["log_mul_ge_quad"]  = math.log1p(f["mul_ge_quad"])
    f["sqrt_mul_ge_quad"] = math.sqrt(max(f["mul_ge_quad"], 0))

    # Adder: carry-ripple = 2N GE; carry-lookahead ≈ N×log2(N)
    f["adder_ge"]         = (n_add + n_sub) * avg_bw * 2
    f["adder_ge_nlogn"]   = (n_add + n_sub) * avg_bw * math.log2(max(avg_bw, 2))
    f["log_adder_ge"]     = math.log1p(f["adder_ge"])

    # Divider: restoring divider ≈ 4N² GE
    f["div_ge"]           = n_div * max_bw * max_bw * 4
    f["log_div_ge"]       = math.log1p(f["div_ge"])

    # Modulo: similar to divider
    f["mod_ge"]           = base.get("num_mod", 0) * max_bw * max_bw * 4

    # Comparator: 2N GE
    f["comp_ge"]          = n_comp * avg_bw * 2
    f["log_comp_ge"]      = math.log1p(f["comp_ge"])

    # Bitwise logic: XOR=2GE, AND/OR=1GE
    f["logic_ge"]         = (n_xor * 2 + n_and + n_or) * avg_bw
    f["xor_ge"]           = n_xor * 2 * avg_bw
    f["log_logic_ge"]     = math.log1p(f["logic_ge"])

    # MUX: MUX2=3GE; decoder scales log2(N)
    f["mux_ge"]           = n_mux * avg_bw * 3
    f["decoder_ge"]       = base["num_if"] * math.log2(max(base["num_if"], 2)) * avg_bw
    f["log_mux_dec_ge"]   = math.log1p(f["mux_ge"] + f["decoder_ge"])

    # Barrel shifter: N×log2(N) MUX stages
    f["shift_ge"]         = n_shift * avg_bw * math.log2(max(avg_bw, 2))
    f["log_shift_ge"]     = math.log1p(f["shift_ge"])

    # Memory: register file ≈ 6 GE/bit (denser than random logic)
    mem_access = len(re.findall(
        r"\w+\s*\[\s*\w+\s*\]\s*(?:<=|=)|\b(?:<=|=)\s*\w+\s*\[\s*\w+\s*\]", rtl))
    f["num_memory_access"] = mem_access
    f["mem_ge"]            = mem_access * max_bw * 6
    f["log_mem_ge"]        = math.log1p(f["mem_ge"])

    # For loop unrolling: each iteration duplicates hardware
    # Detect loop bounds when constant: for(i=0;i<N;i++) → N copies
    loop_bounds = re.findall(r"for\s*\([^;]+;\s*\w+\s*<\s*(\d+)", rtl)
    loop_factor = sum(int(b) for b in loop_bounds if int(b) < 1000) or n_for * 4
    f["unroll_factor"]    = loop_factor
    f["unroll_ge"]        = loop_factor * avg_bw
    f["log_unroll_ge"]    = math.log1p(f["unroll_ge"])

    # Generate statement unrolling (very common source of large area)
    gen_bounds = re.findall(
        r"for\s*\([^;]+;\s*\w+\s*<\s*(\d+).*?endgenerate",
        rtl, re.DOTALL)
    gen_factor = sum(int(b) for b in gen_bounds if int(b) < 10000) \
                 or base["generate_count"] * 8
    f["generate_factor"]  = gen_factor
    f["generate_ge"]      = gen_factor * avg_bw * 4  # each iteration ≈ 4GE

    # ── TOTAL GATE ESTIMATE (master predictor) ────────────────
    f["total_ge"] = (
        f["seq_ge"]      + f["mul_ge_quad"]  + f["adder_ge"] +
        f["div_ge"]      + f["comp_ge"]      + f["logic_ge"] +
        f["mux_ge"]      + f["shift_ge"]     + f["mem_ge"]   +
        f["unroll_ge"]   + f["generate_ge"]
    )
    f["log_total_ge"]  = math.log1p(f["total_ge"])
    f["sqrt_total_ge"] = math.sqrt(max(f["total_ge"], 0))

    # ── AREA COMPOSITION FRACTIONS (scale-invariant) ──────────
    ge_safe = max(f["total_ge"], EPS)
    f["seq_area_frac"]    = f["seq_ge"]      / ge_safe
    f["mul_area_frac"]    = f["mul_ge_quad"] / ge_safe
    f["logic_area_frac"]  = f["logic_ge"]    / ge_safe
    f["mux_area_frac"]    = f["mux_ge"]      / ge_safe
    f["mem_area_frac"]    = f["mem_ge"]      / ge_safe
    f["arith_area_frac"]  = (f["adder_ge"] + f["mul_ge_quad"] + f["div_ge"]) / ge_safe

    # ── BITWIDTH STRUCTURE ────────────────────────────────────
    f["bw_range"]         = bw["max_bw"] - bw["min_bw"]
    f["bw_uniformity"]    = avg_bw / max(max_bw, 1)
    f["log_total_bits"]   = math.log1p(bw["total_bits"])
    f["sqrt_total_bits"]  = math.sqrt(max(bw["total_bits"], 0))
    f["bits_x_arith"]     = bw["total_bits"] * (n_add + n_sub + n_mul + n_div + 1)

    # ── INTERACTION TERMS ─────────────────────────────────────
    # Pipelined multiplier: mul×reg → large area (both contribute)
    f["mul_x_reg"]        = n_mul * n_reg * avg_bw
    f["log_mul_x_reg"]    = math.log1p(f["mul_x_reg"])

    # Arithmetic depth × bitwidth
    total_arith = n_add + n_sub + n_mul + n_div
    f["total_arithmetic"] = total_arith
    f["arith_x_bw"]       = total_arith * avg_bw
    f["log_arith_x_bw"]   = math.log1p(f["arith_x_bw"])

    # Hierarchy: each module interface adds port buffers
    f["hierarchy_ge"]     = n_mod * (n_inp + n_out) * avg_bw
    f["log_hierarchy_ge"] = math.log1p(f["hierarchy_ge"])

    # ── DENSITY FEATURES (per-line) ───────────────────────────
    f["ge_per_line"]      = f["total_ge"]    / n_lines
    f["mul_ge_per_line"]  = f["mul_ge_quad"] / n_lines
    f["seq_ge_per_line"]  = f["seq_ge"]      / n_lines
    f["arith_bw_density"] = total_arith * avg_bw / n_lines

    # ── DESIGN TYPE INDICATORS ────────────────────────────────
    # Purely combinational designs have very different area behaviour
    n_always_ff = len(re.findall(
        r"always\s*@\s*\(\s*(?:posedge|negedge)", rtl, re.IGNORECASE))
    f["num_always_ff"]    = n_always_ff
    f["is_comb_only"]     = int(n_always_ff == 0)
    f["has_multiplier"]   = int(n_mul > 0)
    f["has_division"]     = int(n_div > 0)
    f["has_memory"]       = int(mem_access > 0)
    f["has_generate"]     = int(base["generate_count"] > 0)

    # Arithmetic intensity (ops per line)
    f["arith_intensity"]  = total_arith / n_lines
    f["mul_intensity"]    = n_mul / n_lines

    # ── SIGNAL FANOUT PROXY ───────────────────────────────────
    ids  = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", rtl)
    freq = Counter(ids)
    vals = list(freq.values())
    f["unique_signals"]          = len(freq)
    f["max_signal_occurrences"]  = max(vals) if vals else 0
    f["mean_signal_occurrences"] = float(np.mean(vals)) if vals else 0.0
    f["num_high_fanout_signals"] = sum(1 for v in vals if v > 10)

    return f


# ══════════════════════════════════════════════════════════════
#  RPT PARSING  (area + other PPA targets from synthesis reports)
# ══════════════════════════════════════════════════════════════

RPT_AREA_PATTERNS = {
    "total_cell_area": [
        r"Total cell area\s*[:=]\s*([\d,\.]+)",
        r"Cell Area\s*[:=]\s*([\d,\.]+)",
        r"Combinational area\s*[:=]\s*([\d,\.]+)",
    ],
    "comb_area": [
        r"Combinational area\s*[:=]\s*([\d,\.]+)",
    ],
    "num_cells": [
        r"Leaf Cell Count\s*[:=]\s*([\d,]+)",
        r"Number of cells\s*[:=]?\s*([\d,]+)",
    ],
    "num_nets": [
        r"Total Number of Nets\s*[:=]\s*([\d,]+)",
        r"Number of nets\s*[:=]?\s*([\d,]+)",
    ],
    "Power": [
        r"Total Dynamic Power\s*=\s*([\d\.]+)\s*([munpMUNP]?[wW])",
    ],
    "critical_path_length": [
        r"Critical Path Length\s*[:=]\s*([\d\.]+)",
    ],
    "levels_of_logic": [
        r"Levels of Logic\s*[:=]\s*([\d\.]+)",
    ],
}

UNIT_MULT = {"pw":1e-9,"nw":1e-6,"uw":1e-3,"mw":1.0,"w":1000.0}


def parse_rpt(rpt_text: str) -> dict:
    out  = {}
    txt  = rpt_text.replace(",", "")
    # max data arrival time = critical path
    arrivals = re.findall(r"data arrival time\s+([\d\.]+)", txt, re.IGNORECASE)
    out["critical_path_length"] = float(max(float(v) for v in arrivals)) \
                                   if arrivals else None
    for key, pats in RPT_AREA_PATTERNS.items():
        found = None
        for p in pats:
            m = re.search(p, txt, re.IGNORECASE)
            if not m:
                continue
            try:
                val  = float(m.group(1))
                unit = (m.group(2).lower().strip()
                        if m.lastindex and m.lastindex >= 2 and m.group(2) else "")
                if key == "Power":
                    found = val * UNIT_MULT.get(unit, 1.0)
                elif key in ("num_cells", "num_nets"):
                    found = int(val)
                else:
                    found = float(val)
                break
            except Exception:
                continue
        out[key] = found
    return out


# ══════════════════════════════════════════════════════════════
#  PER-DESIGN EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_design(design_dir: Path) -> dict | None:
    vfiles = sorted(design_dir.glob("*.v"))
    if not vfiles:
        return None

    # prefer file named same as folder
    preferred = [v for v in vfiles if v.stem == design_dir.name]
    vpath = preferred[0] if preferred else vfiles[0]

    rtl_raw = read_safe(vpath)
    rtl     = clean_verilog(rtl_raw)

    bw_feats  = extract_bitwidths(rtl)
    op_feats  = count_operators(rtl)
    st_feats  = extract_structural(rtl)

    base = {}
    base.update(bw_feats)
    base.update(op_feats)
    base.update(st_feats)

    adv_feats = extract_area_advanced(rtl, base, bw_feats)

    # parse .rpt files
    rpt_files  = sorted(design_dir.glob("*.rpt"))
    rpt_concat = "\n\n".join(read_safe(p) for p in rpt_files)
    rpt_feats  = parse_rpt(rpt_concat) if rpt_concat.strip() else {}

    row = {"Design_Name": design_dir.name}
    row.update(bw_feats)
    row.update(op_feats)
    row.update(st_feats)
    row.update(adv_feats)
    row["rpt_files"]    = ";".join(p.name for p in rpt_files)
    row["rpt_text_len"] = len(rpt_concat)
    row.update(rpt_feats)
    return row


# ══════════════════════════════════════════════════════════════
#  BATCH BUILDER
# ══════════════════════════════════════════════════════════════

def build_dataset(dataset_root: str, out_csv: str) -> pd.DataFrame:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"{dataset_root} not found")

    dirs    = sorted([p for p in root.iterdir() if p.is_dir()])
    total   = len(dirs)
    rows    = []
    errors  = []
    t0      = time.time()

    print(f"[scan] {total} design folders in {dataset_root}")

    for i, d in enumerate(dirs, 1):
        try:
            row = extract_design(d)
            if row:
                rows.append(row)
        except Exception as e:
            errors.append({"design": d.name, "error": str(e)})

        if i % 100 == 0 or i == total:
            elapsed = time.time() - t0
            eta     = (elapsed / i) * (total - i)
            pct     = i / total * 100
            bar     = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  [{bar}] {i}/{total}  {pct:5.1f}%  "
                  f"elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s", end="\r")

    print()

    if errors:
        err_path = str(Path(out_csv).with_suffix("")) + "_errors.csv"
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"[warn] {len(errors)} failed → {err_path}")

    df = pd.DataFrame(rows)

    # safe save
    try:
        df.to_csv(out_csv, index=False)
        print(f"[done] {len(df)} rows × {len(df.columns)} cols → {out_csv}")
    except PermissionError:
        alt = str(Path(out_csv).with_suffix("")) + f".{os.getpid()}.csv"
        df.to_csv(alt, index=False)
        print(f"[warn] Permission error — saved to {alt}")

    return df


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract area-specific RTL features from Verilog files")
    parser.add_argument("--dataset_root", required=True,
                        help="Root folder with per-design subfolders")
    parser.add_argument("--out_csv", required=True,
                        help="Output CSV path")
    args = parser.parse_args()
    build_dataset(args.dataset_root, args.out_csv)


if __name__ == "__main__":
    main()