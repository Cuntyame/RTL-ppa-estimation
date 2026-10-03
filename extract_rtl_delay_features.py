#!/usr/bin/env python3
r"""
extract_rtl_delay_features.py
==============================
Dedicated RTL feature extractor for DELAY (critical path) prediction.

USAGE:
  python extract_rtl_delay_features.py \
      --dataset_root "C:\ml ppa\Final_Clean_Dataset" \
      --out_csv "C:\Users\Admin\Documents\rtl_delay_features.csv"

WHY DELAY NEEDS A SEPARATE EXTRACTOR FROM AREA/POWER:
  Power features: switching activity (XOR density, FF bits, clock enables)
  Area features:  structural cell count (mul×N², DFF×8GE, decoder)
  Delay features: CRITICAL PATH DEPTH — how many gate delays in series

  Critical path delay = Σ(gate_delays along worst combinational path)
  From RTL this means:
    1. What slow operations exist? (mul, div >> add >> logic)
    2. How wide are they? (32-bit mul chain >> 8-bit)
    3. Are there pipeline registers? (registers BREAK the path → reduce delay)
    4. How deep is the combinational logic? (nested if/case → mux chains)
    5. What is the fanout? (high fanout → buffer insertion → added delay)

DELAY PHYSICS (sky130 standard cell library, typical corner):
  Gate type          Delay    Scales with
  ───────────────────────────────────────────────────────────
  Inverter (INV)     0.1 ns   1 stage
  AND2/OR2           0.15 ns  1 stage
  XOR2               0.2 ns   2 stages (A⊕B = NOT(A XNOR B))
  Full adder bit     0.3 ns   ripple: N bits → N×0.3 ns
  Carry-lookahead    0.15 ns  log4(N) stages
  Multiplier bit     0.4 ns   N stages → N×0.4 ns (Booth = N/2)
  Divider bit        0.5 ns   N² worst case
  MUX2               0.15 ns  1 stage
  DFF setup+clk-Q    0.5 ns   pipeline register (RESETS counter)
  MAJ3               0.3 ns   used in adder carry chains

  Critical path = longest chain of these delays.
  A 32-bit ripple adder: 32×0.3 = 9.6 ns
  A 32-bit multiplier:   32×0.4 = 12.8 ns
  After pipelining:      one register every N stages → delay/N

NEW DELAY-SPECIFIC FEATURES:
  slow_ops_weighted     : 3×div + 2×mul + 1×add  (delay weights)
  ripple_carry_depth    : num_add × max_bw × 0.3 (ns proxy)
  mul_delay_proxy       : num_mul × max_bw × 0.4
  div_delay_proxy       : num_div × max_bw² × 0.5 (N² for div)
  pipeline_stages       : num_always_ff (each breaks critical path)
  pipeline_reduction    : delay ÷ (pipeline_stages + 1)
  logic_depth_proxy     : nested if/case depth (mux chain length)
  fanout_delay          : high-fanout signals need buffers → added delay
  combinational_depth   : operations between register stages
  is_purely_comb        : no registers → one long path
  maj3_chain_proxy      : MAJ3 gates form carry chains in adders
  carry_chain_depth     : estimated ripple carry length
  critical_op_mix       : fraction of ops that are delay-critical
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

# Sky130 gate delay estimates (ns, typical corner)
DELAY_INV   = 0.10
DELAY_AND2  = 0.15
DELAY_XOR2  = 0.20
DELAY_FA    = 0.30   # full adder per bit (ripple carry)
DELAY_CLA   = 0.15   # carry-lookahead per stage
DELAY_MUL   = 0.40   # multiplier per bit (Booth encoded)
DELAY_DIV   = 0.50   # divider per stage (restoring)
DELAY_MUX2  = 0.15
DELAY_DFF   = 0.50   # setup + clk-to-Q (pipeline register)
DELAY_MAJ3  = 0.30


# ══════════════════════════════════════════════════════════════
#  FILE I/O
# ══════════════════════════════════════════════════════════════

def read_safe(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def clean_verilog(code: str) -> str:
    code = re.sub(r"//.*?$",            "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/",         "", code, flags=re.DOTALL)
    code = re.sub(r'"(?:\\.|[^"\\])*"', "", code, flags=re.DOTALL)
    return code


# ══════════════════════════════════════════════════════════════
#  CANONICAL RTL FEATURES  (same 31 as before)
# ══════════════════════════════════════════════════════════════

BITRANGE_RE = re.compile(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]")


def extract_bitwidths(rtl: str) -> dict:
    matches = BITRANGE_RE.findall(rtl)
    if not matches:
        return dict(max_bw=1, min_bw=1, avg_bw=1.0, total_bits=1,
                    num_bw_decls=0, n32bit=0, n16bit=0, n8bit=0, n1bit=0)
    bw = [abs(int(a) - int(b)) + 1 for a, b in matches]
    return dict(
        max_bw=int(max(bw)), min_bw=int(min(bw)),
        avg_bw=float(np.mean(bw)), total_bits=int(sum(bw)),
        num_bw_decls=len(bw),
        n32bit=bw.count(32), n16bit=bw.count(16),
        n8bit=bw.count(8),   n1bit=bw.count(1),
    )


def count_operators(rtl: str) -> dict:
    return dict(
        num_add=rtl.count("+"), num_sub=rtl.count("-"),
        num_mul=rtl.count("*"), num_div=rtl.count("/"),
        num_logic_and=rtl.count("&"), num_logic_or=rtl.count("|"),
        num_logic_xor=rtl.count("^"), num_shifts=rtl.count("<<")+rtl.count(">>"),
        num_mod=rtl.count("%"),
        num_comparisons=len(re.findall(
            r"==|!=|<=|>=|(?<![<>=])<(?![<>=])|(?<![<>=])>(?![<>=])", rtl)),
        num_ternary=rtl.count("?"),
    )


def extract_structural(rtl: str) -> dict:
    f = dict(
        num_lines=rtl.count("\n")+1, rtl_code_len=len(rtl),
        num_always=len(re.findall(r"\balways\b",  rtl)),
        num_assign=len(re.findall(r"\bassign\b",  rtl)),
        num_if=len(re.findall(r"\bif\b",          rtl)),
        num_case=len(re.findall(r"\bcase\b",       rtl)),
        num_for=len(re.findall(r"\bfor\b",         rtl)),
        num_wire=len(re.findall(r"\bwire\b",       rtl)),
        num_reg=len(re.findall(r"\breg\b",         rtl)),
        num_input=len(re.findall(r"\binput\b",     rtl)),
        num_output=len(re.findall(r"\boutput\b",   rtl)),
        num_modules=max(0,
            len(re.findall(r"\bmodule\b",    rtl)) -
            len(re.findall(r"\bendmodule\b", rtl))),
        num_parameter=len(re.findall(r"\bparameter\b",  rtl)),
        nb_assign=len(re.findall(r"<=(?!=)", rtl)),
        bl_assign=len(re.findall(r"=(?!>|=)", rtl)),
        begin_count=len(re.findall(r"\bbegin\b", rtl)),
    )
    f["num_branches"] = f["num_if"] + f["num_case"] + f["num_ternary"] \
                        if "num_ternary" in f else f["num_if"] + f["num_case"]
    return f


# ══════════════════════════════════════════════════════════════
#  DELAY-SPECIFIC FEATURES
# ══════════════════════════════════════════════════════════════

def extract_delay_features(rtl: str, base: dict, bw: dict) -> dict:
    """
    All features that directly relate to critical path delay.
    """
    f = {}
    avg_bw  = max(bw["avg_bw"],  1.0)
    max_bw  = max(bw["max_bw"],  1.0)
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
    n_if    = base["num_if"]
    n_case  = base["num_case"]
    n_for   = base["num_for"]
    n_always= base["num_always"]

    # ── 1. PIPELINE STAGES  (registers BREAK the critical path) ──
    # This is the most important delay feature — pipelining reduces
    # critical path by inserting registers between logic stages.
    n_always_ff = len(re.findall(
        r"always\s*@\s*\(\s*(?:posedge|negedge)", rtl, re.IGNORECASE))
    f["num_always_ff"]      = n_always_ff
    f["num_always_comb"]    = max(0, n_always - n_always_ff)
    f["is_pipelined"]       = int(n_always_ff > 1)
    f["is_purely_comb"]     = int(n_always_ff == 0)
    f["pipeline_stages"]    = n_always_ff  # more stages = shorter critical path

    # clock domain count (CDC synchronisers add delay)
    clk_sigs = set(re.findall(
        r"(?:posedge|negedge)\s+([a-zA-Z_]\w*)", rtl, re.IGNORECASE))
    f["num_clk_domains"] = len(clk_sigs) if clk_sigs else 0

    # ── 2. SLOW OPERATION DELAY ESTIMATES  (ns proxies) ──────────
    # These directly estimate how much delay each operation contributes
    # to the critical path.

    # Ripple-carry adder: N bits × 0.3 ns/bit
    f["adder_delay_proxy"]  = (n_add + n_sub) * max_bw * DELAY_FA
    f["log_adder_delay"]    = math.log1p(f["adder_delay_proxy"])

    # Multiplier: Booth-encoded N bits × 0.4 ns/bit (dominant delay source)
    f["mul_delay_proxy"]    = n_mul * max_bw * DELAY_MUL
    f["log_mul_delay"]      = math.log1p(f["mul_delay_proxy"])

    # Divider: N² worst case (rarely pipelined)
    f["div_delay_proxy"]    = n_div * max_bw * max_bw * DELAY_DIV
    f["log_div_delay"]      = math.log1p(f["div_delay_proxy"])

    # XOR chain: used in adders, parity, CRC — each adds 0.2 ns
    f["xor_delay_proxy"]    = n_xor * avg_bw * DELAY_XOR2
    f["log_xor_delay"]      = math.log1p(f["xor_delay_proxy"])

    # Barrel shifter: log2(N) MUX stages × 0.15 ns
    f["shift_delay_proxy"]  = n_shift * math.log2(max(avg_bw, 2)) * DELAY_MUX2
    f["log_shift_delay"]    = math.log1p(f["shift_delay_proxy"])

    # Comparator: uses adder or XOR chain
    f["comp_delay_proxy"]   = n_comp * avg_bw * DELAY_FA
    f["log_comp_delay"]     = math.log1p(f["comp_delay_proxy"])

    # ── 3. CRITICAL PATH DEPTH ESTIMATE  (key predictor) ─────────
    # Weighted sum of all delay contributions
    f["critical_path_estimate"] = (
        f["mul_delay_proxy"]   +
        f["adder_delay_proxy"] +
        f["div_delay_proxy"]   +
        f["xor_delay_proxy"]   +
        f["shift_delay_proxy"] +
        n_mux   * avg_bw * DELAY_MUX2 +
        n_and   * avg_bw * DELAY_AND2  +
        n_or    * avg_bw * DELAY_AND2
    )
    f["log_cp_estimate"]    = math.log1p(f["critical_path_estimate"])
    f["sqrt_cp_estimate"]   = math.sqrt(max(f["critical_path_estimate"], 0))

    # ── 4. PIPELINING REDUCTION FACTOR ────────────────────────────
    # In a pipelined design, delay ≈ total_logic_delay / num_stages
    # This is the most physically correct delay predictor from RTL.
    stages = max(n_always_ff, 1)
    f["pipeline_reduced_delay"] = f["critical_path_estimate"] / stages
    f["log_pipeline_reduced"]   = math.log1p(f["pipeline_reduced_delay"])

    # ── 5. SLOW OPS WEIGHTED COUNT  (from original project model) ─
    # This was in the original PPA code: slow_ops = 2×mul + 3×div
    # Validated as a strong delay predictor in the literature.
    f["slow_ops"]           = 2 * n_mul + 3 * n_div
    f["slow_ops_weighted"]  = (3 * n_div + 2 * n_mul + 1 * (n_add + n_sub) +
                               0.5 * n_xor + 0.3 * n_comp)
    f["slow_ops_x_bw"]      = f["slow_ops_weighted"] * max_bw
    f["log_slow_ops_x_bw"]  = math.log1p(f["slow_ops_x_bw"])

    # ── 6. CARRY CHAIN DEPTH  (ripple carry adder) ────────────────
    # A 32-bit ripple-carry adder has a 32-bit carry chain.
    # This is often the critical path for ALU-heavy designs.
    f["carry_chain_depth"]  = (n_add + n_sub) * max_bw
    f["log_carry_chain"]    = math.log1p(f["carry_chain_depth"])
    f["has_long_carry"]     = int((n_add + n_sub) * max_bw > 32)

    # ── 7. MUX / CONTROL LOGIC DEPTH  (multiplexer chains) ───────
    # Deeply nested if/case → long MUX chains on the critical path
    # Estimated nesting depth from begin/end count
    nesting_depth = base.get("begin_count", 0) / max(n_if + n_case, 1)
    f["mux_chain_depth"]    = n_mux * avg_bw * nesting_depth
    f["log_mux_chain"]      = math.log1p(f["mux_chain_depth"])
    f["nesting_depth"]      = float(nesting_depth)

    # control_depth: how many levels of if/case (approximated by else branches)
    n_else = len(re.findall(r"\belse\b", rtl))
    f["control_depth"]      = n_else / max(n_if, 1)  # avg else-chain depth

    # ── 8. FANOUT DELAY PROXY  (high fanout → buffer insertion) ───
    ids  = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", rtl)
    freq = Counter(ids)
    vals = list(freq.values())
    f["unique_signals"]          = len(freq)
    f["max_signal_occurrences"]  = max(vals) if vals else 0
    f["mean_signal_occurrences"] = float(np.mean(vals)) if vals else 0.0
    f["num_high_fanout"]         = sum(1 for v in vals if v > 10)

    # High fanout = more buffer stages = added delay
    f["fanout_delay_proxy"]      = f["num_high_fanout"] * DELAY_INV * 2
    f["signal_fanout_pressure"]  = f["max_signal_occurrences"] * DELAY_INV

    # ── 9. COMBINATIONAL DEPTH  (ops between registers) ───────────
    # Between any two registers, the total combinational logic
    # determines the critical path in that stage.
    total_comb_ops = n_add + n_sub + n_mul + n_div + n_xor + n_and + n_or + n_comp
    f["comb_ops_total"]     = total_comb_ops
    f["comb_ops_per_stage"] = total_comb_ops / stages
    f["log_comb_per_stage"] = math.log1p(f["comb_ops_per_stage"])

    # ── 10. DESIGN TYPE INDICATORS ────────────────────────────────
    f["has_multiplier"]     = int(n_mul > 0)
    f["has_division"]       = int(n_div > 0)
    f["has_accumulator"]    = int(bool(re.search(
        r"(\w+)\s*<=\s*\1\s*\+", rtl)))   # a <= a + x pattern
    f["has_priority_enc"]   = int(bool(re.search(
        r"casez|casex|\bpriority\b", rtl, re.IGNORECASE)))
    f["has_barrel_shift"]   = int(n_shift > 0 and bool(re.search(
        r"<<\s*[a-zA-Z_]\w*|>>\s*[a-zA-Z_]\w*", rtl)))

    # ── 11. BITWIDTH-DELAY INTERACTION ────────────────────────────
    # Wider operations → longer critical paths
    f["max_bw_x_slow_ops"]  = max_bw * f["slow_ops_weighted"]
    f["log_bw_x_slow"]      = math.log1p(f["max_bw_x_slow_ops"])
    f["bw_range"]           = max_bw - bw["min_bw"]

    # ── 12. MEMORY ACCESS DELAY ───────────────────────────────────
    # Synchronous reads through a MUX → adds a MUX stage
    mem_access = len(re.findall(
        r"\w+\s*\[\s*\w+\s*\]\s*(?:<=|=)|\b(?:<=|=)\s*\w+\s*\[\s*\w+\s*\]", rtl))
    f["num_memory_access"]  = mem_access
    f["mem_delay_proxy"]    = mem_access * DELAY_MUX2 * avg_bw

    # ── 13. CRITICAL OP MIX (fraction of ops that are delay-critical) ──
    total_ops = max(total_comb_ops, 1)
    f["critical_op_fraction"] = (n_mul + n_div) / total_ops
    f["high_delay_op_frac"]   = (n_mul * 3 + n_div * 4 +
                                  (n_add + n_sub) * 1) / max(total_ops * 4, 1)

    # ── 14. MAJ3 PROXY  (carry-save adder chains) ─────────────────
    # MAJ3 gates form carry-save structures in fast adder trees.
    # High MAJ3 count = Wallace tree multiplier (faster but complex)
    maj3_count = len(re.findall(r"\bmaj3\b|\bmajority\b", rtl, re.IGNORECASE))
    f["maj3_count"]         = maj3_count
    f["maj3_chain_proxy"]   = maj3_count * DELAY_MAJ3
    f["log_maj3_chain"]     = math.log1p(f["maj3_chain_proxy"])

    # ── 15. LOOP UNROLLING DELAY IMPACT ───────────────────────────
    # Unrolled loops put all iterations on the critical path
    loop_bounds = re.findall(r"for\s*\([^;]+;\s*\w+\s*<\s*(\d+)", rtl)
    unroll_factor = sum(int(b) for b in loop_bounds if int(b) < 1000) or n_for
    f["unroll_delay_factor"] = unroll_factor
    f["log_unroll_delay"]    = math.log1p(unroll_factor * avg_bw * DELAY_AND2)

    return f


# ══════════════════════════════════════════════════════════════
#  RPT PARSING
# ══════════════════════════════════════════════════════════════

RPT_DELAY_PATTERNS = {
    "critical_path_length": [
        r"Critical Path Length\s*[:=]\s*([\d\.]+)",
    ],
    "levels_of_logic": [
        r"Levels of Logic\s*[:=]\s*([\d\.]+)",
    ],
    "wns": [r"\bWNS\s*[:=]\s*([-\d\.]+)"],
    "tns": [r"\bTNS\s*[:=]\s*([-\d\.]+)"],
    "total_cell_area": [
        r"Total cell area\s*[:=]\s*([\d,\.]+)",
        r"Combinational area\s*[:=]\s*([\d,\.]+)",
    ],
    "Power": [
        r"Total Dynamic Power\s*=\s*([\d\.]+)\s*([munpMUNP]?[wW])",
    ],
    "num_cells": [r"Leaf Cell Count\s*[:=]\s*([\d,]+)"],
}

UNIT_MULT = {"pw":1e-9,"nw":1e-6,"uw":1e-3,"mw":1.0,"w":1000.0}


def parse_rpt(rpt_text: str) -> dict:
    out = {}
    txt = rpt_text.replace(",", "")

    # Critical path = MAX data arrival time across all paths
    arrivals = re.findall(r"data arrival time\s+([\d\.]+)", txt, re.IGNORECASE)
    out["critical_path_length"] = (
        float(max(float(v) for v in arrivals)) if arrivals else None)

    for key, pats in RPT_DELAY_PATTERNS.items():
        if key == "critical_path_length":
            continue
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
                elif key in ("num_cells",):
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

    preferred = [v for v in vfiles if v.stem == design_dir.name]
    vpath = preferred[0] if preferred else vfiles[0]

    rtl_raw = read_safe(vpath)
    rtl     = clean_verilog(rtl_raw)

    bw_f  = extract_bitwidths(rtl)
    op_f  = count_operators(rtl)
    st_f  = extract_structural(rtl)

    base = {}
    base.update(bw_f); base.update(op_f); base.update(st_f)

    delay_f = extract_delay_features(rtl, base, bw_f)

    rpt_files  = sorted(design_dir.glob("*.rpt"))
    rpt_concat = "\n\n".join(read_safe(p) for p in rpt_files)
    rpt_f      = parse_rpt(rpt_concat) if rpt_concat.strip() else {}

    row = {"Design_Name": design_dir.name}
    row.update(bw_f); row.update(op_f); row.update(st_f)
    row.update(delay_f)
    row["rpt_files"]    = ";".join(p.name for p in rpt_files)
    row["rpt_text_len"] = len(rpt_concat)
    row.update(rpt_f)
    return row


#  BATCH BUILDER
# # ══════════════════════════════════════════════════════════════

# def build_dataset(dataset_root: str, out_csv: str) -> pd.DataFrame:
#     root  = Path(dataset_root)
#     dirs  = sorted([p for p in root.iterdir() if p.is_dir()])
#     total = len(dirs)
#     rows, errors = [], []
#     t0 = time.time()

#     print(f"[scan] {total} design folders in {dataset_root}")
#     for i, d in enumerate(dirs, 1):
#         try:
#             row = extract_design(d)
#             if row:
#                 rows.append(row)
#         except Exception as e:
#             errors.append({"design": d.name, "error": str(e)})

#         i
# ════════════════════════════════════════════════════
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
        description="Extract delay-specific RTL features from Verilog files")
    parser.add_argument("--dataset_root", required=True,
                        help="Root folder with per-design subfolders")
    parser.add_argument("--out_csv", required=True,
                        help="Output CSV path")
    args = parser.parse_args()
    build_dataset(args.dataset_root, args.out_csv)


if __name__ == "__main__":
    main()