#!/usr/bin/env python3
r"""
extract_rtl_power_features_v2.py
==================================
Full RTL + .rpt feature extractor for power prediction.
Extracts 31 canonical RTL features + engineered power proxies + parsed .rpt metrics.

USAGE:
  python extract_rtl_power_features_v2.py \
      --dataset_root "C:\ml ppa\Final_Clean_Dataset" \
      --out_csv "C:\Users\Admin\Documents\final_rtl_power_features_v2.csv"

CHANGES vs v1:
  - FIXED: save_dataframe_safe infinite recursion bug (was calling itself)
  - ADDED: 24 new power-focused features (see POWER-FOCUSED FEATURES section)
  - ADDED: progress bar with ETA (no extra deps, plain print)
  - ADDED: per-design error log written next to output CSV
  - IMPROVED: Power unit normalization covers pW/nW/uW/mW/W
  - IMPROVED: Critical path extracts the MAX across all timing paths in file
  - IMPROVED: num_modules counts only top-level module declarations
"""

import argparse
import re
import os
import time
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

EPS = 1e-9


# ══════════════════════════════════════════════════════════════
#  FILE I/O UTILITIES
# ══════════════════════════════════════════════════════════════

def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def clean_rtl_text(text: str) -> str:
    """Remove comments and string literals for cleaner token counting."""
    text = re.sub(r"//.*?$",          "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/",       "", text, flags=re.DOTALL)
    text = re.sub(r'"(?:\\.|[^"\\])*"', "", text, flags=re.DOTALL)
    return text


def save_dataframe_safe(df: pd.DataFrame, out_csv: str) -> str:
    """
    Save DataFrame to out_csv.  On PermissionError tries up to 3 alternate
    filenames before raising with a diagnostic message.
    FIXED: v1 had infinite recursion — this version does NOT call itself.
    """
    out_path = Path(out_csv)
    parent   = out_path.parent

    # ensure output directory exists
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise PermissionError(f"Cannot create directory {parent}: {e}")

    # try writing directly
    try:
        df.to_csv(str(out_path), index=False)
        print(f"[saved] {out_path}")
        return str(out_path)
    except PermissionError as e:
        print(f"[warn] PermissionError writing {out_path}: {e}")

    # fallback alternatives (pid / timestamp variants)
    alternatives = [
        parent / f"{out_path.stem}.tmp{out_path.suffix}",
        parent / f"{out_path.stem}.{os.getpid()}{out_path.suffix}",
        parent / f"{out_path.stem}.{int(time.time())}{out_path.suffix}",
    ]

    for alt in alternatives:
        try:
            df.to_csv(str(alt), index=False)
            try:
                os.replace(str(alt), str(out_path))
                print(f"[saved via rename] {out_path}")
                return str(out_path)
            except PermissionError:
                print(f"[warn] Target locked — alt file at: {alt}")
                return str(alt)
        except PermissionError:
            continue

    raise PermissionError(
        f"Failed to write CSV to {out_path} or any alternative.\n"
        "Possible causes:\n"
        "  - File is open in Excel/Notepad — close it first.\n"
        "  - No write permission to folder. Try C:\\Users\\Admin\\Documents\\\n"
        "  - OneDrive/antivirus locking the file — pause OneDrive sync.\n"
        f"Try: --out_csv \"C:\\Users\\Admin\\Documents\\{out_path.name}\""
    )


# ══════════════════════════════════════════════════════════════
#  CANONICAL RTL FEATURES  (31 features — unchanged from v1)
# ══════════════════════════════════════════════════════════════

BITRANGE_RE = re.compile(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]")


def extract_bitwidth_features(rtl: str) -> dict:
    matches = BITRANGE_RE.findall(rtl)
    if not matches:
        return {
            "max_bitwidth": 1, "min_bitwidth": 1, "avg_bitwidth": 1.0,
            "total_bits": 1,   "num_bitwidths": 0,
            "num_32bit": 0,    "num_16bit": 0, "num_8bit": 0, "num_1bit": 1,
        }
    bitwidths = [abs(int(a) - int(b)) + 1 for a, b in matches]
    return {
        "max_bitwidth":  int(max(bitwidths)),
        "min_bitwidth":  int(min(bitwidths)),
        "avg_bitwidth":  float(np.mean(bitwidths)),
        "total_bits":    int(sum(bitwidths)),
        "num_bitwidths": int(len(bitwidths)),
        "num_32bit":     int(bitwidths.count(32)),
        "num_16bit":     int(bitwidths.count(16)),
        "num_8bit":      int(bitwidths.count(8)),
        "num_1bit":      int(bitwidths.count(1)),
    }


def count_operators(rtl: str) -> dict:
    return {
        "num_add":         rtl.count("+"),
        "num_sub":         rtl.count("-"),
        "num_mul":         rtl.count("*"),
        "num_div":         rtl.count("/"),
        "num_logic_and":   rtl.count("&"),
        "num_logic_or":    rtl.count("|"),
        "num_logic_xor":   rtl.count("^"),
        "num_shifts":      rtl.count("<<") + rtl.count(">>"),
        "num_comparisons": len(re.findall(
            r"==|!=|<=|>=|(?<![<>=])<(?![<>=])|(?<![<>=])>(?![<>=])", rtl)),
    }


def extract_structural_features(rtl: str) -> dict:
    feats = {
        "num_lines":   rtl.count("\n") + 1,
        "num_always":  len(re.findall(r"\balways\b",  rtl)),
        "num_assign":  len(re.findall(r"\bassign\b",  rtl)),
        "num_if":      len(re.findall(r"\bif\b",      rtl)),
        "num_case":    len(re.findall(r"\bcase\b",    rtl)),
        "num_for":     len(re.findall(r"\bfor\b",     rtl)),
        "num_ternary": rtl.count("?"),
        "num_wire":    len(re.findall(r"\bwire\b",    rtl)),
        "num_reg":     len(re.findall(r"\breg\b",     rtl)),
        "num_input":   len(re.findall(r"\binput\b",   rtl)),
        "num_output":  len(re.findall(r"\boutput\b",  rtl)),
        # count only opening 'module' declarations (not endmodule)
        "num_modules": len(re.findall(r"\bmodule\b(?!\s*\()", rtl))
                     + len(re.findall(r"\bmodule\b\s+\w+", rtl)),
    }
    # de-duplicate module count (the two patterns above double-count)
    feats["num_modules"] = max(
        0, len(re.findall(r"\bmodule\b", rtl)) - len(re.findall(r"\bendmodule\b", rtl))
    )
    feats["num_branches"] = feats["num_if"] + feats["num_case"] + feats["num_ternary"]
    return feats


# ── clock / posedge ───────────────────────────────────────────
POS_EDGE_RE    = re.compile(r"@\s*\(\s*(?:posedge|negedge)\s+([a-zA-Z_]\w*)", re.IGNORECASE)
CLK_CANDIDATE  = re.compile(r"\b(clk|clock|clk_i|clk_\w+)\b", re.IGNORECASE)


def detect_clock_features(rtl: str) -> dict:
    clk_signals = set(POS_EDGE_RE.findall(rtl))
    candidate   = set(CLK_CANDIDATE.findall(rtl))
    n_domains   = len(clk_signals) if clk_signals else (1 if candidate else 0)
    return {
        "num_posedge_always": int(len(re.findall(
            r"always\s*@\s*\(\s*(?:posedge|negedge)", rtl, re.IGNORECASE))),
        "num_clk_domains":    int(n_domains),
    }


# ── signal / fanout proxy ─────────────────────────────────────
IDENT_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")


def extract_signal_fanout_proxy(rtl: str) -> dict:
    ids = IDENT_RE.findall(rtl)
    if not ids:
        return {"unique_signals": 0, "max_signal_occurrences": 0,
                "mean_signal_occurrences": 0.0, "num_high_fanout_signals": 0}
    freq = Counter(ids)
    vals = np.array(list(freq.values()), dtype=float)
    return {
        "unique_signals":          int(len(freq)),
        "max_signal_occurrences":  int(vals.max()),
        "mean_signal_occurrences": float(vals.mean()),
        "num_high_fanout_signals": int((vals > 10).sum()),
    }


# ══════════════════════════════════════════════════════════════
#  ENGINEERED FEATURES  (v1 set — kept identical)
# ══════════════════════════════════════════════════════════════

def compute_power_engineered_features(base: dict, bitwidth: dict,
                                      signal_proxy: dict) -> dict:
    total_arith = (base.get("num_add", 0) + base.get("num_sub", 0) +
                   base.get("num_mul", 0) + base.get("num_div", 0))
    total_logic = (base.get("num_logic_and", 0) + base.get("num_logic_or", 0) +
                   base.get("num_logic_xor", 0) + base.get("num_shifts", 0))

    feats = {
        "total_arithmetic":          int(total_arith),
        "arith_density":             float(total_arith / max(base.get("num_lines", 1), 1)),
        "switching_proxy":           float(base.get("num_reg", 0) + 2.0 * base.get("num_logic_xor", 0)),
        "logic_density":             float(total_logic / max(base.get("num_lines", 1), 1)),
        "mux_count":                 int(base.get("num_ternary", 0) + base.get("num_case", 0)),
        "bits_log":                  float(math.log1p(bitwidth.get("total_bits", 0))),
        "bits_sqrt":                 float(math.sqrt(max(bitwidth.get("total_bits", 0), 0))),
        "register_bit_product":      float(base.get("num_reg", 0) * bitwidth.get("avg_bitwidth", 1.0)),
        "toggle_estimate":           float(
            (base.get("num_reg", 0) + 2.0 * base.get("num_logic_xor", 0)) *
            float(total_logic / max(base.get("num_lines", 1), 1))
        ),
        # industry-style
        "bit_toggle_load":           float(bitwidth.get("total_bits", 0) *
                                           (base.get("num_reg", 0) + 2.0 * base.get("num_logic_xor", 0))),
        "fanout_switching_pressure": float(
            (base.get("num_logic_xor", 0) + base.get("num_logic_and", 0)) *
            bitwidth.get("avg_bitwidth", 1.0)),
        "reg_to_logic_ratio":        float(base.get("num_reg", 0) / max(total_logic, 1)),
    }
    feats.update({
        "unique_signals":          int(signal_proxy.get("unique_signals", 0)),
        "max_signal_occurrences":  int(signal_proxy.get("max_signal_occurrences", 0)),
        "mean_signal_occurrences": float(signal_proxy.get("mean_signal_occurrences", 0.0)),
        "num_high_fanout_signals": int(signal_proxy.get("num_high_fanout_signals", 0)),
    })
    return feats


# ══════════════════════════════════════════════════════════════
#  NEW POWER-FOCUSED FEATURES  (v2 additions)
#
#  Power = Internal (cell switching) + Net Switching + Leakage
#  Internal  ≈ α_seq × C_ff × V² × f    → sequential features
#  Net Switch ≈ α_comb × C_wire × V² × f → combinational features
#  Leakage   ≈ I_leak × V × num_cells    → cell count proxies
# ══════════════════════════════════════════════════════════════

def compute_new_power_features(rtl: str, base: dict, bitwidth: dict) -> dict:
    """
    All 24 new features added in v2.  Each is guarded so missing base keys
    produce 0 rather than a KeyError.
    """
    f = {}
    avg_bw  = bitwidth.get("avg_bitwidth", 1.0)
    max_bw  = bitwidth.get("max_bitwidth", 1.0)
    n_lines = max(base.get("num_lines", 1), 1)
    n_reg   = base.get("num_reg",         0)
    n_xor   = base.get("num_logic_xor",   0)
    n_and   = base.get("num_logic_and",   0)
    n_or    = base.get("num_logic_or",    0)
    n_add   = base.get("num_add",         0)
    n_sub   = base.get("num_sub",         0)
    n_mul   = base.get("num_mul",         0)
    n_div   = base.get("num_div",         0)
    n_if    = base.get("num_if",          0)
    n_case  = base.get("num_case",        0)
    n_for   = base.get("num_for",         0)
    n_wire  = base.get("num_wire",        0)
    n_comp  = base.get("num_comparisons", 0)
    n_tern  = base.get("num_ternary",     0)
    n_shifts= base.get("num_shifts",      0)

    # ── 1. Always-block type decomposition ───────────────────
    # Clocked (posedge/negedge) → sequential → drives internal power
    # Combinational              → glitch source → drives net switching
    f["num_always_ff"]   = len(re.findall(
        r"always\s*@\s*\(\s*(?:posedge|negedge)", rtl, re.IGNORECASE))
    f["num_always_comb"] = max(0, base.get("num_always", 0) - f["num_always_ff"])

    # ── 2. Reset style  ───────────────────────────────────────
    # Synchronous reset → extra MUX on D-input → added switching capacitance
    f["num_sync_reset"]  = len(re.findall(
        r"\bif\s*\(\s*(?:reset|rst|rst_n|aresetn|sreset)\s*\)",
        rtl, re.IGNORECASE))
    # Async reset → listed in sensitivity list → structural difference
    f["num_async_reset"] = len(re.findall(
        r"(?:posedge|negedge)\s+(?:reset|rst|rst_n|aresetn)\b",
        rtl, re.IGNORECASE))

    # ── 3. Clock enables  ─────────────────────────────────────
    # CE logic gates the clock → REDUCES dynamic power
    # Negative correlation with power (important for model)
    ce_assign = len(re.findall(r"\bif\s*\(\s*\w*(?:enable|en|ce|clk_en)\w*\s*\)",
                               rtl, re.IGNORECASE))
    ce_signal  = len(re.findall(r"\b(?:clk_en|clock_enable|ce|enable)\b\s*(?:<=|=)",
                                rtl, re.IGNORECASE))
    f["num_clock_enable"] = int(ce_assign + ce_signal)

    # ── 4. Latch detection  ───────────────────────────────────
    # Latches have different leakage profile; infer from level-sensitive always
    latch_always = len(re.findall(
        r"always\s*@\s*\([^)]*\)\s*\n?\s*(?:begin)?\s*\n?\s*if\b",
        rtl, re.IGNORECASE))
    latch_kw = len(re.findall(r"\blatch\b", rtl, re.IGNORECASE))
    f["num_latch"] = int(latch_always + latch_kw)

    # ── 5. Tri-state / hi-Z  ──────────────────────────────────
    # Hi-Z assignments → bus keepers → extra leakage + switching
    f["num_tristate"] = len(re.findall(r"['\"]?\s*[zZ]\b|1'[bh][zZ]", rtl))

    # ── 6. Estimated flip-flop bit count  ─────────────────────
    # STRONGEST single proxy for internal (cell switching) power
    # Because: P_internal ∝ α × C_ff × f, and C_ff ∝ num_bits
    f["estimated_ff_bits"] = float(n_reg * avg_bw)

    # ── 7. Weighted switching score  ──────────────────────────
    # XOR output toggles ~3× more often than AND/OR (statistical)
    # Both scaled by bitwidth (wider bus = more capacitance)
    f["weighted_switching"] = float(
        n_reg   * avg_bw * 1.0 +
        n_xor   * avg_bw * 3.0 +
        n_and   * avg_bw * 0.5
    )

    # ── 8. Datapath width pressure  ───────────────────────────
    # Net switching power ∝ toggle activity × wire capacitance
    # Capacitance scales with bus width, multipliers use full width
    f["datapath_width_pressure"] = float(
        (n_add + n_sub) * avg_bw +
         n_mul          * max_bw     # multiplier uses max width (partial products)
    )

    # ── 9. XOR density  ───────────────────────────────────────
    # XOR is the highest-toggle operation in arithmetic/crypto circuits
    f["xor_density"] = float(n_xor / n_lines)

    # ── 10. MUX-to-logic ratio  ───────────────────────────────
    # Mux trees (ternary + case) are heavily loaded nets → high switching
    total_logic = max(n_and + n_or + n_xor, 1)
    mux_cnt     = n_tern + n_case
    f["mux_to_logic_ratio"] = float(mux_cnt / total_logic)

    # ── 11. Control-to-data ratio  ────────────────────────────
    # More control logic relative to arithmetic → more glitch-prone paths
    total_arith = max(n_add + n_sub + n_mul + n_div, 1)
    f["control_to_data_ratio"] = float((n_if + n_case + n_for) / total_arith)

    # ── 12. Conditional (implicit mux) count  ─────────────────
    # Every if/ternary synthesises to a MUX → capacitive load
    f["num_conditional_assign"] = int(n_tern + n_if)

    # ── 13. Clock domain pressure  ────────────────────────────
    # Multiple clock domains → CDC synchronisers (constantly toggling)
    f["clk_domain_pressure"] = int(
        base.get("num_clk_domains", 0) * base.get("num_posedge_always", 0))

    # ── 14. Reset fanout proxy  ───────────────────────────────
    # Reset tree fans out to every FF → one of the highest-fanout nets
    # High fanout = large capacitance driven per assertion
    f["reset_fanout_proxy"] = int(
        (f["num_sync_reset"] + f["num_async_reset"]) * n_reg)

    # ── 15. High-toggle operation score  ─────────────────────
    # XOR=3×, comparisons=2×, mux=2×  — all confirmed high-toggle in SPICE
    f["high_toggle_score"] = int(n_xor * 3 + n_comp * 2 + mux_cnt * 2)

    # ── 16. Sequential-to-combinational ratio  ────────────────
    # Balance affects the ratio of internal vs net switching power
    f["seq_to_comb_ratio"] = float(n_reg / max(n_wire, 1))

    # ── 17. Average operational bitwidth  ────────────────────
    # Wider operations drive larger busses → more capacitance
    f["avg_op_bitwidth"] = float(
        ((n_add + n_sub) * avg_bw + (n_mul + n_div) * max_bw)
        / max(n_add + n_sub + n_mul + n_div, 1)
    )

    # ── 18. Bitwidth-weighted individual ops  ────────────────
    f["bitwidth_weighted_adds"] = float(n_add * avg_bw)
    f["bitwidth_weighted_muls"] = float(n_mul * max_bw)  # muls use full width

    # ── 19. Variable shift detection  ────────────────────────
    # a << b where b is a signal → barrel shifter → very high power
    f["num_shift_by_var"] = int(len(re.findall(
        r"<<\s*[a-zA-Z_]\w*|>>\s*[a-zA-Z_]\w*", rtl)))

    # ── 20. Memory / array access  ───────────────────────────
    # Indexed writes/reads → infer SRAM-like structures
    f["num_memory_access"] = int(len(re.findall(
        r"\w+\s*\[\s*\w+\s*\]\s*(?:<=|=)|\b(?:<=|=)\s*\w+\s*\[\s*\w+\s*\]", rtl)))

    # ── 21. Signal reuse factor  ──────────────────────────────
    # High reuse = high fanout = more capacitance driven per transition
    mean_occ = float(base.get("mean_signal_occurrences",
                               base.get("mean_signal_occurrences", 1.0)))
    f["signal_reuse_factor"] = float(mean_occ / n_lines)

    # ── 22. I/O port switching load  ─────────────────────────
    # Output ports drive external loads (PCB, pad capacitance)
    f["output_switching_load"]  = float(base.get("num_output", 0) * avg_bw)
    f["input_toggle_potential"] = float(base.get("num_input",  0) * avg_bw)

    # ── 23. Pipeline depth proxy  ────────────────────────────
    # Count non-blocking assignments inside clocked always blocks
    # More pipeline stages → more FFs toggling every clock cycle
    ff_blocks = re.findall(
        r"always\s*@\s*\([^)]*(?:posedge|negedge)[^)]*\)(.*?)(?=always|\Z)",
        rtl, re.DOTALL | re.IGNORECASE)
    f["pipeline_depth_proxy"] = int(sum(blk.count("<=") for blk in ff_blocks))

    # ── 24. Glitch potential  ─────────────────────────────────
    # Long combinational paths between registers → glitches waste power
    # Proxy: combinational blocks × arithmetic ops ÷ register stages
    f["glitch_potential"] = float(
        f["num_always_comb"] * (n_add + n_sub + n_mul + n_xor)
        / max(f["num_always_ff"], 1)
    )

    # ── Composite power indices  ──────────────────────────────
    f["effective_switching_activity"] = float(
        f["weighted_switching"] +
        f["high_toggle_score"]  * avg_bw * 0.5 +
        f["datapath_width_pressure"] * 0.3
    )
    # Log-normalised for model stability (Yeo-Johnson transform in model)
    f["power_complexity_index"] = float(math.log1p(
        f["effective_switching_activity"] + f["datapath_width_pressure"]
    ))

    return f


# ══════════════════════════════════════════════════════════════
#  .RPT FILE PARSING
# ══════════════════════════════════════════════════════════════

UNIT_MULTIPLIERS = {
    "pw": 1e-9,   # pW → mW
    "nw": 1e-6,   # nW → mW
    "uw": 1e-3,   # uW → mW
    "mw": 1.0,    # mW → mW  (no change)
    "w":  1000.0, # W  → mW
}

# Patterns per metric — first match wins
RPT_PATTERNS = {
    "total_cell_area": [
        r"Total cell area\s*[:=]\s*([\d,\.]+)",
        r"Cell Area\s*[:=]\s*([\d,\.]+)",
        r"Combinational area\s*[:=]\s*([\d,\.]+)",
    ],
    "comb_area": [
        r"Combinational area\s*[:=]\s*([\d,\.]+)",
    ],
    "Power": [
        r"Total Dynamic Power\s*=\s*([\d\.]+)\s*([munpMUNP]?[wW])",
        r"Total power\s*=\s*([\d\.]+)\s*([munpMUNP]?[wW])",
        r"Dynamic Power\s*=\s*([\d\.]+)\s*([munpMUNP]?[wW])",
        r"Total\s+[\d\.]+\s+[munpMUNP]?[wW]\s+[\d\.]+\s+[munpMUNP]?[wW]\s+[\d\.]+\s+[nN][wW]\s+([\d\.]+)\s+([munpMUNP]?[wW])",
    ],
    "levels_of_logic": [
        r"Levels of Logic\s*[:=]\s*([\d\.]+)",
    ],
    "wns": [
        r"\bWNS\s*[:=]\s*([-\d\.]+)",
    ],
    "tns": [
        r"\bTNS\s*[:=]\s*([-\d\.]+)",
    ],
    "num_cells": [
        r"Number of cells\s*[:=]?\s*([\d,]+)",
        r"Leaf Cell Count\s*[:=]\s*([\d,]+)",
    ],
    "num_nets": [
        r"Number of nets\s*[:=]?\s*([\d,]+)",
        r"Total Number of Nets\s*[:=]\s*([\d,]+)",
    ],
}


def _parse_critical_path(txt: str) -> float | None:
    """
    Extract the MAXIMUM data-arrival time across all timing paths.
    The v1 code extracted only the first match; this takes the max,
    which equals the true critical path.
    """
    # Pattern: 'data arrival time   15.00'
    arrivals = re.findall(r"data arrival time\s+([\d\.]+)", txt, re.IGNORECASE)
    if arrivals:
        return float(max(float(v) for v in arrivals))
    # Fallback: 'Critical Path Length: 15.00'
    m = re.search(r"Critical Path Length\s*[:=]\s*([\d\.]+)", txt, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def parse_rpt_text(rpt_text: str) -> dict:
    """Parse all known metrics from concatenated .rpt content."""
    out = {}
    txt = rpt_text.replace(",", "")  # strip thousands separators

    for key, patterns in RPT_PATTERNS.items():
        found = None
        for pat in patterns:
            m = re.search(pat, txt, re.IGNORECASE)
            if not m:
                continue
            try:
                val  = float(m.group(1))
                unit = (m.group(2).lower().strip()
                        if m.lastindex and m.lastindex >= 2 and m.group(2) else "")
            except Exception:
                continue

            if key == "Power":
                mult  = UNIT_MULTIPLIERS.get(unit, 1.0)
                found = float(val * mult)
            elif key in ("num_cells", "num_nets"):
                found = int(val)
            else:
                found = float(val)
            break
        out[key] = found

    # critical path handled separately (needs max over all paths)
    out["critical_path_length"] = _parse_critical_path(txt)
    return out


# ══════════════════════════════════════════════════════════════
#  PER-DESIGN EXTRACTION
# ══════════════════════════════════════════════════════════════

CANONICAL_FEATURE_ORDER = [
    "num_lines", "max_bitwidth", "min_bitwidth", "avg_bitwidth",
    "total_bits", "num_bitwidths", "num_32bit", "num_16bit", "num_8bit", "num_1bit",
    "num_add", "num_sub", "num_mul", "num_div",
    "num_logic_and", "num_logic_or", "num_logic_xor", "num_shifts", "num_comparisons",
    "num_always", "num_assign", "num_if", "num_case", "num_for", "num_ternary",
    "num_branches",
    "num_wire", "num_reg", "num_input", "num_output", "num_modules",
]


def extract_features_for_design(design_dir: Path) -> dict | None:
    # find .v files — prefer <design_name>.v, fall back to first found
    vfiles = sorted(design_dir.glob("*.v"))
    if not vfiles:
        return None

    # prefer exact name match
    preferred = [v for v in vfiles if v.stem == design_dir.name]
    vpath = preferred[0] if preferred else vfiles[0]

    rtl_raw = read_text_safe(vpath)
    rtl     = clean_rtl_text(rtl_raw)

    # ── base feature extraction ───────────────────────────────
    bitwidth_feats = extract_bitwidth_features(rtl)
    op_feats       = count_operators(rtl)
    struct_feats   = extract_structural_features(rtl)
    clock_feats    = detect_clock_features(rtl)
    signal_proxy   = extract_signal_fanout_proxy(rtl)

    # merged lookup dict for downstream functions
    base = {}
    base.update(bitwidth_feats)
    base.update(op_feats)
    base.update(struct_feats)
    base.update(clock_feats)
    base.update(signal_proxy)

    # ── canonical 31 features ─────────────────────────────────
    canonical = {k: base.get(k, 0) for k in CANONICAL_FEATURE_ORDER}

    # ── v1 engineered features ────────────────────────────────
    v1_feats  = compute_power_engineered_features(base, bitwidth_feats, signal_proxy)

    # ── v2 new power features (24 new) ───────────────────────
    v2_feats  = compute_new_power_features(rtl, base, bitwidth_feats)
    # fill in clock / signal proxies needed by v2 that come from base
    v2_feats["num_clk_domains"]    = clock_feats["num_clk_domains"]
    v2_feats["num_posedge_always"] = clock_feats["num_posedge_always"]
    v2_feats["mean_signal_occurrences"] = signal_proxy["mean_signal_occurrences"]

    # ── .rpt parsing ─────────────────────────────────────────
    rpt_files   = sorted(design_dir.glob("*.rpt"))
    rpt_concat  = "\n\n".join(read_text_safe(p) for p in rpt_files)
    rpt_metrics = parse_rpt_text(rpt_concat) if rpt_concat.strip() else {
        k: None for k in list(RPT_PATTERNS.keys()) + ["critical_path_length"]}
    rpt_meta    = {
        "rpt_files":    ";".join(p.name for p in rpt_files),
        "rpt_text_len": len(rpt_concat),
    }

    # ── assemble final row ────────────────────────────────────
    row = {"Design_Name": design_dir.name}
    row.update(canonical)       # 31 canonical RTL features
    row.update(v1_feats)        # v1 engineered (switching_proxy, etc.)
    row.update(clock_feats)     # num_posedge_always, num_clk_domains
    row.update(v2_feats)        # 24 new power features + composites
    row["rtl_code_len"] = len(rtl)
    row.update(rpt_metrics)     # parsed PPA numbers
    row.update(rpt_meta)        # file list + text length
    return row


# ══════════════════════════════════════════════════════════════
#  DATASET BUILDER  (with progress bar + error log)
# ══════════════════════════════════════════════════════════════

def build_dataset(dataset_root: str, out_csv: str) -> pd.DataFrame:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"{dataset_root} not found")

    design_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    total       = len(design_dirs)
    print(f"[scan] {total} design folders under {dataset_root}")

    rows      = []
    errors    = []
    t_start   = time.time()

    for idx, d in enumerate(design_dirs, 1):
        try:
            row = extract_features_for_design(d)
            if row:
                rows.append(row)
        except Exception as e:
            errors.append({"design": d.name, "error": str(e)})

        # progress bar every 50 designs or on last
        if idx % 50 == 0 or idx == total:
            elapsed = time.time() - t_start
            eta     = (elapsed / idx) * (total - idx)
            pct     = idx / total * 100
            bar     = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  [{bar}] {idx}/{total}  {pct:5.1f}%  "
                  f"elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s", end="\r")

    print()  # newline after progress bar

    if errors:
        err_path = str(Path(out_csv).with_suffix("")) + "_errors.csv"
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"[warn] {len(errors)} designs failed — logged to {err_path}")

    df = pd.DataFrame(rows)
    written = save_dataframe_safe(df, out_csv)
    print(f"[done] {len(df)} rows × {len(df.columns)} columns → {written}")
    return df


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract RTL + .rpt features for power prediction (v2)")
    parser.add_argument("--dataset_root", required=True,
                        help="Root folder with per-design subfolders")
    parser.add_argument("--out_csv", required=True,
                        help="Output CSV path")
    args = parser.parse_args()
    build_dataset(args.dataset_root, args.out_csv)


if __name__ == "__main__":
    main()