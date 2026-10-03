'''#!/usr/bin/env python3'''

import os
import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------- RTL HELPERS ---------------- #

def clean_rtl_text(text):
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'".*?"', '', text, flags=re.DOTALL)
    return text


def extract_bitwidth_features(rtl):
    features = {}
    matches = re.findall(r'\[(\d+)\s*:\s*(\d+)\]', rtl)

    if matches:
        bitwidths = [abs(int(a) - int(b)) + 1 for a, b in matches]
        features.update({
            "max_bitwidth": max(bitwidths),
            "min_bitwidth": min(bitwidths),
            "avg_bitwidth": np.mean(bitwidths),
            "total_bits": sum(bitwidths),
            "num_bitwidths": len(bitwidths),
            "num_32bit": bitwidths.count(32),
            "num_16bit": bitwidths.count(16),
            "num_8bit": bitwidths.count(8),
            "num_1bit": bitwidths.count(1),
        })
    else:
        features.update({
            "max_bitwidth": 1,
            "min_bitwidth": 1,
            "avg_bitwidth": 1,
            "total_bits": 1,
            "num_bitwidths": 0,
            "num_32bit": 0,
            "num_16bit": 0,
            "num_8bit": 0,
            "num_1bit": 1,
        })

    return features


def count_operators(rtl):
    rtl = clean_rtl_text(rtl)
    feats = {
        "num_add": rtl.count("+"),
        "num_sub": rtl.count("-"),
        "num_mul": rtl.count("*"),
        "num_div": rtl.count("/"),
        "num_logic_and": rtl.count("&"),
        "num_logic_or": rtl.count("|"),
        "num_logic_xor": rtl.count("^"),
        "num_shifts": rtl.count("<<") + rtl.count(">>"),
        "num_comparisons": len(re.findall(r'==|!=|<=|>=|<|>', rtl)),
    }
    return feats


def extract_structural_features(rtl):
    rtl = clean_rtl_text(rtl)
    feats = {
        "num_always": rtl.count("always"),
        "num_assign": rtl.count("assign"),
        "num_if": rtl.count("if"),
        "num_case": rtl.count("case"),
        "num_for": rtl.count("for"),
        "num_wire": rtl.count("wire"),
        "num_reg": rtl.count("reg"),
        "num_input": rtl.count("input"),
        "num_output": rtl.count("output"),
        "num_modules": rtl.count("module"),
        "num_ternary": rtl.count("?"),
    }

    feats["num_branches"] = feats["num_if"] + feats["num_case"] + feats["num_ternary"]
    return feats


# ---------------- REPORT PARSING ---------------- #

def extract_report_metrics(text):
    feats = {}

    patterns = {
        "num_cells": r"Number of cells:\s+(\d+)",
        "num_nets": r"Number of nets:\s+(\d+)",
        "comb_area": r"Combinational area:\s+([\d\.]+)",
        "total_cell_area": r"Total cell area:\s+([\d\.]+)",
        "critical_path_length": r"Critical Path Length:\s+([\d\.]+)",
        "levels_of_logic": r"Levels of Logic:\s+([\d\.]+)",
        "wns": r"WNS:\s+([\d\.\-]+)",
        "tns": r"TNS:\s+([\d\.\-]+)",
    }

    for k, p in patterns.items():
        m = re.search(p, text)
        if m:
            feats[k] = float(m.group(1))

    pm = re.search(r"Total Dynamic Power\s*=\s*([\d\.]+)", text)
    if pm:
        feats["Power"] = float(pm.group(1))

    return feats


# ---------------- NETLIST PARSING ---------------- #

def extract_netlist_features(text):
    feats = {
        "netlist_num_gates": text.count("sky130_fd_sc_hd__"),
        "netlist_num_nets": text.count("wire"),
        "netlist_num_instances": len(re.findall(r'\s+\w+\s+\w+\s*\(', text)),
    }

    for gate in ["inv", "and", "or", "nand", "nor", "xor", "mux", "buf", "maj3"]:
        feats[f"netlist_{gate}_count"] = text.count(f"__{gate}")

    return feats


# ---------------- BUILD DATASET ---------------- #

def build_dataset(final_root, output_csv):
    final_root = Path(final_root)
    rows = []

    for design_dir in final_root.iterdir():
        if not design_dir.is_dir():
            continue

        name = design_dir.name
        print(f"Processing: {name}")

        rtl_path = design_dir / f"{name}.v"
        netlist_path = design_dir / f"netlist_{name}.v"

        reports = list(design_dir.glob("*.rpt"))
        all_report_text = "".join(p.read_text(errors="ignore") for p in reports)

        row = {"Design_Name": name}

        # RTL
        if rtl_path.exists():
            rtl = rtl_path.read_text(errors="ignore")
            row["num_lines"] = rtl.count("\n") + 1
            row["RTL_Code"] = rtl
            row.update(extract_bitwidth_features(rtl))
            row.update(count_operators(rtl))
            row.update(extract_structural_features(rtl))

        # REPORTS
        row.update(extract_report_metrics(all_report_text))

        # NETLIST
        if netlist_path.exists():
            netlist = netlist_path.read_text(errors="ignore")
            row.update(extract_netlist_features(netlist))

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"\n✅ FINAL CSV CREATED: {output_csv}")
    print(f"✅ TOTAL DESIGNS: {len(df)}")
    print(f"✅ TOTAL FEATURES: {len(df.columns)}")


# ---------------- CLI ---------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_clean_root", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    build_dataset(args.final_clean_root, args.output_csv)
