import shutil
from pathlib import Path

# ================== CONFIG ==================
RTL_REPORT_ROOT = Path(r"C:\ml ppa\Final_Clean_Dataset")
NETLIST_ROOT = Path(r"C:\Users\Admin\OneDrive - MSFT\Desktop\BTech-Project-Power-Estimation-main\dc_out_filtered_cleaned")
# ============================================

print("\n--- COPYING NETLIST FILES (SMART SEARCH MODE) ---\n")

# Build a lookup table of ALL netlists
netlist_lookup = {}

for design_group in NETLIST_ROOT.iterdir():
    if not design_group.is_dir():
        continue

    for netlist_folder in design_group.iterdir():
        if not netlist_folder.is_dir():
            continue

        for vfile in netlist_folder.glob("*.v"):
            netlist_lookup[vfile.name] = vfile

print(f"✅ Indexed {len(netlist_lookup)} netlists\n")

# Now match and copy
for rtl_folder in RTL_REPORT_ROOT.iterdir():
    if not rtl_folder.is_dir():
        continue

    rtl_files = list(rtl_folder.glob("*.v"))
    if not rtl_files:
        print(f"⚠️ No RTL found in: {rtl_folder.name}")
        continue

    rtl_name = rtl_files[0].name

    if rtl_name not in netlist_lookup:
        print(f"❌ Netlist NOT found for: {rtl_name}")
        continue

    source_netlist = netlist_lookup[rtl_name]
    destination = rtl_folder / f"netlist_{rtl_name}"

    try:
        shutil.copy2(source_netlist, destination)
        print(f"✅ Netlist copied for: {rtl_folder.name}")
    except Exception as e:
        print(f"❌ Copy failed for {rtl_folder.name}: {e}")

print("\n✅ NETLIST COPYING COMPLETE")
