import pandas as pd

CSV_PATH = r"C:\ml ppa\final_maximal_dataset.csv"

df = pd.read_csv(CSV_PATH, nrows=1)
print("\nTOTAL FEATURES:", len(df.columns))
print("\nCOLUMN NAMES:\n")
for i, col in enumerate(df.columns, start=1):
    print(f"{i:02d}. {col}")
