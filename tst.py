import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import numpy as np

# CONFIG
MODEL_DIR = r"C:\ml ppa\models_multi"
CSV_PATH = r"C:\ml ppa\final_maximal_dataset.csv"
OUTPUT_IMG_DIR = r"C:\ml ppa\visualizations"
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

def visualize_results():
    # 1. LOAD DATA
    print("Loading data for visualization...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return
        
    df = pd.read_csv(CSV_PATH).dropna()
    print(f"Loaded {len(df)} rows.")

    # 2. Re-create the Test Split (Must match training random_state)
    # We use indices to ensure we pick the exact same rows as the test set
    indices = np.arange(len(df))
    _, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
    df_test = df.iloc[idx_test]
    
    print(f"Test set size: {len(df_test)}")

    # 3. Targets Helper
    # Map friendly names to actual CSV column names
    target_map = {
        'Area': 'total_cell_area' if 'total_cell_area' in df.columns else 'comb_area',
        'Power': 'Power',
        'Delay': 'critical_path_length'
    }

    # 4. PLOTTING FUNCTION
    def plot_prediction(model_name, target_name):
        model_filename = f"{model_name}_{target_name}.joblib"
        path = os.path.join(MODEL_DIR, model_filename)
        
        if not os.path.exists(path):
            print(f"Skipping {model_filename} (not found)")
            return

        print(f"Loading model: {model_filename}...")
        try:
            # Load the full object (Model + Metadata)
            saved_obj = joblib.load(path)
            model = saved_obj['model']
            meta = saved_obj['meta']
            
            # --- CRITICAL FIX ---
            # Use the EXACT features list saved during training
            trained_features = meta['feature_cols']
            
            # Check if features exist in CSV (in case of renaming)
            missing_cols = [c for c in trained_features if c not in df_test.columns]
            if missing_cols:
                print(f"Error: The following features expected by the model are missing from CSV: {missing_cols}")
                return

            # Select X and Y
            X_test = df_test[trained_features].values
            y_true = df_test[target_map[target_name]].values
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Plot
            plt.figure(figsize=(7, 6))
            sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor=None)
            
            # Perfect alignment line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
            
            plt.title(f"{target_name}: {model_name}\n(Test Set Predictions)")
            plt.xlabel(f"Actual {target_name}")
            plt.ylabel(f"Predicted {target_name}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_filename = f"{target_name}_{model_name}_scatter.png"
            save_path = os.path.join(OUTPUT_IMG_DIR, save_filename)
            plt.savefig(save_path)
            plt.close()
            print(f"   -> Saved plot: {save_path}")
            
        except Exception as e:
            print(f"   -> Failed to plot {model_name}: {e}")

    # GENERATE PLOTS
    print("-" * 40)
    # Check the Suspicious Delay Model
    plot_prediction("MLP", "Delay")
    plot_prediction("RandomForest", "Delay")

    # Check the Best Area Model
    plot_prediction("XGBoost", "Area")

    # Check the Best Power Model
    plot_prediction("CatBoost", "Power")
    print("-" * 40)
    print("Visualization complete.")

if __name__ == "__main__":
    visualize_results()