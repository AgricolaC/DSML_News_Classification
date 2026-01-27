import sys
import os
import pandas as pd
import numpy as np

# Add src to system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ensemble import preprocess_dataset, get_voting_ensemble, load_params
from src.seed import set_global_seed

def run_final_prediction():
    set_global_seed(42)
    load_params()
    
    print("="*60)
    print("FINAL MODEL TRAINING & SUBMISSION")
    print("="*60)
    
    # 1. Load Data
    print("\n[1/4] Loading Data...")
    try:
        df_dev = pd.read_csv("./dataset/development.csv")
        df_eval = pd.read_csv("./dataset/evaluation.csv")
    except FileNotFoundError:
        print("Error: Dataset files not found in ./dataset/")
        return

    # 2. Preprocess
    print("\n[2/4] Preprocessing Data...")
    print("  Processing Development Set...")
    df_dev, src_transformer = preprocess_dataset(df_dev, is_train=True)
    print(f"    Rows: {len(df_dev)}")
    
    print("  Processing Evaluation Set...")
    ids = df_eval['Id']
    # Explicitly pass the fitted transformer
    df_eval, _ = preprocess_dataset(df_eval, is_train=False, source_transformer=src_transformer)
    print(f"    Rows: {len(df_eval)}")

    # 3. Train
    print("\n[3/4] Training Ensemble on Full Development Set...")
    ensemble = get_voting_ensemble()
    
    ensemble.fit(df_dev, df_dev['label'])
    print("  Training Complete.")

    # 4. Predict
    print("\n[4/4] Generating Predictions...")
    preds = ensemble.predict(df_eval)
    
    submission = pd.DataFrame({'Id': ids, 'Predicted': preds})
    submission.to_csv("submission.csv", index=False)
    print(f"  Saved to 'submission.csv' ({len(submission)} rows).")

if __name__ == "__main__":
    run_final_prediction()
