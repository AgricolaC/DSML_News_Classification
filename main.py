import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ensemble import preprocess_dataset, get_voting_ensemble, load_params
from src.seed import set_global_seed

def run_final_prediction():
    set_global_seed(42)
    load_params()
    
    print("="*60)
    print("FINAL MODEL TRAINING & SUBMISSION")
    print("="*60)
    
    with tqdm(total=4, desc="Overall Progress", unit="step", ncols=80) as pbar:
        # 1. Load Data
        pbar.set_description("[1/4] Loading Data")
        try:
            df_dev = pd.read_csv("./dataset/development.csv")
            df_eval = pd.read_csv("./dataset/evaluation.csv")
        except FileNotFoundError:
            print("Error: Dataset files not found in ./dataset/")
            return
        pbar.update(1)

        # 2. Preprocess
        pbar.set_description("[2/4] Preprocessing")
        
        # Extract IDs before preprocessing
        ids = df_eval['Id'].copy()
        
        # Sort training data by time (for CV), but preserve eval order (for submission)
        df_dev, src_transformer = preprocess_dataset(df_dev, is_train=True, sort_by_time=True)
        df_eval, _ = preprocess_dataset(df_eval, is_train=False, source_transformer=src_transformer, sort_by_time=False)
        pbar.update(1)

        # 3. Train
        pbar.set_description("[3/4] Training Ensemble")
        ensemble = get_voting_ensemble()
        ensemble.fit(df_dev, df_dev['label'])
        pbar.update(1)

        # 4. Predict
        pbar.set_description("[4/4] Predicting")
        preds = ensemble.predict(df_eval)
        
        submission = pd.DataFrame({'Id': ids, 'Predicted': preds})
        submission.to_csv("submission.csv", index=False)
        pbar.update(1)
    
    print(f"\n✓ Submission saved to 'submission.csv' ({len(submission)} rows)")

if __name__ == "__main__":
    run_final_prediction()
