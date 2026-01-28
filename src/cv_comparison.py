"""
Compare TimeAnchoredCV vs StratifiedKFold validation scores.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer

from .ensemble import preprocess_dataset, get_voting_ensemble, load_params
from .cv_utils import AnchoredTimeSeriesSplit
from .seed import set_global_seed

def run_comparison():
    set_global_seed(42)
    load_params()
    
    print("="*80)
    print("CV STRATEGY COMPARISON: TimeAnchoredCV vs StratifiedKFold")
    print("="*80)
    
    # Load and preprocess data
    print("\n[1/3] Loading and preprocessing data...")
    df = pd.read_csv('./dataset/development.csv')
    df, _ = preprocess_dataset(df, is_train=True, sort_by_time=True)
    
    X = df
    y = df['label']
    
    # Get ensemble
    print("\n[2/3] Building ensemble...")
    ensemble = get_voting_ensemble()
    
    # Test 1: TimeAnchoredCV (Current approach)
    print("\n[3/3] Running CV strategies...")
    print("\n" + "-"*80)
    print("STRATEGY 1: TimeAnchoredCV (Respects temporal order)")
    print("-"*80)
    
    time_cv = AnchoredTimeSeriesSplit(df, n_splits=5)
    time_scores = cross_val_score(
        ensemble, X, y, 
        cv=time_cv, 
        scoring='f1_macro', 
        n_jobs=1,
        verbose=1
    )
    
    print(f"\nFold Scores: {[f'{s:.4f}' for s in time_scores]}")
    print(f"Mean F1: {time_scores.mean():.4f} (+/- {time_scores.std():.4f})")
    
    # Test 2: StratifiedKFold (Random splits)
    print("\n" + "-"*80)
    print("STRATEGY 2: StratifiedKFold (Random splits, ignores time)")
    print("-"*80)
    
    strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    strat_scores = cross_val_score(
        ensemble, X, y, 
        cv=strat_cv, 
        scoring='f1_macro', 
        n_jobs=1,
        verbose=1
    )
    
    print(f"\nFold Scores: {[f'{s:.4f}' for s in strat_scores]}")
    print(f"Mean F1: {strat_scores.mean():.4f} (+/- {strat_scores.std():.4f})")
    
    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"TimeAnchoredCV:   {time_scores.mean():.4f} (+/- {time_scores.std():.4f})")
    print(f"StratifiedKFold:  {strat_scores.mean():.4f} (+/- {strat_scores.std():.4f})")
    print(f"Difference:       {abs(strat_scores.mean() - time_scores.mean()):.4f}")
    print(f"Evaluation Score: 0.734 (actual test set)")
    print(f"\nTimeAnchoredCV Gap:   {abs(time_scores.mean() - 0.734):.4f}")
    print(f"StratifiedKFold Gap:  {abs(strat_scores.mean() - 0.734):.4f}")
    
    if abs(time_scores.mean() - 0.734) < abs(strat_scores.mean() - 0.734):
        print("\n✓ TimeAnchoredCV is MORE realistic (closer to test score)")
    else:
        print("\n✓ StratifiedKFold is MORE realistic (closer to test score)")
    
    # Save results
    import os
    import json
    os.makedirs('results/cv_comparison', exist_ok=True)
    
    results = {
        'time_anchored': {
            'mean': float(time_scores.mean()),
            'std': float(time_scores.std()),
            'scores': [float(s) for s in time_scores]
        },
        'stratified_kfold': {
            'mean': float(strat_scores.mean()),
            'std': float(strat_scores.std()),
            'scores': [float(s) for s in strat_scores]
        },
        'evaluation_score': 0.734
    }
    
    with open('results/cv_comparison/comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results/cv_comparison/comparison.json")

if __name__ == '__main__':
    run_comparison()
