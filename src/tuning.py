"""
Usage:

    If you are using a virtual environment, first run 
    source .venv/bin/activate

    python -m src.tuning --model svc
    python -m src.tuning --model hgb
    python -m src.tuning --model all  # Tune all models sequentially / takes a long time !!
"""
import pandas as pd
import numpy as np
import json
import argparse
from sklearn.model_selection import GridSearchCV
from .preprocessing import DatasetCleaner, DatasetDeduplicator, FeatureExtractor, PageRankTransformer, SourceTransformer
from sklearn.model_selection import StratifiedGroupKFold
from .seed import set_global_seed
from .models import get_pipelines, load_best_params
import os

PARAM_GRIDS = {
    'svc': {
        'clf__C': [0.01, 0.1, 1.0],
        'clf__loss': ['squared_hinge'],  # 'hinge' is not supported by LinearSVC with dual=False
        'clf__tol': [1e-4, 1e-3],
        'clf__max_iter': [2500]
    },
    'lr': {
        'clf__C': [0.1, 1.0, 10.0],
        'clf__max_iter': [2500]
    },
    'hgb': {
        'clf__learning_rate': [0.1],
        'clf__max_depth': [10, 20, None],
        'clf__l2_regularization': [1.0, 10.0],
        'clf__max_leaf_nodes': [31, 63]
    },
    'mnb': {
        'clf__alpha': [0.01, 0.05, 0.1, 0.5, 1.0],
        'clf__fit_prior': [True, False]
    },
    'cnb': {
        'clf__alpha': [0.01, 0.1, 1.0],
        'clf__norm': [True, False],
        'clf__fit_prior': [True, False]
    }
}

def preprocess_data(df):
    """
    Apply preprocessing pipeline
    """
    print("Preprocessing data...")
    
    # Step 1: Clean
    df = DatasetCleaner(verbose=False).transform(df)
    
    # Step 2: Deduplicate
    df = DatasetDeduplicator(mode='advanced', verbose=True).transform(df)
    
    # Step 3: Feature Extraction
    df = FeatureExtractor().transform(df)
    
    # Step 4: Source Transformer (Handled in Pipeline)
    # df = SourceTransformer(top_k=400).fit_transform(df) 
    
    # Step 5: PageRank
    df = PageRankTransformer(mode='onehot').transform(df)
    
    # Extract target
    y = df['label']
    X = df
    
    return X, y

def tune_model(model_name, param_grid=None, n_splits=5, n_jobs=-1, verbose=2):
    """
    Tune hyperparameters for a specific model.       
    """
    set_global_seed(42)
    
    print("="*60)
    print(f"TUNING: {model_name.upper()}")
    print("="*60)
    
    # Load data
    df = pd.read_csv("./dataset/development.csv")
    X, y = preprocess_data(df)
    
    # Get pipeline for this model
    pipelines = get_pipelines(models=[model_name])
    if not pipelines:
        raise ValueError(f"Model '{model_name}' not found")
    
    pipeline = pipelines[0][1]  # Extract pipeline from (name, pipeline) tuple
    
    # Get parameter grid
    if param_grid is None:
        param_grid = PARAM_GRIDS.get(model_name, {})
    
    if not param_grid:
        print(f"No parameter grid defined for {model_name}. Using default parameters.")
        return {'model': model_name, 'best_params': {}, 'best_score': None}
    
    print(f"\nParameter Grid:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    print(f"\nGrid Size: {np.prod([len(v) for v in param_grid.values()])} candidates")
    print(f"CV Splits: {n_splits}")
    
    # Setup CV
    print(f"CV Strategy: StratifiedKFold")
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Run GridSearch
    print(f"\nRunning GridSearchCV...")
    gs = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring='f1_macro', 
        n_jobs=n_jobs, 
        verbose=verbose,
        return_train_score=False
    )
    
    gs.fit(X, y)
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Best Score: {gs.best_score_:.4f}")
    print(f"\nBest Parameters:")
    print(json.dumps(gs.best_params_, indent=2))
    
    return {
        'model': model_name,
        'best_params': gs.best_params_,
        'best_score': float(gs.best_score_),
        'cv_results': {
            'mean_test_score': gs.cv_results_['mean_test_score'].tolist(),
            'std_test_score': gs.cv_results_['std_test_score'].tolist(),
            'params': [str(p) for p in gs.cv_results_['params']]
        }
    }

def save_tuning_results(results, output_dir='results/tuning'):
    """Save tuning results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{output_dir}/{results['model']}_tuning_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results to {filename}")
    
    # Also update best_params.json if better
    update_best_params(results)

def update_best_params(results):
    """Update best_params.json if new results are better."""
    model_name = results['model']
    best_params = results['best_params']
    
    try:
        with open('best_params.json', 'r') as f:
            current_best = json.load(f)
    except FileNotFoundError:
        current_best = {}
    
    # Map model names to keys in best_params.json
    param_keys = {
        'svc': 'svc',
        'lr': 'lr',
        'hgb': 'hgbc',
        'mnb': 'mnb',
        'cnb': 'cnb'
    }
    
    key = param_keys.get(model_name)
    if key:
        current_best[key] = best_params
        
        with open('best_params.json', 'w') as f:
            json.dump(current_best, f, indent=4)
        
        print(f"✓ Updated best_params.json['{key}']")

def tune_all_models(n_splits=3):
    """Tune all 5 models sequentially."""
    results_summary = []
    
    for model_name in ['svc', 'lr', 'hgb', 'mnb', 'cnb']:
        try:
            results = tune_model(model_name, n_splits=n_splits, verbose=1)
            save_tuning_results(results)
            results_summary.append({
                'model': model_name,
                'best_score': results['best_score']
            })
        except Exception as e:
            print(f"Error tuning {model_name}: {e}")
            results_summary.append({
                'model': model_name,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "="*60)
    print("TUNING SUMMARY")
    print("="*60)
    for result in results_summary:
        if 'error' in result:
            print(f"{result['model']:>5}: ERROR - {result['error']}")
        else:
            print(f"{result['model']:>5}: F1 = {result['best_score']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for ensemble models')
    parser.add_argument(
        '--model', 
        type=str, 
        default='hgb',
        choices=['svc', 'lr', 'hgb', 'mnb', 'cnb', 'all'],
        help='Model to tune (default: hgb)'
    )
    parser.add_argument(
        '--splits', 
        type=int, 
        default=3,
        help='Number of CV splits (default: 3)'
    )
    parser.add_argument(
        '--jobs', 
        type=int, 
        default=-1,
        help='Number of parallel jobs (default: -1 = all cores)'
    )
    
    args = parser.parse_args()
    
    if args.model == 'all':
        tune_all_models(n_splits=args.splits)
    else:
        results = tune_model(args.model, n_splits=args.splits, n_jobs=args.jobs)
        save_tuning_results(results)
