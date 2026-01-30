import pandas as pd
import numpy as np
import json
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from .preprocessing import DatasetCleaner, DatasetDeduplicator, FeatureExtractor, PageRankTransformer, SourceTransformer
from sklearn.model_selection import StratifiedGroupKFold
from .seed import set_global_seed



from .models import get_pipelines, load_best_params as load_params


def preprocess_dataset(df, is_train=True, source_transformer=None, verbose=True):
    """
    Applies the standard preprocessing pipeline:
    - Cleaning
    - Deduplication (Train only)
    - Feature Extraction 
    - PageRank One-Hot
    - Source Extraction
    """
    df = DatasetCleaner(verbose=verbose).transform(df)
    if is_train:
        df = DatasetDeduplicator(mode='advanced', verbose=verbose).transform(df)
    df = FeatureExtractor().transform(df)
    df = PageRankTransformer().transform(df)  
    return df, source_transformer

def get_voting_ensemble():
    all_pipelines = get_pipelines()
    strong_models = [p for p in all_pipelines if p[0] in ['svc', 'lr']]
    weak_models = [p for p in all_pipelines if p[0] in ['hgb', 'mnb', 'cnb']]
    weak_ensemble = VotingClassifier(estimators=weak_models, voting='hard')
    final_estimators = strong_models + [('weak_consensus', weak_ensemble)]
    return VotingClassifier(estimators=final_estimators, voting='hard')

def run_cv_evaluation():
    set_global_seed(42)
    print("="*60)
    print("RUNNING CROSS-VALIDATION EVALUATION")
    print("="*60)
    
    print("Loading Development Data...")
    df_dev = pd.read_csv("./dataset/development.csv")    
    df_dev, _ = preprocess_dataset(df_dev, is_train=True)
    ensemble = get_voting_ensemble()
    
    # Validation
    print("Validating (5 folds)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    from sklearn.metrics import f1_score
    
    # Initialize metric storage
    metrics = {
        'LinearSVC': [],
        'LogisticRegression': [],
        'WeakConsensus': [],
        'HistGradientBoosting': [],
        'MultinomialNB': [],
        'ComplementNB': [],
        'Ensemble': []
    }
    
    for i, (train_idx, test_idx) in enumerate(cv.split(df_dev, df_dev['label'])):
        print(f"\n[Fold {i+1}/5]")
        
        X_train, y_train = df_dev.iloc[train_idx], df_dev['label'].iloc[train_idx]
        X_test, y_test = df_dev.iloc[test_idx], df_dev['label'].iloc[test_idx]
        
        # Fit Ensemble
        ensemble.fit(X_train, y_train)
        
        # Extract fitted models
        svc = ensemble.named_estimators_['svc']
        lr = ensemble.named_estimators_['lr']
        weak_ens = ensemble.named_estimators_['weak_consensus']
        hgb = weak_ens.named_estimators_['hgb']
        mnb = weak_ens.named_estimators_['mnb']
        cnb = weak_ens.named_estimators_['cnb']
        
        models_to_eval = [
            ('LinearSVC', svc),
            ('LogisticRegression', lr),
            ('WeakConsensus', weak_ens),
            ('HistGradientBoosting', hgb),
            ('MultinomialNB', mnb),
            ('ComplementNB', cnb)
        ]
        
        # Evaluate Components
        for name, model in models_to_eval:
            pred = model.predict(X_test)
            score = f1_score(y_test, pred, average='macro')
            print(f"  {name}: {score:.4f}")
            metrics[name].append(score)
            
        # Evaluate Ensemble
        pred_ensemble = ensemble.predict(X_test)
        score_ensemble = f1_score(y_test, pred_ensemble, average='macro')
        print(f"  >> Ensemble (Hierarchical): {score_ensemble:.4f}")
        metrics['Ensemble'].append(score_ensemble)
    
    # Report Final Results
    print("\n" + "="*60)
    print("FINAL VALIDATION REPORT")
    print("="*60)
    
    final_stats = {}
    
    for name, scores in metrics.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        final_stats[name] = {'mean': mean_score, 'std': std_score}
        
        # formatted output
        print(f"{name:<25} : {mean_score:.4f} (+/- {std_score:.4f})")
    
    print("\n" + "="*60)
    print("SAVING RESULTS TO results/ensemble_cv/")
    print("="*60)
    
    import os
    os.makedirs('results/ensemble_cv', exist_ok=True)
    
    results_dict = {
        'detailed_metrics': {
            name: {
                'mean': float(stat['mean']),
                'std': float(stat['std']),
                'fold_scores': [float(s) for s in metrics[name]]
            } for name, stat in final_stats.items()
        },
        'mean_f1': float(final_stats['Ensemble']['mean']), # Backward compatibility
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_folds': 5,
        'voting_strategy': 'hierarchical'
    }
    
    with open('results/ensemble_cv/fold_scores.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print("✓ Saved fold_scores.json")
    
if __name__ == "__main__":
    run_cv_evaluation()
