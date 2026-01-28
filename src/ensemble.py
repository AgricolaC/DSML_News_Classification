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
from sklearn.model_selection import cross_val_score
from .preprocessing import DatasetCleaner, DatasetDeduplicator, FeatureExtractor, TimeExtractor, RawTimeExtractor, PageRankOneHot, SourceTransformer
from .cv_utils import AnchoredTimeSeriesSplit
from .seed import set_global_seed



from .models import get_pipelines, load_best_params as load_params


def preprocess_dataset(df, is_train=True, source_transformer=None, sort_by_time=True):
    """
    Applies the standard preprocessing pipeline:
    - Cleaning
    - Deduplication (Train only)
    - Feature Extraction (Text concatenation)
    - Time Extraction (Sin/Cos features)

    Returns:
    - df: Transformed DataFrame
    - source_transformer: The fitted SourceTransformer instance
    """
    # 1. Cleaner
    df = DatasetCleaner(verbose=True).transform(df)
    
    # 2. Deduplicator (Train Only)
    if is_train:
        df = DatasetDeduplicator(mode='advanced', verbose=True).transform(df)
    
    # 3. Text Feature Extractor
    df = FeatureExtractor().transform(df)
    
    # 3a. HTML Removal
    # df = AdvancedTextCleaner().transform(df)

    # 3b. PageRank One-Hot
    df = PageRankOneHot().transform(df)
    
    # 4. Time Extractors (dual: cyclical for SVC, raw for HGB)
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Cyclical time features for LinearSVC (sin/cos, 0-fill)
        time_extractor_cyclical = TimeExtractor()
        df = time_extractor_cyclical.fit_transform(df)
        
        # Raw time features for HGB (native NaN handling)
        # Need to restore timestamp first
        df['timestamp'] = timestamps
        time_extractor_raw = RawTimeExtractor()
        df_raw_time = time_extractor_raw.fit_transform(df[['timestamp']])
        
        # Add raw features with 'raw_' prefix to distinguish
        for col in ['hour', 'day_of_week', 'month', 'quarter', 'year_offset']:
            if col in df_raw_time.columns:
                df[f'raw_{col}'] = df_raw_time[col]
        
        # Restore timestamp for sorting
        df['timestamp'] = timestamps
    else:
        time_extractor_cyclical = TimeExtractor()
        df = time_extractor_cyclical.fit_transform(df)
        
    # 5. Time Sorting (Only for Training/CV)
    if sort_by_time and 'timestamp' in df.columns:
        df = df.sort_values(by='timestamp', na_position='first')
        
    return df, source_transformer

def get_voting_ensemble():
    
    all_pipelines = get_pipelines()
    
    # Split into Strong (Tier 1) and Weak (Tier 2)
    strong_models = [p for p in all_pipelines if p[0] in ['svc', 'lr']]
    weak_models = [p for p in all_pipelines if p[0] in ['hgb', 'mnb', 'cnb']]
    
    # Tier 2 Consensus (Internal Vote)
    weak_ensemble = VotingClassifier(estimators=weak_models, voting='hard')
    
    # Tier 1 Final Vote (Strong + Weak_Consensus)
    final_estimators = strong_models + [('weak_consensus', weak_ensemble)]
    
    return VotingClassifier(estimators=final_estimators, voting='hard')

def run_cv_evaluation():
    set_global_seed(42)

    
    print("="*60)
    print("RUNNING CROSS-VALIDATION EVALUATION")
    print("="*60)
    
    # 1. Load & Prep
    print("Loading Development Data...")
    df_dev = pd.read_csv("./dataset/development.csv")
    
    df_dev, _ = preprocess_dataset(df_dev, is_train=True)

    ensemble = get_voting_ensemble()
    
    # 3. Validation
    print("Validating with AnchoredTimeSeriesSplit (5 folds)...")
    cv = AnchoredTimeSeriesSplit(df_dev, n_splits=5)
    
    from sklearn.metrics import f1_score
    fold_scores = []
    
    for i, (train_idx, test_idx) in enumerate(cv.split(df_dev)):
        print(f"\n[Fold {i+1}/5]")
        
        # Train/Test Split
        X_train, y_train = df_dev.iloc[train_idx], df_dev['label'].iloc[train_idx]
        X_test, y_test = df_dev.iloc[test_idx], df_dev['label'].iloc[test_idx]
        
        # Fit Ensemble
        ensemble.fit(X_train, y_train)
        
        # Extract fitted models from the hierarchy
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
            ('  HistGradientBoosting', hgb),
            ('  MultinomialNB', mnb),
            ('  ComplementNB', cnb)
        ]
        
        for name, model in models_to_eval:
            pred = model.predict(X_test)
            score = f1_score(y_test, pred, average='macro')
            print(f"  {name}: {score:.4f}")
            
        # Evaluate Ensemble
        pred_ensemble = ensemble.predict(X_test)
        score_ensemble = f1_score(y_test, pred_ensemble, average='macro')
        print(f"  >> Ensemble (Hierarchical): {score_ensemble:.4f}")
        
        fold_scores.append(score_ensemble)
    
    scores = np.array(fold_scores)
    mean_f1 = np.mean(scores)
    std_f1 = np.std(scores)
    
    print(f"\nValidation Score: {mean_f1:.4f} (+/- {std_f1:.4f})")
    
    # SAVE ARTIFACTS
    print("\n" + "="*60)
    print("SAVING RESULTS TO results/ensemble_cv/")
    print("="*60)
    
    import os
    os.makedirs('results/ensemble_cv', exist_ok=True)
    
    # Save fold scores as JSON
    results_dict = {
        'mean_f1': float(mean_f1),
        'std_f1': float(std_f1),
        'fold_scores': [float(s) for s in fold_scores],
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_folds': len(fold_scores),
        'voting_strategy': 'hierarchical'
    }
    
    with open('results/ensemble_cv/fold_scores.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print("✓ Saved fold_scores.json")
    
if __name__ == "__main__":
    run_cv_evaluation()
