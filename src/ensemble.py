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
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from preprocessing import DatasetCleaner, DatasetDeduplicator, FeatureExtractor, TimeExtractor, PageRankOneHot, SourceTransformer
from cv_utils import AnchoredTimeSeriesSplit
from seed import set_global_seed

# Default Configuration
BEST_PARAMS = {
    'tfidf': {'ngram_range': (1, 3), 'min_df': 2, 'sublinear_tf': True, 'max_features': 30000, 'lowercase': True},
    'svc': {'clf__C': 0.1},
    'lr': {'clf__C': 1.0, 'clf__penalty': 'l2'},
    'hgbc': {'clf__learning_rate': 0.1, 'clf__max_depth': 10, 'clf__l2_regularization': 10.0, 'clf__max_leaf_nodes': 63}
}

def load_params():
    if os.path.exists("best_params.json"):
        print("Loading best_params.json...")
        with open("best_params.json", "r") as f:
            tuned = json.load(f)
            # Update BEST_PARAMS shallowly
            for k in tuned:
                if k in BEST_PARAMS:
                    # Fix JSON list back to tuple for ngram_range
                    if k == 'tfidf' and 'ngram_range' in tuned[k]:
                        tuned[k]['ngram_range'] = tuple(tuned[k]['ngram_range'])
                    
                    # Fix LR penalty warning
                    if k == 'lr' and 'clf__penalty' in tuned[k] and tuned[k]['clf__penalty'] == 'l2':
                         del tuned[k]['clf__penalty']

                    BEST_PARAMS[k] = tuned[k]
    else:
        print("Using Default BEST_PARAMS.")

def get_pipelines():
    # 1. SVC Pipeline
    # Tfidf -> SVC
    tfidf_args = BEST_PARAMS['tfidf']
    
    from sklearn.compose import make_column_selector
    
    def make_ct(steps):
        return ColumnTransformer(
            transformers=[
                ('txt', Pipeline(steps), 'final_text'),
                ('dense', 'passthrough', make_column_selector(pattern=r'^(?:hour_|day_|month_|week_|is_missing_|rank_|src_).*'))
            ],
            remainder='drop'
        )

    # SVC
    svc_steps = [
        ('vec', TfidfVectorizer(**tfidf_args)),
    ]

    pipe_svc = Pipeline([
        ('prep', make_ct(svc_steps)),
        ('clf', LinearSVC(class_weight='balanced', random_state=42, dual=False))
    ])
    pipe_svc.set_params(**BEST_PARAMS['svc']) 

    # LR
    lr_steps = [
        ('vec', TfidfVectorizer(**tfidf_args))
    ]
    pipe_lr = Pipeline([
        ('prep', make_ct(lr_steps)),
        ('clf', LogisticRegression(class_weight='balanced', solver='saga', random_state=42, max_iter=2500))
    ])
    pipe_lr.set_params(**BEST_PARAMS['lr'])

    # HGB
    # Tfidf -> SVD -> HGB
    hgb_steps = [
        ('vec', TfidfVectorizer(**tfidf_args)),
        ('svd', TruncatedSVD(n_components=400, random_state=42))
    ]
    pipe_hgb = Pipeline([
        ('prep', make_ct(hgb_steps)),
        ('clf', HistGradientBoostingClassifier(class_weight='balanced', random_state=42))
    ])
    # HGB Params might have prefixes like 'clf__learning_rate'
    pipe_hgb.set_params(**BEST_PARAMS['hgbc'])

    return [
        ('svc', pipe_svc),
        ('lr', pipe_lr),
        ('hgb', pipe_hgb)
    ]

def preprocess_dataset(df, is_train=True, source_transformer=None):
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
    df = DatasetCleaner(verbose=False).transform(df)
    
    # 2. Deduplicator (Train Only)
    if is_train:
        df = DatasetDeduplicator(mode='advanced', verbose=True).transform(df)
    
    # 3. Text Feature Extractor
    df = FeatureExtractor().transform(df)
    
    # 3b. Source Tagging (Top 300)
    if source_transformer is None:
        source_transformer = SourceTransformer(top_k=300)
        df = source_transformer.fit_transform(df)
    else:
        df = source_transformer.transform(df)

    # 3c. PageRank One-Hot
    df = PageRankOneHot().transform(df)
    
    # 4. Time Extractor (Preserving timestamp)
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
        df = TimeExtractor().transform(df)
        df['timestamp'] = timestamps # Restore
    else:
        df = TimeExtractor().transform(df)
        
    return df, source_transformer

def get_voting_ensemble():
    load_params()
    estimators = get_pipelines()
    return VotingClassifier(estimators=estimators, voting='hard')

def run_cv_evaluation():
    set_global_seed(42)
    load_params()
    
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
        
        # Evaluate Constituent Models
        # Note: 'estimators_' stores the fitted estimators
        model_names = ['LinearSVC', 'LogisticRegression', 'HistGradientBoosting']
        for name, model in zip(model_names, ensemble.estimators_):
            pred = model.predict(X_test)
            score = f1_score(y_test, pred, average='macro')
            print(f"  {name}: {score:.4f}")
            
        # Evaluate Ensemble
        pred_ensemble = ensemble.predict(X_test)
        score_ensemble = f1_score(y_test, pred_ensemble, average='macro')
        print(f"  >> Ensemble: {score_ensemble:.4f}")
        
        fold_scores.append(score_ensemble)
    
    scores = np.array(fold_scores)
    
    print(f"\nValidation Score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

if __name__ == "__main__":
    run_cv_evaluation()
