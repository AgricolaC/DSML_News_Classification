import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.preprocessing import DatasetCleaner, DatasetDeduplicator, FeatureExtractor, TimeExtractor, AdvancedTextCleaner
from src.cv_utils import AnchoredTimeSeriesSplit
from src.seed import set_global_seed

def get_svc():
    return LinearSVC(C=0.1, class_weight='balanced', random_state=42, dual=False)

def evaluate(pipeline, X, y, name):
    print(f"\nrunning config: {name}")
    
    # 1. Standard Stratified K-Fold
    cv_strat = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores_strat = cross_val_score(pipeline, X, y, cv=cv_strat, scoring='f1_macro', n_jobs=-1)
    mean_strat = np.mean(scores_strat)
    
    # 2. Anchored Time Series Split
    # Requires X to be DataFrame with 'timestamp'
    try:
        cv_anchor = AnchoredTimeSeriesSplit(X, n_splits=3)
        scores_anchor = cross_val_score(pipeline, X, y, cv=cv_anchor, scoring='f1_macro', n_jobs=-1)
        mean_anchor = np.mean(scores_anchor)
    except Exception as e:
        print(f"  [Anchored CV Failed]: {e}")
        mean_anchor = 0.0

    print(f"  -> Stratified: {mean_strat:.4f} | Anchored: {mean_anchor:.4f} | Delta: {mean_anchor - mean_strat:.4f}")

def make_text_pipeline(clf):
    """
    Wraps Tfidf+Clf in a ColumnTransformer that selects 'final_text' 
    and drops everything else (like timestamp).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(), 'final_text')
        ],
        remainder='drop'
    )
    return Pipeline([
        ('prep', preprocessor),
        ('clf', clf)
    ])

def run_ablation():
    set_global_seed(42)
    print("="*80)
    print(f"{'Configuration':<30} | {'Stratified (3)':<15} | {'Anchored (3)':<15} | {'Delta':<10}")
    print("="*80)
    
    # Load
    df_raw = pd.read_csv("./dataset/development.csv")
    # Ensuring timestamp is datetime initially for all steps because AnchoredSplit needs it
    # But DatasetCleaner handles coercion. 
    # For Step 0/1 (Raw), we must manually ensure timestamp exists if we want to run AnchoredSplit.
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], errors='coerce')
    
    # ----------------------------------------------------------------
    # Step 00: Random Guess
    # ----------------------------------------------------------------
    clf_dummy = DummyClassifier(strategy='uniform', random_state=42)
    # Dummy doesn't need features, but pipeline structure expects them.
    # We can pass raw df and a dummy pipeline? 
    # Or just evaluate(clf, df[['label']], ...) 
    # evaluate expects inputs. Dummy ignores X.
    evaluate(clf_dummy, df_raw[['label']], df_raw['label'], "0. Random Guess")

    # ----------------------------------------------------------------
    # Step 0.5: Baseline (CNB)
    # ----------------------------------------------------------------
    # Re-using df_0 (prepared below) but we need to create it here or move it up.
    # FeatureExtractor transform is stateless, so we can do it here.
    df_0 = FeatureExtractor().transform(df_raw) 
    
    pipe_cnb = make_text_pipeline(ComplementNB())
    evaluate(pipe_cnb, df_0, df_0['label'], "0.5 Baseline (CNB)")

    # ----------------------------------------------------------------
    # Step 1: Baseline (SVC)
    # ----------------------------------------------------------------
    # df_0 is already created above.
    pipe_0 = make_text_pipeline(get_svc())
    evaluate(pipe_0, df_0, df_0['label'], "1. Baseline (SVC)")

    # ----------------------------------------------------------------
    # Step 2: + DatasetCleaner
    # ----------------------------------------------------------------
    cleaner = DatasetCleaner(verbose=False)
    # Re-read raw to be clean
    df_raw_2 = pd.read_csv("./dataset/development.csv") 
    df_2 = cleaner.transform(df_raw_2)
    df_2 = FeatureExtractor().transform(df_2)
    
    pipe_2 = make_text_pipeline(get_svc())
    evaluate(pipe_2, df_2, df_2['label'], "2. + Cleaning")
    
    # ----------------------------------------------------------------
    # Step 3: + DatasetDeduplicator
    # ----------------------------------------------------------------
    dedup = DatasetDeduplicator(mode='advanced', verbose=False)
    df_3 = dedup.transform(df_2)
    pipe_3 = make_text_pipeline(get_svc())
    evaluate(pipe_3, df_3, df_3['label'], "3. + Deduplication")

    # ----------------------------------------------------------------
    # Step 4: + TimeExtractor
    # ----------------------------------------------------------------
    # TimeExtractor normally drops 'timestamp'. We must keep it for AnchoredSplit.
    # We will modify the transform manually here or re-attach.
    
    time_ext = TimeExtractor()
    df_4_feats = time_ext.transform(df_3) # Has sin/cos, NO timestamp
    df_4 = df_4_feats.copy()
    df_4['timestamp'] = df_3['timestamp'].values # Re-attach for Splitter
    
    # Pipeline needs to consume text + time cols
    time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    
    preprocessor_4 = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(), 'final_text'),
            ('time', 'passthrough', time_cols)
        ],
        remainder='drop' # drops timestamp
    )
    pipe_4 = Pipeline([
        ('prep', preprocessor_4),
        ('clf', get_svc())
    ])
    evaluate(pipe_4, df_4, df_4['label'], "4. + TimeFeatures")

    # ----------------------------------------------------------------
    # Step 5: + AdvancedTextCleaner
    # ----------------------------------------------------------------
    adv_cleaner = AdvancedTextCleaner(verbose=False)
    df_5 = adv_cleaner.transform(df_4) # Transforms 'final_text' in place or new col
    # df_5 has advanced cleaned final_text + time cols + timestamp
    
    # Reuse pipe_4 structure
    evaluate(pipe_4, df_5, df_5['label'], "5. + AdvCleaning")

if __name__ == "__main__":
    run_ablation()
