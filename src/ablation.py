import pandas as pd
import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
# Import Preprocessing
from .preprocessing import (
    DatasetCleaner, DatasetDeduplicator, FeatureExtractor, 
    TimeExtractor, RawTimeExtractor, PageRankTransformer, SourceTransformer
)
from .models import get_base_models, load_best_params
from .seed import set_global_seed

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def get_models():
    """Returns the classifiers to test."""
    return get_base_models()

def get_pipeline(model_name, model, feature_set='text_only'):
    """
    Constructs the pipeline based on the model and feature set availability.
    """
    params = load_best_params()
    tfidf_args = params.get('tfidf', {'ngram_range': (1, 3), 'min_df': 2, 'sublinear_tf': True, 'lowercase': True})
    
    # Base TF-IDF
    tfidf = TfidfVectorizer(**tfidf_args)
    
    # Text Pipeline: Default is just TF-IDF
    text_pipeline = tfidf
    
    # Special handling for HGB: TF-IDF -> SelectKBest -> TruncatedSVD
    if model_name in ['HistGradientBoosting', 'hgb']:
        text_pipeline = Pipeline([
            ('vec', tfidf),
            ('sel', SelectKBest(chi2, k=5000)),
            ('svd', TruncatedSVD(n_components=300, random_state=42))
        ])
    
    # HistGradientBoosting requires dense input. 
    # With SVD it's already dense, but explicit 0 threshold is safe.
    sparse_threshold = 0 if model_name in ['HistGradientBoosting', 'hgb'] else 0.3
    
    # Define Column Selector Patterns
    # 1. Text Only (Steps 1-2)
    if feature_set == 'text_only':
        preprocessor = ColumnTransformer(
            transformers=[('text_pipe', text_pipeline, 'final_text')],
            remainder='drop',
            sparse_threshold=sparse_threshold
        )
        
    # 2. Base Features (Text + Source + Rank) - Used in Steps 3, 4
    # Also used for NB models in Step 5 (Time) as they don't use time.
    elif feature_set == 'base_features':
        # Regex: matches src_... and rank_...
        pattern = r'^(?:src_|rank_).*'
        preprocessor = ColumnTransformer(
            transformers=[
                ('text_pipe', text_pipeline, 'final_text'),
                ('dense', 'passthrough', make_column_selector(pattern=pattern))
            ],
            remainder='drop',
            sparse_threshold=sparse_threshold
        )
        
    # 3. Cyclical Time (Base + Sin/Cos) - Used for LR/SVC in Step 5
    elif feature_set == 'time_cyclical':
        # Regex: src_, rank_, and time (sin/cos/year/missing)
        pattern = r'^(?:src_|rank_|.*_sin|.*_cos|year_offset|is_missing_date).*'
        preprocessor = ColumnTransformer(
            transformers=[
                ('text_pipe', text_pipeline, 'final_text'),
                ('dense', 'passthrough', make_column_selector(pattern=pattern))
            ],
            remainder='drop',
            sparse_threshold=sparse_threshold
        )
        
    # 4. Raw Time (Base + Integers) - Used for HGB in Step 5
    elif feature_set == 'time_raw':
        pattern = r'^(?:src_|rank_|hour|day_of_week|month|quarter|year_offset|is_missing_date).*'
        preprocessor = ColumnTransformer(
            transformers=[
                ('text_pipe', text_pipeline, 'final_text'),
                ('dense', 'passthrough', make_column_selector(pattern=pattern))
            ],
            remainder='drop',
            sparse_threshold=sparse_threshold
        )
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    return Pipeline([
        ('prep', preprocessor),
        ('clf', model)
    ])

def evaluate(pipeline, X, y, cv_strategy='Stratified'):
    if cv_strategy == 'Stratified':
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    else:
        raise ValueError(f"Unknown CV strategy: {cv_strategy}")
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    return np.mean(scores)

def run_ablation():
    import os
    set_global_seed(42)
    
    os.makedirs('results/ablation', exist_ok=True)
    
    print("="*120)
    print("ABLATION STUDY: [LinearSVC, MNB, CNB, LR, HGB]")
    print(f"Comparison: StratifiedKFold (3 Splits)")
    print("="*120)
    
    all_results = []
    
    # Load Initial Data
    df_raw = pd.read_csv("./dataset/development.csv")
    
    # Define models
    models = get_models()
    
    # Helper to run a step
    def run_step(df_current, step_name, feature_mode_map):
        """
        feature_mode_map: dict mapping model_name -> feature_set string.
        If 'default' key exists, it applies to unlisted models.
        """
        print(f"\n--- {step_name} ---")
        
        step_results = {'Step': step_name}
        
        # We run only Stratified
        for cv_name in ['Stratified']:
            print(f"  CV: {cv_name}")
            for name, model in models.items():
                # Determine feature set for this model
                fset = feature_mode_map.get(name, feature_mode_map.get('default', 'text_only'))
                
                
                target_df = df_current
                if isinstance(df_current, dict):
                    target_df = df_current.get(name, df_current.get('default'))
                
                # Construct Pipeline
                pipe = get_pipeline(name, model, fset)
                
                # Create 'final_text' if missing (Step 1 handling)
                if 'final_text' not in target_df.columns:
                    df_temp = target_df.copy()
                    df_temp['title'] = df_temp['title'].fillna('')
                    df_temp['article'] = df_temp['article'].fillna('')
                    df_temp['final_text'] = df_temp['title'] + " " + df_temp['article']
                    score = evaluate(pipe, df_temp, df_temp['label'], cv_name)
                else:
                    score = evaluate(pipe, target_df, target_df['label'], cv_name)
                
                print(f"    {name:<25}: {score:.4f}")
                step_results[f"{name}_{cv_name}"] = score
        
        all_results.append(step_results)

    # Baseline (Raw Text)
    # Use default text_only
    run_step(df_raw, "1. Baseline (Raw)", {'default': 'text_only'})

    # Step 2: + Dataset Cleaner & Deduplicator
    cleaner = DatasetCleaner(verbose=False)
    dedup = DatasetDeduplicator(mode='advanced', verbose=False) # Assuming 'advanced' based on previous script
    
    df_step2 = cleaner.fit_transform(df_raw)
    df_step2 = dedup.fit_transform(df_step2)
    
    run_step(df_step2, "2. + Cleaner & Dedup", {'default': 'text_only'})


    # Step 3: + Feature Extractor & Source OHE
    extractor = FeatureExtractor()
    src_trans = SourceTransformer(top_k=300)
    
    df_step3 = extractor.fit_transform(df_step2)
    df_step3 = src_trans.fit_transform(df_step3)
    
    # Feature set becomes 'base_features' (Text + Source)
    run_step(df_step3, "3. + Features & Source", {'default': 'base_features'})

    # Step 4: + Page Rank OHE
    pr_trans = PageRankTransformer(mode='onehot')
    df_step4 = pr_trans.fit_transform(df_step3)
    
    run_step(df_step4, "4. + PageRank", {'default': 'base_features'})

    # Step 5: + Time Features
    
    # Create Cyclical Branch
    time_cyc = TimeExtractor()
    df_cyclical = time_cyc.fit_transform(df_step4)
    
    # Create Raw Branch
    time_raw = RawTimeExtractor()
    df_raw_features = time_raw.fit_transform(df_step4)
    
    # Dictionary to route DFs to models
    df_map = {
        'LinearSVC': df_cyclical,
        'LogisticRegression': df_cyclical,
        'HistGradientBoosting': df_raw_features,
        'MultinomialNB': df_step4,
        'ComplementNB': df_step4,
        'default': df_step4
    }
    
    # Feature Map
    feat_map = {
        'LinearSVC': 'time_cyclical',
        'LogisticRegression': 'time_cyclical',
        'HistGradientBoosting': 'time_raw',
        'MultinomialNB': 'base_features',
        'ComplementNB': 'base_features'
    }
    
    run_step(df_map, "5. + Time Features", feat_map)
    
    # Save Output
    results_df = pd.DataFrame(all_results)
    cols = ['Step'] + [c for c in results_df.columns if c != 'Step']
    results_df = results_df[cols]
    
    results_df.to_csv('results/ablation/ablation_results.csv', index=False)
    print("\nResults saved to results/ablation/ablation_results.csv")

if __name__ == "__main__":
    run_ablation()
