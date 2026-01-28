import pandas as pd
import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.exceptions import ConvergenceWarning

# Import Preprocessing
from preprocessing import (
    DatasetCleaner, DatasetDeduplicator, FeatureExtractor, 
    TimeExtractor, PageRankOneHot, SourceTransformer, AdvancedTextCleaner
)
from cv_utils import AnchoredTimeSeriesSplit
from seed import set_global_seed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def get_models():
    """Returns the 3 classifiers to test."""
    return {
        'LinearSVC': LinearSVC(C=0.1, class_weight='balanced', random_state=42, dual=False),
        'MultinomialNB': MultinomialNB(alpha=0.1),
        'ComplementNB': ComplementNB(alpha=0.1)
    }

def get_pipeline(model_name, model, feature_set='text_only'):
    """
    Constructs the pipeline based on the model and feature set availability.
    NB models cannot handle negative values (Time Features).
    """
    tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True, lowercase=True)
    
    if feature_set == 'text_only':
        # Simple TF-IDF on 'final_text'
        preprocessor = ColumnTransformer(
            transformers=[('tfidf', tfidf, 'final_text')],
            remainder='drop'
        )
    elif feature_set == 'all_features':
        # Text + Dense
        # Handling Negative Values for NB
        if model_name in ['MultinomialNB', 'ComplementNB']:
            # NB: Exclude Time (sin/cos) features which are negative
            # Select only Text + OHE (Rank/Source/Missing Flags)
            # Regex: is_missing_, rank_, src_
            pattern = r'^(?:is_missing_|rank_|src_).*'
            preprocessor = ColumnTransformer(
                transformers=[
                    ('tfidf', tfidf, 'final_text'),
                    ('dense', 'passthrough', make_column_selector(pattern=pattern))
                ],
                remainder='drop'
            )
        else:
            # SVC: Full Features (Time included)
            # Regex: hour_, day_, month_, week_, is_missing_, rank_, src_
            pattern = r'^(?:hour_|day_|month_|week_|is_missing_|rank_|src_).*'
            preprocessor = ColumnTransformer(
                transformers=[
                    ('tfidf', tfidf, 'final_text'),
                    ('dense', 'passthrough', make_column_selector(pattern=pattern))
                ],
                remainder='drop'
            )
            
    return Pipeline([
        ('prep', preprocessor),
        ('clf', model)
    ])

def evaluate(pipeline, X, y, state_name, model_name):
    # Standard Stratified K-Fold
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    return np.mean(scores)

def run_ablation():
    set_global_seed(42)
    print("="*100)
    print("INCREMENTAL ABLATION STUDY: [CMB, MNB, LinearSVC]")
    print(f"{'State':<35} | {'MNB':<10} | {'CMB':<10} | {'LinearSVC':<10}")
    print("="*100)
    
    # 0. Load Raw
    df = pd.read_csv("./dataset/development.csv")
    
    # Helper to print row
    def run_step(df_current, step_name, use_all_features=False):
        results = {}
        for name, model in get_models().items():
            feat_set = 'all_features' if use_all_features else 'text_only'
            pipe = get_pipeline(name, model, feat_set)
            
            # Ensure final_text exists (for Step 0, we create it manually from title+article)
            if 'final_text' not in df_current.columns:
                 # Temporary fill for raw step
                df_temp = df_current.copy()
                df_temp['title'] = df_temp['title'].fillna('')
                df_temp['article'] = df_temp['article'].fillna('')
                df_temp['final_text'] = df_temp['title'] + " " + df_temp['article']
                score = evaluate(pipe, df_temp, df_temp['label'], step_name, name)
            else:
                score = evaluate(pipe, df_current, df_current['label'], step_name, name)
            results[name] = score
            
        print(f"{step_name:<35} | {results['MultinomialNB']:.4f}     | {results['ComplementNB']:.4f}     | {results['LinearSVC']:.4f}")

    # ----------------------------------------------------------------
    # State 1: Baseline (Raw Text)
    # ----------------------------------------------------------------
    # Just raw title+article, no cleaning.
    run_step(df, "1. Raw Text (Baseline)")
    
    # ----------------------------------------------------------------
    # State 2: + DatasetCleaner (Basic Chars)
    # ----------------------------------------------------------------
    cleaner = DatasetCleaner(verbose=False)
    df = cleaner.fit_transform(df)
    run_step(df, "2. + DatasetCleaner")

    # ----------------------------------------------------------------
    # State 3: + Deduplication
    # ----------------------------------------------------------------
    dedup = DatasetDeduplicator(mode='advanced', verbose=False)
    df = dedup.fit_transform(df)
    run_step(df, "3. + Deduplicator")

    # ----------------------------------------------------------------
    # State 4: + FeatureExtractor (Missing Tokens)
    # ----------------------------------------------------------------
    # Adds 'final_text' with 'src_unknown', 'title_unknown', and source tokens
    extractor = FeatureExtractor()
    df = extractor.fit_transform(df)
    run_step(df, "4. + FeatureExtractor (Tokens)")

    # ----------------------------------------------------------------
    # State 5: (REMOVED - AdvancedTextCleaner only for explainability)
    # ----------------------------------------------------------------
    # Skipping this state - advanced cleaning not used in production
    # run_step(df, "5. (SKIPPED - AdvTextCleaner)")

    # ----------------------------------------------------------------
    # State 6: + SourceTransformer (Metadata)
    # ----------------------------------------------------------------
    # This adds Dense features. We switch to use_all_features=True
    src_trans = SourceTransformer(top_k=300)
    df = src_trans.fit_transform(df)
    run_step(df, "6. + Source (OHE)", use_all_features=True)

    # ----------------------------------------------------------------
    # State 7: + PageRank (Metadata)
    # ----------------------------------------------------------------
    pr = PageRankOneHot()
    df = pr.fit_transform(df)
    run_step(df, "7. + PageRank (OHE)", use_all_features=True)

    # ----------------------------------------------------------------
    # State 8: + TimeFeatures (Final)
    # ----------------------------------------------------------------
    # Note: NB models will ignore these in get_pipeline/ColumnTransformer
    # SVC will use them.
    time_ext = TimeExtractor()
    df = time_ext.fit_transform(df)
    run_step(df, "8. + Time (Full Pipeline)", use_all_features=True)

if __name__ == "__main__":
    run_ablation()
