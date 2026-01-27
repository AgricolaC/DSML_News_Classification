import pandas as pd
import numpy as np
import json
import time
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from preprocessing import DatasetCleaner, DatasetDeduplicator, FeatureExtractor, TimeExtractor, PageRankOneHot, SourceTransformer
from cv_utils import AnchoredTimeSeriesSplit
from seed import set_global_seed
from sklearn.compose import make_column_selector

def run_hgb_tuning():
    set_global_seed(42)
    print("="*60)
    print("AGGRESSIVE HGB TUNING (SVD=400)")
    print("="*60)
    
    # 1. Load Data
    df = pd.read_csv("./dataset/development.csv")
    
    # 2. Advanced Preprocessing (Matching ensemble.py)
    # Cleaning & Dedup
    df = DatasetCleaner(verbose=False).transform(df)
    df = DatasetDeduplicator(mode='advanced', verbose=False).transform(df)
    
    # Feature Engineering
    df = FeatureExtractor().transform(df)
    df = SourceTransformer(top_k=200).fit_transform(df) # Fit on dev set
    df = PageRankOneHot().transform(df)
    
    # Handle Time (Preserve timestamp for CV)
    timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
    df = TimeExtractor().transform(df)
    df['timestamp'] = timestamps 
    
    y = df['label']
    X = df # Pass full DF to pipeline
    
    # 3. Define Pipeline with Dense Features
    # Tfidf args (Best so far)
    tfidf_args = {'ngram_range': (1, 3), 'min_df': 2, 'sublinear_tf': True, 'max_features': 30000}
    
    # Steps for Text: Vectorize -> SVD (400)
    txt_steps = [
        ('vec', TfidfVectorizer(**tfidf_args)),
        ('svd', TruncatedSVD(n_components=400, random_state=42))
    ]
    txt_pipe = Pipeline(txt_steps)
    
    # Column Transformer
    # Combines SVD-Text + Dense Features (Time/Week/Rank/Source)
    # Using the REGEX selector to pick up all new features
    preprocessor = ColumnTransformer(
        transformers=[
            ('txt', txt_pipe, 'final_text'),
            ('dense', 'passthrough', make_column_selector(pattern=r'^(?:hour_|day_|month_|week_|is_missing_|rank_|src_).*'))
        ],
        remainder='drop'
    )
    
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', HistGradientBoostingClassifier(class_weight='balanced', random_state=42))
    ])
    
    # 4. QGrid
    param_grid = {
        'clf__learning_rate': [0.1],
        'clf__max_depth': [10,50,None],
        'clf__l2_regularization': [1.0,5.0,10.0],
        'clf__max_leaf_nodes': [63]
    }
    
    print(f"Grid Size: {np.prod([len(v) for v in param_grid.values()])} candidates")
    
    # 5. Run GridSearch
    cv = AnchoredTimeSeriesSplit(df, n_splits=2) # Use 3 splits for speed
    
    gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=2)
    gs.fit(X, y)
    
    print("\nBest Parameters:")
    print(json.dumps(gs.best_params_, indent=4))
    print(f"Best Score: {gs.best_score_:.4f}")
    
    # Save partial result (merging with existing best_params if possible, or just print)
    # For now, just print.
    
if __name__ == "__main__":
    run_hgb_tuning()
