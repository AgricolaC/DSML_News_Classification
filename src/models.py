import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import FunctionTransformer
from .preprocessing import SourceTransformer, RawTimeExtractor

def to_dense(x):
    return x.toarray()

# Load hyperparameters
def load_best_params(path='best_params.json'):
    """Load hyperparameters from JSON file."""
    try:
        with open(path, 'r') as f:
            params = json.load(f)
        if 'tfidf' in params and 'ngram_range' in params['tfidf']:
            if isinstance(params['tfidf']['ngram_range'], list):
                params['tfidf']['ngram_range'] = tuple(params['tfidf']['ngram_range'])
        
        return params
    except FileNotFoundError:
        print(f"Warning: {path} not found. Using defaults.")
        return get_default_params()

def get_default_params():
    """Default hyperparameters"""
    return {
        'tfidf': {'ngram_range': (1, 3), 'min_df': 2, 'sublinear_tf': True, 'max_features': 30000, 'lowercase': True},
        'svc': {'clf__C': 0.1},
        'lr': {'clf__C': 1.0, 'clf__penalty': 'l2'},
        'hgbc': {'clf__learning_rate': 0.1, 'clf__max_depth': 10, 'clf__l2_regularization': 10.0, 'clf__max_leaf_nodes': 63}
    }

def make_column_transformer(steps, feature_pattern=r'^(?:hour_|day_|month_|week_|is_missing_|rank_|src_).*'):
    """
    Factory for creating ColumnTransformer.
    """
    return ColumnTransformer(
        transformers=[
            ('txt', Pipeline(steps), 'final_text'),
            ('dense', 'passthrough', make_column_selector(pattern=feature_pattern))
        ],
        remainder='drop'
    )

def get_base_models():
    """
    Get a dictionary of base model instances.
    """
    params = load_best_params()
    return {
        'svc': LinearSVC(class_weight='balanced', random_state=42, dual=False, C=0.1),
        'lr': LogisticRegression(class_weight='balanced', solver='saga', random_state=42, max_iter=2500),
        'hgb': HistGradientBoostingClassifier(class_weight='balanced', random_state=42),
        'mnb': MultinomialNB(alpha=0.1),
        'cnb': ComplementNB(alpha=0.01)
    }

def get_pipelines(models=None, params=None):
    """
    Build sklearn pipelines for specified models.
    """
    
    params = load_best_params()
    
    if models is None:
        models = ['svc', 'lr', 'hgb', 'mnb', 'cnb']
    
    tfidf_args = params.get('tfidf', {})
    pipelines = []
    
    if 'svc' in models:
        svc_steps = [('vec', TfidfVectorizer(**tfidf_args))]
        svc_base = LinearSVC(class_weight='balanced', random_state=42, dual=False)
        svc_params = {k.replace('clf__', ''): v for k, v in params.get('svc', {}).items() if k.startswith('clf__')}
        svc_base.set_params(**svc_params)
        
        pipe_svc = Pipeline([
            ('src_gen', SourceTransformer(top_k=300)),
            ('prep', make_column_transformer(svc_steps, feature_pattern=r'^(?:hour_|day_|week_|month_|quarter_|year_|is_missing_|rank_|src_).*')),
            ('clf', svc_base)
        ], memory='.cache')
        pipelines.append(('svc', pipe_svc))
    
    if 'lr' in models:
        lr_steps = [('vec', TfidfVectorizer(**tfidf_args))]
        pipe_lr = Pipeline([
            ('src_gen', SourceTransformer(top_k=300)),
            ('prep', make_column_transformer(lr_steps, feature_pattern=r'^(?:is_missing_|rank_|src_).*')),
            ('clf', LogisticRegression(class_weight='balanced', solver='saga', random_state=42, max_iter=2500))
        ], memory='.cache')
        pipe_lr.set_params(**params.get('lr', {}))
        pipelines.append(('lr', pipe_lr))
    
    if 'hgb' in models:
        hgb_steps = [
            ('vec', TfidfVectorizer(**tfidf_args)),
            ('sel', SelectKBest(chi2, k=5000)),
            ('svd', TruncatedSVD(n_components=300, random_state=42))
        ]
        hgb_ct = make_column_transformer(
            hgb_steps,
            feature_pattern=r'^(?:raw_hour|raw_day_of_week|raw_month|raw_quarter|raw_year_offset|is_missing_|rank_|src_).*'
        )
        pipe_hgb = Pipeline([
            ('src_gen', SourceTransformer(top_k=300)),
            ('prep', hgb_ct),
            ('clf', HistGradientBoostingClassifier(class_weight='balanced', random_state=42))
        ], memory='.cache')
        pipe_hgb.set_params(**params.get('hgbc', {}))
        pipelines.append(('hgb', pipe_hgb))
    
    # MultinomialNB Pipeline (Text + OHE only, no time features)
    if 'mnb' in models:
        mnb_steps = [('vec', TfidfVectorizer(**tfidf_args))]
        mnb_ct = make_column_transformer(
            mnb_steps,
            feature_pattern=r'^(?:is_missing_|src_).*'  
        )
        pipe_mnb = Pipeline([
            ('src_gen', SourceTransformer(top_k=300)),
            ('prep', mnb_ct),
            ('clf', MultinomialNB(alpha=0.1))
        ], memory='.cache')
        pipelines.append(('mnb', pipe_mnb))
    
    # ComplementNB Pipeline (same features as MNB)
    if 'cnb' in models:
        cnb_steps = [('vec', TfidfVectorizer(**tfidf_args))]
        cnb_ct = make_column_transformer(
            cnb_steps,
            feature_pattern=r'^(?:is_missing_|src_).*' 
        )
        pipe_cnb = Pipeline([
            ('src_gen', SourceTransformer(top_k=300)),
            ('prep', cnb_ct),
            ('clf', ComplementNB(alpha=0.1))
        ], memory='.cache')
        pipelines.append(('cnb', pipe_cnb))
    
    return pipelines
