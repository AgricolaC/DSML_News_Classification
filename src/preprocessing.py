import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
import ftfy

class DatasetCleaner(BaseEstimator, TransformerMixin):
    r"""
    This is the entry point for raw data cleaning. We handle type conversion and 
    remove artifacts like "\N" or "\\" before any feature engineering happens. 
    We also use the 'ftfy' library that helps fix encoding errors in text.
    """

    def __init__(self, text_cols=['source', 'title', 'article'], 
                 date_col='timestamp', rank_col='page_rank', verbose=False):
        self.text_cols = text_cols
        self.date_col = date_col
        self.rank_col = rank_col
        self.artifacts = ['\\N', '\\', 'nan', 'NULL', '']
        self.verbose = verbose

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        X = X.copy()
        
        if self.verbose:
            before_stats = X[self.text_cols + [self.date_col, self.rank_col]].isnull().sum()
            print("-" * 40)
            print("[DatasetCleaner] Starting Cleaning Process...")
        
        for col in self.text_cols:
            if col in X.columns:
                X[col] = X[col].replace(self.artifacts, np.nan)
                X[col] = X[col].replace('', np.nan)
                X[col] = X[col].apply(lambda x: ftfy.fix_text(str(x)) if pd.notna(x) else x).str.strip()

        if self.date_col in X.columns:
            X[self.date_col] = pd.to_datetime(X[self.date_col], errors='coerce')

        if self.rank_col in X.columns:
            X[self.rank_col] = pd.to_numeric(X[self.rank_col], errors='coerce')

        if self.verbose:
            after_stats = X[self.text_cols + [self.date_col, self.rank_col]].isnull().sum()
            summary = pd.DataFrame({
                'Missing (Raw)': before_stats,
                'Missing (Clean)': after_stats,
                'Uncovered': after_stats - before_stats
            })
            display_mask = (summary['Missing (Clean)'] > 0) | (summary['Uncovered'] > 0)
            
            print(f"[DatasetCleaner] Report: Uncovered {summary['Uncovered'].sum()} hidden missing values.")
            if display_mask.any():
                print(summary[display_mask])
            else:
                print("No missing values found.")
            print("-" * 40)
        return X


class DatasetDeduplicator(BaseEstimator, TransformerMixin):
    """
    Removes duplicate rows based on a configurable strategy.

    Mode 'none' returns the dataset as is

    Mode 'simple' drops duplicates, keeping the first occurrence.

    Mode 'advanced' does prioritization & conflict resolution. This strategy handles cases
    where the same article appears multiple times but with different labels. We prioritize specific labels over generic labels
    and remove confusing articles that claim to be two different specific things.
    """
    def __init__(self, mode='advanced', verbose=True):
        self.mode = mode
        self.verbose = verbose
        self.content_cols = ['title', 'article']
        
        # Validate mode
        valid_modes = ['advanced', 'simple', 'none']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of {valid_modes}")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        initial_count = len(X)
        
        # Mode 1: No Deduplication
        if self.mode == 'none':
            if self.verbose:
                print("[DatasetDeduplicator] Mode='none': No rows removed.")
            return X
            
        # Mode 2: Simple Deduplication
        # just keep the first occurrence found in the file.
        if self.mode == 'simple':
            X = X.drop_duplicates(subset=self.content_cols, keep='first')
            if self.verbose:
                print(f"[DatasetDeduplicator] Mode='simple': Removed {initial_count - len(X)} rows based on file order.")
            return X

        # Mode 3: Advanced Deduplication
        # Strict logic: Drop ALL conflicting labels, Keep Earliest timestamp for matches.
        if self.mode == 'advanced':
            if 'label' not in X.columns:
                if self.verbose:
                    print("[DatasetDeduplicator] No 'label' column found. Skipping deduplication")
                return X
            
            # Sort by Content and Time (Earliest first)
            X = X.sort_values(by=['title', 'timestamp'], ascending=[True, True])

            # Group by content -> Check number of unique labels
            label_counts = X.groupby(self.content_cols)['label'].nunique()
            conflicting_content = label_counts[label_counts > 1].index
            
            # Drop Conflicts
            if len(conflicting_content) > 0:
                if self.verbose:
                    print(f"[DatasetDeduplicator] Dropping {len(conflicting_content)} content groups with conflicting labels.")
                X = X.set_index(self.content_cols).drop(index=conflicting_content).reset_index()    
            
            # 4. Deduplicate (Keep Earliest)
            X = X.drop_duplicates(subset=self.content_cols, keep='first')

            if self.verbose:
                print(f"[DatasetDeduplicator] Mode='advanced': Removed {initial_count - len(X)} rows total.")
                
            return X

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Constructs the primary text feature used for classification.
    Combines 'source', 'title', and 'article' into a single 'final_text' column.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # We have to fill NaN with empty strings so the function doesn't crash.
        X['source'] = X['source'].fillna('')
        source_tokens = X['source'].apply(self._tokenize_source)
        
        # Handle Text Missingness with explicit tokens
        X['title'] = X['title'].fillna('title_unknown').replace('', 'title_unknown')
        X['article'] = X['article'].fillna('article_unknown').replace('', 'article_unknown')
        
        # Concatenate
        X['final_text'] = source_tokens + " " + X['title'] + " " + X['article']
        return X

    def _tokenize_source(self, text):
        if not text:
            return "src_unknown"
        clean = re.sub(r'[^a-z0-9]', '', str(text).lower())
        if not clean:
            return "src_unknown"
        return f"src_{clean}"

class TimeExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts cyclical temporal features from a timestamp column.
    """
    def __init__(self, date_col='timestamp'):
        self.date_col = date_col
        self.median_year_ = None  # Fitted on train

    def fit(self, X, y=None):
        dates = pd.to_datetime(X[self.date_col], errors='coerce')
        valid_dates = dates[dates.notna()]
        
        self.median_year_ = int(valid_dates.dt.year.median())
        return self

    def transform(self, X):
        X = X.copy()
        
        dates = pd.to_datetime(X[self.date_col], errors='coerce')
        
        # Hour 
        hours = dates.dt.hour
        X['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        X['hour_cos'] = np.cos(2 * np.pi * hours / 24)

        # Day of Week (0=Monday, 6=Sunday)
        dow = dates.dt.dayofweek
        X['day_sin'] = np.sin(2 * np.pi * dow / 7)
        X['day_cos'] = np.cos(2 * np.pi * dow / 7)

        # Week (1-53)
        week = dates.dt.isocalendar().week.astype(float)
        X['week_sin'] = np.sin(2 * np.pi * week / 53)
        X['week_cos'] = np.cos(2 * np.pi * week / 53)

        # Month (1-12)
        month = dates.dt.month
        X['month_sin'] = np.sin(2 * np.pi * month / 12)
        X['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        # Quarter (1-4, cyclical)
        quarter = dates.dt.quarter
        X['quarter_sin'] = np.sin(2 * np.pi * quarter / 4)
        X['quarter_cos'] = np.cos(2 * np.pi * quarter / 4)
        
        # Year Offset (median-based)
        # Use 0 for missing dates (assumes median period)
        year = dates.dt.year
        X['year_offset'] = ((year - self.median_year_).fillna(0)).astype(int)

        # Missing date indicator. We explicitly want to tell the model that 
        # we don't know the date for this row.
        X['is_missing_date'] = dates.isna().astype(int)

        # Fill NaN values for all time features (for missing timestamps)
        cols_to_fill = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
            'month_sin', 'month_cos', 'week_sin', 'week_cos',
            'quarter_sin', 'quarter_cos'
        ]
        X[cols_to_fill] = X[cols_to_fill].fillna(0)

        X = X.drop(columns=[self.date_col])
        return X

class RawTimeExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts raw temporal features for models with native NaN handling
    
    Unlike TimeExtractor, this keeps NaN values instead of filling them,
    allowing models like HistGradientBoosting to treat missing timestamps
    as a separate category.
    """
    def __init__(self, date_col='timestamp'):
        self.date_col = date_col
        self.median_year_ = None

    def fit(self, X, y=None):
        """Learn median year from training data for symmetric year normalization."""
        dates = pd.to_datetime(X[self.date_col], errors='coerce')
        valid_dates = dates[dates.notna()]
        
        self.median_year_ = int(valid_dates.dt.year.median())
        return self

    def transform(self, X):
        X = X.copy()
        
        # Ensure it is datetime
        dates = pd.to_datetime(X[self.date_col], errors='coerce')
        
        X['hour'] = dates.dt.hour
        X['day_of_week'] = dates.dt.dayofweek
        X['month'] = dates.dt.month
        X['quarter'] = dates.dt.quarter
        
        # Year offset (median-based, keep NaN for missing - HGB handles it)
        year = dates.dt.year
        X['year_offset'] = (year - self.median_year_)
        
        # Missing date indicator (still useful as explicit feature)
        X['is_missing_date'] = dates.isna().astype(int)
        
        X = X.drop(columns=[self.date_col])
        return X

class PageRankTransformer(BaseEstimator, TransformerMixin):
    """
    Encodes the PageRank column.
    
    'onehot': One-Hot encodes (1-5), treating it as categorical.
    'ordinal': Keeps it as a single integer column (0-5).
    """
    def __init__(self, mode='onehot', rank_col='page_rank'):
        self.mode = mode
        self.rank_col = rank_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        ranks = pd.to_numeric(X[self.rank_col], errors='coerce').fillna(0).astype(int)
        
        if self.mode == 'onehot':
            for r in range(1, 6):
                X[f'rank_{r}'] = (ranks == r).astype(int)
        elif self.mode == 'ordinal':
            # Name must start with 'rank_' to be picked up by model regex
            X['rank_ordinal'] = ranks
            
        if self.rank_col in X.columns:
            X = X.drop(columns=[self.rank_col])
            
        return X

class SourceTransformer(BaseEstimator, TransformerMixin):
    """
    Encodes 'source' column:
    - Keeps top K frequent sources.
    - Hashes or marks others as 'Other'.
    - Returns OHE columns.
    """
    def __init__(self, top_k=300):
        self.top_k = top_k
        self.top_sources_ = None

    def fit(self, X, y=None):
        if 'source' in X.columns:
            # Calculate top K sources
            counts = X['source'].value_counts()
            self.top_sources_ = counts.index[:self.top_k].tolist()
        else:
            self.top_sources_ = []
        return self

    def transform(self, X):
        X = X.copy()
        if 'source' not in X.columns:
            return X
            
        X['top_sources'] = X['source'].where(X['source'].isin(self.top_sources_), 'Other')
        
        # One-Hot Encode
        dummies = pd.get_dummies(X['top_sources'], prefix='src')
        
        # align with fitted top_sources + other
        expected_cols = [f'src_{s}' for s in self.top_sources_] + ['src_Other']
        
        # Identify and create missing columns
        missing_cols = [col for col in expected_cols if col not in dummies.columns]
        if missing_cols:
            missing_data = pd.DataFrame(0, index=dummies.index, columns=missing_cols)
            dummies = pd.concat([dummies, missing_data], axis=1)
            
        # Select columns in correct order using a copy 
        dummies = dummies[expected_cols].copy()
        
        X = pd.concat([X, dummies], axis=1)
        X = X.drop(columns=['source', 'top_sources'])
        return X
