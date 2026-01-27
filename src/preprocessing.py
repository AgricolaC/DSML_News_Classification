import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import ftfy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

class DatasetCleaner(BaseEstimator, TransformerMixin):
    """
    Handles cleaning and standardization of raw dataset columns.
    
    Parameters:
    ----------
    text_cols : list of str, default=['source', 'title', 'article']
        List of columns containing text data to clean (strip whitespace, remove artifacts).
    date_col : str, default='timestamp'
        Name of the column containing timestamp data to be coerced to datetime.
    rank_col : str, default='page_rank'
        Name of the column containing PageRank data to be coerced to numeric.
    verbose : bool, default=False
        If True, prints a report of uncovered missing values after cleaning.
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
    Handles duplicate removal with configurable strategies for ablation studies.
    
    Parameters:
    ----------
    mode : {'advanced', 'simple', 'none'}, default='advanced'
        - 'none': Retains all duplicates (Base case).
        - 'simple': Drops content duplicates based on file order (Naïve approach).
        - 'advanced': Prioritizes specific labels over generic ones, resolves conflicts, 
                      and keeps earliest timestamps (Smart approach).
    verbose : bool, default=True
        If True, prints statistics about dropped rows.
    """
    def __init__(self, mode='advanced', verbose=True):
        self.mode = mode
        self.verbose = verbose
        self.content_cols = ['title', 'article', 'source']
        
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
        # Custom logic: Specific > Generic, Conflict Resolution, Time sort.
        if self.mode == 'advanced':
            if 'label' not in X.columns:
                if self.verbose:
                    print("[DatasetDeduplicator] No 'label' column found. Skipping deduplication")
                return X
            
            # Sort Priority (Specific Label > Generic Label > Timestamp)
            X['is_generic_label'] = (X['label'] == 5)
            X = X.sort_values(
                by=['title', 'is_generic_label', 'timestamp'], 
                ascending=[True, True, True]
            )
            
            has_specific_label_mask = X['label'] != 5
            rows_with_specific_labels = X[has_specific_label_mask]
            
            label_counts = rows_with_specific_labels.groupby(self.content_cols)['label'].nunique()
            content_with_contradictions = label_counts[label_counts > 1].index
        
            if len(content_with_contradictions) > 0:
                if self.verbose:
                    print(f"[DatasetDeduplicator] Dropping {len(content_with_contradictions)} articles with impossible label conflicts.")
                X = X.set_index(self.content_cols).drop(index=content_with_contradictions).reset_index()    
            
            # Final Deduplication
            X = X.drop_duplicates(subset=self.content_cols, keep='first')
            X = X.drop(columns=['is_generic_label'])

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
        
        # 1. Source Tokenization (Lowercase, Unknown handling)
        X['source'] = X['source'].fillna('')
        source_tokens = X['source'].apply(self._tokenize_source)
        
        # 2. Handle Text Missingness with explicit tokens
        X['title'] = X['title'].fillna('title_unknown').replace('', 'title_unknown')
        X['article'] = X['article'].fillna('article_unknown').replace('', 'article_unknown')
        
        # 3. Concatenate
        X['final_text'] = source_tokens + " " + X['title'] + " " + X['article']
        return X

    def _tokenize_source(self, text):
        if not text:
            return "src_unknown"
        # Standardize: Lowercase + Alphanumeric
        clean = re.sub(r'[^a-z0-9]', '', str(text).lower())
        if not clean:
            return "src_unknown"
        return f"src_{clean}"

class TimeExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts cyclical temporal features from a timestamp column.
    
    Parameters:
    ----------
    date_col : str, default='timestamp'
        Name of the timestamp column to extract features from.
    """
    def __init__(self, date_col='timestamp'):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Ensure it is datetime
        dates = pd.to_datetime(X[self.date_col], errors='coerce')
        
        # Hour
        hours = dates.dt.hour
        X['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        X['hour_cos'] = np.cos(2 * np.pi * hours / 24)

        # Day of Week (0=Monday, 6=Sunday)
        dow = dates.dt.dayofweek
        X['day_sin'] = np.sin(2 * np.pi * dow / 7)
        X['day_cos'] = np.cos(2 * np.pi * dow / 7)

        # Helper for missing dates
        X['is_missing_date'] = dates.isna().astype(int)

        # Week (1-53)
        week = dates.dt.isocalendar().week.astype(float)
        X['week_sin'] = np.sin(2 * np.pi * week / 53)
        X['week_cos'] = np.cos(2 * np.pi * week / 53)

        # Month
        month = dates.dt.month
        X['month_sin'] = np.sin(2 * np.pi * month / 12)
        X['month_cos'] = np.cos(2 * np.pi * month / 12)

        valid_mask = dates.notna()
        
        cols_to_fill = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'week_sin', 'week_cos']
        X[cols_to_fill] = X[cols_to_fill].fillna(0)

        X = X.drop(columns=[self.date_col])
        return X

class AdvancedTextCleaner(BaseEstimator, TransformerMixin):
    """
    Performs text cleaning and normalization.
    
    Parameters:
    ----------
    text_col : str, default='final_text'
        Name of the column containing text to clean.
    verbose : bool, default=False
        If True, prints progress of the cleaning operation.
    """
    def __init__(self, text_col='final_text', verbose=False):
        self.text_col = text_col
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.text_col in X.columns:
            if self.verbose:
                print("[AdvancedTextCleaner] Starting text cleaning (URL Extraction + HTML Strip)...")
            X[self.text_col] = X[self.text_col].astype(str).apply(self._clean_text)
        return X

    def _clean_text(self, text):
        if pd.isna(text): return ""
        text = str(text)
        
        # 1. Extract URLs to keep the fingerprint
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        
        # 2. Strip HTML Tags completely
        clean_text = re.sub(r'<[^>]+>', ' ', text)
        
        # 3. Append extracted URLs to the end
        final_content = clean_text + " " + " ".join(urls)
        
        # 4. Normalize Whitespace and fix safe text
        final_content = ftfy.fix_text(final_content)
        return re.sub(r'\s+', ' ', final_content).strip()

class PageRankOneHot(BaseEstimator, TransformerMixin):
    """
    One-Hot encodes the PageRank column (1-5), treating it as categorical.
    """
    def __init__(self, rank_col='page_rank'):
        self.rank_col = rank_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Ensure numeric first (handled by DatasetCleaner, but safe to double check)
        ranks = pd.to_numeric(X[self.rank_col], errors='coerce').fillna(0).astype(int)
        
        # Create Dummy Columns for ranks 1-5
        # We process manually to ensure all columns exist even if a rank is missing in a specific batch
        for r in range(1, 6):
            X[f'rank_{r}'] = (ranks == r).astype(int)
            
        # (Rank 0/NaN becomes all zeros)
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
            
        # Mask everything else as 'Other'
        X['source_clean'] = X['source'].where(X['source'].isin(self.top_sources_), 'Other')
        
        # One-Hot Encode
        dummies = pd.get_dummies(X['source_clean'], prefix='src')
        
        # Ensure we have consistent columns (align with fitted top_sources + Other)
        expected_cols = [f'src_{s}' for s in self.top_sources_] + ['src_Other']
        
        for col in expected_cols:
            if col not in dummies.columns:
                dummies[col] = 0
                
        # Filter to only expected columns to avoid train/test mismatch
        dummies = dummies[expected_cols]
        
        X = pd.concat([X, dummies], axis=1)
        X = X.drop(columns=['source', 'source_clean'])
        return X
