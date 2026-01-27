from sklearn.model_selection import TimeSeriesSplit
import numpy as np

class AnchoredTimeSeriesSplit:
    """
    Custom Cross-Validation iterator.
    - Anchors 'NaT' missing date indices in EVERY training fold.
    - Performs TimeSeriesSplit (expanding window) on dated indices.
    """
    def __init__(self, df, n_splits=5):
        self.df = df
        self.n_splits = n_splits
        
    def split(self, X, y=None, groups=None):
        
        if not hasattr(X, 'iloc'): # check if dataframe
            raise ValueError("X must be a pandas DataFrame containing 'timestamp'")
            
        mask_nat = X['timestamp'].isna()
        nat_indices = np.where(mask_nat)[0]
        valid_indices = np.where(~mask_nat)[0]
         
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        for train_valid_idx, test_valid_idx in tscv.split(valid_indices):
            # Map back to original X indices
            train_valid_original_idx = valid_indices[train_valid_idx]
            test_valid_original_idx = valid_indices[test_valid_idx]
            
            # Combine NaT + Valid Train
            train_idx = np.concatenate([nat_indices, train_valid_original_idx])
            test_idx = test_valid_original_idx
            
            yield train_idx, test_idx
            
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
