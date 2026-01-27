import numpy as np
import random
import os

def set_global_seed(seed=42):
    """
    Sets random seeds for Python, NumPy, and environment for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[Reproducibility] Global Seed set to {seed}")