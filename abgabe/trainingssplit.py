import pandas as pd
from sklearn.model_selection import train_test_split

def random_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits DataFrame into train and evaluation sets randomly
    using a specified test_size.
    Returns (train_df, eval_df).
    """
    train_df, eval_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, eval_df

def sequential_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Example of a sequential or 'time-based' split (if your data
    is sorted in some chronological order) or a simple top-slice approach.
    Returns (train_df, eval_df).
    """
    cutoff = int(len(df) * train_ratio)
    train_df = df.iloc[:cutoff].copy()
    eval_df = df.iloc[cutoff:].copy()
    return train_df, eval_df
