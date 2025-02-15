import pandas as pd
import numpy as np

def remove_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Removes rows from the DataFrame that have any feature with a Z-score
    above a certain threshold (default=3.0). This is a simple outlier removal approach.
    Returns a new DataFrame without outliers.
    """
    # Select only numeric columns for outlier checking
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Compute Z-scores
    df_numeric = df[numeric_cols].copy()
    z_scores = (df_numeric - df_numeric.mean()) / df_numeric.std()

    # Keep rows where all absolute Z-scores are below the threshold
    mask = (z_scores.abs() < z_thresh).all(axis=1)
    df_clean = df[mask].copy()
    return df_clean

def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example data augmentation: create new columns or transformations.
    For instance: price_per_area = price / area
    Returns a new DataFrame with augmented data.
    """
    df_aug = df.copy()
    #if 'price' in df.columns and 'area' in df.columns:
    #    df_aug['price_per_area'] = df_aug['price'] / df_aug['area']
    # You can add more transformations as needed
    return df_aug

def remove_low_correlation_cols(df: pd.DataFrame, target_col: str = 'price', threshold: float = 0.1) -> pd.DataFrame:
    """
    Calculates the correlation of each feature with the target_col.
    Removes columns that have a correlation (in absolute value)
    less than the specified threshold.
    """
    # Ensure target_col is in the DataFrame
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    corr_matrix = df.corr(numeric_only=True)
    target_corr = corr_matrix[target_col].drop(labels=[target_col])  # correlation of all features with price

    # Keep columns that meet the threshold
    cols_to_keep = target_corr[abs(target_corr) >= threshold].index.tolist()
    # Always keep the target col
    cols_to_keep.append(target_col)

    # Also keep any non-numeric columns if you want them to remain
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    cols_to_keep = list(set(cols_to_keep + list(non_numeric_cols)))

    df_reduced = df[cols_to_keep].copy()
    return df_reduced
