import pandas as pd
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut

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


# Trainingsdatensplit nach der "Holdout Method": 80% Trainingsdaten und 20% Testdaten
def holdoutMethod(data):
    X = data.drop(['price'], axis=1)
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    splits = []
    splits.append((X_train, X_test, y_train, y_test))

    return splits

# Trainingsdatensplit nach "k-cross-validation": Einteilung in 5 gleich große subsets, jedes subset ist einmal der Testdaten subset
def k_cross_validation(data):
    X = data.drop(['price'], axis=1)
    y = data['price']

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    splits = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    splits.append((X_train, X_test, y_train, y_test))

    return splits


# Trainingsdatensplit nach "leave-one-out": Trainingsdaten alle bis auf Einer. Jeder Datenpunkt ist einmal Testdatenpunkt
def leave_one_out(data):
    X = data.drop(['price'], axis=1)
    y = data['price']

    loo = LeaveOneOut()

    splits = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    splits.append((X_train, X_test, y_train, y_test))

    return splits

import pandas as pd

def unify_splits_to_dataframes(splits, target_col='price'):
    """
    Ensures we always get a list of (train_df, eval_df) tuples.

    'splits' can be:
      - A single tuple of length 2: (train_df, eval_df)
      - A single tuple of length 4: (X_train, X_test, y_train, y_test)
      - A list of such tuples (for k-fold, leave-one-out, etc.)

    Returns: List of (train_df, eval_df)
    """

    # If the splitting function returned a single tuple, wrap it in a list
    if isinstance(splits, tuple):
        splits = [splits]

    final_splits = []
    for s in splits:
        if len(s) == 2:
            # We assume it is (train_df, eval_df) directly
            train_df, eval_df = s
            final_splits.append((train_df, eval_df))

        elif len(s) == 4:
            # We assume it is (X_train, X_test, y_train, y_test)
            X_train, X_test, y_train, y_test = s

            # Merge X and y back into DataFrames
            train_df = pd.concat([X_train, y_train.rename(target_col)], axis=1)
            eval_df = pd.concat([X_test,  y_test.rename(target_col)],  axis=1)
            final_splits.append((train_df, eval_df))

        else:
            raise ValueError(
                f"Unsupported split format with length={len(s)}. "
                f"Expected 2 or 4 elements in the tuple."
            )

    return final_splits
