import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Keras / TensorFlow imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from trainingssplit import unify_splits_to_dataframes

# -------------------------------------------------------------------------
# Single-split model training functions
# -------------------------------------------------------------------------
def train_linear_regression(train_df: pd.DataFrame, eval_df: pd.DataFrame, target_col: str = 'price'):
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_eval = eval_df.drop(columns=[target_col])
    y_eval = eval_df[target_col]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)
    metrics = {
        'MSE': mean_squared_error(y_eval, y_pred),
        'AMSE': np.sqrt(mean_squared_error(y_eval, y_pred)),
        'R2': r2_score(y_eval, y_pred)
    }

    return model, metrics

def train_decision_tree(train_df: pd.DataFrame, eval_df: pd.DataFrame,
                        target_col: str = 'price', max_depth: int = 4):
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_eval = eval_df.drop(columns=[target_col])
    y_eval = eval_df[target_col]

    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)
    metrics = {
        'MSE': mean_squared_error(y_eval, y_pred),
        'AMSE': np.sqrt(mean_squared_error(y_eval, y_pred)),
        'R2': r2_score(y_eval, y_pred)
    }

    return model, metrics

def train_neural_network(train_df: pd.DataFrame, eval_df: pd.DataFrame, target_col: str = 'price'):
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_eval = eval_df.drop(columns=[target_col])
    y_eval = eval_df[target_col]

    model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=20000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)
    metrics = {
        'MSE': mean_squared_error(y_eval, y_pred),
        'R2': r2_score(y_eval, y_pred)
    }

    return model, metrics

def train_keras_network(
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        target_col: str = 'price',
        epochs: int = 1000,
        batch_size: int = 16
):
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values

    X_eval = eval_df.drop(columns=[target_col])
    y_eval = eval_df[target_col].values

    # Convert all features to float (in case of leftover non-numeric)
    X_train = X_train.select_dtypes(include=[np.number]).astype(float)
    X_eval  = X_eval.select_dtypes(include=[np.number]).astype(float)

    # Build Keras model
    n_features = X_train.shape[1]
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(n_features,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Single output for regression

    model.compile(
        optimizer=Adam(learning_rate=0.002),
        loss='mean_squared_error',
        metrics=['mae']
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0  # set to 1 if you want training output
    )

    y_pred = model.predict(X_eval).flatten()
    mse = mean_squared_error(y_eval, y_pred)
    r2 = r2_score(y_eval, y_pred)

    metrics = {
        'MSE': mse,
        'R2': r2
    }
    return model, metrics, history

# -------------------------------------------------------------------------
# Multi-split wrappers
# -------------------------------------------------------------------------
def train_linear_regression_splits(splits, target_col: str = 'price'):
    unified_splits = unify_splits_to_dataframes(splits, target_col=target_col)
    results = []

    for (train_df, eval_df) in unified_splits:
        model, metrics = train_linear_regression(train_df, eval_df, target_col=target_col)
        results.append((model, metrics))

    if len(results) == 1:
        # Return (model, metrics) directly
        return results[0]
    else:
        # Compute average metrics across splits
        metric_keys = results[0][1].keys()
        avg_metrics = {}
        for k in metric_keys:
            avg_metrics[k] = np.mean([r[1][k] for r in results])
        return results, avg_metrics

def train_decision_tree_splits(splits, target_col: str = 'price', max_depth=5):
    unified_splits = unify_splits_to_dataframes(splits, target_col=target_col)
    results = []

    for (train_df, eval_df) in unified_splits:
        model, metrics = train_decision_tree(train_df, eval_df, target_col=target_col, max_depth=max_depth)
        results.append((model, metrics))

    if len(results) == 1:
        return results[0]
    else:
        metric_keys = results[0][1].keys()
        avg_metrics = {}
        for k in metric_keys:
            avg_metrics[k] = np.mean([r[1][k] for r in results])
        return results, avg_metrics

def train_neural_network_splits(splits, target_col='price'):
    unified_splits = unify_splits_to_dataframes(splits, target_col=target_col)
    results = []

    for (train_df, eval_df) in unified_splits:
        model, metrics = train_neural_network(train_df, eval_df, target_col=target_col)
        results.append((model, metrics))

    if len(results) == 1:
        return results[0]
    else:
        metric_keys = results[0][1].keys()
        avg_metrics = {}
        for k in metric_keys:
            avg_metrics[k] = np.mean([r[1][k] for r in results])
        return results, avg_metrics

def train_keras_network_splits(splits, target_col='price', epochs=1000, batch_size=16):
    unified_splits = unify_splits_to_dataframes(splits, target_col=target_col)
    results = []

    for (train_df, eval_df) in unified_splits:
        model, metrics, history = train_keras_network(
            train_df, eval_df,
            target_col=target_col,
            epochs=epochs,
            batch_size=batch_size
        )
        results.append((model, metrics, history))

    if len(results) == 1:
        return results[0]  # (model, metrics, history)
    else:
        metric_keys = results[0][1].keys()  # results[i] = (model, metrics, history)
        avg_metrics = {}
        for k in metric_keys:
            avg_metrics[k] = np.mean([r[1][k] for r in results])
        return results, avg_metrics
