import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_linear_regression(train_df: pd.DataFrame, eval_df: pd.DataFrame, target_col: str = 'price'):
    """
    Train a Linear Regression model and evaluate it on the eval set.
    Returns the trained model and a dictionary of evaluation metrics.
    """
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

def train_decision_tree(train_df: pd.DataFrame, eval_df: pd.DataFrame, target_col: str = 'price', max_depth: int = 5):
    """
    Train a Decision Tree Regressor and evaluate it.
    Returns the trained model and a dictionary of evaluation metrics.
    """
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
    """
    Train a simple Neural Network (MLPRegressor) and evaluate it.
    Returns the trained model and a dictionary of evaluation metrics.
    """
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_eval = eval_df.drop(columns=[target_col])
    y_eval = eval_df[target_col]

    model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)
    metrics = {
        'MSE': mean_squared_error(y_eval, y_pred),
        'R2': r2_score(y_eval, y_pred)
    }

    return model, metrics
