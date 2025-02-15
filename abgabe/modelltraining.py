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

def train_keras_network(
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        target_col: str = 'price',
        epochs: int = 1000,
        batch_size: int = 16
):
    """
    Build and train a Keras Sequential model to predict `target_col`.
    Returns the trained model and a dict containing MSE and R^2 on the eval set.
    """
    print(train_df)

    # -------------------------------------------------------------------------
    # 1. Split the DataFrame into features (X) and target (y)
    # -------------------------------------------------------------------------
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values

    X_eval = eval_df.drop(columns=[target_col])
    y_eval = eval_df[target_col].values

    # Force float or drop non-numeric columns
    X_train = X_train.select_dtypes(include=[np.number])
    X_train = X_train.astype(float)

    X_eval = eval_df.drop(columns=[target_col])
    y_eval = eval_df[target_col]

    X_eval = X_eval.select_dtypes(include=[np.number])
    X_eval = X_eval.astype(float)

    # -------------------------------------------------------------------------
    # 2. Define the Keras model architecture
    # -------------------------------------------------------------------------
    n_features = X_train.shape[1]

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(n_features,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Single output for regression

    # -------------------------------------------------------------------------
    # 3. Compile the model (choose optimizer, loss, and optional metrics)
    # -------------------------------------------------------------------------
    model.compile(
        optimizer=Adam(learning_rate=0.002),
        loss='mean_squared_error',
        metrics=['mae']  # You can track additional metrics if desired
    )

    # -------------------------------------------------------------------------
    # 4. Train the model, with validation split
    #    - If your dataset is large, you might prefer a separate eval set
    #      but here we'll do a quick 10% split from the training data itself
    #      for a "mini-validation."
    #    - The final evaluation with `X_eval` is separate.
    # -------------------------------------------------------------------------
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,  # 10% of train data used for validation
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # -------------------------------------------------------------------------
    # 5. Evaluate the model on the held-out evaluation set
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_eval).flatten()  # Keras returns 2D array, flatten to 1D

    mse = mean_squared_error(y_eval, y_pred)
    r2 = r2_score(y_eval, y_pred)

    metrics = {
        'MSE': mse,
        'R2': r2
    }

    return model, metrics, history
