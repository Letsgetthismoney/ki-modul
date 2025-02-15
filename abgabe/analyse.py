import pandas as pd

from abgabe.normalisierung import normalize_data
from trainingssplit import holdoutMethod, k_cross_validation
import modelltraining

df = normalize_data('Housing.csv')

# Suppose we want to do the holdout method
splits_holdout = holdoutMethod(df)
model, metrics = modelltraining.train_linear_regression_splits(splits_holdout)
print("Holdout method metrics:", metrics)

# Suppose we want to do 5-fold cross validation
splits_kfold = k_cross_validation(df)
results, avg_metrics = modelltraining.train_linear_regression_splits(splits_kfold)
print("K-Fold average metrics:", avg_metrics)