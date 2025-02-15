import sys
import argparse

import normalisierung
import aufbereitung
import trainingssplit
import modelltraining

def main(args):
    # -------------------------------------------------------------------------
    # 1. Normalize data
    # -------------------------------------------------------------------------
    df_normalized = normalisierung.normalize_data(args.csv_file_path)

    # -------------------------------------------------------------------------
    # 2. Aufbereitung (preprocessing):
    #    You can call these in any order, nest them, or conditionally.
    # -------------------------------------------------------------------------
    if args.remove_correlation:
        df_normalized = aufbereitung.remove_low_correlation_cols(df_normalized, target_col='price', threshold=0.1)
    if args.augment:
        df_normalized = aufbereitung.augment_data(df_normalized)
    if args.remove_outliers:
        df_normalized = aufbereitung.remove_outliers(df_normalized, z_thresh=3.0)

    # -------------------------------------------------------------------------
    # 3. Trainings split
    #    Depending on the user argument, we pick one of two split methods
    # -------------------------------------------------------------------------
    if args.split_method == 'random':
        train_df, eval_df = trainingssplit.random_split(df_normalized, test_size=0.2)
    else:
        train_df, eval_df = trainingssplit.sequential_split(df_normalized, train_ratio=0.8)

    # -------------------------------------------------------------------------
    # 4. Modelltraining
    #    We can train one or multiple models depending on the arguments
    # -------------------------------------------------------------------------
    if args.train_model == 'linear':
        model, metrics = modelltraining.train_linear_regression(train_df, eval_df, target_col='price')
    elif args.train_model == 'tree':
        model, metrics = modelltraining.train_decision_tree(train_df, eval_df, target_col='price', max_depth=5)
    elif args.train_model == 'nn':
        model, metrics = modelltraining.train_neural_network(train_df, eval_df, target_col='price')
    else:
        print("No valid model specified to train.")
        sys.exit(1)

    print(f"Trained {args.train_model} model. Metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file_path', type=str, required=True,
                        help='Path to the input CSV file.')
    parser.add_argument('--remove_correlation', action='store_true',
                        help='If set, remove columns with low correlation to price.')
    parser.add_argument('--augment', action='store_true',
                        help='If set, augment data (e.g., create new features).')
    parser.add_argument('--remove_outliers', action='store_true',
                        help='If set, remove outliers from the dataset.')
    parser.add_argument('--split_method', type=str, default='random', choices=['random', 'sequential'],
                        help='Method to split the data: random or sequential.')
    parser.add_argument('--train_model', type=str, default='linear', choices=['linear', 'tree', 'nn'],
                        help='Which model to train: linear, tree, or nn.')

    args = parser.parse_args()
    main(args)
