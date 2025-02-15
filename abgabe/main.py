import sys
import argparse
from itertools import product

import normalisierung
import aufbereitung
import trainingssplit
import modelltraining

def run_pipeline(
        csv_file_path: str,
        remove_correlation: bool,
        augment: bool,
        remove_outliers: bool,
        split_method: str,
        train_model: str,
        target_col: str = "price"
):
    """
    Executes the data pipeline once, given the specified flags and options.
    Returns (model, metrics) from the final training step.
    """

    # -------------------------------------------------------------------------
    # 1. Normalize data
    # -------------------------------------------------------------------------
    df = normalisierung.normalize_data(csv_file_path)

    # -------------------------------------------------------------------------
    # 2. Aufbereitung (preprocessing)
    # -------------------------------------------------------------------------
    # remove low-correlation columns
    if remove_correlation:
        df = aufbereitung.remove_low_correlation_cols(df, target_col=target_col, threshold=0.1)

    # augment data
    if augment:
        df = aufbereitung.augment_data(df)

    # remove outliers
    if remove_outliers:
        df = aufbereitung.remove_outliers(df, z_thresh=3.0)

    # -------------------------------------------------------------------------
    # 3. Trainings split
    # -------------------------------------------------------------------------
    if split_method == "random":
        train_df, eval_df = trainingssplit.random_split(df, test_size=0.2)
    else:
        train_df, eval_df = trainingssplit.sequential_split(df, train_ratio=0.8)

    # -------------------------------------------------------------------------
    # 4. Modelltraining
    # -------------------------------------------------------------------------
    if train_model == "linear":
        model, metrics = modelltraining.train_linear_regression(train_df, eval_df, target_col=target_col)
    elif train_model == "tree":
        model, metrics = modelltraining.train_decision_tree(train_df, eval_df, target_col=target_col, max_depth=5)
    elif train_model == "nn":
        model, metrics = modelltraining.train_neural_network(train_df, eval_df, target_col=target_col)
    elif train_model == "keras":
        model, metrics, history = modelltraining.train_keras_network(
            train_df, eval_df, target_col='price'
        )
    else:
        raise ValueError(f"Unknown model type: {train_model}")

    return model, metrics

def run_all_combinations(csv_file_path: str):
    """
    Runs every possible combination of:
      - remove_correlation in [False, True]
      - augment in [False, True]
      - remove_outliers in [False, True]
      - split_method in ['random', 'sequential']
      - train_model in ['linear', 'tree', 'nn']

    Returns a list of results sorted by ascending MSE.
    """
    results = []

    bool_options = [False, True]
    split_options = ["random", "sequential"]
    model_options = ["linear", "tree", "nn", "keras"]

    # Generate all combinations using itertools.product:
    for remove_correlation, augment, remove_outliers in product(bool_options, bool_options, bool_options):
        for split_method in split_options:
            for train_model in model_options:
                # Run the pipeline
                try:
                    _, metrics = run_pipeline(
                        csv_file_path=csv_file_path,
                        remove_correlation=remove_correlation,
                        augment=augment,
                        remove_outliers=remove_outliers,
                        split_method=split_method,
                        train_model=train_model
                    )

                    # Collect everything in a dict
                    combination_info = {
                        "remove_correlation": remove_correlation,
                        "augment": augment,
                        "remove_outliers": remove_outliers,
                        "split_method": split_method,
                        "train_model": train_model,
                        "MSE": metrics["MSE"],
                        "R2": metrics["R2"]
                    }
                    results.append(combination_info)

                except Exception as e:
                    # If something fails (e.g., due to data shape issues), store the error
                    results.append({
                        "remove_correlation": remove_correlation,
                        "augment": augment,
                        "remove_outliers": remove_outliers,
                        "split_method": split_method,
                        "train_model": train_model,
                        "MSE": float("inf"),
                        "R2": float("-inf"),
                        "error": str(e),
                    })

    # Sort results by ascending MSE (best -> worst)
    results_sorted = sorted(results, key=lambda x: x["MSE"])
    return results_sorted

def main(args):
    if args.run_all_combinations:
        # ---------------------------------------------------------------------
        # Run the exhaustive search of all combinations
        # ---------------------------------------------------------------------
        all_results = run_all_combinations(args.csv_file_path)

        print("\nAll combinations sorted by ascending MSE:\n")
        for i, res in enumerate(all_results, start=1):
            print(
                f"{i:2d}) "
                f"[remove_corr={res['remove_correlation']}, "
                f"augment={res['augment']}, "
                f"outliers={res['remove_outliers']}, "
                f"split={res['split_method']}, "
                f"model={res['train_model']}] "
                f"=> MSE={res['MSE']:.5f}, R2={res['R2']:.5f}, "
                f"error={res.get('error', None)}"
            )

    else:
        # ---------------------------------------------------------------------
        # Normal single-run pipeline based on command-line arguments
        # ---------------------------------------------------------------------
        model, metrics = run_pipeline(
            csv_file_path=args.csv_file_path,
            remove_correlation=args.remove_correlation,
            augment=args.augment,
            remove_outliers=args.remove_outliers,
            split_method=args.split_method,
            train_model=args.train_model,
        )
        print(f"Metrics => MSE: {metrics['MSE']:.5f}, R2: {metrics['R2']:.5f}")

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
    parser.add_argument('--train_model', type=str, default='linear', choices=['linear', 'tree', 'nn', 'keras'],
                        help='Which model to train: linear, tree, or nn.')
    parser.add_argument('--run_all_combinations', action='store_true',
                        help='Run the pipeline for every possible combination of flags/models.')
    args = parser.parse_args()
    main(args)
