import sys
import argparse
import csv
from itertools import product

import normalisierung  # your normalization.py
import aufbereitung    # your aufbereitung.py
import trainingssplit  # your trainingssplit.py
import modelltraining  # your modelltraining.py

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
    Returns either:
       (model, metrics) for single-split
       or (list_of_results, avg_metrics) for multi-split
       or (model, metrics, history) if it's Keras with a single split
    """

    # 1) Normalize data
    df = normalisierung.normalize_data(csv_file_path)

    # 2) Aufbereitung (preprocessing)
    if remove_correlation:
        df = aufbereitung.remove_low_correlation_cols(df, target_col=target_col, threshold=0.1)
    if augment:
        df = aufbereitung.augment_data(df)
    if remove_outliers:
        df = aufbereitung.remove_outliers(df, z_thresh=3.0)

    # 3) Get splits based on method
    splits = trainingssplit.get_splits(df, split_method)

    # 4) Train the model on the splits
    if train_model == "linear":
        result = modelltraining.train_linear_regression_splits(splits, target_col=target_col)
    elif train_model == "tree":
        result = modelltraining.train_decision_tree_splits(splits, target_col=target_col, max_depth=5)
    elif train_model == "nn":
        result = modelltraining.train_neural_network_splits(splits, target_col=target_col)
    elif train_model == "keras":
        result = modelltraining.train_keras_network_splits(splits, target_col=target_col,
                                                           epochs=1000, batch_size=16)
    else:
        raise ValueError(f"Unknown model type: {train_model}")

    return result

def run_all_combinations(csv_file_path: str):
    """
    Runs every possible combination of:
      - remove_correlation in [False, True]
      - augment in [False, True]
      - remove_outliers in [False, True]
      - split_method in ['random', 'sequential'] (can extend if you want holdout/kfold/loo)
      - train_model in ['linear', 'tree', 'nn', 'keras']

    Returns a list of results sorted by ascending MSE.
    """
    results = []

    bool_options = [False, True]
    split_options = ["holdout", "kfold"]
    model_options = ["linear", "tree", "nn"]

    for remove_correlation, augment, remove_outliers in product(bool_options, bool_options, bool_options):
        for split_method in split_options:
            for train_model in model_options:
                try:
                    # Run the pipeline
                    result = run_pipeline(
                        csv_file_path=csv_file_path,
                        remove_correlation=remove_correlation,
                        augment=augment,
                        remove_outliers=remove_outliers,
                        split_method=split_method,
                        train_model=train_model
                    )

                    # 'result' could be several forms:
                    #  1) (model, metrics)
                    #  2) (list_of_results, avg_metrics)
                    #  3) (model, metrics, history)
                    # We'll unify them to extract MSE, R2.

                    mse_val = None
                    r2_val = None

                    if isinstance(result, tuple) and len(result) == 2:
                        (obj1, obj2) = result
                        # Could be (model, metrics) OR (list_of_results, avg_metrics)
                        if isinstance(obj1, list):
                            # => (list_of_results, avg_metrics)
                            avg_metrics = obj2
                            mse_val = avg_metrics.get("MSE", float("inf"))
                            r2_val = avg_metrics.get("R2", float("-inf"))
                        else:
                            # => (model, metrics)
                            metrics = obj2
                            mse_val = metrics.get("MSE", float("inf"))
                            r2_val = metrics.get("R2", float("-inf"))

                    elif isinstance(result, tuple) and len(result) == 3:
                        # => (model, metrics, history)
                        model, metrics, history = result
                        mse_val = metrics.get("MSE", float("inf"))
                        r2_val = metrics.get("R2", float("-inf"))
                    else:
                        # Unexpected format
                        mse_val = float("inf")
                        r2_val = float("-inf")

                    combination_info = {
                        "remove_correlation": remove_correlation,
                        "augment": augment,
                        "remove_outliers": remove_outliers,
                        "split_method": split_method,
                        "train_model": train_model,
                        "MSE": mse_val,
                        "R2": r2_val
                    }
                    results.append(combination_info)

                except Exception as e:
                    # If something fails, store inf MSE so it sorts at the bottom
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
        # 1. Run all combinations
        # ---------------------------------------------------------------------
        all_results = run_all_combinations(args.csv_file_path)

        # 2. Print to console
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

        # 3. Write full details into a CSV result file
        output_file = "results3_log.csv"
        fieldnames = [
            "remove_correlation", "augment", "remove_outliers",
            "split_method", "train_model", "MSE", "R2", "error"
        ]
        with open(output_file, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                # Ensure 'error' key exists even if None
                if 'error' not in row:
                    row['error'] = None
                writer.writerow(row)

        print(f"\nAll results have been logged to '{output_file}'.\n")

    else:
        # ---------------------------------------------------------------------
        # Normal single-run pipeline
        # ---------------------------------------------------------------------
        result = run_pipeline(
            csv_file_path=args.csv_file_path,
            remove_correlation=args.remove_correlation,
            augment=args.augment,
            remove_outliers=args.remove_outliers,
            split_method=args.split_method,
            train_model=args.train_model,
        )

        # We'll interpret and print the result depending on its type
        if isinstance(result, tuple) and len(result) == 2:
            first, second = result
            if isinstance(first, list):
                # => multiple splits => first is list_of_results, second is avg_metrics
                avg_metrics = second
                print(f"Multiple-split => Average MSE: {avg_metrics['MSE']:.5f}, "
                      f"Average R2: {avg_metrics['R2']:.5f}")
            else:
                # => single-split => (model, metrics)
                metrics = second
                print(f"Single-split => MSE: {metrics['MSE']:.5f}, R2: {metrics['R2']:.5f}")
        elif isinstance(result, tuple) and len(result) == 3:
            # => (model, metrics, history) from Keras
            model, metrics, history = result
            print(f"Keras single-split => MSE: {metrics['MSE']:.5f}, R2: {metrics['R2']:.5f}")
        else:
            # Unexpected format
            print("Unexpected result format:", result)

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
    parser.add_argument('--split_method', type=str, default='random',
                        choices=['random', 'sequential', 'holdout', 'kfold', 'loo'],
                        help='Method to split the data.')
    parser.add_argument('--train_model', type=str, default='linear',
                        choices=['linear', 'tree', 'nn', 'keras'],
                        help='Which model to train: linear, tree, nn, or keras.')
    parser.add_argument('--run_all_combinations', action='store_true',
                        help='Run the pipeline for every possible combination of flags/models.')
    args = parser.parse_args()
    main(args)
