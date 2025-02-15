import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def main():
    # -------------------------------------------------------------------------
    # 1. Read the CSV file
    # -------------------------------------------------------------------------
    df = pd.read_csv("results3_log.csv")  # Change to your results file name

    # -------------------------------------------------------------------------
    # 2. Filter out rows with errors or infinite MSE
    # -------------------------------------------------------------------------
    # Drop rows where "error" is not NaN (meaning a run failure)
    df = df[df["error"].isna()]

    # Also remove any rows with infinite MSE (in case they were set that way on error)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["MSE", "R2"])  # drop rows where MSE or R2 is NaN

    # Optional: convert boolean columns from string ("True"/"False") if needed
    # If your CSV stored them as actual booleans, you can skip this step
    # df["remove_correlation"] = df["remove_correlation"].astype(bool)
    # df["augment"] = df["augment"].astype(bool)
    # df["remove_outliers"] = df["remove_outliers"].astype(bool)

    # -------------------------------------------------------------------------
    # 3. Plot 1: Bar Plot of MSE by Model and Split Method
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="train_model",
        y="MSE",
        hue="split_method",
        errorbar=("ci", 95)  # or use errorbar=None for no error bars
    )
    plt.title("MSE by Model and Split Method")
    plt.legend(title="Split Method")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 4. Plot 2: Box Plot of R² by Model, grouped by remove_outliers
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="train_model",
        y="R2",
        hue="remove_outliers"
    )
    plt.title("R² by Model, with/without Outlier Removal")
    plt.legend(title="remove_outliers")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 4.1. Plot 2.1: Box Plot of R² by Model, grouped by augment
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="train_model",
        y="R2",
        hue="augment"
    )
    plt.title("R² by Model, with/without Augmentation")
    plt.legend(title="augment")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 5. Plot 3: Heatmap showing average MSE for (train_model) vs. (remove_correlation)
    # -------------------------------------------------------------------------
    # We'll aggregate by the average MSE across any duplicates.
    pivot_df = df.pivot_table(
        index="train_model",
        columns="remove_correlation",
        values="MSE",
        aggfunc="mean"
    )

    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".1f")
    plt.title("Avg MSE: Model vs. remove_correlation")
    plt.xlabel("remove_correlation")
    plt.ylabel("train_model")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # You can add more plots as desired:
    #  - Compare "augment" or "remove_outliers"
    #  - Compare R² instead of MSE
    #  - Plot distributions, pairplots, etc.
    # -------------------------------------------------------------------------

if __name__ == "__main__":
    main()
