import pandas as pd

def normalize_data(csv_file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file and normalizes certain columns:
    - yes/no -> 1/0
    - One-hot encode furnishingstatus.

    Returns a pandas DataFrame with normalized data.
    """
    df = pd.read_csv(csv_file_path)

    # Convert yes/no columns to 1/0
    yes_no_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
    for col in yes_no_cols:
        df[col] = df[col].map({"yes": 1, "no": 0})

    # One-hot encode furnishingstatus
    # furnishingstatus: typically "furnished", "semi-furnished", "unfurnished"
    df = pd.get_dummies(df, columns=["furnishingstatus"], prefix="furnishingstatus")

    return df
