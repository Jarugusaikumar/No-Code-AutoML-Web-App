"""Data loading, profiling, and problem-type inference."""

import pandas as pd


def load_data(file) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(file)


def profile_data(df: pd.DataFrame) -> dict:
    """Return a quick summary profile of the dataset."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_types": df.dtypes.astype(str).to_dict(),
    }


def infer_problem_type(target_series: pd.Series) -> str:
    """
    Heuristic to decide classification vs regression:
    - Non-numeric target -> classification
    - Numeric target with few unique values (<= 20 or < 5% of rows) -> classification
    - Otherwise -> regression
    """
    if not pd.api.types.is_numeric_dtype(target_series):
        return "classification"

    n_unique = target_series.nunique()
    n_rows = len(target_series)

    if n_unique <= 20 or (n_unique / max(n_rows, 1)) < 0.05:
        return "classification"

    return "regression"
