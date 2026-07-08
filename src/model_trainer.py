"""AutoML training pipeline: preprocessing, model comparison, leaderboard."""

import io
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
}

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "SVR": SVR(),
}


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric + categorical columns."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])


def run_automl(df: pd.DataFrame, target_col: str, problem_type: str, test_size: float) -> dict:
    """Train multiple models, evaluate them, and return leaderboard + best model."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        y = y.astype("category").cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    preprocessor = build_preprocessor(X)
    models = CLASSIFICATION_MODELS if problem_type == "classification" else REGRESSION_MODELS

    leaderboard_rows = []
    fitted_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        if problem_type == "classification":
            metrics = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred, average="weighted"),
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted"),
            }
            sort_metric = "F1"
        else:
            metrics = {
                "Model": name,
                "R2": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": mean_squared_error(y_test, y_pred, squared=False),
            }
            sort_metric = "R2"

        leaderboard_rows.append(metrics)
        fitted_pipelines[name] = pipeline

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        by=sort_metric, ascending=False
    ).reset_index(drop=True)

    best_name = leaderboard.iloc[0]["Model"]
    best_pipeline = fitted_pipelines[best_name]
    y_pred_best = best_pipeline.predict(X_test)

    # Serialize best model for download
    buffer = io.BytesIO()
    pickle.dump(best_pipeline, buffer)
    model_bytes = buffer.getvalue()

    return {
        "leaderboard": leaderboard,
        "best_model": best_pipeline,
        "feature_names": X.columns.tolist(),
        "y_test": y_test,
        "y_pred": y_pred_best,
        "model_bytes": model_bytes,
    }
