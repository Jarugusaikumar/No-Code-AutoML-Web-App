"""Plotly visualizations for the AutoML app."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix


def plot_leaderboard(leaderboard, problem_type):
    metric = "F1" if problem_type == "classification" else "R2"
    fig = px.bar(
        leaderboard, x="Model", y=metric, color="Model",
        title=f"Model Comparison by {metric}",
        text_auto=".3f",
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title="Confusion Matrix",
    )
    return fig


def plot_feature_importance(pipeline, feature_names):
    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocessor")

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = getattr(model, "coef_", None)
        if importances is not None:
            importances = np.abs(importances).flatten()

    if importances is None:
        return None

    try:
        transformed_names = preprocessor.get_feature_names_out()
    except Exception:
        transformed_names = feature_names

    n = min(len(transformed_names), len(importances))
    fig = px.bar(
        x=importances[:n], y=transformed_names[:n], orientation="h",
        title="Feature Importance",
        labels={"x": "Importance", "y": "Feature"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def plot_regression_residuals(y_test, y_pred):
    residuals = np.array(y_test) - np.array(y_pred)
    fig = px.scatter(
        x=y_pred, y=residuals,
        labels={"x": "Predicted", "y": "Residual"},
        title="Residual Plot",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    return fig
