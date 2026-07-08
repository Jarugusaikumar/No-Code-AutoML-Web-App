"""
No-Code AutoML Web App
Upload a dataset, pick a target column, and let the app train and
compare multiple ML models automatically.
"""

import streamlit as st
import pandas as pd

from src.data_utils import load_data, profile_data, infer_problem_type
from src.model_trainer import run_automl
from src.visualizations import (
    plot_leaderboard,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_regression_residuals,
)

st.set_page_config(page_title="No-Code AutoML Web App", layout="wide", page_icon="🤖")

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; color: #f8fafc; }
    .stApp { background: #0a192f; color: #f8fafc; }
    .css-18e3th9 { background-color: #0a192f; }
    .css-1d391kg { background-color: #020617; }
    .css-1wrcr25 { background-color: #020617; }
    .stSidebar { background: #091426; }
    .stSidebar .css-1d391kg { background-color: #091426; }
    .stButton>button { background-color: #0f4c81; color: #ffffff; }
    .stButton>button:hover { background-color: #1368b2; }
    .section-header { font-size: 2.8rem; font-weight: 800; color: #ffffff; margin-bottom: 0.2rem; }
    .section-subtitle { font-size: 1rem; color: #cbd5e1; margin-bottom: 1.5rem; }
    .hero-card { background: #10263b; border-radius: 26px; padding: 1.5rem; box-shadow: 0 25px 80px rgba(0, 0, 0, 0.25); }
    .author-text { color: #38bdf8; font-weight: 700; margin-bottom: 0.75rem; }
    .status-banner { background: #064e3b; color: #d1fae5; border-radius: 14px; padding: 1rem 1.25rem; margin-bottom: 1.35rem; }
    .streamlit-expanderHeader { font-weight: 600 !important; }
    .stDownloadButton>button { background-color: #0f4c81; color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='section-header'>No-Code AutoML Web App</div>", unsafe_allow_html=True)
st.markdown("<div class='section-subtitle'>Upload a CSV, pick a target, and get a trained model leaderboard in seconds.</div>", unsafe_allow_html=True)
st.markdown("<div class='author-text'>Developed by JARUGU SAI KUMAR</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("About the Project")
    st.write("A polished AutoML web app for classification and regression tasks.")
    st.write("**Author:** JARUGU SAI KUMAR")
    st.write("**Tech:** Streamlit, scikit-learn, Plotly, pandas")
    st.write("**Instructions:** Upload, choose target, adjust test size and run.")

# ---------------------------------------------------------------------
# 1. Upload
# ---------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    with st.expander("Data Profile"):
        profile = profile_data(df)
        st.write(profile)

    # -------------------------------------------------------------
    # 2. Target selection
    # -------------------------------------------------------------
    st.subheader("🎯 Select Target Column")
    target_col = st.selectbox("Column to predict", df.columns)

    problem_type = infer_problem_type(df[target_col])
    st.info(f"Detected problem type: **{problem_type.upper()}**")

    test_size = st.slider("Test set size (%)", 10, 40, 20) / 100

    # -------------------------------------------------------------
    # 3. Run AutoML
    # -------------------------------------------------------------
    if st.button("🚀 Run AutoML", type="primary"):
        with st.spinner("Training models... this may take a moment"):
            results = run_automl(df, target_col, problem_type, test_size)

        st.session_state["results"] = results
        st.session_state["problem_type"] = problem_type

    # -------------------------------------------------------------
    # 4. Results
    # -------------------------------------------------------------
    if "results" in st.session_state:
        results = st.session_state["results"]
        problem_type = st.session_state["problem_type"]

        st.subheader("🏆 Leaderboard")
        st.dataframe(results["leaderboard"], use_container_width=True)
        st.plotly_chart(plot_leaderboard(results["leaderboard"], problem_type),
                         use_container_width=True)

        best_name = results["leaderboard"].iloc[0]["Model"]
        st.success(f"Best model: **{best_name}**")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔍 Feature Importance")
            fig = plot_feature_importance(results["best_model"], results["feature_names"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Feature importance not available for this model.")

        with col2:
            if problem_type == "classification":
                st.subheader("🧩 Confusion Matrix")
                st.plotly_chart(
                    plot_confusion_matrix(results["y_test"], results["y_pred"]),
                    use_container_width=True,
                )
            else:
                st.subheader("📉 Residuals")
                st.plotly_chart(
                    plot_regression_residuals(results["y_test"], results["y_pred"]),
                    use_container_width=True,
                )

        # ---------------------------------------------------------
        # 5. Download
        # ---------------------------------------------------------
        st.subheader("⬇️ Export")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download Best Model (.pkl)",
                data=results["model_bytes"],
                file_name="best_model.pkl",
                mime="application/octet-stream",
            )
        with c2:
            preds_df = pd.DataFrame({
                "actual": results["y_test"],
                "predicted": results["y_pred"],
            })
            st.download_button(
                "Download Predictions (.csv)",
                data=preds_df.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv",
            )
else:
    st.info("👆 Upload a CSV file to get started.")
