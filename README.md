# No-Code AutoML Web App

Upload a CSV, pick a target column, and get a trained model leaderboard with metrics, visualizations, and a downloadable model — no code required.

## Features
- CSV upload with data preview and profiling
- Auto-detects classification vs regression
- Trains 5–6 models automatically (Logistic/Linear Regression, Random Forest, Gradient Boosting, Decision Tree, SVM)
- Leaderboard sorted by best metric (F1 or R²)
- Confusion matrix / residual plots
- Feature importance chart
- Download best model (.pkl) and predictions (.csv)

## Project Structure
```
automl-webapp/
├── app.py                 # Streamlit UI
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── data_utils.py       # loading, profiling, problem-type inference
    ├── model_trainer.py     # preprocessing + model training + leaderboard
    └── visualizations.py    # Plotly charts
```

## Setup

```bash
# 1. Clone / unzip the project, then cd into it
cd automl-webapp

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Usage
1. Upload a CSV file.
2. Select the target column you want to predict.
3. Adjust the test set size if needed.
4. Click **Run AutoML**.
5. Review the leaderboard, charts, and download the best model or predictions.

## Extending This Project
- **Hyperparameter tuning**: wrap models in `GridSearchCV`/`Optuna` inside `model_trainer.py`
- **More model types**: add XGBoost/LightGBM/CatBoost to the model dicts
- **Explainability**: integrate SHAP for per-prediction explanations
- **Predict on new data**: add a second uploader to score a fresh dataset with the saved model
- **Persistence/history**: add a database (SQLite/Postgres) to save past runs per user
- **Production backend**: split into FastAPI backend + React frontend, use Celery/Redis for async training on large datasets
- **Deployment**: containerize with Docker, deploy to Render, AWS, or Hugging Face Spaces

## Notes
- Designed for small-to-medium tabular datasets. For very large files, add sampling or chunked processing.
- Classification vs regression is inferred heuristically (numeric target with few unique values → classification). You can override this by editing `infer_problem_type` in `src/data_utils.py`.

