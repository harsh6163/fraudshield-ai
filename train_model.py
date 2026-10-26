
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------- 1. Paths and configuration ----------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
THRESHOLD = 0.50


# ---------- 2. Load and validate data ----------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Download creditcard.csv into the data folder."
    )

df = pd.read_csv(DATA_PATH)

if "Class" not in df.columns:
    raise ValueError("The dataset must contain a 'Class' target column.")

if df.empty or df.isnull().any().any():
    raise ValueError("The dataset is empty or contains missing values.")

if not set(df["Class"].unique()).issubset({0, 1}):
    raise ValueError("Class must contain only 0 and 1.")

if df["Class"].nunique() != 2:
    raise ValueError("Both fraud and non-fraud examples are required.")

feature_columns = [c for c in df.columns if c != "Class"]

X = df[feature_columns]
y = df["Class"].astype(int)

print(f"Dataset shape: {df.shape}")
print("\nClass distribution:")
print(y.value_counts())
print(f"\nFraud percentage: {y.mean() * 100:.4f}%")


# ---------- 3. Split before fitting preprocessing ----------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)


# ---------- 4. Build the ML pipeline ----------

# Time and Amount use different scales.
# The V1-V28 features are already transformed in this dataset.
scaled_columns = [
    col for col in ["Time", "Amount"] if col in feature_columns
]

preprocessor = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), scaled_columns),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# ---------- 5. Train ----------

print("\nTraining Logistic Regression baseline...")
model.fit(X_train, y_train)


# ---------- 6. Evaluate on untouched test data ----------

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= THRESHOLD).astype(int)

metrics = {
    "model_name": "Logistic Regression",
    "threshold": THRESHOLD,
    "test_rows": int(len(y_test)),
    "fraud_cases_in_test": int(y_test.sum()),
    "precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, predictions, zero_division=0)
    ),
    "average_precision": float(
        average_precision_score(y_test, probabilities)
    ),
    "roc_auc": float(roc_auc_score(y_test, probabilities)),
    "confusion_matrix": confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).tolist(),
    "classification_report": classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Non-fraud", "Fraud"],
        output_dict=True,
        zero_division=0,
    ),
    "feature_columns": feature_columns,
}

print("\nEvaluation results")
print("------------------")
for key in [
    "precision",
    "recall",
    "f1_score",
    "average_precision",
    "roc_auc",
]:
    print(f"{key}: {metrics[key]:.4f}")

print("\nConfusion matrix:")
print(metrics["confusion_matrix"])


# ---------- 7. Save model and metrics ----------

joblib.dump(model, MODEL_DIR / "fraud_model.joblib")

with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nTraining complete!")
print(f"Model saved to: {MODEL_DIR / 'fraud_model.joblib'}")
print(f"Metrics saved to: {MODEL_DIR / 'metrics.json'}")


# Save unlabeled examples from the held-out test set for the UI demo.
demo_path = MODEL_DIR / "demo_transactions.csv"
X_test.head(100).to_csv(demo_path, index=False)

print(f"Demo transactions saved to: {demo_path}")