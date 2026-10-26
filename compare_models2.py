"""
FraudShield AI — Step 16: Model Comparison
-----------------------------------------
Compares Logistic Regression and Random Forest on the same stratified holdout set.
This script does NOT overwrite the model currently used by app.py.

Expected dataset:
    data/creditcard.csv

Outputs:
    models/model_comparison.csv
    models/model_comparison_metrics.json

Run:
    python compare_models.py
"""

from pathlib import Path
import json
import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLDS = [0.10, 0.25, 0.50]


def build_models(feature_columns):
    scaled_columns = [col for col in ["Time", "Amount"] if col in feature_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), scaled_columns),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    logistic = Pipeline(
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

    # Random Forest does not require feature scaling, but using the same
    # split ensures the evaluation comparison is fair and reproducible.
    forest = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return {
        "Logistic Regression": logistic,
        "Random Forest": forest,
    }


def evaluate_at_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place creditcard.csv inside the project's data folder."
        )

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("The dataset is empty.")
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain the target column 'Class'.")
    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values; clean it before training.")
    if not set(df["Class"].unique()).issubset({0, 1}) or df["Class"].nunique() != 2:
        raise ValueError("'Class' must contain both 0 and 1.")

    feature_columns = [col for col in df.columns if col != "Class"]
    X = df[feature_columns]
    y = df["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=" * 66)
    print("FRAUDSHIELD AI | MODEL COMPARISON")
    print("=" * 66)
    print(f"Dataset rows: {len(df):,}")
    print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"Fraud prevalence: {y.mean():.4%}")
    print("Models are evaluated on the same held-out test split.")
    print("The active app model will not be changed by this script.\n")

    rows = []
    details = {
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "fraud_cases_in_test": int(y_test.sum()),
        "fraud_prevalence": float(y.mean()),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "thresholds": THRESHOLDS,
        "models": {},
    }

    for name, candidate in build_models(feature_columns).items():
        print(f"Training {name}...")
        candidate.fit(X_train, y_train)
        classes = list(candidate.classes_)
        if 1 not in classes:
            raise RuntimeError(f"{name} did not produce the fraud class.")
        probabilities = candidate.predict_proba(X_test)[:, classes.index(1)]

        base_metrics = {
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "threshold_results": [],
        }

        for threshold in THRESHOLDS:
            metrics = evaluate_at_threshold(y_test, probabilities, threshold)
            row = {"model": name, **metrics, **{
                "average_precision": base_metrics["average_precision"],
                "roc_auc": base_metrics["roc_auc"],
            }}
            rows.append(row)
            base_metrics["threshold_results"].append(metrics)

        details["models"][name] = base_metrics
        print(
            f"  Average precision: {base_metrics['average_precision']:.4f} | "
            f"ROC-AUC: {base_metrics['roc_auc']:.4f}"
        )
        for item in base_metrics["threshold_results"]:
            print(
                f"  Threshold {item['threshold']:.2f}: "
                f"Precision={item['precision']:.4f}, "
                f"Recall={item['recall']:.4f}, "
                f"F1={item['f1_score']:.4f}, "
                f"FP={item['false_positives']}, "
                f"FN={item['false_negatives']}"
            )
        print()

    comparison = pd.DataFrame(rows)
    csv_path = MODEL_DIR / "model_comparison.csv"
    json_path = MODEL_DIR / "model_comparison_metrics.json"

    comparison.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(details, file, indent=2)

    # A useful starting ranking: average precision first, then F1 at 0.25.
    ranking = comparison.loc[comparison["threshold"].eq(0.25)].sort_values(
        ["average_precision", "f1_score"], ascending=False
    )
    print("=" * 66)
    print("SUMMARY (ranked by average precision, then F1 at threshold 0.25)")
    print("=" * 66)
    print(
        ranking[
            ["model", "threshold", "average_precision", "roc_auc",
             "precision", "recall", "f1_score", "false_positives", "false_negatives"]
        ].to_string(index=False)
    )
    print(f"\nSaved comparison table: {csv_path}")
    print(f"Saved detailed metrics: {json_path}")
    print("\nReview the precision/recall trade-off before choosing a model.")
    print("Do not select a threshold from the test set for final performance claims.")


if __name__ == "__main__":
    main()
