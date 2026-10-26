"""
FraudShield AI — Step 17: Validation-Based Model and Threshold Selection
------------------------------------------------------------------------
This script:
1. Creates separate training, validation, and test sets.
2. Trains Logistic Regression and Random Forest on the training set only.
3. Selects the model and threshold using validation results only.
4. Reports one final evaluation on the untouched test set.
5. NEVER overwrites app.py, fraud_model.joblib, random_forest_model.joblib,
   metrics.json, or any active application model/configuration.

Expected input:
    data/creditcard.csv

New outputs only:
    models/validation_model_selection.csv
    models/validation_model_selection_metrics.json

Run from the project folder:
    python select_model_validation.py
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

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_FRACTION_OF_REMAINING = 0.25  # 20% of all data after initial 20% test split
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
              0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
TARGET_MIN_RECALL = 0.80


def build_models(feature_columns):
    scaled_columns = [c for c in ["Time", "Amount"] if c in feature_columns]
    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), scaled_columns)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    logistic = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ])

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


def get_positive_probabilities(model, X):
    classes = list(model.classes_)
    if 1 not in classes:
        raise RuntimeError("The trained model does not contain fraud class 1.")
    return model.predict_proba(X)[:, classes.index(1)]


def evaluate_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place creditcard.csv inside the project's data folder."
        )

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Dataset is empty.")
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain the target column 'Class'.")
    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values. Clean it before training.")
    if not set(df["Class"].unique()).issubset({0, 1}) or df["Class"].nunique() != 2:
        raise ValueError("'Class' must contain both 0 and 1.")

    feature_columns = [c for c in df.columns if c != "Class"]
    X = df[feature_columns]
    y = df["Class"].astype(int)

    # First create the untouched test set. This matches the initial split used
    # by Step 16 (same test rows with the same dataset and random state).
    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Split the remaining 80% into 60% train and 20% validation overall.
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_remaining,
        y_remaining,
        test_size=VALIDATION_FRACTION_OF_REMAINING,
        random_state=RANDOM_STATE + 1,
        stratify=y_remaining,
    )

    print("=" * 72)
    print("FRAUDSHIELD AI | VALIDATION-BASED MODEL SELECTION")
    print("=" * 72)
    print(f"Dataset rows: {len(df):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Untouched test rows: {len(X_test):,}")
    print(f"Overall fraud prevalence: {y.mean():.4%}")
    print(f"Validation fraud cases: {int(y_validation.sum())}")
    print(f"Test fraud cases (used only for final evaluation): {int(y_test.sum())}")
    print("\nThe active app model and its files will NOT be changed.\n")

    rows = []
    model_results = {}
    fitted_models = {}
    validation_probabilities = {}

    for name, model in build_models(feature_columns).items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        probabilities = get_positive_probabilities(model, X_validation)
        fitted_models[name] = model
        validation_probabilities[name] = probabilities

        ap = float(average_precision_score(y_validation, probabilities))
        auc = float(roc_auc_score(y_validation, probabilities))
        threshold_results = [
            evaluate_threshold(y_validation, probabilities, t)
            for t in THRESHOLDS
        ]

        model_results[name] = {
            "average_precision_validation": ap,
            "roc_auc_validation": auc,
            "threshold_results": threshold_results,
        }

        print(f"  Validation Average Precision: {ap:.4f}")
        print(f"  Validation ROC-AUC: {auc:.4f}")
        for item in threshold_results:
            print(
                f"  Threshold {item['threshold']:.2f}: "
                f"Precision={item['precision']:.4f}, "
                f"Recall={item['recall']:.4f}, "
                f"F1={item['f1_score']:.4f}, "
                f"FP={item['false_positives']}, "
                f"FN={item['false_negatives']}"
            )
            rows.append({
                "model": name,
                "average_precision_validation": ap,
                "roc_auc_validation": auc,
                **item,
            })
        print()

    # Select the model using validation Average Precision, not test performance.
    selected_model_name = max(
        model_results,
        key=lambda n: model_results[n]["average_precision_validation"]
    )

    # For the selected model, prefer thresholds meeting the target recall;
    # among them maximize precision, then F1. If none meet the target,
    # maximize F1 and clearly report that recall target was not met.
    selected_threshold_results = model_results[selected_model_name]["threshold_results"]
    eligible = [r for r in selected_threshold_results if r["recall"] >= TARGET_MIN_RECALL]

    if eligible:
        selected_threshold_result = max(
            eligible,
            key=lambda r: (r["precision"], r["f1_score"], -r["false_positives"])
        )
        selection_rule = (
            f"Among thresholds with validation recall >= {TARGET_MIN_RECALL:.0%}, "
            "maximize precision; tie-break by F1, then fewer false positives."
        )
        recall_target_met = True
    else:
        selected_threshold_result = max(
            selected_threshold_results,
            key=lambda r: (r["f1_score"], r["precision"])
        )
        selection_rule = (
            f"No tested threshold reached validation recall >= {TARGET_MIN_RECALL:.0%}; "
            "selected threshold by F1, then precision."
        )
        recall_target_met = False

    selected_threshold = selected_threshold_result["threshold"]
    selected_model = fitted_models[selected_model_name]

    # Final test evaluation happens only after model and threshold selection.
    # The test set is not used to choose or alter the model/threshold.
    test_probabilities = get_positive_probabilities(selected_model, X_test)
    final_test_metrics = evaluate_threshold(y_test, test_probabilities, selected_threshold)
    final_test_metrics.update({
        "average_precision": float(average_precision_score(y_test, test_probabilities)),
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "test_rows": int(len(y_test)),
        "fraud_cases_in_test": int(y_test.sum()),
    })

    details = {
        "purpose": "Validation-based model and threshold selection; active app model untouched.",
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "test_rows": int(len(X_test)),
        "fraud_prevalence": float(y.mean()),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation_fraction_of_remaining": VALIDATION_FRACTION_OF_REMAINING,
        "thresholds_tested_on_validation": THRESHOLDS,
        "target_min_recall": TARGET_MIN_RECALL,
        "model_selection_rule": "Highest validation Average Precision",
        "selected_model": selected_model_name,
        "selected_threshold": float(selected_threshold),
        "selected_threshold_validation_metrics": selected_threshold_result,
        "threshold_selection_rule": selection_rule,
        "validation_recall_target_met": recall_target_met,
        "validation_results_by_model": model_results,
        "final_test_evaluation": final_test_metrics,
        "note": (
            "The model was fitted on the training subset and selected using validation data. "
            "The test set was used only once for final evaluation. No model artifact is saved "
            "or activated by this script."
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = MODEL_DIR / "validation_model_selection.csv"
    json_path = MODEL_DIR / "validation_model_selection_metrics.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)

    print("=" * 72)
    print("SELECTION RESULT (based on validation data)")
    print("=" * 72)
    print(f"Selected model: {selected_model_name}")
    print(f"Selected threshold: {selected_threshold:.2f}")
    print(f"Selection rule: {selection_rule}")
    print("Validation metrics at selected threshold:")
    for key in ["precision", "recall", "f1_score", "false_positives", "false_negatives"]:
        print(f"  {key}: {selected_threshold_result[key]}")

    print("\nFINAL TEST EVALUATION (threshold already selected)")
    print("-" * 72)
    for key in [
        "precision", "recall", "f1_score", "average_precision", "roc_auc",
        "false_positives", "false_negatives", "true_positives", "true_negatives",
    ]:
        print(f"  {key}: {final_test_metrics[key]}")

    print(f"\nSaved validation table: {csv_path}")
    print(f"Saved detailed report: {json_path}")
    print("\nNo active model file was saved, replaced, or modified.")
    print("Review the validation results before manually deciding whether to update the app.")


if __name__ == "__main__":
    main()
