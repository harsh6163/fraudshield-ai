
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Load dataset
DATA_PATH = Path("data/creditcard.csv")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

# First, reserve 20% for final evaluation
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Split the remaining data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev,
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

print("Training Random Forest...")
model.fit(X_train, y_train)

# Get validation probabilities
val_probs = model.predict_proba(X_val)[:, 1]

# Compare thresholds using validation data only
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30,
              0.35, 0.40, 0.45, 0.50]

results = []

print("\nVALIDATION THRESHOLD COMPARISON")
print("-" * 75)
print(
    f"{'Threshold':<12}{'Precision':<12}"
    f"{'Recall':<12}{'F1 Score':<12}"
    f"{'False Alarms':<15}"
)

for threshold in thresholds:
    preds = (val_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, preds, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_val, preds, zero_division=0
    )
    recall = recall_score(
        y_val, preds, zero_division=0
    )
    f1 = f1_score(
        y_val, preds, zero_division=0
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_alarms": int(fp),
        "missed_fraud": int(fn),
    })

    print(
        f"{threshold:<12.2f}{precision:<12.3f}"
        f"{recall:<12.3f}{f1:<12.3f}"
        f"{fp:<15}"
    )

# Choose the highest-precision threshold that achieves
# at least 80% recall on validation data.
eligible = [r for r in results if r["recall"] >= 0.80]

if eligible:
    best = max(
        eligible,
        key=lambda r: (r["precision"], r["f1_score"])
    )
else:
    best = max(results, key=lambda r: r["f1_score"])

best_threshold = best["threshold"]

print(f"\nSelected threshold: {best_threshold:.2f}")
print("Selection rule: target recall >= 80%, then maximize precision.")
print("If no threshold reaches 80% recall, maximize validation F1.")

# Evaluate once on the held-out test set
test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_preds, labels=[0, 1]
).ravel()

test_metrics = {
    "model": "Random Forest",
    "threshold": best_threshold,
    "precision": float(precision_score(
        y_test, test_preds, zero_division=0
    )),
    "recall": float(recall_score(
        y_test, test_preds, zero_division=0
    )),
    "f1_score": float(f1_score(
        y_test, test_preds, zero_division=0
    )),
    "average_precision": float(
        average_precision_score(y_test, test_probs)
    ),
    "roc_auc": float(roc_auc_score(y_test, test_probs)),
    "true_positives": int(tp),
    "false_positives": int(fp),
    "true_negatives": int(tn),
    "false_negatives": int(fn),
}

print("\nFINAL TEST RESULTS")
for key, value in test_metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# Save the candidate model and its evaluation
Path("models").mkdir(exist_ok=True)

joblib.dump(model, "models/random_forest_model.joblib")

with open("models/random_forest_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=4)

pd.DataFrame(results).to_csv(
    "models/threshold_comparison.csv", index=False
)

print("\nSaved:")
print("models/random_forest_model.joblib")
print("models/random_forest_metrics.json")
print("models/threshold_comparison.csv")
