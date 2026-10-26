
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Load dataset
DATA_PATH = Path("data/creditcard.csv")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

# Split data while preserving the fraud/non-fraud ratio
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Scale Time and Amount for Logistic Regression
preprocessor = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), ["Time", "Amount"])
    ],
    remainder="passthrough",
)

models = {
    "Logistic Regression": Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    results.append({
        "Model": name,
        "Average Precision": average_precision_score(
            y_test, probabilities
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Average Precision", ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

Path("models").mkdir(exist_ok=True)
results_df.to_csv("models/model_comparison.csv", index=False)

print("\nResults saved to models/model_comparison.csv")
