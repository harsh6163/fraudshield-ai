
import json
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

model_files = [
    "random_forest_model.joblib",
    "fraud_model.joblib",
]

print("=" * 60)
print("FRAUDSHIELD AI — SAVED MODEL INSPECTION")
print("Read-only: no models or application files will be changed")
print("=" * 60)

for filename in model_files:
    path = MODELS_DIR / filename

    print(f"\n{'-' * 50}")
    print(f"File: {filename}")

    if not path.exists():
        print("File missing.")
        continue

    try:
        model = joblib.load(path)

        print("Loaded object type:", type(model).__name__)
        print("Python class:", type(model).__module__ + "." + type(model).__name__)

        if hasattr(model, "n_features_in_"):
            print("Expected feature count:", model.n_features_in_)

        if hasattr(model, "feature_names_in_"):
            names = list(model.feature_names_in_)
            print("Feature names available:", True)
            print("Feature name count:", len(names))
            print("First 10 feature names:", names[:10])
        else:
            print("Feature names available: False")

        if hasattr(model, "classes_"):
            print("Classes:", model.classes_.tolist())

        if hasattr(model, "get_params"):
            params = model.get_params()
            keys = [
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "class_weight",
                "random_state",
            ]
            print("Selected model parameters:")
            for key in keys:
                if key in params:
                    print(f"  {key}: {params[key]}")

        if hasattr(model, "steps"):
            print("Pipeline steps:", [
                name for name, _ in model.steps
            ])

    except Exception as error:
        print("Inspection error:", repr(error))

print("\n" + "=" * 60)
print("Saved-model inspection finished.")
print("No model files were written or modified.")
print("=" * 60)
