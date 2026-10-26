
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

files_to_check = [
    "random_forest_model.joblib",
    "fraud_model.joblib",
    "random_forest_metrics.json",
    "metrics.json",
    "validation_model_selection_metrics.json",
]

print("=" * 60)
print("FRAUDSHIELD AI — READ-ONLY MODEL CONFIGURATION CHECK")
print("=" * 60)

for filename in files_to_check:
    path = MODELS_DIR / filename

    print(f"\nFile: {filename}")

    if not path.exists():
        print("Status: MISSING")
        continue

    print("Status: PRESENT")
    print(f"Size: {path.stat().st_size:,} bytes")

    if path.suffix.lower() != ".json":
        print("Contents: Not opened (model file left untouched)")
        continue

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            print("JSON structure:", type(data).__name__)
            print("Contents:", repr(data)[:1000])
            continue

        print("Available keys:", ", ".join(data.keys()))

        for key, value in data.items():
            key_lower = key.lower()

            if (
                "threshold" in key_lower
                or "precision" in key_lower
                or "recall" in key_lower
                or "f1" == key_lower
                or "average_precision" in key_lower
                or "roc_auc" in key_lower
                or "model" in key_lower
                or "selected" in key_lower
            ):
                print(f"  {key}: {value}")

    except (json.JSONDecodeError, OSError) as error:
        print(f"Could not read JSON: {error}")

print("\n" + "=" * 60)
print("Inspection complete. No project files were modified.")
print("No model was loaded, retrained, or replaced.")
print("=" * 60)
