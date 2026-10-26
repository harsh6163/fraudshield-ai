
import pandas as pd
from pathlib import Path

dataset_path = Path("data/creditcard.csv")
output_path = Path("data/demo_transactions.csv")

if not dataset_path.exists():
    raise FileNotFoundError(
        "Dataset not found. Check that data/creditcard.csv exists."
    )

df = pd.read_csv(dataset_path)

# Include a mix of legitimate and fraudulent transactions when possible.
fraud = df[df["Class"] == 1].head(20)
legitimate = df[df["Class"] == 0].head(480)

demo = pd.concat([fraud, legitimate])
demo = demo.sample(frac=1, random_state=42).reset_index(drop=True)

demo.to_csv(output_path, index=False)

print(f"Demo dataset created: {output_path}")
print(f"Rows: {len(demo)}")
print(f"Fraud examples: {(demo['Class'] == 1).sum()}")
print(f"Legitimate examples: {(demo['Class'] == 0).sum()}")
