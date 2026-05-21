import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import random
import sys

EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
]


def validate(input_path: str, report_path: str) -> None:
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)

    report = {}

    # shape
    report["rows"] = int(df.shape[0])
    report["cols"] = int(df.shape[1])
    print(f"Shape: {df.shape}")

    # missing values
    missing = df.isnull().sum()
    missing_dict = {col: int(v) for col, v in missing.items() if v > 0}
    report["missing_values"] = missing_dict
    if missing_dict:
        print(f"WARNING — missing values detected: {missing_dict}")
    else:
        print("No missing values found.")

    # expected columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    report["missing_columns"] = missing_cols
    if missing_cols:
        print(f"ERROR — missing columns: {missing_cols}")
        sys.exit(1)

    # class distribution
    counts = df["isFraud"].value_counts().to_dict()
    counts = {int(k): int(v) for k, v in counts.items()}
    total = sum(counts.values())
    fraud_pct = round(counts.get(1, 0) / total * 100, 4)
    report["class_distribution"] = counts
    report["fraud_percentage"] = fraud_pct
    print(f"Class distribution: {counts}")
    print(f"Fraud percentage: {fraud_pct}%")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report["numeric_columns"] = numeric_cols

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV dataset")
    parser.add_argument(
        "--report", required=True, help="Path to save validation_report.json"
    )
    parser.add_argument("--benchmark", type=bool, default=False)
    args = parser.parse_args()

    if (not args.benchmark) and random.random() < 0.3:
        print("Random error happened!")
        sys.exit(1)

    validate(args.input, args.report)
