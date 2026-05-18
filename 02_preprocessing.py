import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

DROP_COLS = ["nameOrig", "nameDest", "isFraudster"]  

def preprocess(input_path: str, out_dir: str) -> None:
    print(f"[Preprocessing] Loading: {input_path}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    y = df["isFraud"].values.astype(int)
    df.drop(columns=["isFraud"], inplace=True)

    df["orig_balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["dest_balance_diff"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["dest_was_zero"] = (df["oldbalanceDest"]==0).astype(int)
    df["orig_was_zero"]  = (df["oldbalanceOrg"]==0).astype(int)

    df = pd.get_dummies(df, columns=["type"], drop_first=False)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    scaler= StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    X = df.values.astype(np.float32)

    print(f"[Preprocessing] Feature matrix shape: {X.shape}")
    print(f"[Preprocessing] Columns: {list(df.columns)}")

    df.to_csv(f"{out_dir}/X_processed.csv", index=False)
    np.save(f"{out_dir}/X_processed.npy", X)
    np.save(f"{out_dir}/y.npy", y)
    joblib.dump(scaler, f"{out_dir}/scaler.joblib")


    import json
    with open(f"{out_dir}/feature_names.json", "w") as f:
        json.dump(list(df.columns), f)

    print(f"[Preprocessing] Outputs saved to: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV dataset")
    parser.add_argument("--out-dir", required=True, help="Directory to save processed arrays")
    args = parser.parse_args()
    preprocess(args.input, args.out_dir)
