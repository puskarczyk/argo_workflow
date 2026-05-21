import argparse
import numpy as np
from imblearn.over_sampling import SMOTE
from pathlib import Path
from sklearn.model_selection import train_test_split


def handle_imbalance(data_dir: str, out_dir: str, test_size: float, random_state: int) -> None:
    print(f"[Imbalance] Loading arrays from: {data_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    X = np.load(f"{data_dir}/X_processed.npy")
    y = np.load(f"{data_dir}/y.npy")

    # train-test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"[Imbalance] Train size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")
    print(f"[Imbalance] Fraud in train before SMOTE: {y_train.sum()} / {len(y_train)}")

    # SMOTE on training 
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"[Imbalance] Train size after SMOTE: {X_train_res.shape[0]}")
    unique, counts = np.unique(y_train_res, return_counts=True)
    print(f"[Imbalance] Class distribution after SMOTE: {dict(zip(unique.tolist(), counts.tolist()))}")

    np.save(f"{out_dir}/X_train_res.npy", X_train_res)
    np.save(f"{out_dir}/y_train_res.npy", y_train_res)
    np.save(f"{out_dir}/X_test.npy", X_test)
    np.save(f"{out_dir}/y_test.npy", y_test)
    print(f"[Imbalance] Outputs saved to: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Dir with X_processed.npy / y.npy")
    parser.add_argument("--out-dir", required=True, help="Dir to save resampled arrays")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    handle_imbalance(args.data_dir, args.out_dir, args.test_size, args.random_state)
