import argparse
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def train(data_dir: str, out_dir: str, max_iter: int, random_state: int) -> None:
    print(f"[Training] Loading data from: {data_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    X_train = np.load(f"{data_dir}/X_train_res.npy")
    y_train = np.load(f"{data_dir}/y_train_res.npy")
    X_test = np.load(f"{data_dir}/X_test.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")

    print(f"[Training] X_train: {X_train.shape}  X_test: {X_test.shape}")

    model = LogisticRegression(
        solver="saga",
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
    )
    print("[Training] Logistic Regression")
    model.fit(X_train, y_train)
    print("[Training] Training complete.")

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    pr_auc = average_precision_score(y_test, y_pred_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    prc, rec, _ = precision_recall_curve(y_test, y_pred_prob)

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "confusion_matrix": cm,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "pr_curve": {
            "prc": prc.tolist(),
            "rec": rec.tolist(),
        },
    }

    print("\n[Training] Evaluation on test set")
    print(f"ROC-AUC: {metrics['roc_auc']}")
    print(f"PR-AUC: {metrics['pr_auc']}")
    print(f"F1: {metrics['f1']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"Confusion matrix:\n{np.array(cm)}")
    print("\n" + classification_report(y_test, y_pred, target_names=["legit", "fraud"]))

    joblib.dump(model, f"{out_dir}/model_lr.joblib")
    with open(f"{out_dir}/metrics_lr.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Training] Model {out_dir}/model_lr.joblib")
    print(f"[Training] Metrics {out_dir}/metrics_lr.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Dir with resampled arrays")
    parser.add_argument("--out-dir", required=True, help="Dir to save model + metrics")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    train(args.data_dir, args.out_dir, args.max_iter, args.random_state)
