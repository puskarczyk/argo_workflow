import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot(data_dir: str, out_dir: str) -> None:
    print(f"[Evaluation] Loading results from: {data_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    try:
        with open(f"{data_dir}/metrics_lr.json", "r") as file:
            metrics_lr = json.load(file)
        print("Logistic Regression metrics loaded")
    except FileNotFoundError:
        print(f"Error: The file '{data_dir}/metrics_lr.json' was not found.")

    try:
        with open(f"{data_dir}/metrics_xgb.json", "r") as file:
            metrics_xgb = json.load(file)
        print("XGBoost metrics loaded")
    except FileNotFoundError:
        print(f"Error: The file '{data_dir}/metrics_xgb.json' was not found.")

    # convert to percentage
    cm_lr_pct = (
        np.array(metrics_lr["confusion_matrix"])
        / np.array(metrics_lr["confusion_matrix"]).sum()
        * 100
    )
    cm_xgb_pct = (
        np.array(metrics_xgb["confusion_matrix"])
        / np.array(metrics_xgb["confusion_matrix"]).sum()
        * 100
    )

    # plot confusion matrix
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(
        cm_lr_pct,
        cmap="Blues",
        cbar=False,
        annot=[[f"{v:.2f}%" for v in row] for row in cm_lr_pct],
        fmt="",
    )
    plt.title("Logistic Regression")

    plt.subplot(1, 2, 2)
    sns.heatmap(
        cm_xgb_pct,
        cmap="Blues",
        cbar=False,
        annot=[[f"{v:.2f}%" for v in row] for row in cm_xgb_pct],
        fmt="",
    )
    plt.title("XGBoost")
    plt.suptitle("Confusion Matrix (%)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cm.png")

    # ROC Curve
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(
        metrics_lr["roc_curve"]["fpr"],
        metrics_lr["roc_curve"]["tpr"],
        label="ROC curve (area = %0.2f)" % metrics_lr["roc_auc"],
    )
    plt.title("Logistic Regression")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        metrics_xgb["roc_curve"]["fpr"],
        metrics_xgb["roc_curve"]["tpr"],
        label="ROC curve (area = %0.2f)" % metrics_xgb["roc_auc"],
    )
    plt.title("XGBoost")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.suptitle("ROC Curve")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_curve.png")

    # PR Curve
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(
        metrics_lr["pr_curve"]["rec"],
        metrics_lr["pr_curve"]["prc"],
        label="PR curve (area = %0.2f)" % metrics_lr["pr_auc"],
    )
    plt.title("Logistic Regression")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        metrics_xgb["pr_curve"]["rec"],
        metrics_xgb["pr_curve"]["prc"],
        label="PR curve (area = %0.2f)" % metrics_xgb["pr_auc"],
    )
    plt.title("XGBoost")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.suptitle("PR Curve")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/pr_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Dir with resampled arrays")
    parser.add_argument("--out-dir", required=True, help="Dir to save model + metrics")
    args = parser.parse_args()
    plot(args.data_dir, args.out_dir)
