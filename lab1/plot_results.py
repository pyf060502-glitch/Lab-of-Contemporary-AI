"""Generate only confusion matrices, model comparison, loss and accuracy curves."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_preprocessing import OUTPUT, read_json


def plot(output=OUTPUT, keep_svm=False, keep_mlp=False):
    output = Path(output)
    results = pd.read_csv(output / "evaluation/results.csv")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    best = []
    split_seed = read_json(output / "data/split.json")["seed"]
    for kind in ("svm", "mlp"):
        row = results[(results.model == kind) & (results.seed == split_seed)].iloc[0]
        best.append(row)
        if kind == "svm" and keep_svm:
            continue
        if kind == "mlp" and keep_mlp:
            continue
        run = output / "runs" / row.run_id
        if kind == "svm":
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(["Train", "Validation"], [row.train_accuracy, row.validation_accuracy])
            ax.set(ylim=(0, 1), ylabel="Accuracy", title=f"SVC accuracy - {row.run_id}")
            fig.tight_layout()
            fig.savefig(figures / "svm_accuracy.png", dpi=160)
            plt.close(fig)
            (figures / "svm_loss.png").unlink(missing_ok=True)
        history = pd.read_csv(run / "history.csv") if kind == "mlp" else None
        for metric in (("loss", "accuracy") if kind == "mlp" else ()):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(history.epoch, history[f"train_{metric}"], label="Train")
            ax.plot(history.epoch, history[f"validation_{metric}"], label="Validation")
            ax.axvline(row.selected_epoch, color="gray", linestyle="--", label="Selected epoch")
            detail = "mean OVR hinge" if kind == "svm" else "cross-entropy"
            ax.set(
                xlabel="Epoch", ylabel="Accuracy" if metric == "accuracy" else f"Data loss ({detail})",
                title=f"{kind.upper()} MBGD - {metric} - {row.run_id}",
            )
            if metric == "accuracy":
                ax.set_ylim(0, 1.02)
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / f"{kind}_{metric}.png", dpi=160)
            plt.close(fig)
        matrix = np.asarray(read_json(run / "metrics.json")["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, cmap="Blues")
        for i in range(10):
            for j in range(10):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set(xticks=np.arange(10), yticks=np.arange(10), xlabel="Predicted label",
               ylabel="True label", title=f"{kind.upper()} validation confusion matrix")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(figures / f"{kind}_confusion_matrix.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(2)
    ax.bar(positions - 0.18, [row.validation_accuracy for row in best], width=0.36, label="Accuracy")
    ax.bar(positions + 0.18, [row.validation_macro_f1 for row in best], width=0.36, label="Macro-F1")
    ax.set(xticks=positions, xticklabels=["Linear SVM", "MLP"], ylim=(0, 1),
           ylabel="Validation score", title="Best validation checkpoint per model")
    ax.legend()
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "model_comparison.png", dpi=160)
    plt.close(fig)
    print(f"Updated figures in {figures}; SVC has no epoch loss/accuracy curves.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--keep-svm", action="store_true", help="Do not regenerate SVM images.")
    parser.add_argument("--keep-mlp", action="store_true", help="Do not regenerate MLP images.")
    args = parser.parse_args()
    plot(args.output, args.keep_svm, args.keep_mlp)
