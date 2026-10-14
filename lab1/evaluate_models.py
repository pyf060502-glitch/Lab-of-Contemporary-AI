"""Recompute validation metrics and freeze the best candidate before loading test data."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from data_preprocessing import OUTPUT, read_json, save_json, sha256
from train_models import load_features, measure, predict


def evaluate(output=OUTPUT, mlp_run_prefix=None):
    output = Path(output)
    x, y, xv, yv = load_features(output)
    split = read_json(output / "data/split.json")
    rows = []
    with threadpool_limits(limits=1):
        for run in sorted((output / "runs").iterdir()):
            if not (run / "run.json").exists():
                continue
            meta = read_json(run / "run.json")
            if meta["features_sha256"] != sha256(output / "features/features.json"):
                raise ValueError(f"Stale features in {run.name}")
            if meta["model_sha256"] != sha256(run / "model.joblib"):
                raise ValueError("Saved model hash mismatch.")
            if meta["history_sha256"] and meta["history_sha256"] != sha256(run / "history.csv"):
                raise ValueError("Saved history hash mismatch.")
            bundle = joblib.load(run / "model.joblib")
            model, config = bundle["model"], bundle["config"]
            kind = "svm" if isinstance(model, SVC) else model["kind"]
            if kind == "svm" and not isinstance(model, SVC):
                continue
            if kind == "mlp" and mlp_run_prefix and not run.name.startswith(mlp_run_prefix):
                continue
            train = measure(model, x, y, config.get("regularization", 0))
            validation = measure(model, xv, yv, config.get("regularization", 0))
            predictions = predict(model, xv)
            if kind == "mlp":
                selected = pd.read_csv(run / "history.csv").set_index("epoch").loc[meta["selected_epoch"]]
                assert np.isclose(selected.validation_loss, validation["loss"])
                assert np.isclose(selected.validation_accuracy, validation["accuracy"])
            row = {
                "run_id": run.name, "model": kind, "seed": config["seed"],
                "learning_rate": config.get("learning_rate"), "regularization": config.get("regularization"),
                "hidden": config.get("hidden") if kind == "mlp" else None,
                "batch_size": config.get("batch_size"), "selected_epoch": meta["selected_epoch"],
                "executed_epochs": meta["executed_epochs"],
                "train_accuracy": train["accuracy"], "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": f1_score(yv, predictions, average="macro", zero_division=0),
                "train_loss": train["loss"], "validation_loss": validation["loss"],
                "accuracy_gap": train["accuracy"] - validation["accuracy"],
                "fit_and_monitor_seconds": meta["fit_and_monitor_seconds"],
            }
            metrics = {
                **row, "confusion_matrix": confusion_matrix(yv, predictions, labels=np.arange(10)).tolist(),
                "classification_report": classification_report(
                    yv, predictions, labels=np.arange(10), output_dict=True, zero_division=0,
                ),
            }
            if (run / "metrics.json").exists():
                existing = read_json(run / "metrics.json")
                for key in ("train_accuracy", "validation_accuracy", "validation_macro_f1"):
                    assert np.isclose(existing[key], metrics[key]), "Existing metrics changed."
                saved = pd.read_csv(run / "validation_predictions.csv")
                assert np.array_equal(saved.predicted_label, predictions)
                assert np.array_equal(saved.true_label, yv)
                row = {key: existing[key] for key in row}
            else:
                save_json(run / "metrics.json", metrics)
                pd.DataFrame({
                    "row_id": split["validation_row_ids"], "true_label": yv,
                    "predicted_label": predictions,
                }).to_csv(run / "validation_predictions.csv", index=False)
            rows.append(row)
    if not rows or {row["model"] for row in rows} != {"svm", "mlp"}:
        raise ValueError("Complete at least one successful SVM and MLP run before selection.")
    frame = pd.DataFrame(rows).sort_values(
        ["validation_accuracy", "validation_macro_f1", "run_id"],
        ascending=[False, False, True],
    )
    destination = output / "evaluation"
    destination.mkdir(exist_ok=True)
    frame.to_csv(destination / "results.csv", index=False)
    # Repeated seeds measure uncertainty; do not deploy the luckiest seed.
    candidates = frame[frame.seed == split["seed"]]
    if candidates.empty:
        raise ValueError("Selection requires a candidate with the fixed split seed.")
    winner = candidates.iloc[0]
    run = output / "runs" / winner.run_id
    selected_bundle = joblib.load(run / "model.joblib")
    save_json(destination / "selected.json", {
        "source_run": winner.run_id, "model": winner.model,
        "config": selected_bundle["config"],
        "selected_epoch": int(winner.selected_epoch) if pd.notna(winner.selected_epoch) else None,
        "validation_accuracy": float(winner.validation_accuracy),
        "validation_macro_f1": float(winner.validation_macro_f1),
        "selection_rule": "Fixed split seed candidates only: Accuracy then Macro-F1, then run_id; epoch by validation data loss",
        "split_sha256": sha256(output / "data/split.json"),
        "features_sha256": sha256(output / "features/features.json"),
        "model_sha256": sha256(run / "model.joblib"),
        "mlp_run_prefix": mlp_run_prefix,
    })
    repeats = frame[
        (frame.model == winner.model)
        & (frame.regularization == winner.regularization)
        & (frame.learning_rate == winner.learning_rate)
        & (frame.batch_size == winner.batch_size)
    ]
    if winner.model == "svm":
        repeats = frame[frame.run_id == winner.run_id]
    if winner.model == "mlp":
        repeats = repeats[repeats.hidden == winner.hidden]
    repeats = repeats.drop_duplicates("seed")
    save_json(destination / "repeatability.json", {
        "configuration_source": winner.run_id, "seeds": repeats.seed.astype(int).tolist(),
        "validation_accuracy": repeats.validation_accuracy.tolist(),
        "mean_accuracy": float(repeats.validation_accuracy.mean()),
        "sample_std_accuracy": float(repeats.validation_accuracy.std()) if len(repeats) > 1 else None,
        "limit": "Same split; initialization and batch-order variation only. Not a new test estimate.",
    })
    print(frame[["run_id", "selected_epoch", "train_accuracy", "validation_accuracy",
                 "validation_macro_f1", "accuracy_gap"]].to_string(index=False))
    print(f"Frozen candidate: {winner.run_id}. No test data loaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--mlp-run-prefix", help="Only include MLP runs with this prefix; keep all SVM runs.")
    args = parser.parse_args()
    evaluate(args.output, args.mlp_run_prefix)
