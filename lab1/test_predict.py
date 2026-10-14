"""Refit a frozen validation choice on all labeled data, then predict unlabeled test."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from data_preprocessing import (
    DATA_DIR, OUTPUT, ROOT, clean_text, environment, load_csv, read_json, save_json, sha256,
)
from tfidf_features import make_vectorizer
from train_models import fit_epochs, predict


def test_predict(data_dir=DATA_DIR, output=OUTPUT):
    output, data_dir = Path(output), Path(data_dir)
    final = output / "final"
    if final.exists():
        raise FileExistsError("Final result already exists; use a new --output for a new experiment.")
    selection = read_json(output / "evaluation/selected.json")
    split = read_json(output / "data/split.json")
    if selection["split_sha256"] != sha256(output / "data/split.json"):
        raise ValueError("Frozen split changed.")
    if selection["features_sha256"] != sha256(output / "features/features.json"):
        raise ValueError("Frozen feature configuration changed.")
    if selection["model_sha256"] != sha256(output / "runs" / selection["source_run"] / "model.joblib"):
        raise ValueError("Frozen validation model changed.")
    if split["source_train_sha256"] != sha256(data_dir / "train_data.csv"):
        raise ValueError("Original labeled data changed.")
    source_hashes = {name: sha256(data_dir / name)
                     for name in ("train_data.csv", "test_data_unlabeled.csv")}
    train = load_csv(data_dir / "train_data.csv")
    test = load_csv(data_dir / "test_data_unlabeled.csv", labeled=False)
    cleaning = split["cleaning"]
    train_text = [clean_text(text, cleaning) for text in train.text]
    test_text = [clean_text(text, cleaning) for text in test.text]
    feature_config = read_json(output / "features/features.json")["config"]
    vectorizer = make_vectorizer(feature_config)
    x_train = vectorizer.fit_transform(train_text).tocsr()
    x_test = vectorizer.transform(test_text).tocsr()
    config = dict(selection["config"])
    if selection["model"] == "mlp":
        config["epochs"] = selection["selected_epoch"]
        config["patience"] = 0
    with threadpool_limits(limits=1):
        model, _, diagnostics = fit_epochs(
            selection["model"], x_train, train.target.to_numpy(), config, validation=None,
        )
        predictions = predict(model, x_test)
    final.mkdir(parents=True)
    bundle = {"model": model, "vectorizer": vectorizer, "cleaning": cleaning,
              "config": config, "feature_config": feature_config}
    joblib.dump(bundle, final / "model.joblib")
    pd.DataFrame(predictions).to_csv(final / "predictions.csv", index=False, header=False)
    loaded = joblib.load(final / "model.joblib")
    with threadpool_limits(limits=1):
        replay = predict(loaded["model"], loaded["vectorizer"].transform(test_text).tocsr())
    submitted = pd.read_csv(final / "predictions.csv", header=None)
    if submitted.shape != (len(test), 1) or not pd.api.types.is_integer_dtype(submitted[0]):
        raise AssertionError("Prediction CSV must be a headerless single integer column.")
    if not submitted[0].isin(range(10)).all() or not np.array_equal(submitted[0], predictions):
        raise AssertionError("Invalid labels or row order.")
    if not np.array_equal(replay, predictions):
        raise AssertionError("Reloaded model predictions differ.")
    assert all(sha256(data_dir / name) == digest for name, digest in source_hashes.items())
    save_json(final / "submission_check.json", {
        "shape": list(submitted.shape), "labels_0_to_9": True, "reload_predictions_match": True,
        "original_training_count": len(train), "test_count": len(test),
        "selection": selection, "training": diagnostics,
        "source_sha256": source_hashes, "predictions_sha256": sha256(final / "predictions.csv"),
        "model_sha256": sha256(final / "model.joblib"), "environment": environment(),
        "code_sha256": {path.name: sha256(path) for path in ROOT.glob("*.py")},
        "test_accuracy": None, "note": "Test labels are unavailable; no test-score or tuning.",
    })
    print(f"Validated {len(test)} ordered test predictions. Test accuracy is unavailable.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    test_predict(args.data_dir, args.output)
