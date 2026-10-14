"""Fit TF-IDF on training text only and save sparse matrices."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from data_preprocessing import OUTPUT, read_json, save_json, sha256


def make_vectorizer(config):
    return TfidfVectorizer(
        max_features=config["max_features"], ngram_range=(1, config["ngram_max"]),
        min_df=config["min_df"], max_df=config["max_df"],
        lowercase=True, stop_words=None, sublinear_tf=config["sublinear_tf"],
        smooth_idf=True, norm="l2", dtype=np.float64,
    )


def extract(output=OUTPUT, max_features=5000, ngram_max=1, min_df=2, sublinear_tf=True):
    output = Path(output)
    destination = output / "features"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}.")
    split = read_json(output / "data/split.json")
    for name in ("train", "validation"):
        if sha256(output / f"data/{name}.csv") != split[f"{name}_csv_sha256"]:
            raise ValueError("Prepared data changed after splitting.")
    train = pd.read_csv(output / "data/train.csv")
    validation = pd.read_csv(output / "data/validation.csv")
    config = {
        "max_features": max_features, "ngram_max": ngram_max, "min_df": min_df,
        "max_df": 1.0, "sublinear_tf": sublinear_tf,
    }
    vectorizer = make_vectorizer(config)
    x_train = vectorizer.fit_transform(train.text).tocsr()
    vocab, idf = dict(vectorizer.vocabulary_), vectorizer.idf_.copy()
    x_val = vectorizer.transform(validation.text).tocsr()
    assert vocab == vectorizer.vocabulary_ and np.array_equal(idf, vectorizer.idf_)
    for matrix in (x_train, x_val):
        if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
            raise ValueError("Invalid TF-IDF values.")
        norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
        if not np.allclose(norms[norms > 0], 1):
            raise AssertionError("Non-empty vectors must have L2 norm 1.")
    destination.mkdir(parents=True)
    sparse.save_npz(destination / "train.npz", x_train)
    sparse.save_npz(destination / "validation.npz", x_val)
    np.save(destination / "y_train.npy", train.target.to_numpy(dtype=np.int64))
    np.save(destination / "y_validation.npy", validation.target.to_numpy(dtype=np.int64))
    joblib.dump(vectorizer, destination / "vectorizer.joblib")
    paths = ["train.npz", "validation.npz", "y_train.npy", "y_validation.npy", "vectorizer.joblib"]
    meta = {
        "config": config, "split_sha256": sha256(output / "data/split.json"),
        "train_shape": list(x_train.shape), "validation_shape": list(x_val.shape),
        "train_density": x_train.nnz / np.prod(x_train.shape),
        "train_zero_rows": int((x_train.getnnz(axis=1) == 0).sum()),
        "validation_zero_rows": int((x_val.getnnz(axis=1) == 0).sum()),
        "files_sha256": {name: sha256(destination / name) for name in paths},
        "fit_scope": "cleaned training originals only; no test or augmentation",
    }
    save_json(destination / "features.json", meta)
    print(f"TF-IDF: train {x_train.shape}, validation {x_val.shape}, L2 normalization.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--ngram-max", type=int, choices=[1, 2], default=1)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--sublinear-tf", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_features <= 0 or args.min_df <= 0:
        parser.error("max-features and min-df must be positive integers.")
    extract(args.output, args.max_features, args.ngram_max, args.min_df, args.sublinear_tf)
