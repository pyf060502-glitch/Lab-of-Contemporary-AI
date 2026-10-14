"""Train scikit-learn SVC and a one-hidden-layer MLP using MBGD."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from data_preprocessing import OUTPUT, ROOT, environment, read_json, save_json, sha256


def initialize(kind, features, hidden, seed):
    rng = np.random.default_rng(seed)
    if kind == "svm":
        return {"kind": kind, "W": np.zeros((features, 10)), "b": np.zeros(10)}
    return {
        "kind": kind, "W1": rng.normal(0, np.sqrt(2 / features), (features, hidden)),
        "b1": np.zeros(hidden), "W2": rng.normal(0, np.sqrt(1 / hidden), (hidden, 10)),
        "b2": np.zeros(10),
    }


def forward(model, x):
    if model["kind"] == "svm":
        return np.asarray(x @ model["W"]) + model["b"], None
    hidden = np.maximum(np.asarray(x @ model["W1"]) + model["b1"], 0)
    return hidden @ model["W2"] + model["b2"], hidden


def loss_gradient(model, x, y, regularization, gradient=True):
    """Both objectives use mean data loss + lambda/2 * sum(W**2), no bias penalty."""
    scores, hidden = forward(model, x)
    rows = np.arange(len(y))
    if model["kind"] == "svm":
        target = -np.ones_like(scores)
        target[rows, y] = 1
        margins = 1 - target * scores
        data_loss = float(np.maximum(margins, 0).mean())
        penalty = regularization * np.square(model["W"]).sum() / 2
        if not gradient:
            return data_loss, float(data_loss + penalty), scores
        delta = -target * (margins > 0) / scores.size
        grads = {
            "W": np.asarray(x.T @ delta) + regularization * model["W"],
            "b": delta.sum(axis=0),
        }
    else:
        shifted = scores - scores.max(axis=1, keepdims=True)
        exponential = np.exp(shifted)
        probabilities = exponential / exponential.sum(axis=1, keepdims=True)
        data_loss = float(np.mean(np.log(exponential.sum(axis=1)) - shifted[rows, y]))
        penalty = regularization * (
            np.square(model["W1"]).sum() + np.square(model["W2"]).sum()
        ) / 2
        if not gradient:
            return data_loss, float(data_loss + penalty), scores
        delta = probabilities.copy()
        delta[rows, y] -= 1
        delta /= len(y)
        hidden_delta = (delta @ model["W2"].T) * (hidden > 0)
        grads = {
            "W2": hidden.T @ delta + regularization * model["W2"],
            "b2": delta.sum(axis=0),
            "W1": np.asarray(x.T @ hidden_delta) + regularization * model["W1"],
            "b1": hidden_delta.sum(axis=0),
        }
    return float(data_loss + penalty), grads


def measure(model, x, y, regularization, batch_size=1024):
    if isinstance(model, SVC):
        return {"loss": None, "objective": None,
                "accuracy": float(np.mean(model.predict(x) == y))}
    total_loss, correct = 0.0, 0
    for start in range(0, len(y), batch_size):
        stop = min(start + batch_size, len(y))
        data_loss, _, scores = loss_gradient(
            model, x[start:stop], y[start:stop], regularization, gradient=False
        )
        total_loss += data_loss * (stop - start)
        correct += int((scores.argmax(axis=1) == y[start:stop]).sum())
    matrices = ("W",) if model["kind"] == "svm" else ("W1", "W2")
    penalty = regularization * sum(np.square(model[key]).sum() for key in matrices) / 2
    return {
        "loss": total_loss / len(y), "objective": total_loss / len(y) + penalty,
        "accuracy": correct / len(y),
    }


def predict(model, x, batch_size=1024):
    if isinstance(model, SVC):
        return model.predict(x).astype(np.int64)
    return np.concatenate([
        forward(model, x[start:start + batch_size])[0].argmax(axis=1)
        for start in range(0, x.shape[0], batch_size)
    ]).astype(np.int64)


def fit_epochs(kind, x, y, config, validation=None, verbose=True):
    if kind == "svm":
        start = time.perf_counter()
        model = SVC(kernel="linear", C=config["C"], random_state=config["seed"])
        model.fit(x, y)
        return model, [], {
            "selected_epoch": None, "executed_epochs": None, "mbgd_updates": None,
            "fit_and_monitor_seconds": time.perf_counter() - start,
            "solver": "libsvm SVC, one-vs-one training; not MBGD",
            "support_vectors": int(model.n_support_.sum()),
        }
    model = initialize(kind, x.shape[1], config["hidden"], config["seed"])
    rng = np.random.default_rng(config["seed"])
    history, best_model, best_epoch, best_loss = [], None, 0, float("inf")
    stale, updates = 0, 0
    start_time = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        order = rng.permutation(len(y))
        for start in range(0, len(y), config["batch_size"]):
            indices = order[start:start + config["batch_size"]]
            loss, grads = loss_gradient(model, x[indices], y[indices], config["regularization"])
            if not np.isfinite(loss) or any(not np.isfinite(value).all() for value in grads.values()):
                raise FloatingPointError("Non-finite MBGD update; lower the learning rate.")
            for key, derivative in grads.items():
                model[key] -= config["learning_rate"] * derivative
            updates += 1
        train = measure(model, x, y, config["regularization"])
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train.items()}}
        if validation is not None:
            val = measure(model, *validation, config["regularization"])
            row.update({f"validation_{key}": value for key, value in val.items()})
            if val["loss"] < best_loss - config["min_delta"]:
                best_model, best_epoch, best_loss, stale = copy.deepcopy(model), epoch, val["loss"], 0
            else:
                stale += 1
        history.append(row)
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == config["epochs"]):
            suffix = "" if validation is None else (
                f" val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f}"
            )
            print(f"{kind} epoch={epoch:03d} train_loss={train['loss']:.4f}"
                  f" train_acc={train['accuracy']:.4f}{suffix}", flush=True)
        if validation is not None and config["patience"] > 0 and stale >= config["patience"]:
            break
    if validation is None:
        best_model, best_epoch = model, len(history)
    return best_model, history, {
        "selected_epoch": best_epoch, "executed_epochs": len(history),
        "mbgd_updates": updates, "fit_and_monitor_seconds": time.perf_counter() - start_time,
    }


def load_features(output):
    output = Path(output)
    meta = read_json(output / "features/features.json")
    if meta["split_sha256"] != sha256(output / "data/split.json"):
        raise ValueError("Feature split hash mismatch.")
    for name, expected in meta["files_sha256"].items():
        if sha256(output / "features" / name) != expected:
            raise ValueError(f"Feature artifact changed: {name}")
    return (
        sparse.load_npz(output / "features/train.npz"),
        np.load(output / "features/y_train.npy"),
        sparse.load_npz(output / "features/validation.npz"),
        np.load(output / "features/y_validation.npy"),
    )


def train_run(output, kind, config, run_id, threads=1):
    output = Path(output)
    run = output / "runs" / run_id
    if run.exists():
        raise FileExistsError(f"Run already exists: {run}")
    x, y, xv, yv = load_features(output)
    run.mkdir(parents=True)
    save_json(run / "config.json", {"model": kind, **config, "threads": threads})
    try:
        with threadpool_limits(limits=threads):
            model, history, diagnostics = fit_epochs(kind, x, y, config, (xv, yv))
        if history:
            pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        joblib.dump({"model": model, "config": config}, run / "model.joblib")
        save_json(run / "run.json", {
            "status": "success", **diagnostics,
            "selection": "SVC fit on training only" if kind == "svm" else "minimum validation data loss with min_delta; validation never in gradients",
            "features_sha256": sha256(output / "features/features.json"),
            "model_sha256": sha256(run / "model.joblib"),
            "history_sha256": sha256(run / "history.csv") if history else None,
            "code_sha256": {path.name: sha256(path) for path in ROOT.glob("*.py")},
            "environment": environment(),
        })
    except Exception as error:
        save_json(run / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        raise
    print(f"Saved {run_id}, selected epoch {diagnostics['selected_epoch']}.")


def self_check():
    """In-file checks: finite-difference gradients, sparse/dense equality and reproducibility."""
    rng = np.random.default_rng(19)
    x = rng.normal(size=(6, 4))
    y = np.arange(6)
    for kind in ("mlp",):
        model = initialize(kind, 4, 3, 9)
        if kind == "svm":
            model["W"] = rng.normal(0, 0.1, model["W"].shape)
        objective, derivatives = loss_gradient(model, sparse.csr_matrix(x), y, 0.003)
        dense_objective, dense_grads = loss_gradient(model, x, y, 0.003)
        assert np.isclose(objective, dense_objective)
        for key, gradient in derivatives.items():
            assert np.allclose(gradient, dense_grads[key])
            for index in list(np.ndindex(gradient.shape))[:12]:
                original = model[key][index]
                model[key][index] = original + 1e-6
                plus = loss_gradient(model, x, y, 0.003, False)[1]
                model[key][index] = original - 1e-6
                minus = loss_gradient(model, x, y, 0.003, False)[1]
                model[key][index] = original
                assert np.isclose((plus - minus) / 2e-6, gradient[index], atol=1e-6), (kind, key)
        config = {"hidden": 3, "seed": 5, "epochs": 3, "batch_size": 4,
                  "learning_rate": 0.1, "regularization": 0.003, "patience": 0, "min_delta": 0}
        first, history, _ = fit_epochs(kind, sparse.csr_matrix(x), y, config, verbose=False)
        second, _, _ = fit_epochs(kind, sparse.csr_matrix(x), y, config, verbose=False)
        assert all(np.array_equal(first[k], second[k]) for k in first if k != "kind")
        assert history[-1]["train_loss"] <= history[0]["train_loss"]
    svc = SVC(kernel="linear", C=1).fit(sparse.csr_matrix(x), y)
    assert np.array_equal(predict(svc, sparse.csr_matrix(x)), svc.predict(x))
    print("PASS: MLP gradients and MBGD reproducibility; SVC sparse prediction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--model", choices=["both", "svm", "mlp"], default="both")
    parser.add_argument("--run-id")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--regularization", type=float, default=0.0001)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, help="Default: MLP 0 (full epochs), SVM 20.")
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        if min(args.epochs, args.batch_size, args.hidden, args.threads) <= 0:
            parser.error("epochs, batch-size, hidden and threads must be positive.")
        if not np.isfinite(args.svm_c) or args.svm_c <= 0:
            parser.error("svm-c must be finite and positive.")
        if args.regularization < 0 or args.min_delta < 0 or (args.patience is not None and args.patience < 0):
            parser.error("regularization, min-delta and patience cannot be negative.")
        if args.learning_rate is not None and (args.learning_rate <= 0 or not np.isfinite(args.learning_rate)):
            parser.error("learning-rate must be finite and positive.")
        for kind in (("svm", "mlp") if args.model == "both" else (args.model,)):
            config = {
                "epochs": args.epochs, "batch_size": args.batch_size,
                "learning_rate": args.learning_rate or (2.0 if kind == "svm" else 1.0),
                "regularization": args.regularization, "hidden": args.hidden,
                "seed": args.seed,
                "patience": args.patience if args.patience is not None else (0 if kind == "mlp" else 20),
                "min_delta": args.min_delta,
            }
            if kind == "svm":
                config = {"C": args.svm_c, "seed": args.seed}
            run_id = f"{args.run_id}-{kind}" if args.run_id and args.model == "both" else args.run_id or f"{kind}-default"
            train_run(args.output, kind, config, run_id, args.threads)
