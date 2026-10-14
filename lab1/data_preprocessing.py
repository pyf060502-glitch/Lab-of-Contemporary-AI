"""Read original labeled data, group-split it, and clean text without augmentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data-Project1" / "data-Project1"
OUTPUT = ROOT / "outputs"
IDENTITY_HEADERS = {
    "from", "reply-to", "sender", "organization", "nntp-posting-host",
    "path", "x-newsreader",
}
KNOWN_HEADERS = IDENTITY_HEADERS | {
    "subject", "date", "lines", "message-id", "references", "newsgroups",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment():
    import joblib
    import matplotlib
    import scipy
    import sklearn
    from threadpoolctl import threadpool_info

    return {
        "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__, "matplotlib": matplotlib.__version__,
        "platform": platform.platform(), "processor": platform.processor(),
        "blas": threadpool_info(),
    }


def load_csv(path, labeled=True):
    frame = pd.read_csv(path)
    if "text" not in frame or frame.text.isna().any():
        raise ValueError("CSV must contain non-missing text.")
    if not frame.text.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Text must be a non-empty string; no rows are silently removed.")
    if labeled:
        if "target" not in frame:
            raise ValueError("Labeled CSV requires target.")
        labels = pd.to_numeric(frame.target, errors="coerce")
        if labels.isna().any() or not np.isfinite(labels).all() or (labels % 1 != 0).any():
            raise ValueError("Labels must be finite integers.")
        frame["target"] = labels.astype(np.int64)
        if sorted(frame.target.unique()) != list(range(10)):
            raise ValueError("Expected the ten labels 0 through 9.")
    else:
        if "target" in frame:
            raise ValueError("This prediction workflow expects an unlabeled test CSV.")
    frame.insert(0, "row_id", np.arange(len(frame)))
    return frame


def clean_text(text, mode="light"):
    if mode == "raw":
        return text
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if mode == "body":
        head, separator, body = normalized.partition("\n\n")
        lines = head.split("\n")
        fields = [
            line.split(":", 1)[0].strip().lower()
            for line in lines if ":" in line and not line[:1].isspace()
        ]
        # Only parse an identifiable mail header. Folded lines inherit their field.
        if separator and sum(field in KNOWN_HEADERS for field in fields) >= 2:
            kept, removing = [], False
            for line in lines:
                if line[:1].isspace():
                    if not removing:
                        kept.append(line)
                    continue
                if ":" not in line:
                    kept.append(line)
                    removing = False
                    continue
                removing = line.split(":", 1)[0].strip().lower() in IDENTITY_HEADERS
                if not removing:
                    kept.append(line)
            normalized = "\n".join(kept) + "\n\n" + body
    elif mode != "light":
        raise ValueError(f"Unknown cleaning mode: {mode}")
    result = "\n".join(re.sub(r"[ \t\f\v]+", " ", line).strip() for line in normalized.split("\n"))
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result or text


def duplicate_groups(texts, threshold=0.9):
    """Fixed shingle signatures for grouping, not a learned classification feature."""
    parent = list(range(len(texts)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    exact, shingles, buckets = {}, [], defaultdict(list)
    exact_pairs = []
    for index, text in enumerate(texts):
        canonical = " ".join(text.casefold().split())
        if canonical in exact:
            union(exact[canonical], index)
            exact_pairs.append([exact[canonical], index])
        else:
            exact[canonical] = index
        tokens = re.findall(r"[a-z0-9_]+", text.casefold())
        items = {" ".join(tokens[i:i + 5]) for i in range(max(0, len(tokens) - 4))}
        shingles.append(items)
        for signature in sorted(
            int.from_bytes(hashlib.sha1(item.encode()).digest()[:8], "big") for item in items
        )[:8]:
            buckets[signature].append(index)
    candidates = set()
    for indices in buckets.values():
        if 1 < len(indices) <= 100:
            candidates.update((left, right) for i, left in enumerate(indices) for right in indices[i + 1:])
    near_pairs = []
    for left, right in sorted(candidates):
        combined = shingles[left] | shingles[right]
        if combined:
            similarity = len(shingles[left] & shingles[right]) / len(combined)
            if similarity >= threshold:
                union(left, right)
                near_pairs.append({"left": left, "right": right, "jaccard": similarity})
    return np.array([find(i) for i in range(len(texts))]), exact_pairs, near_pairs


def prepare(data_dir=DATA_DIR, output=OUTPUT, seed=42, cleaning="light", threshold=0.9):
    path = Path(data_dir) / "train_data.csv"
    destination = Path(output) / "data"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}. Use a new --output directory.")
    frame = load_csv(path)
    groups, exact, near = duplicate_groups(frame.text.tolist(), threshold)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_ids, val_ids = next(splitter.split(np.zeros(len(frame)), frame.target, groups))
    if set(groups[train_ids]) & set(groups[val_ids]):
        raise AssertionError("A duplicate source group crosses the split.")
    frame["group_id"] = groups
    # The fixed audit above only defines groups; cleaning and TF-IDF follow the split.
    train, validation = frame.iloc[train_ids].copy(), frame.iloc[val_ids].copy()
    for part in (train, validation):
        part["text"] = [clean_text(text, cleaning) for text in part.text]
    destination.mkdir(parents=True)
    train.to_csv(destination / "train.csv", index=False)
    validation.to_csv(destination / "validation.csv", index=False)
    meta = {
        "source_train_sha256": sha256(path), "source_rows": len(frame),
        "seed": seed, "cleaning": cleaning, "augmentation": False,
        "split_method": "first fold of seeded StratifiedGroupKFold(n_splits=5)",
        "train_count": len(train), "validation_count": len(validation),
        "train_row_ids": train_ids.tolist(), "validation_row_ids": val_ids.tolist(),
        "train_class_counts": np.bincount(train.target, minlength=10).tolist(),
        "validation_class_counts": np.bincount(validation.target, minlength=10).tolist(),
        "exact_duplicate_pairs": exact, "near_duplicate_pairs": near,
        "jaccard_threshold": threshold, "group_overlap": 0,
        "audit_limit": "8 minimum shingle hashes, buckets <=100; not exhaustive semantic deduplication",
        "train_csv_sha256": sha256(destination / "train.csv"),
        "validation_csv_sha256": sha256(destination / "validation.csv"),
    }
    save_json(destination / "split.json", meta)
    print(f"Prepared train={len(train)}, validation={len(validation)}, test NOT loaded.")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cleaning", choices=["raw", "light", "body"], default="light")
    parser.add_argument("--duplicate-threshold", type=float, default=0.9)
    args = parser.parse_args()
    if not 0 < args.duplicate_threshold <= 1:
        parser.error("duplicate-threshold must be in (0, 1].")
    prepare(args.data_dir, args.output, args.seed, args.cleaning, args.duplicate_threshold)
