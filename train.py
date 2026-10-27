"""Reproduce the Colab's 70/30 split, baseline tree and five-fold grid search."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

from domain import FEATURES, LABELS, MONTHS, encode, load_data

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "INNHotelsGroup.csv"
DEFAULT_ARTIFACTS = ROOT / "artifacts"
MODEL_REVISION = 1


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(model, x, y):
    predicted = model.predict(x)
    # Locate class 1 explicitly: this is always P(cancellation), regardless of prediction.
    probabilities = model.predict_proba(x)[:, list(model.classes_).index(1)]
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "rows": len(y),
    }


def summarize(data):
    frame = data.copy()
    frame["cancelled"] = frame["booking_status"].eq("Canceled").astype(int)
    frame["value"] = frame["avg_price_per_room"] * (frame["no_of_week_nights"] + frame["no_of_weekend_nights"])
    monthly = []
    for month in range(1, 13):
        group = frame[frame["arrival_month"].eq(month)]
        cancelled = int(group["cancelled"].sum())
        monthly.append({"month": month, "label": MONTHS[month-1], "bookings": len(group), "cancelled": cancelled,
                        "rate": cancelled / len(group) if len(group) else 0})
    segments = []
    for name, group in frame.groupby("market_segment_type"):
        segments.append({"label": name, "bookings": len(group), "rate": float(group["cancelled"].mean())})
    lead_time = []
    bins = [-1, 7, 30, 90, 180, float("inf")]
    labels = ["0–7 días", "8–30 días", "31–90 días", "91–180 días", ">180 días"]
    grouped = frame.groupby(pd.cut(frame["lead_time"], bins=bins, labels=labels), observed=False)
    for label, group in grouped:
        lead_time.append({"label": str(label), "bookings": len(group), "rate": float(group["cancelled"].mean()) if len(group) else 0})
    by_status = []
    for name, group in frame.groupby("booking_status"):
        by_status.append({"label": name, "price": float(group["avg_price_per_room"].mean()),
                          "lead_time": float(group["lead_time"].mean())})
    return {
        "rows": len(frame), "cancelled": int(frame["cancelled"].sum()), "rate": float(frame["cancelled"].mean()),
        "avg_price": float(frame["avg_price_per_room"].mean()),
        "cancelled_value": float(frame.loc[frame["cancelled"].eq(1), "value"].sum()),
        "date_min": frame["arrival_date"].min(), "date_max": frame["arrival_date"].max(),
        "missing": int(data.isna().sum().sum()), "duplicates": int(data.duplicated().sum()),
        "monthly": monthly, "segments": segments, "lead_time": lead_time, "by_status": by_status,
        "statistics": data.select_dtypes(include="number").describe().round(2).reset_index().to_dict("records"),
    }


def train(data_path=DEFAULT_DATA, artifact_dir=DEFAULT_ARTIFACTS):
    data_path, artifact_dir = Path(data_path), Path(artifact_dir)
    data = load_data(data_path)
    x = encode(data)
    y = data["booking_status"].eq("Canceled").astype(int)
    # Keep the exact split and seeds used in the notebook, without stratification.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    baseline = DecisionTreeClassifier(random_state=1).fit(x_train, y_train)
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=1, class_weight="balanced"),
        {"max_leaf_nodes": [150, 250], "min_samples_split": [10, 30]},
        scoring="f1", cv=5, n_jobs=1,
    ).fit(x_train, y_train)
    model = grid.best_estimator_
    report = {
        "revision": MODEL_REVISION, "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": fingerprint(data_path), "dataset_name": data_path.name,
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "features": FEATURES, "split_seed": 42, "tree_seed": 1, "train_rows": len(y_train), "test_rows": len(y_test),
        "best_params": grid.best_params_, "cv_f1": float(grid.best_score_),
        "baseline": {"train": evaluate(baseline, x_train, y_train), "test": evaluate(baseline, x_test, y_test)},
        "tuned": {"train": evaluate(model, x_train, y_train), "test": evaluate(model, x_test, y_test)},
        "grid": [{"params": p, "mean_f1": float(mean), "std_f1": float(std)} for p, mean, std in zip(
            grid.cv_results_["params"], grid.cv_results_["mean_test_score"], grid.cv_results_["std_test_score"])],
        "importance": sorted([{"feature": f, "label": LABELS[f], "value": float(v)}
                              for f, v in zip(FEATURES, model.feature_importances_)], key=lambda item: -item["value"]),
        "ranges": {f: [float(x[f].min()), float(x[f].max())] for f in FEATURES},
        "summary": summarize(data),
    }
    bundle = {"model": model, "report": report, "test_indices": x_test.index.to_numpy()}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    temporary = artifact_dir / "model.tmp"
    joblib.dump(bundle, temporary)
    os.replace(temporary, artifact_dir / "model.joblib")
    temporary_report = artifact_dir / "report.tmp"
    temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_report, artifact_dir / "report.json")
    return bundle


def load_or_train(data_path=DEFAULT_DATA, artifact_dir=DEFAULT_ARTIFACTS):
    artifact = Path(artifact_dir) / "model.joblib"
    if artifact.exists():
        # Load only the artifact generated by this project, never a user-uploaded pickle.
        bundle = joblib.load(artifact)
        report = bundle["report"]
        if (report.get("revision") == MODEL_REVISION and report["dataset_sha256"] == fingerprint(Path(data_path))
                and report["sklearn_version"] == sklearn.__version__):
            return bundle
    return train(data_path, artifact_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    result = train(args.data, args.artifacts)["report"]
    print(f"Entrenamiento: {result['train_rows']:,} | Prueba: {result['test_rows']:,}")
    print(f"F1 prueba — base: {result['baseline']['test']['f1']:.4f} | ajustado: {result['tuned']['test']['f1']:.4f}")
    print(f"Parámetros: {result['best_params']}")
