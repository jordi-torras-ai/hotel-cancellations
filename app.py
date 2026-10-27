"""Local Python + HTML application for the EADA hotel cancellation case study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

from domain import FEATURES, INPUTS, LABELS, MONTHS, NUMERIC_FIELDS, InputError, encode, load_data, normalize_inputs
from train import DEFAULT_ARTIFACTS, DEFAULT_DATA, load_or_train, train


def cancellation_probabilities(model, features):
    return model.predict_proba(features)[:, list(model.classes_).index(1)]


def predict_booking(bundle, payload):
    if not isinstance(payload, dict):
        raise InputError("Envía un objeto JSON con los datos de la reserva.")
    inputs = normalize_inputs(pd.DataFrame([payload]))
    features = encode(inputs)
    model = bundle["model"]
    probability = float(cancellation_probabilities(model, features)[0])
    predicted = int(model.predict(features)[0])
    path = model.decision_path(features).indices
    rules = []
    for node in path:
        feature_index = model.tree_.feature[node]
        if feature_index < 0:
            continue
        feature = FEATURES[feature_index]
        value = float(features.iloc[0, feature_index])
        threshold = float(model.tree_.threshold[node])
        rules.append({"label": LABELS[feature], "operator": "≤" if value <= threshold else ">",
                      "threshold": round(threshold, 2), "value": value})
    warnings = []
    for feature in FEATURES:
        low, high = bundle["report"]["ranges"][feature]
        if not low <= float(features.iloc[0][feature]) <= high:
            warnings.append(f"{LABELS[feature]} queda fuera del rango histórico ({low:g}–{high:g}).")
    row = inputs.iloc[0]
    booking_value = float(row["avg_price_per_room"] * (row["no_of_week_nights"] + row["no_of_weekend_nights"]))
    leaf = int(model.apply(features)[0])
    return {"cancelled": bool(predicted), "probability": probability, "booking_value": booking_value,
            "expected_exposure": probability * booking_value, "leaf_samples": int(model.tree_.n_node_samples[leaf]),
            "rules": rules, "warnings": warnings}


def create_app(data_path=DEFAULT_DATA, artifact_dir=DEFAULT_ARTIFACTS):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=10 * 1024 * 1024, JSON_SORT_KEYS=False)
    data_path, artifact_dir = Path(data_path), Path(artifact_dir)
    data = load_data(data_path)
    state = {"bundle": load_or_train(data_path, artifact_dir)}
    train_lock = threading.Lock()

    @app.get("/api/health")
    def health():
        return jsonify(application="inn-hotel-intelligence", status="ready")

    @app.context_processor
    def context():
        return {"months": MONTHS, "numeric_fields": NUMERIC_FIELDS, "labels": LABELS,
                "report": state["bundle"]["report"]}

    @app.get("/")
    def dashboard():
        return render_template("dashboard.html", active="dashboard", title="Vista general")

    @app.get("/data")
    def explore():
        return render_template("data.html", active="data", title="Explorar datos", columns=data.columns.tolist())

    @app.get("/model")
    def model():
        return render_template("model.html", active="model", title="El modelo")

    @app.get("/predict")
    def predictor():
        return render_template("predict.html", active="predict", title="Nueva predicción")

    @app.get("/guide")
    def guide():
        return render_template("guide.html", active="guide", title="Del dato a la decisión")

    @app.get("/api/report")
    def report():
        return jsonify(state["bundle"]["report"])

    @app.get("/api/data")
    def rows():
        try:
            page = max(1, int(request.args.get("page", 1)))
            month = int(request.args.get("month", 0))
        except ValueError:
            raise InputError("Página y mes deben ser números enteros.")
        if not 0 <= month <= 12:
            raise InputError("El mes debe estar entre 1 y 12.")
        selected = data
        if month:
            selected = selected.loc[selected["arrival_month"].eq(month)]
        status = request.args.get("status", "")
        if status:
            if status not in ["Canceled", "Not_Canceled"]:
                raise InputError("Estado de reserva no válido.")
            selected = selected.loc[selected["booking_status"].eq(status)]
        size = 20
        pages = max(1, (len(selected) + size - 1) // size)
        page = min(page, pages)
        return jsonify({"rows": selected.iloc[(page-1)*size:page*size].to_dict("records"),
                        "total": len(selected), "page": page, "pages": pages})

    @app.post("/api/predict")
    def predict():
        return jsonify(predict_booking(state["bundle"], request.get_json()))

    @app.post("/api/train")
    def retrain():
        if not train_lock.acquire(blocking=False):
            return jsonify(error="Ya hay un entrenamiento en curso."), 409
        try:
            state["bundle"] = train(data_path, artifact_dir)
        finally:
            train_lock.release()
        return jsonify(state["bundle"]["report"])

    @app.post("/api/batch")
    def batch():
        upload = request.files.get("file")
        if not upload or not upload.filename:
            raise InputError("Selecciona un archivo CSV.")
        try:
            frame = pd.read_csv(upload, nrows=50001)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError):
            raise InputError("No se puede leer el CSV. Usa UTF-8 y columnas separadas por comas.")
        if len(frame) > 50000:
            raise InputError("El límite es de 50.000 reservas por archivo.")
        normalized = normalize_inputs(frame)
        features = encode(normalized)
        # Use a single model snapshot even if another request finishes retraining.
        model = state["bundle"]["model"]
        output = normalized[INPUTS].copy()
        output["predicted_booking_status"] = np.where(model.predict(features) == 1, "Canceled", "Not_Canceled")
        output["cancellation_probability"] = cancellation_probabilities(model, features).round(6)
        output["booking_value_eur"] = (output["avg_price_per_room"] * (output["no_of_week_nights"] + output["no_of_weekend_nights"])).round(2)
        return Response(output.to_csv(index=False), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=predicciones.csv"})

    @app.get("/download/<kind>")
    def download(kind):
        if kind == "sample":
            sample = data.head(8).drop(columns=["booking_status", "arrival_date"])
            return Response(sample.to_csv(index=False), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=ejemplo_reservas.csv"})
        if kind == "data":
            return send_file(data_path, as_attachment=True)
        if kind == "report":
            return send_file(artifact_dir / "report.json", as_attachment=True)
        if kind == "model":
            return send_file(artifact_dir / "model.joblib", as_attachment=True)
        return jsonify(error="Archivo no encontrado."), 404

    @app.errorhandler(InputError)
    def invalid(error):
        return jsonify(error=str(error)), 422

    @app.errorhandler(HTTPException)
    def http_error(error):
        if request.path.startswith("/api/"):
            messages = {413: "El archivo supera el límite de 10 MB.", 400: "El JSON enviado no es válido.",
                        415: "Envía los datos como application/json."}
            return jsonify(error=messages.get(error.code, error.description)), error.code
        return error

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--port", type=int, default=5055)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("El puerto debe estar entre 1 y 65535.")
    address = f"http://127.0.0.1:{args.port}"
    try:
        with urlopen(f"{address}/api/health", timeout=1) as response:
            existing = json.load(response)
        if existing.get("application") == "inn-hotel-intelligence":
            print(f"El servidor ya está iniciado: {address}")
            raise SystemExit(0)
    except (URLError, TimeoutError, ValueError):
        pass
    create_app(args.data, args.artifacts).run(host="127.0.0.1", port=args.port, debug=False)
