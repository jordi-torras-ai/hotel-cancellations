"""Integration checks against the course data and real trained estimator."""
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app
from domain import FEATURES, InputError, encode, load_data
from train import DEFAULT_DATA, load_or_train


@pytest.fixture(scope="module")
def bundle():
    return load_or_train()


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def booking():
    return {"lead_time": 211, "market_segment_type": "Online", "avg_price_per_room": 100,
            "no_of_adults": 2, "no_of_weekend_nights": 0, "no_of_week_nights": 2,
            "no_of_special_requests": 0, "arrival_month": 5, "required_car_parking_space": 0}


def test_notebook_features_and_held_out_score(bundle):
    """Independently reproduce the notebook preprocessing and evaluate saved model."""
    original = pd.read_csv(DEFAULT_DATA)
    original["arrival_month"] = pd.to_datetime(original["arrival_date"]).dt.month
    x = pd.get_dummies(original.drop(columns=["booking_status", "arrival_date"]), drop_first=True)
    y = original["booking_status"].eq("Canceled").astype(int)
    _, test, _, labels = train_test_split(x, y, test_size=0.3, random_state=42)
    np.testing.assert_allclose(encode(original), x[FEATURES].astype(float))
    assert len(original) == 36275
    assert len(test) == 10883
    assert f1_score(labels, bundle["model"].predict(test[FEATURES])) == pytest.approx(0.8020, abs=.0001)
    assert bundle["report"]["best_params"] == {"max_leaf_nodes": 250, "min_samples_split": 10}


@pytest.mark.parametrize("path", ["/", "/data", "/model", "/predict", "/guide"])
def test_pages_render(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert b'lang="es"' in response.data


def test_both_probability_branches(client, bundle, booking):
    cases = [booking, {**booking, "lead_time": 5, "arrival_month": 11, "no_of_special_requests": 1,
                       "avg_price_per_room": 106.68, "no_of_weekend_nights": 2, "no_of_week_nights": 3}]
    responses = []
    for payload in cases:
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200
        result = response.get_json()
        expected = bundle["model"].predict_proba(encode(pd.DataFrame([payload])))[0, 1]
        assert result["probability"] == pytest.approx(expected)
        assert result["expected_exposure"] == pytest.approx(expected * result["booking_value"])
        assert result["rules"]
        responses.append(result)
    assert responses[0]["cancelled"] is True
    assert responses[1]["cancelled"] is False
    assert responses[0]["probability"] > responses[1]["probability"]


@pytest.mark.parametrize("field,value", [("lead_time", -1), ("arrival_month", 13), ("no_of_adults", 1.5),
                                        ("avg_price_per_room", "nan"), ("market_segment_type", "Unknown"),
                                        ("required_car_parking_space", 2), ("no_of_week_nights", None)])
def test_invalid_inputs(client, booking, field, value):
    response = client.post("/api/predict", json={**booking, field: value})
    assert response.status_code == 422
    assert "error" in response.get_json()


@pytest.mark.parametrize("payload", [None, [], "abc", 42, {}])
def test_invalid_json_shape(client, payload):
    response = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 422


def test_batch_matches_individual_predictions(client, booking):
    frame = pd.DataFrame([booking, {**booking, "market_segment_type": "Offline", "lead_time": 5}])
    response = client.post("/api/batch", data={"file": (io.BytesIO(frame.to_csv(index=False).encode()), "bookings.csv")})
    assert response.status_code == 200
    output = pd.read_csv(io.BytesIO(response.data))
    assert len(output) == 2
    assert "attachment" in response.headers["Content-Disposition"]
    for index, payload in enumerate(frame.to_dict("records")):
        single = client.post("/api/predict", json=payload).get_json()
        assert output.iloc[index]["cancellation_probability"] == pytest.approx(single["probability"], abs=1e-6)
        assert output.iloc[index]["predicted_booking_status"] == ("Canceled" if single["cancelled"] else "Not_Canceled")


def test_batch_accepts_notebook_dates_and_downloadable_sample(client):
    frame = pd.read_csv(DEFAULT_DATA).head(10)
    response = client.post("/api/batch", data={"file": (io.BytesIO(frame.to_csv(index=False).encode()), "course.csv")})
    assert response.status_code == 200
    sample = client.get("/download/sample")
    response = client.post("/api/batch", data={"file": (io.BytesIO(sample.data), "example.csv")})
    assert response.status_code == 200


def test_batch_errors(client):
    for data in [{}, {"file": (io.BytesIO(b""), "empty.csv")}, {"file": (io.BytesIO(b"x,y\n1,2"), "bad.csv")}]:
        response = client.post("/api/batch", data=data)
        assert response.status_code == 422


def test_filters_and_pagination(client):
    data = load_data(DEFAULT_DATA)
    response = client.get("/api/data?month=7&status=Canceled&page=2")
    result = response.get_json()
    assert result["page"] == 2
    assert result["total"] == len(data.loc[data.arrival_month.eq(7) & data.booking_status.eq("Canceled")])
    assert all(row["arrival_month"] == 7 and row["booking_status"] == "Canceled" for row in result["rows"])
    assert client.get("/api/data?month=99").status_code == 422
    assert client.get("/api/data?page=abc").status_code == 422


def test_exported_artifacts(client, bundle):
    assert client.get("/download/data").status_code == 200
    assert client.get("/download/model").status_code == 200
    report = client.get("/download/report").get_json()
    assert report["dataset_sha256"] == bundle["report"]["dataset_sha256"]


def test_training_endpoint(client):
    response = client.post("/api/train")
    assert response.status_code == 200
    assert response.get_json()["tuned"]["test"]["f1"] == pytest.approx(.8020, abs=.0001)
