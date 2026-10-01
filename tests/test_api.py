"""Pruebas de contrato básico para la API local."""

from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health_endpoint_reports_loaded_model():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_returns_probability_and_binary_class():
    payload = {
        "features": {
            "Age": 54,
            "Sex": "M",
            "ChestPainType": "ATA",
            "RestingBP": 140,
            "Cholesterol": 240,
            "FastingBS": 0,
            "RestingECG": "Normal",
            "MaxHR": 150,
            "ExerciseAngina": "N",
            "Oldpeak": 0.5,
            "ST_Slope": "Up",
        }
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert 0 <= result["heart_disease_probability"] <= 1
    assert result["prediction"] in (0, 1)
    assert result["threshold"] == 0.5


def test_prediction_rejects_unknown_category():
    payload = {
        "features": {
            "Age": 54,
            "Sex": "X",
            "ChestPainType": "ATA",
            "RestingBP": 140,
            "Cholesterol": 240,
            "FastingBS": 0,
            "RestingECG": "Normal",
            "MaxHR": 150,
            "ExerciseAngina": "N",
            "Oldpeak": 0.5,
            "ST_Slope": "Up",
        }
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
