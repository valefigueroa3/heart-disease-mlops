"""API local que entrega una predicción a partir de variables clínicas."""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


class HeartFeatures(BaseModel):
    """Variables esperadas por el modelo, con categorías verificadas."""

    model_config = ConfigDict(extra="forbid")

    Age: float = Field(gt=0)
    Sex: Literal["M", "F"]
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"]
    RestingBP: float | None = Field(default=None, gt=0)
    Cholesterol: float | None = Field(default=None, gt=0)
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: float = Field(gt=0)
    ExerciseAngina: Literal["N", "Y"]
    Oldpeak: float
    ST_Slope: Literal["Up", "Flat", "Down"]


class PredictionRequest(BaseModel):
    """Solicitud con las variables nombradas para evitar errores de orden."""

    model_config = ConfigDict(extra="forbid")

    features: HeartFeatures


if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"No se encontró el modelo en {MODEL_PATH}. "
        "Ejecute primero el cuaderno 2_model_pipeline_cv.ipynb."
    )

model = joblib.load(MODEL_PATH)
app = FastAPI(
    title="Predicción académica de enfermedad cardíaca",
    description=(
        "API de demostración para el proyecto de MLOps. "
        "El resultado no constituye un diagnóstico médico."
    ),
    version="1.0.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Confirma que el servicio y el artefacto del modelo están disponibles."""
    return {"status": "ok", "model": MODEL_PATH.name}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float | int]:
    """Devuelve la probabilidad estimada y la clase usando umbral 0,5."""
    row = pd.DataFrame([request.features.model_dump()])
    probability = float(model.predict_proba(row)[0, 1])
    return {
        "heart_disease_probability": probability,
        "prediction": int(probability >= 0.5),
        "threshold": 0.5,
    }
