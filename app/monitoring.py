"""Genera el reporte de deriva de datos solicitado en la etapa 6."""

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.model_selection import train_test_split

from app.modeling import prepare_features_and_target


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "heart.csv"
REPORT_PATH = ROOT / "drift_report.html"


def generate_report() -> Path:
    """Compara entrenamiento (referencia) con prueba (muestra nueva simulada)."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}. Guarde allí el archivo heart.csv."
        )

    data = pd.read_csv(DATA_PATH)
    X, y = prepare_features_and_target(data)
    reference_data, current_data, _, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # El conjunto de prueba representa datos nuevos en este ejercicio.
    # No es un registro real de predicciones en producción.
    report = Report([DataDriftPreset()])
    snapshot = report.run(
        current_data=current_data,
        reference_data=reference_data,
    )
    # Evidently 0.7 acepta una ruta de texto; si recibe Path puede terminar
    # sin escribir el archivo y sin lanzar una excepción.
    snapshot.save_html(str(REPORT_PATH))
    if not REPORT_PATH.is_file() or REPORT_PATH.stat().st_size == 0:
        raise RuntimeError(f"Evidently no generó el reporte en {REPORT_PATH}.")
    return REPORT_PATH


if __name__ == "__main__":
    print(f"Reporte guardado en: {generate_report()}")
