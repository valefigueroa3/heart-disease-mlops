"""Entrena el candidato mejor ubicado por AUC de validación cruzada."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from app.modeling import (
    add_probability_calibration,
    compare_models,
    prepare_features_and_target,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "heart.csv"
MODEL_PATH = ROOT / "app" / "model.joblib"


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}. Guarde allí el archivo heart.csv."
        )

    data = pd.read_csv(DATA_PATH)
    X, y = prepare_features_and_target(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    results, searches = compare_models(X_train, y_train, X_test, y_test)
    selected_name = results.iloc[0]["Modelo"]
    selected_estimator = searches[selected_name].best_estimator_
    model_for_api = add_probability_calibration(
        selected_estimator, X_train, y_train
    )
    joblib.dump(model_for_api, MODEL_PATH)

    print(results[[
        "Posición", "Modelo", "AUC promedio CV", "AUC prueba", "Exactitud prueba"
    ]].round(3).to_string(index=False))
    print(f"Modelo guardado en: {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
