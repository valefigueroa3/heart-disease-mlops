"""Funciones de preparación, búsqueda y comparación de modelos."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


TARGET_COLUMN = "HeartDisease"
ZERO_AS_MISSING_COLUMNS = ("RestingBP", "Cholesterol")


def prepare_features_and_target(
    data: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa la respuesta y marca como faltantes los ceros no válidos.

    Los ceros de RestingBP y Cholesterol no representan mediciones clínicas
    posibles en este conjunto. La imputación posterior se aprende dentro del
    pipeline, usando únicamente los datos de entrenamiento de cada partición.
    """
    if target_column not in data.columns:
        raise ValueError(
            f"No se encontró la variable objetivo '{target_column}'. "
            f"Columnas disponibles: {list(data.columns)}"
        )

    if data[target_column].isna().any():
        raise ValueError("La variable objetivo contiene valores faltantes.")

    y = pd.to_numeric(data[target_column], errors="raise").astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("La variable objetivo debe contener únicamente 0 y 1.")

    X = data.drop(columns=[target_column]).copy()
    for column in ZERO_AS_MISSING_COLUMNS:
        if column in X.columns:
            X[column] = X[column].mask(X[column].eq(0), np.nan)

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construye imputación, escalamiento y codificación según el tipo de dato."""
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in X.columns if column not in numeric_columns
    ]

    if not numeric_columns and not categorical_columns:
        raise ValueError("No hay variables predictoras para procesar.")

    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
                ("escalar", StandardScaler()),
            ]
        )
        transformers.append(("numericas", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputar", SimpleImputer(strategy="most_frequent")),
                ("codificar", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categoricas", categorical_pipeline, categorical_columns)
        )

    # La salida densa permite comparar también los modelos de árboles de sklearn.
    return ColumnTransformer(transformers=transformers, sparse_threshold=0)


def model_searches() -> dict[str, tuple[object, dict[str, list]]]:
    """Modelos y rangos de parámetros propuestos para la comparación."""
    return {
        "SVC": (
            SVC(random_state=42),
            {
                "clasificador__C": [0.1, 1, 10],
                "clasificador__gamma": ["scale", 0.01, 0.1],
            },
        ),
        "Regresión logística": (
            LogisticRegression(max_iter=2000, random_state=42),
            {"clasificador__C": [0.1, 1, 10]},
        ),
        "Bosque aleatorio": (
            RandomForestClassifier(random_state=42),
            {
                "clasificador__n_estimators": [100, 200],
                "clasificador__max_depth": [None, 5, 10],
                "clasificador__min_samples_leaf": [1, 2],
            },
        ),
        "K vecinos (KNN)": (
            KNeighborsClassifier(),
            {
                "clasificador__n_neighbors": [5, 9, 15],
                "clasificador__weights": ["uniform", "distance"],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "clasificador__n_estimators": [100, 200],
                "clasificador__learning_rate": [0.05, 0.1],
                "clasificador__max_depth": [2, 3],
            },
        ),
    }


def positive_class_scores(estimator, X: pd.DataFrame) -> np.ndarray:
    """Devuelve probabilidades o puntajes para ordenar casos por riesgo."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    raise TypeError("El modelo no produce probabilidades ni puntajes de decisión.")


def add_probability_calibration(estimator, X: pd.DataFrame, y: pd.Series):
    """Agrega probabilidades al SVC solo cuando la API las necesita."""
    if hasattr(estimator, "predict_proba"):
        return estimator
    calibrated = CalibratedClassifierCV(
        estimator=estimator, method="sigmoid", cv=5
    )
    return calibrated.fit(X, y)


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    n_splits: int = 5,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict[str, GridSearchCV]]:
    """Ajusta los modelos con validación cruzada y evalúa la prueba reservada.

    El ranking se ordena por AUC promedio de validación cruzada. El conjunto de
    prueba no participa en la búsqueda de parámetros ni en ese ordenamiento.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fitted_searches: dict[str, GridSearchCV] = {}
    rows = []

    for model_name, (classifier, parameter_grid) in model_searches().items():
        pipeline = Pipeline(
            steps=[
                ("preprocesamiento", build_preprocessor(X_train)),
                ("clasificador", classifier),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring={"AUC": "roc_auc", "Exactitud": "accuracy"},
            refit="AUC",
            cv=cv,
            n_jobs=n_jobs,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)

        scores = positive_class_scores(search.best_estimator_, X_test)
        predictions = search.best_estimator_.predict(X_test)
        best_index = search.best_index_
        rows.append(
            {
                "Modelo": model_name,
                "AUC promedio CV": search.best_score_,
                "AUC prueba": float(roc_auc_score(y_test, scores)),
                "Exactitud prueba": float(accuracy_score(y_test, predictions)),
                "AUC entrenamiento CV": float(
                    search.cv_results_["mean_train_AUC"][best_index]
                ),
                "Mejores parámetros": search.best_params_,
            }
        )
        fitted_searches[model_name] = search

    results = pd.DataFrame(rows).sort_values(
        by=["AUC promedio CV", "AUC prueba"], ascending=False
    )
    results.insert(0, "Posición", range(1, len(results) + 1))
    return results.reset_index(drop=True), fitted_searches
