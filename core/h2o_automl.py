"""Wrapper de H2O AutoML con folds Purged K-Fold para series temporales.

H2O AutoML aplica por defecto KFold convencional para su validación cruzada
interna, lo cual viola la causalidad temporal en series financieras. Este
módulo precalcula los folds y los pasa a ``H2OAutoML`` mediante
``fold_column='fold_id'``: chunks contiguos en orden temporal y embargo
(eliminación de las últimas ``embargo`` filas de cada chunk) para evitar fuga
entre vecinos (López de Prado 2018, sec. 7.4).

Configuraciones que usan este wrapper:

- **M3** (configuración 3): **excepción** — usa KFold convencional sin
  ``fold_column``. Réplica del sesgo metodológico que el TFG denuncia.
- **M4** y **M9**: chunks contiguos + embargo.

El módulo es defensivo respecto a la dependencia de H2O: arranca el clúster
una sola vez por proceso, devuelve un *leader* serializable y limpia recursos
a petición.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_H2O_INIT = False


def _ensure_h2o() -> Any:
    """Inicializa el clúster H2O local una sola vez por proceso y lo devuelve."""
    global _H2O_INIT
    import h2o

    if not _H2O_INIT:
        h2o.init(nthreads=-1, max_mem_size="4G", verbose=False)
        _H2O_INIT = True
    return h2o


def shutdown_h2o() -> None:
    """Apaga el clúster H2O. Llamar solo al final del pipeline."""
    global _H2O_INIT
    if not _H2O_INIT:
        return
    import h2o

    try:
        h2o.cluster().shutdown(prompt=False)
    finally:
        _H2O_INIT = False


def purged_kfold_fold_ids(
    n: int, n_splits: int = 5, embargo: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Asigna folds contiguos en orden temporal con embargo entre chunks.

    Devuelve ``(keep_idx, fold_ids)``:

    - ``keep_idx``: índices a conservar (las últimas ``embargo`` filas de cada
      chunk se descartan para evitar fuga al chunk siguiente).
    - ``fold_ids``: para cada fila conservada, su id de fold en ``[0, n_splits)``.

    El descarte por embargo solo se aplica entre chunks; el último chunk no
    tiene embargo posterior.
    """
    if n_splits < 2:
        raise ValueError("n_splits debe ser ≥ 2.")
    edges = np.linspace(0, n, n_splits + 1, dtype=int)
    keep: list[int] = []
    fold_ids: list[int] = []
    for k in range(n_splits):
        start, end = int(edges[k]), int(edges[k + 1])
        end_keep = end - embargo if k < n_splits - 1 else end
        if end_keep <= start:
            continue
        for i in range(start, end_keep):
            keep.append(i)
            fold_ids.append(k)
    return np.asarray(keep, dtype=int), np.asarray(fold_ids, dtype=int)


@dataclass
class H2OLeaderResult:
    """Predicción y metadatos del líder de H2O AutoML.

    ``leaderboard_summary`` es una lista de dicts con (model_id, métrica) en
    el orden devuelto por H2O; útil para la figura 13.
    """

    leader_id: str
    leader_metric: float
    metric_name: str
    leaderboard: list[dict]
    train_n: int
    valid_n: int | None
    used_fold_column: bool


def _to_h2o_frame(h2o_mod, X: pd.DataFrame, y: pd.Series, fold_ids: np.ndarray | None):
    """Construye un H2OFrame a partir de un DataFrame de pandas + target."""
    df = X.copy()
    df["y"] = y.astype(int).values
    if fold_ids is not None:
        df["fold_id"] = fold_ids.astype(int)
    h2o_df = h2o_mod.H2OFrame(df)
    h2o_df["y"] = h2o_df["y"].asfactor()
    return h2o_df


def train_h2o(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    use_fold_column: bool,
    max_runtime_secs: int = 600,
    n_splits: int = 5,
    embargo: int = 5,
    seed: int = 42,
) -> tuple[Any, H2OLeaderResult]:
    """Entrena H2OAutoML y devuelve ``(leader, H2OLeaderResult)``.

    Args:
        X: features de entrenamiento (DataFrame con columnas numéricas).
        y: target binario (0/1).
        use_fold_column: si ``True``, precalcula ``fold_id`` con
            ``purged_kfold_fold_ids`` y lo pasa a H2O vía ``fold_column``.
            Si ``False``, deja a H2O usar su KFold convencional (M3).
        max_runtime_secs: presupuesto de búsqueda AutoML.
        n_splits, embargo: parámetros de Purged K-Fold cuando aplica.
        seed: semilla H2O para reproducibilidad.

    Returns:
        ``(leader, H2OLeaderResult)``. ``leader`` es el modelo H2O para
        predecir; ``H2OLeaderResult`` lleva metadatos serializables.
    """
    h2o = _ensure_h2o()
    from h2o.automl import H2OAutoML

    if use_fold_column:
        keep_idx, fold_ids = purged_kfold_fold_ids(len(X), n_splits=n_splits, embargo=embargo)
        X_trim = X.iloc[keep_idx]
        y_trim = y.iloc[keep_idx]
        train_df = _to_h2o_frame(h2o, X_trim, y_trim, fold_ids)
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            seed=seed,
            sort_metric="AUC",
            nfolds=0,  # H2O usa fold_column para la xval, no nfolds.
            verbosity=None,
        )
        aml.train(x=list(X.columns), y="y", training_frame=train_df, fold_column="fold_id")
    else:
        train_df = _to_h2o_frame(h2o, X, y, None)
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            seed=seed,
            sort_metric="AUC",
            nfolds=5,  # KFold convencional — sesgo intencional para M3.
            verbosity=None,
        )
        aml.train(x=list(X.columns), y="y", training_frame=train_df)

    lb = aml.leaderboard.as_data_frame(use_pandas=True)
    leader = aml.leader
    metric_col = "auc" if "auc" in lb.columns else lb.columns[1]
    result = H2OLeaderResult(
        leader_id=str(lb.iloc[0]["model_id"]),
        leader_metric=float(lb.iloc[0][metric_col]),
        metric_name=metric_col,
        leaderboard=[
            {"model_id": str(r["model_id"]), metric_col: float(r[metric_col])}
            for _, r in lb.head(15).iterrows()
        ],
        train_n=int(train_df.nrows),
        valid_n=None,
        used_fold_column=use_fold_column,
    )
    return leader, result


def predict_class1_proba(leader: Any, X_oos: pd.DataFrame) -> np.ndarray:
    """Devuelve P(y=1|x) del líder sobre ``X_oos``."""
    h2o = _ensure_h2o()
    test_df = h2o.H2OFrame(X_oos.copy())
    preds = leader.predict(test_df).as_data_frame(use_pandas=True)
    # Columnas típicas: predict, p0, p1.
    if "p1" in preds.columns:
        return preds["p1"].to_numpy()
    # Fallback para modelos sin probabilidades calibradas: usa la columna predict.
    return preds.iloc[:, -1].to_numpy()


def save_leader_metadata(result: H2OLeaderResult, path: Path) -> None:
    """Persiste metadatos del leader en un JSON legible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "leader_id": result.leader_id,
                "leader_metric": result.leader_metric,
                "metric_name": result.metric_name,
                "leaderboard": result.leaderboard,
                "train_n": result.train_n,
                "valid_n": result.valid_n,
                "used_fold_column": result.used_fold_column,
            },
            indent=2,
        )
    )
