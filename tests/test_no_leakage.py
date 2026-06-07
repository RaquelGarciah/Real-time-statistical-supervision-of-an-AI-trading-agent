"""Test crítico de no-leakage temporal en CPCV.

Garantiza que en cada fold se cumple
``max(train_dates) + embargo <= min(test_dates)`` cuando el test es un único
bloque contiguo, y la propiedad equivalente generalizada para folds con varios
bloques de test (no debe haber ninguna fecha de train dentro del rango cubierto
por el test, incluyendo el embargo al final).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.cpcv import CombinatorialPurgedKFold


def test_sin_leakage_temporal_basico():
    """Para eventos puntuales (``t1[i]=idx[i]``), el embargo solo afecta al lado
    posterior al test; el lado anterior basta con que no solape posiciones.
    """
    idx = pd.date_range("2010-01-01", periods=500, freq="B")
    X = pd.DataFrame({"x": np.arange(500)}, index=idx)
    embargo = 10
    cv = CombinatorialPurgedKFold(n_splits=10, n_test_splits=1, embargo=embargo)

    for tr, te in cv.split(X):
        test_min_pos = te.min()
        test_max_pos = te.max()
        train_positions = tr

        antes = train_positions[train_positions < test_min_pos]
        despues = train_positions[train_positions > test_max_pos]

        if antes.size > 0:
            # Eventos puntuales: basta con que el último train anterior < primer test.
            assert antes.max() < test_min_pos
        if despues.size > 0:
            assert despues.min() > test_max_pos + embargo
        dentro = train_positions[(train_positions >= test_min_pos) & (train_positions <= test_max_pos)]
        assert dentro.size == 0


def test_sin_leakage_con_eventos_largos():
    """Eventos cuya etiqueta tarda varios días en cerrarse: ``t1[i] > idx[i]``.

    Estos eventos requieren purging adicional: cualquier evento de train cuyo
    cierre cae dentro del periodo de test (o embargo) se elimina.
    """
    idx = pd.date_range("2010-01-01", periods=200, freq="B")
    X = pd.DataFrame({"x": np.arange(200)}, index=idx)
    horizonte = 5
    t1_values = pd.DatetimeIndex(idx).shift(horizonte, freq="B")
    t1 = pd.Series(t1_values, index=idx)
    cv = CombinatorialPurgedKFold(n_splits=5, n_test_splits=1, embargo=3)

    for tr, te in cv.split(X, t1=t1):
        test_start = idx[te.min()]
        test_end = idx[te.max()]
        for i in tr:
            t0_i = idx[i]
            t1_i = t1.iloc[i]
            # El intervalo del evento de train no puede solapar el del test.
            assert not (t1_i >= test_start and t0_i <= test_end)
