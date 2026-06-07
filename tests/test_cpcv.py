"""Tests del módulo ``core.cpcv``."""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest

from core.cpcv import CombinatorialPurgedKFold


@pytest.fixture()
def serie_fechada() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    return pd.DataFrame({"x": np.arange(200)}, index=idx)


def test_numero_de_folds_combinatorio(serie_fechada):
    cv = CombinatorialPurgedKFold(n_splits=10, n_test_splits=2, embargo=0)
    folds = list(cv.split(serie_fechada))
    assert len(folds) == comb(10, 2) == 45


def test_train_y_test_disjuntos(serie_fechada):
    cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, embargo=5)
    for tr, te in cv.split(serie_fechada):
        assert set(tr).isdisjoint(set(te))


def test_purge_elimina_solape_temporal():
    """Si t1[i] solapa con el test, ese sample del train debe ser purgado."""
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    X = pd.DataFrame({"x": np.arange(20)}, index=idx)
    t1 = pd.Series(idx + pd.Timedelta(days=3), index=idx)  # eventos a 4 días
    cv = CombinatorialPurgedKFold(n_splits=4, n_test_splits=1, embargo=0)
    for tr, te in cv.split(X, t1=t1):
        test_start = idx[te.min()]
        test_end = idx[te.max()]
        for i in tr:
            t0_i = idx[i]
            t1_i = t1.iloc[i]
            # Ningún sample de train puede solapar [test_start, test_end].
            assert not (t1_i >= test_start and t0_i <= test_end)


def test_n_test_splits_invalido():
    with pytest.raises(ValueError):
        CombinatorialPurgedKFold(n_splits=5, n_test_splits=5)


def test_split_determinista(serie_fechada):
    cv = CombinatorialPurgedKFold()
    a = [(tr.tolist(), te.tolist()) for tr, te in cv.split(serie_fechada)]
    b = [(tr.tolist(), te.tolist()) for tr, te in cv.split(serie_fechada)]
    assert a == b
