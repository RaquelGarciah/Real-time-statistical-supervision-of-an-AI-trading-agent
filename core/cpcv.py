"""Combinatorial Purged Cross-Validation (López de Prado, 2018, sec. 7.4).

Genera todas las particiones de ``n_splits`` grupos contiguos en las que se
toman ``n_test_splits`` grupos como test, aplicando *purging* (eliminar del
train los samples cuyo evento solapa temporalmente con el test) y *embargo*
(ventana opcional adicional tras el final del test) para evitar fuga de
información entre muestras vecinas.

El número total de folds es ``C(n_splits, n_test_splits)``. Para el ajuste
estándar ``n_splits=10, n_test_splits=2`` da 45 folds.

Referencia:

- López de Prado, M. (2018). "Advances in Financial Machine Learning",
  Wiley, capítulo 7 (Cross-Validation in Finance), sección 7.4.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


def _purge(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    t1: pd.Series,
    embargo: int,
) -> np.ndarray:
    """Elimina del train los samples cuyo evento solapa con el rango del test.

    ``t1`` es la serie de tiempos de cierre por sample: ``t1.index[i]`` es el
    inicio del evento ``i`` y ``t1.iloc[i]`` su cierre. El "rango contaminado"
    del test es ``[min(t1.index[test]), max(max(t1.iloc[test]), idx[last_test+embargo])]``.
    Se purga cualquier evento de train cuyo intervalo ``(t0_i, t1_i)`` solape
    con ese rango.
    """
    test_starts = t1.index[test_idx]
    test_ends = t1.iloc[test_idx]
    test_start = test_starts.min()
    test_end_by_t1 = test_ends.max()

    if embargo > 0:
        last_test_pos = int(np.max(test_idx))
        embargo_pos = min(last_test_pos + embargo, len(t1) - 1)
        test_end_by_embargo = t1.index[embargo_pos]
        test_end = max(test_end_by_t1, test_end_by_embargo)
    else:
        test_end = test_end_by_t1

    keep = []
    for i in train_idx:
        t0_i = t1.index[i]
        t1_i = t1.iloc[i]
        overlaps = (t1_i >= test_start) and (t0_i <= test_end)
        if not overlaps:
            keep.append(i)
    return np.asarray(keep, dtype=int)


@dataclass
class CombinatorialPurgedKFold:
    """CPCV con purging y embargo aplicado al final de cada test.

    Parámetros:
        n_splits: número de grupos contiguos en que se divide la serie.
        n_test_splits: cuántos grupos se toman como test en cada fold.
        embargo: nº de pasos adicionales tras el test que también se purgan.
    """

    n_splits: int = 10
    n_test_splits: int = 2
    embargo: int = 10

    def __post_init__(self) -> None:
        if self.n_test_splits >= self.n_splits:
            raise ValueError("n_test_splits debe ser estrictamente menor que n_splits.")

    def _group_ranges(self, n: int) -> list[tuple[int, int]]:
        """Particiona ``range(n)`` en ``n_splits`` grupos contiguos."""
        edges = np.linspace(0, n, self.n_splits + 1, dtype=int)
        return [(int(edges[i]), int(edges[i + 1])) for i in range(self.n_splits)]

    @property
    def n_folds(self) -> int:
        """Número total de folds combinatorios."""
        from math import comb

        return comb(self.n_splits, self.n_test_splits)

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        t1: pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Genera ``(train_idx, test_idx)`` para cada fold combinatorio.

        Si ``t1`` es None se asume ``t1[i] = índice[i]`` (eventos puntuales que
        terminan en el mismo paso en que empiezan).
        """
        n = len(X)
        if t1 is None:
            if isinstance(X, pd.DataFrame):
                t1 = pd.Series(X.index, index=X.index)
            else:
                idx = pd.RangeIndex(n)
                t1 = pd.Series(idx, index=idx)

        groups = self._group_ranges(n)
        all_idx = np.arange(n)

        for combo in combinations(range(self.n_splits), self.n_test_splits):
            test_idx_parts: list[Iterable[int]] = []
            for g in combo:
                start, end = groups[g]
                test_idx_parts.append(range(start, end))
            test_idx = np.fromiter(
                (i for part in test_idx_parts for i in part), dtype=int
            )

            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            train_idx = all_idx[train_mask]

            train_idx = _purge(train_idx, test_idx, t1, self.embargo)
            yield train_idx, test_idx
