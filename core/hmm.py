"""Hidden Markov Model gaussiano de tres estados para detección de régimen.

Implementación basada en ``hmmlearn.GaussianHMM`` con covarianza completa.
El ajuste sigue la receta descrita en ``replicar_regimen_mercado.md``:

1. **Estandarización por feature.** ``ret_log`` (std ≈ 0,01) y
   ``rv_21_ann`` (≈ 0,20) difieren en escala unas 20×. Sin estandarizar,
   la covarianza emisora queda dominada por la volatilidad y el signo
   del retorno se diluye como ruido: Crisis (caída + vol alta) se vuelve
   indistinguible de Estrés (vol alta sin caída). Por eso ``fit`` resta
   la media y divide por la desviación típica de cada columna, y ese
   mismo escalado se aplica en ``predict_states`` / ``predict_proba``.
2. **Múltiples inicializaciones.** Se entrenan ``n_seeds=10`` modelos con
   ``random_state=seed+k`` y ``n_iter=1000``; se conserva el que maximiza
   ``model.score(X)`` (log-verosimilitud). Esto evita mínimos locales del
   algoritmo Baum-Welch.
3. **Ordenamiento por volatilidad.** Tras la selección, los estados se
   reordenan por la **media de la segunda columna** del input (que se
   asume es la volatilidad realizada) ascendente. Las etiquetas finales
   son ``Calma`` (vol menor), ``Estrés`` (media) y ``Crisis`` (mayor).
   La estandarización es monótona por columna, así que el ordenamiento
   se puede calcular indistintamente sobre la escala cruda o la
   estandarizada.

La feature de volatilidad esperada es ``realized_vol_21d`` anualizada
(``core.features.realized_vol_annualized``), no ``log(VIX)``. Esto purifica
la señal de régimen al ruido observable real, sin primas de riesgo de las
opciones.

Referencias:

- Rabiner (1989), "A tutorial on hidden Markov models", *Proc. IEEE 77*.
- Hamilton (1989), "A new approach to the economic analysis of nonstationary
  time series and the business cycle", *Econometrica 57*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

from config import HMM_COVARIANCE_TYPE, HMM_N_ITER, HMM_N_STATES, SEED


@dataclass
class RegimeHMM:
    """HMM de tres estados con etiquetado determinista por varianza emisora.

    Atributos públicos tras ``fit``:

    - ``model``: instancia interna de ``hmmlearn.GaussianHMM``.
    - ``state_order``: array con la permutación que ordena los estados
      del modelo crudo a los ordenados por varianza ascendente.
    - ``state_labels``: diccionario ``{índice ordenado: etiqueta}``.
    """

    n_states: int = HMM_N_STATES
    covariance_type: str = HMM_COVARIANCE_TYPE
    n_iter: int = HMM_N_ITER
    seed: int = SEED
    n_seeds: int = 10

    def __post_init__(self) -> None:
        self.model: GaussianHMM | None = None
        self.state_order: np.ndarray | None = None
        self.state_labels: dict[int, str] = {}
        self.best_seed: int | None = None
        self.best_score: float | None = None
        self.feature_means_: np.ndarray | None = None
        self.feature_stds_: np.ndarray | None = None

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        """Aplica el escalado ``(x - μ) / σ`` por columna usando los parámetros del fit."""
        if self.feature_means_ is None or self.feature_stds_ is None:
            raise RuntimeError("Llama a fit() antes de estandarizar.")
        return (features - self.feature_means_) / self.feature_stds_

    def fit(self, features: np.ndarray) -> RegimeHMM:
        """Estima el HMM con ``n_seeds`` inicializaciones y aplica el etiquetado.

        ``features`` debe tener forma ``(T, d)`` con ``d >= 2``. La **segunda
        columna** se interpreta como volatilidad realizada y determina el
        ordenamiento de los estados (Calma = vol media menor). Las features
        se estandarizan internamente columna a columna antes del ajuste.
        """
        if features.ndim != 2:
            raise ValueError("features debe ser 2D con shape (T, d).")
        if features.shape[1] < 2:
            raise ValueError(
                "features debe tener al menos 2 columnas; la segunda se usa "
                "para ordenar estados por volatilidad."
            )

        # Estandarización por columna (ver docstring del módulo).
        self.feature_means_ = features.mean(axis=0)
        stds = features.std(axis=0, ddof=0)
        # Evita división por cero si alguna columna es constante.
        stds = np.where(stds > 0, stds, 1.0)
        self.feature_stds_ = stds
        X = (features - self.feature_means_) / self.feature_stds_

        best_model: GaussianHMM | None = None
        best_score: float = -np.inf
        best_seed: int = self.seed
        for k in range(self.n_seeds):
            seed_k = self.seed + k
            model_k = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=seed_k,
            )
            try:
                model_k.fit(X)
                score_k = float(model_k.score(X))
            except Exception:
                continue
            if score_k > best_score:
                best_model = model_k
                best_score = score_k
                best_seed = seed_k

        if best_model is None:
            raise RuntimeError("Ninguna de las inicializaciones del HMM convergió.")

        # Etiquetado determinista: ordenar estados por media de la segunda
        # columna (realized_vol) ascendente (Calma → Crisis). La
        # estandarización es monótona por columna, así que ordenar por la
        # escala cruda o por la estandarizada da el mismo resultado.
        raw_states = best_model.predict(X)
        means_by_state = np.array([
            features[raw_states == s, 1].mean() if (raw_states == s).any() else np.inf
            for s in range(self.n_states)
        ])
        order = np.argsort(means_by_state)

        self.model = best_model
        self.best_seed = best_seed
        self.best_score = best_score
        self.state_order = order
        labels = ("Calma", "Estrés", "Crisis")
        if self.n_states != 3:
            labels = tuple(f"S{i}" for i in range(self.n_states))
        self.state_labels = dict(enumerate(labels))
        return self

    def _map_to_ordered(self, raw_states: np.ndarray) -> np.ndarray:
        """Traduce estados crudos del modelo a la indexación ordenada."""
        if self.state_order is None:
            raise RuntimeError("Llama a fit() antes de predecir.")
        inverse = np.argsort(self.state_order)
        return inverse[raw_states]

    def predict_states(self, features: np.ndarray) -> np.ndarray:
        """Estados más probables vía Viterbi, ya en la indexación ordenada."""
        if self.model is None:
            raise RuntimeError("Llama a fit() antes de predecir.")
        X = self._standardize(features)
        raw = self.model.predict(X)
        return self._map_to_ordered(raw)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Posteriores SUAVIZADOS ``gamma_t(s) = P(s_t | x_{1:T})`` (forward-backward).

        Usa toda la serie, incluido el futuro de ``t``. Sirve para describir el
        régimen *ex-post* (gráficos, calibración), NO para supervisar en OOS:
        para eso está ``predict_proba_filtered``, que es causal.
        """
        if self.model is None:
            raise RuntimeError("Llama a fit() antes de predecir.")
        X = self._standardize(features)
        raw_proba = self.model.predict_proba(X)
        return raw_proba[:, self.state_order]

    def predict_proba_filtered(self, features: np.ndarray) -> np.ndarray:
        """Posteriores FILTRADOS ``gamma^f_t(s) = P(s_t | x_{1:t})``, causales.

        Solo propaga el mensaje forward: la fila ``t`` depende exclusivamente de
        ``x_{1:t}``, nunca del futuro. Es el régimen admisible para supervisar
        en tiempo real sin look-ahead, frente al suavizado de ``predict_proba``.

        Forward: ``alpha_t(j) = b_j(x_t) * sum_i alpha_{t-1}(i) a_{ij}``, en
        espacio log. La fila normalizada es ``P(s_t | x_{1:t})``.
        Referencia: Rabiner (1989), algoritmo forward, ecs. (18)-(20).
        """
        if self.model is None:
            raise RuntimeError("Llama a fit() antes de predecir.")
        X = self._standardize(features)
        framelogprob = self.model._compute_log_likelihood(X)  # (T, n_states): log b_j(x_t)
        with np.errstate(divide="ignore"):
            log_startprob = np.log(self.model.startprob_)
            log_transmat = np.log(self.model.transmat_)
        log_alpha = np.empty_like(framelogprob)
        log_alpha[0] = log_startprob + framelogprob[0]
        for t in range(1, framelogprob.shape[0]):
            log_alpha[t] = framelogprob[t] + logsumexp(
                log_alpha[t - 1][:, None] + log_transmat, axis=0
            )
        filtered = np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
        return filtered[:, self.state_order]

    @property
    def transition_matrix(self) -> np.ndarray:
        """Matriz de transición reordenada (filas y columnas) por varianza."""
        if self.model is None:
            raise RuntimeError("Llama a fit() antes de consultar A.")
        a = self.model.transmat_
        return a[np.ix_(self.state_order, self.state_order)]
