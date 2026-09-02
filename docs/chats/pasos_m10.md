# Pasos de M10 y dónde mejorar — conversación

## Pregunta (Raquel)

> antes de hacer elegir como activo SMCI definitivamente quiero que pruebes varias cosas. vamos a intentar
> mejorar el accuracy de m10. dime exactamente los pasos que sigues y que me devuelves en m10 y dime que cosas
> se podrian cambiar o mejorar en cada paso.

## Respuesta

Perfecto — antes de cerrar SMCI, vamos a entender **exactamente** qué hace M10 y dónde se puede mejorar. Te lo desgloso leyendo el código real (la versión **desplegable walk-forward**, la del 0.524).

## Los 5 pasos de M10 (y qué devuelve)

**Paso 1 — Estado de mercado por día** (`build_states_onthefly`)
- Carga precios 2000→hoy; calcula `r` (log-retorno) y `RV21` (vol realizada 21d).
- Ajusta en **calibración (2000→2024-09)**: HMM K=3 (régimen) y GARCH(1,1)-t (volatilidad).
- Por cada día OOS: posterior de régimen **filtrado/causal** → `calm_prob, stress_prob, crisis_prob`; y `σ_t` GARCH.

**Paso 2 — Matriz de 22 features + etiqueta** (`run_master`)
- **15 del agente:** por cada personalidad (5) × {signo, size, confianza}.
- **3 scores STRATA:** `ram_score, psa_score, gso_score`.
- **4 de régimen/vol:** `calm_prob, stress_prob, crisis_prob, garch_sigma`.
- **Etiqueta:** `y = signo(r_{t+1})` → binaria (sube/baja).

**Paso 3 — Entrenamiento walk-forward** (`expanding_wf_p1`)
- Burn-in 150 d; reentreno cada 21 d; embargo 5 d.
- Cada reentreno: **XGBoost** (300 árboles, prof 4, lr 0.05, subsample 0.8, colsample 0.8, reg_lambda 1) sobre **todo el pasado**; predice el bloque siguiente.

**Paso 4 — Probabilidad → decisión**
- `p1 = P(sube mañana)` ∈ [0,1]; posición `w = signo(p1 − 0.5)` (+1 largo / −1 corto).

**Paso 5 — Evaluación**
- accuracy = media de aciertos `signo(p1−0.5) == signo(r_{t+1})` en el tramo de test.

**Qué devuelve M10:** por cada día, `p1` (probabilidad de subida); de ahí la **dirección** y la **accuracy** (más AUC, Brier). Internamente: una secuencia de XGBoost reentrenados (solo pasado).

---

## Qué se puede cambiar/mejorar en CADA paso

🟢 = prometedor y honesto · 🟡 = posible pero cambia la tesis o arriesgado · 🔴 = lo evitamos (p-hacking/look-ahead)

**Paso 1 (régimen/vol):**
- 🟡 **Feature del HMM.** Hoy el régimen se construye con (retorno, **volatilidad**) → captura *régimen de volatilidad*, no de **dirección**. Añadir un rasgo direccional (momentum/tendencia) podría hacerlo informativo — pero **cambia la definición de STRATA**.
- 🟢 **K del HMM** (2/3/4) y **ventana de vol** (10/21/63): elegir en validación.
- 🟡 GARCH asimétrico (GJR/EGARCH) — afecta sizing/σ, poco a la accuracy.

**Paso 2 (features + etiqueta):**
- 🟢 **Selección de features.** Las 15 del agente son señal perdedora y pueden **arrastrar** a M10 (la ablación dio agent-only 0.476). Quitar las flojas / quedarse con régimen+STRATA puede subir accuracy.
- 🟢 **Añadir features con señal real:** momentum (retornos pasados), rachas, vol relativa. Es la vía honesta a más accuracy (si hay señal).
- 🟢 **Etiqueta menos ruidosa:** *triple-barrier / meta-labeling* (López de Prado) o excluir días de retorno ≈0 → etiquetas más limpias.
- 🟢 **Pesos por magnitud:** dar más peso a los días de movimiento grande.

**Paso 3 (modelo/entreno):**
- 🟢 **Pesos por recencia:** dar más peso a datos recientes (SMCI es no estacionario) → puede ayudar específicamente aquí.
- 🟢 **Modelo más simple/regularizado:** logístico o árboles muy podados generalizan mejor con N pequeño; o más `reg_lambda`/`min_child_weight`.
- 🟢 **Ensemble de semillas:** promediar `p1` de ~10 semillas → menos varianza, más estable (bajo riesgo).
- 🟡 **Ventana rodante** (en vez de expandible): se adapta más al cambio de régimen.

**Paso 4 (umbral/decisión):**
- 🟢 **Umbral ≠ 0.5.** Tu tutor lo dijo literal ("el corte es 0.47, no 0.5"). Ajustarlo **en validación** puede subir accuracy directamente.
- 🟡 **Calibración** (isotónica/Platt): la isotónica empeoró en SMCI, pero un umbral ajustado en validación es más seguro.
- 🔴 Abstención agresiva: ya vimos que sin calibración real no ayuda; y mide accuracy en menos días (no comparable a B&H).

**Paso 5 (evaluación):** 🟢 más datos / OOS más largo es lo que de verdad daría potencia — pero el agente solo existe en el OOS → eso es **trabajo futuro**.

---

## La regla de oro (para no hundirte en la defensa)
Mejorar accuracy es legítimo **solo si** los cambios se (a) deciden **a priori** y (b) se eligen en un **set de validación**, NO mirando el test y quedándote con lo que sale mejor. Eso último es p-hacking → el tribunal lo tumba. Así que: **conjunto pequeño y pre-registrado de mejoras → validación elige → test (intacto) reporta.** Y expectativa realista: ganancias modestas (la dirección diaria es casi eficiente y la muestra es corta).

## Mi propuesta (un set pequeño y honesto para probar)
Las 5 palancas 🟢 de mayor impacto/menor riesgo:
1. **Umbral óptimo en validación** (no 0.5).
2. **Selección de features** (con/sin las 15 del agente; régimen+STRATA).
3. **Pesos por recencia** (no estacionariedad de SMCI).
4. **Ensemble de semillas** (estabilidad).
5. **Etiqueta meta-labeling** (menos ruido) — opcional, más trabajo.

Lo monto como estudio **pre-registrado**: split validación/test (o walk-forward con validación interna), pruebo estas mejoras, elijo en validación, reporto en test, y te lo enseño con gráficos de barras (accuracy de cada mejora vs el M10 base vs B&H/M5/M8).

¿Quieres que lo lance con estas 5? ¿Quitas/añades alguna antes de que lo pre-registre y lo corra? Tú eliges el menú; yo lo ejecuto con disciplina validación≠test.
