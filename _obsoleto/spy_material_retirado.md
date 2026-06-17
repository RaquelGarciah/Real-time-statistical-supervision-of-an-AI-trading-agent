# SPY — material retirado del guion (recuperable)

**Por qué está aquí.** El 2026-06-17 Raquel decidió enfocar la memoria **solo en el caso de estudio SMCI**
("olvídate de SPY"). Este documento guarda **todo lo que se quitó de `memoria/`** referente a SPY, por si más
adelante se retoma SPY como segundo caso. **Los datos de SPY NO se han borrado**: siguen en los ficheros que se
listan al final. Esto es solo el *framing* retirado + el índice.

---

## 1. La tesis vieja (con SPY) que se retiró

**Hipótesis falsable en tres niveles** (de `CLAUDE.md §2`, marco original):
1. **Estadístico:** McNemar pareado M8 vs M5 (la regla rescata al agente).
2. **Mecánico:** atribución de P&L por detector → **RAM domina**.
3. **Universalidad:** un XGBoost (M10) **no debe batir** a la regla (M8) significativamente, y **SHAP** debe
   señalar las features STRATA como las informativas. (En SPY: M10 ≈ M8 en P&L, DM p≈0,67 → el XGBoost
   *redescubre* STRATA, no la bate.)

**Objetivo retirado del MANUAL §2:**
> "Mostrar que STRATA **rescata la dirección** del agente — **significativo** donde el agente discrepa de un
> régimen que acierta: **SPY** (M10 vs M5 **p = 0,0041**, leverage effect)."

**Frase retirada del MANUAL §3** (por qué SMCI no separa, con contraste SPY):
> "…STRATA solo rescata donde el agente va a contracorriente de un régimen que acierta (**= SPY**)."

## 2. Cifras canónicas de SPY (el "caso-mecanismo")

- **Accuracy escalera (SPY):** M5 **0,384** → M8 **0,436** → M10 **0,539** (CPCV) / **0,534** (walk-forward
  desplegable) → B&H **0,569** (techo; toro fuerte, "siempre largo" gana).
- **Rescate del agente significativo:** McNemar **M10-WF vs M5 p < 0,001** (también citado p = 0,0041 según el
  contraste); M8 vs M5 p ≈ 0,05–0,07.
- **M8 Sharpe causal** ≈ +0,67 (frágil, DSR ≈ 0,10).
- **SHAP top-5 (M10 SPY):** `ram_score`, `psa_score`, `garch_sigma`, `stress_prob`, `calm_prob` (las 3 STRATA +
  2 de régimen; ninguna personalidad del agente).
- **Leverage effect (Black 1976; Christie 1982):** en índices, Crisis (alta vol) ≈ mercado a la baja y Calma ≈
  al alza, así que el régimen del HMM adquiere **contenido direccional**. Es lo que daba sentido a RAM en SPY;
  en acciones individuales (SMCI) el efecto es débil → por eso ahora el prior RAM es data-driven por activo.

## 3. Respuestas a objeciones retiradas (`objeciones_tribunal.md`)

- **"Un XGBoost debería batir tu regla"** (versión SPY): *M10 y M8 son equivalentes en P&L (DM); SHAP señala las
  features STRATA; el XGBoost redescubre STRATA, no la bate → universalidad (CLAUDE.md §2, nivel 3).* (En el
  marco SMCI esto cambia: M10 **sí** supera a M8 nominalmente, 0,552 vs 0,496.)
- **"Si M8/M10 no baten a B&H en SPY, ¿qué rescataste?"**: *"Rescate" = recuperar la dirección del agente, no
  batir al mercado. Es **significativo en SPY** (M10 vs M5 p = 0,0041) porque el agente discrepa de un régimen
  que acierta (leverage effect). En SMCI no hay margen porque el agente ya va alineado con el régimen.*
- **Anti-trivial (versión SPY):** en SPY "siempre largo" da 0,569 → por eso SPY **no** sirve como benchmark
  justo y se eligió SMCI.

## 4. Dónde viven los datos completos de SPY (no borrados)

| Material | Fichero |
|---|---|
| SPY como "caso-mecanismo" del método (recorrido) | `docs/chats/decision_activo/spy_understandStrata.md` |
| Cifras SPY (método): §1 y §2 | `RESULTADOS_OBJETIVO.md` |
| Narrativa SPY (escalera, regímenes) | `CONOCIMIENTO_ACUMULADO.md` |
| Defensa walk-forward (planos accuracy/Sharpe, falsificación) | `docs/defensa_walkforward.md` |
| Experimentos SPY | `experiments/spy_m10_full_report.py`, `spy_ablation_robustness.py`, `spy_momentum_ablation.py`, `walkforward_robustez.py` (+ sus JSON en `outputs/experiments/`) |
| Q&A de defensa con cifras SPY | `docs/chats/questions_and_answers.md` |

## 5. Cómo retomar SPY (si se decide)

1. Reintroducir SPY como **segundo caso ("caso-mecanismo")** en el cap. 4: donde el rescate del agente **sí es
   significativo** (a diferencia de SMCI, que es el benchmark justo nominal).
2. Restaurar en `memoria/MANUAL.md §2` el objetivo del rescate significativo, y en `objeciones_tribunal.md` las
   filas 4 y 7 en su versión SPY (arriba).
3. En `estructura_cap3.md §3.2`, recuperar el *leverage effect* como justificación direccional fuerte para
   índices (no solo motivación).
4. Reconciliar la cifra SPY: **0,534 (WF, desplegable)** vs **0,539 (CPCV, contraste)** — usar la WF como
   canónica.
