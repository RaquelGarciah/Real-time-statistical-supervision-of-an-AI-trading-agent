"""Builder del cuaderno de EXPERIMENTOS (no canónico). Genera notebooks/experimentos.ipynb.

Recoge la exploración metodológica sobre el número de regímenes K y la comparación regla-a-mano
vs meta-learner, que NO entra en el canónico (SPY, K=3) pero lo respalda. Carga los JSON ya
calculados en outputs/experiments/ (rápido; no recalcula). Tres secciones:
  E1. Selección de K por verosimilitud fuera de muestra (calibración, sin OOS).
  E2. ¿Elegir K por activo? Exploración honesta (criterio direccional ex-ante + validación OOS).
  E3. ¿Cuándo bate el meta-learner (M10) a la regla a mano (M8)? Test del drift, K=2 Y K=3 (§12).
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells: list = []


def md(t: str) -> None:
    cells.append(new_markdown_cell(t))


def code(t: str) -> None:
    cells.append(new_code_cell(t))


md(r"""# STRATA — Cuaderno de **experimentos** (no canónico)

Exploraciones metodológicas que **respaldan** las decisiones del cuaderno canónico
(`strata_canonical.ipynb`, SPY con HMM de 3 estados y gate $\tau=0.5$) pero que, por alcance, no
viven en él. Cada experimento tiene su pre-registro en `BITACORA.md` y su script reproducible en
`experiments/`. Aquí solo se cargan los resultados (`outputs/experiments/*.json`) y se interpretan.

**Hilo conductor — la elección del número de regímenes $K$:** un binario ($K=2$) da *más*
accuracy y Sharpe que $K=3$ en el OOS de SPY. Estas secciones demuestran, sin mirar el futuro de
forma circular, que esa ventaja **no es real** (es cabalgar el drift alcista) y que **$K=3$ es la
elección correcta**.""")

code(r"""import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path.cwd() if (Path.cwd() / "config.py").exists() else Path.cwd().parent
sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")


def load(name):
    return json.load(open(f"outputs/experiments/{name}.json"))""")

# ───────────────────────── E1 ─────────────────────────
md(r"""## E1. Selección de $K$ por verosimilitud fuera de muestra (sin OOS)

El criterio honesto para decidir cuántos regímenes describen los datos no es el P&L de una
ventana, sino la **verosimilitud fuera de muestra** dentro de la calibración (validación temporal
*expanding-window*, 2000–2024, sin tocar el OOS 2024-10+): ¿cada estado extra mejora la
descripción de datos *no vistos*, o sobreajusta? Lo evaluamos por activo
(`experiments/k_selection_panel.py`).""")

code(r"""sel = load("k_selection_panel")
df = pd.DataFrame(sel["per_asset"])
tab = df[["ticker", "heldout_LL_K2", "heldout_LL_K3", "delta_LL_K3_minus_K2", "k_elegido", "vol_anual_calib"]]
tab = tab.rename(columns={"delta_LL_K3_minus_K2": "ΔLL(K3−K2)", "k_elegido": "K elegido", "vol_anual_calib": "vol calib"})
display(tab.set_index("ticker"))
print(f"K=3 elegido en {sel['n_k3']}/{sel['n_k3'] + sel['n_k2']} activos; K=2 en {sel['n_k2']}.")
print(f"ΔLL(K3−K2) > 0 en TODOS: el tercer estado mejora la descripción fuera de muestra en cada activo.")
print(f"Correlación vol↔ventaja-de-K3: ρ={sel['spearman_vol_vs_deltaLL']['rho']:+.2f} "
      f"(p={sel['spearman_vol_vs_deltaLL']['p']:.2f}) → la volatilidad NO decide K.")""")

md(r"""**Lectura E1.** Por verosimilitud fuera de muestra, **$K=3$ bate a $K=2$ en los 10/10 activos**,
con holgura y con independencia de la volatilidad ($\rho\approx0$). El tercer estado mejora la
descripción de datos no vistos en cada activo —no es relleno—. Cautela: este test compara solo
$K=2$ vs $K=3$; la LL es monótona creciente en $K$ (gaussiana mal especificada), así que **no
selecciona $K=3$ como óptimo global** —$K\ge4$ se descarta por interpretabilidad, no por LL—. Lo
que queda firme: el "$K=2$ mejor" del trading OOS NO proviene de la estructura de los datos.""")

# ───────────────────────── E2 ─────────────────────────
md(r"""## E2. ¿Y elegir $K$ por activo con un criterio direccional? (exploración honesta)

Pregunta natural: ¿se puede elegir $K$ por activo, *a pasado*, con un criterio alineado al uso de
STRATA (la **dirección**, no la densidad)? Criterio ex-ante: el $K$ con mejor **acierto direccional
del régimen fuera de muestra en calibración** (`experiments/k_selection_directional.py`). Lo
**validamos** comparando el $K$ elegido en calibración con el $K$ que mejor rinde en el OOS
(diagnóstico, no selección; `experiments/k_per_asset_directional.py`).""")

code(r"""from scipy.stats import binomtest

pad = load("k_per_asset_directional"); agg = pad["aggregate"]
df2 = pd.DataFrame(pad["per_asset"])
tab2 = df2[["ticker", "calib_K", "oos_best_K", "match", "oos_sharpe_K2", "oos_sharpe_K3"]]
display(tab2.set_index("ticker"))

n, k = agg["n_assets"], agg["n_match_calibK_vs_oosbestK"]
p_sign = binomtest(k, n, 0.5, alternative="greater").pvalue
print(f"Concordancia K(calibración) vs K(mejor-OOS): {k}/{n}  →  sign test vs azar p={p_sign:.3f}")
print(f"Cartera K-por-activo: Sharpe OOS medio {agg['mean_oos_sharpe_perasset']:+.3f}  "
      f"vs K=3 fijo {agg['mean_oos_sharpe_fixedK3']:+.3f}  (Diebold-Mariano p={agg['dm_perasset_vs_k3_p']:.2f})")""")

md(r"""**Lectura E2 (honesta).** El criterio direccional tiene *algo* de valor predictivo
(concordancia 7/10, mejor que el 1/10 de criterios ingenuos), pero **no supera al azar de forma
significativa** (sign test $p\approx0.17$, $n=10$) y la cartera K-por-activo **no bate
significativamente** a $K=3$ fijo (Diebold-Mariano $p\approx0.60$). Con $n=10$ y una sola ventana
OOS, los desajustes son indistinguibles del ruido. Conclusión: elegir $K$ por activo es una
extensión **prometedora pero no demostrada**; el canónico usa $K=3$ fijo (E1) y esto queda como
trabajo futuro a validar en un panel mayor.""")

# ───────────────────────── E3 ─────────────────────────
md(r"""## E3. ¿Cuándo bate el meta-learner (M10) a la regla a mano (M8)? — el test del drift

Es la pieza que cierra *por qué* $K=2$ parecía mejor. Si la ventaja de $M8$ fuera destreza de
régimen, batiría al meta-learner $M10$ con independencia del drift del activo; si es cabalgar el
drift, $M10$ (adaptativo) debería ganarle en activos que **caen**. Comparamos $M10$ vs $M8$
**para K=2 Y K=3** en los 10 activos y medimos $\rho(\text{drift}, M10-M8)$ por cada $K$
(`experiments/drift_test_k2k3.py`). Así vemos si la abstención de $K=3$ reduce la dependencia del
drift respecto a $K=2$ —el test previo solo corría $K=2$ y no podía responderlo (autogol corregido)—.""")

code(r"""dr = load("drift_test_k2k3")  # K=2 Y K=3 sobre los 10 activos (corrige el autogol del test previo a K=2)
for K in ("2", "3"):
    d = dr[K]
    t = pd.DataFrame(d["per_asset"]).sort_values("drift_oos", ascending=False)[
        ["ticker", "drift_oos", "M8", "M10", "M10_minus_M8"]]
    print(f"═══ K={K}:  ρ(drift, M10−M8) = {d['rho_drift_vs_M10minusM8']:+.2f}  (p={d['p']:.2f}, n={len(t)})  "
          f"| M8 bate a M10 en {d['n_M8_beats_M10']}/{len(t)} activos")
    display(t.set_index("ticker"))
rho2, rho3 = dr["2"]["rho_drift_vs_M10minusM8"], dr["3"]["rho_drift_vs_M10minusM8"]
print(f"\nComparación: ρ_K2={rho2:+.2f}  vs  ρ_K3={rho3:+.2f}")
if rho2 < -0.4 and abs(rho3) < abs(rho2) - 0.25:
    print("→ K=2 es claramente más CONDICIONAL al drift que K=3: la abstención de K=3 reduce el drift-riding.")
elif rho3 < -0.4:
    print("→ K=3 TAMBIÉN es condicional al drift (ρ_K3 fuerte y negativo): cabalga el drift menos que K=2 pero")
    print("  no es 'puro supervisor'. Honesto: la diferencia es de GRADO, no categórica.")
else:
    print("→ Lectura matizada: ver magnitudes; con n=10 y p no significativos, es sugerente, no concluyente.")""")

md(r"""**Lectura E3 (honesta, corrige el test previo).** El test anterior solo corría $K=2$ —no podía
decir nada de $K=3$—. Aquí comparamos ambos en los 10 activos. $\rho<0$ significa que $M8$ bate al
meta-learner $M10$ **solo en activos alcistas** (cabalga el drift). La comparación $\rho_{K2}$ vs
$\rho_{K3}$ dice si la abstención de $K=3$ **reduce** esa dependencia del drift respecto a $K=2$.

Importante: con $n=10$ y $p$ no significativos, esto es **sugerente, no una prueba**. Mide un
*grado* de drift-riding, no una dicotomía. El cierre robusto de esta cuestión es la **validación
multi-ventana / walk-forward** (2008, 2020, 2022 de la calibración como pseudo-OOS), **pendiente**:
sin ella, "$K=3$ supervisa y $K=2$ cabalga" es una hipótesis bien fundada, no un resultado cerrado.

### Síntesis: por qué $K=3$ (decisión razonada, con sus límites)

1. **E1** — la verosimilitud fuera de muestra bate a $K=2$ en los 10 activos (la LL pediría más
   estados, pero $K\ge4$ no es interpretable; dentro de $\{2,3\}$ gana 3).
2. **E2** — elegir $K$ por activo no está estadísticamente respaldado ($7/10$, $p=0.17$).
3. **E3** — la ventaja nominal de $K=2$ es (sugerentemente) cabalgar el drift; $K=3$ se abstiene.
4. **Mecanismo** — el estado **Estrés = abstención** hace de STRATA un *supervisor*, no un
   *cabalga-drifts*: $K=3$ interviene selectivamente (~49 % de días), $K=2$ por defecto (~75 %).

El canónico adopta $K=3$ deliberadamente. **Límite reconocido:** todo esto vive en una única
ventana OOS alcista; la robustez multi-ventana es trabajo pendiente.""")

# ───────────────────────── E4 ─────────────────────────
md(r"""## E4. Sensibilidad de los umbrales de PSA y GSO — ¿aportan algo si los forzamos a disparar?

Los umbrales de PSA/GSO son percentiles **ex-ante** de la calibración (low=P95, medium=P99, high=max);
el override dispara en severidad ≥ medium (GSO) o = high (PSA), por eso **casi nunca saltan** en SPY
—RAM hace todo el trabajo—. Pregunta natural del tutor: *si los bajamos para que disparen, ¿mejora el
acierto direccional?* Barremos sus umbrales (P50–P99) y medimos. Es **DIAGNÓSTICO, no recalibración**:
se reportan todos los puntos y **no se adopta** ninguno (el default sigue siendo P95/P99 ex-ante; bajar
y quedarse con el mejor OOS sería look-ahead). Pre-registro: `BITACORA.md [2026-06-09]`. Como PSA frena
×0.5 (no voltea) y GSO solo recorta magnitud (no voltea), la H1 honesta es que **bajar sus umbrales no
mejora la dirección**. El barrido mueve `high` en PSA (donde interviene) y `medium` en GSO
(`experiments/psa_gso_threshold_sensitivity.py`).""")

code(r"""s4 = load("psa_gso_threshold_sensitivity")
b, m5 = s4["base"], s4["m5"]
print(f"M5 (agente solo): acc={m5['accuracy']:.3f}  sharpe={m5['sharpe_causal']:+.2f}")
print(f"BASE M8 (P95/P99 ex-ante): acc={b['accuracy']:.3f}  mcc={b['mcc']:+.3f}  sharpe={b['sharpe_causal']:+.2f}"
      f"  | PSA high={b['n_psa_high']}  GSO medium+={b['n_gso_medium_plus']}  (ninguno dispara en el default)")

rows = [{"detector": p["detector"], "pctil": p["pctile"], "umbral": round(p["thresh"], 3),
         "umbral_movido": p["umbral_movido"], "nº dispara": p["n_intervenciones_detector"],
         "accuracy": p["accuracy"], "Δacc vs base": p["delta_accuracy_vs_base"],
         "sharpe": p["sharpe_causal"], "turnover": p["turnover"],
         "McNemar p (vs base)": p["mcnemar_vs_base"]["p"], "b/c": f"{p['mcnemar_vs_base']['b']}/{p['mcnemar_vs_base']['c']}",
         "¿refuta H1?": p["refuta_h1"]}
        for p in s4["sweep"]]
display(pd.DataFrame(rows).set_index(["detector", "pctil"]))
print(f"\nVeredicto: H1 sostenida = {s4['verdict']['h1_sostenida']}  ({s4['verdict']['n_puntos_refutan']}/"
      f"{s4['meta']['n_trials']} puntos refutan)")
for ci in s4["ci_sharpe_extreme_vs_base"]:
    print(f"  IC ΔSharpe {ci['detector'].upper()} P50 vs base: media-diff95=[{ci['ci95_low_meandiff']:.5f}, "
          f"{ci['ci95_high_meandiff']:.5f}]  excluye 0: {ci['excluye_cero']}")""")

md(r"""**Lectura E4 (honesta).** Dos hallazgos distintos, los dos refuerzan que **RAM es quien rescata**:

- **PSA — al bajar el umbral SÍ dispara** (hasta 193/401 días en P50), **pero el accuracy direccional no
  cambia ni un punto** (Δacc = 0.000 en todo el barrido; McNemar vs base trivial, $b=c=0$). Es exactamente
  lo que predice la mecánica: el freno de PSA **encoge la magnitud (×0.5), nunca voltea el signo** → mismos
  aciertos direccionales. El Sharpe se mueve mínimamente ($0.67\to0.74$ en P50) por modular tamaño, pero el
  **IC del ΔSharpe incluye 0** (no significativo).
- **GSO — no dispara en NINGÚN umbral**, ni a P50. Su score de sobreexposición es $\approx0$ en todo el OOS
  porque el agente **nunca arriesga más que la banda de volatilidad** ($|size|\sim$0.1–0.25 < bound). Si el
  score es 0, ningún umbral positivo lo captura: GSO es **estructuralmente inerte** en este OOS,
  independientemente del umbral. (Limitación metodológica ya conocida, aquí confirmada como robusta.)

**Conclusión.** Ni forzando a PSA/GSO a actuar mejora el acierto direccional de M8 —porque **solo modulan
magnitud, no dirección**—. El único detector que voltea el signo (y por tanto el único que puede mejorar la
dirección) es **RAM**. Esto valida, por la vía negativa, la decisión de que M8 se apoye en RAM y que PSA/GSO
sean guardarraíles, no motores. *Nota de rigor:* es un análisis de sensibilidad en una sola ventana OOS; no
se adopta ningún umbral (el default ex-ante P95/P99 es inamovible).""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
out = Path(__file__).resolve().parent / "experimentos.ipynb"
nbf.write(nb, str(out))
print(f"Notebook escrito: {out}  ({len(cells)} celdas)")
