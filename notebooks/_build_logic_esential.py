"""Construye notebooks/logic_esential.ipynb — notebook DIDÁCTICO (no de análisis).

Su único objetivo: que Raquel entienda con claridad qué entra y qué sale de
STRATA, y por qué M5/M8/M10 son comparables (los tres devuelven una posición).
No produce cifras canónicas; las celdas de código corren las funciones reales de
STRATA sobre un día ILUSTRATIVO para ver la mecánica. El notebook canónico de
análisis sigue siendo strata_canonical.ipynb (CLAUDE.md §9).

Correr:  python notebooks/_build_logic_esential.py
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

nb = new_notebook()
C: list = []


def md(text: str) -> None:
    C.append(new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    C.append(new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------------------
md(r"""
# STRATA — La lógica esencial

**Notebook didáctico.** No calcula cifras del TFG: existe para fijar *qué entra y
qué sale* de STRATA y por qué M5, M8 y M10 son comparables. Las celdas de código
ejecutan las funciones **reales** del repositorio sobre un día *ilustrativo*, para
ver la mecánica con los ojos, no de memoria.

> Idea ancla, repítela hasta que sea automática:
> **STRATA no predice el retorno de mañana. Decide una posición para hoy.**
> Y M5, M8 y M10 entregan los tres *el mismo objeto*: una posición $w_t\in[-1,+1]$.
""")

# ---------------------------------------------------------------------------
md(r"""
## 0. El problema en una frase

Un **agente LLM** (AI Hedge Fund, 5 personalidades: Buffett, Wood, Druckenmiller,
Burry, Ackman) decide cada día sobre SPY. Sin supervisar **pierde dinero** y acierta
la dirección **menos del 50 %** de los días.

**STRATA** es una capa de supervisión estadística que se interpone entre el agente y
el mercado. Formalmente es una **función determinista**:

$$f:\ \underbrace{(a_t,\,s_t,\,c_t)}_{\text{decisión del agente hoy}}\ \times\
\underbrace{x_t}_{\text{estado del mercado hoy}}\ \longrightarrow\ \tilde s_t\in[-1,+1]$$

- $(a_t,s_t,c_t)$ = (acción, tamaño, confianza) que el agente decidió **hoy**.
- $x_t$ = estado del mercado **hoy** (régimen, volatilidad, historia de sizing).
- $\tilde s_t$ = el tamaño **supervisado**: la posición con la que nos quedamos.

Fíjate en lo que **no** aparece en $f$: el retorno de mañana $r_{t+1}$. STRATA nunca
lo ve. Esa es la garantía de que no hay *look-ahead*.
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. El objeto que se decide cada día: la tupla del agente

Cada día, las 5 personalidades opinan, y el *Portfolio Manager* las agrega en **una
sola tupla** `AgentOutput`:

- `action` ∈ {long, short, hold}
- `size` ∈ $[-1,+1]$ — **el signo es la dirección** (+ = long, − = short), **el módulo
  es la convicción** (cuánto).
- `confidence` ∈ $[0,1]$.

**Punto crítico que confunde a todo el mundo:** STRATA **no** consume las 5
personalidades por separado. Consume solo la tupla **agregada** `(action, size,
conf)`. Las 5 personalidades individuales solo se usan luego como *features* en M10.
""")

code(r"""
import sys, os
# El notebook vive en notebooks/; añadimos la raíz del repo al path.
ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from strata.types import AgentOutput

# Día ILUSTRATIVO. El agente quiere LONG con convicción media-alta.
agent = AgentOutput(date="2025-03-12", ticker="SPY",
                    action="long", size=+0.30, confidence=0.85)
print("Tupla del agente (input de STRATA):")
print(f"  action = {agent.action}")
print(f"  size   = {agent.size:+.2f}   (signo = dirección, módulo = convicción)")
print(f"  conf   = {agent.confidence:.2f}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. Lo que STRATA mira: el estado del mercado

STRATA resume el mercado con tres modelos estadísticos clásicos, todos calibrados
*una sola vez* sobre 2000–2024 (nunca con datos del futuro):

| Modelo | Qué da | Lo usa |
|---|---|---|
| **HMM gaussiano 3 estados** | régimen $\gamma_t(k)=P(z_t{=}k)$, $k\in\{$Calma, Estrés, Crisis$\}$ | RAM |
| **GARCH(1,1)-t** | volatilidad condicional anualizada $\sigma_t$ | GSO |
| **BOCPD** | ¿el agente acaba de cambiar de opinión? (sobre su propio sizing) | PSA |

En el código, el estado del mercado entra como un diccionario simple:
""")

code(r"""
# Estado del mercado ILUSTRATIVO para ese día: el HMM dice CRISIS dominante.
market_state = {
    "regime": {"calm_prob": 0.05, "stress_prob": 0.15,
               "crisis_prob": 0.80, "viterbi_state": "Crisis"},
    "garch_vol_annualized": 0.233,   # sigma_t anualizada = 23.3 %
}
# Historial de sizing del agente (lo que PSA vigila). Termina en el de hoy.
sizing_history = [0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

print("Estado del mercado (input de STRATA):")
print(f"  Régimen HMM: Calma=0.05  Estrés=0.15  Crisis=0.80  -> CRISIS dominante")
print(f"  sigma_t GARCH = {market_state['garch_vol_annualized']:.3f} (anualizada)")
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. Los tres detectores

Cada detector mira un eje distinto y emite un *score* continuo, un *flag* y una
*severidad* (none/low/medium/high). Son ortogonales por diseño.

### RAM — Regime-Action Mismatch (el detector direccional, el que manda)

Asigna un *sentido permitido* por régimen, apoyándose en el **leverage effect**
(Black 1976; Christie 1982): en índices, la alta volatilidad coincide con caídas, así
que el régimen sirve de **proxy de la dirección**.

- **Calma** → permitido **long** (short es incoherente)
- **Crisis** → permitido **short** (long es incoherente)
- **Estrés** → ambos

$$\text{RAM}_t=\min\!\Big(1,\ \mathbb 1[s_t<0]\,\gamma_t(\text{Calma})
+\mathbb 1[s_t>0]\,\gamma_t(\text{Crisis})\Big)$$

es decir, **la masa de probabilidad de los regímenes donde la acción del agente es
incoherente**. Nuestro agente va **long** (+0.30) en **Crisis** (0.80) → incoherente →
RAM = 0.80. Lo confirmamos:
""")

code(r"""
from strata.detectors import ram_detector

regime_probs = {"Calma": 0.05, "Estrés": 0.15, "Crisis": 0.80}
ram = ram_detector(agent.size, regime_probs)
print(f"RAM score = {ram.score:.3f}  (= crisis_prob, porque el agente va long en Crisis)")
print(f"RAM severidad = {ram.severity}   flag = {ram.flag}")
print(f"Dirección implícita del régimen: regime_sign = {ram.extra['regime_sign']:+.0f}  (Crisis -> short)")
""")

md(r"""
### PSA — Position Sizing Anomaly (consistencia temporal del agente)

- **Qué vigila:** la propia **historia de sizing del agente**, no el mercado.
- **Modelo:** BOCPD (*Bayesian Online Change-Point Detection*, Adams & MacKay 2007).
- **Pregunta:** *¿el agente acaba de cambiar bruscamente de opinión? ¿es inconsistente
  consigo mismo?*
- **Score:** probabilidad posterior de un **punto de cambio reciente** (que la *run-length*
  —días seguidos con el mismo régimen de sizing— sea corta). Alto = volantazo reciente.
- **Qué hace:** si dispara fuerte → **freno** (reduce el size a la mitad) para amortiguar la
  transición ruidosa.
- **En la práctica:** apenas dispara (~2 % del P&L atribuible). Guardarraíl de consistencia.

### GSO — GARCH-bounded Sizing Override (magnitud vs volatilidad)

- **Qué vigila:** la **volatilidad del mercado** (continua), vía GARCH(1,1)-t.
- **Pregunta:** *¿el tamaño del agente es compatible con lo volátil que está el mercado?*
- **La banda:** $b_t=\min(1,\ \text{target\_vol}/\sigma_t)$ con target = 10 %. Si $|s_t|$
  **supera** la banda → sobreexposición → GSO lo **recorta** a la banda (*volatility targeting*).
- **En la práctica:** **nunca** dispara medium+ en el panel (el agente ya es conservador,
  $|s_t|\sim$0.1–0.25). Es un **hallazgo negativo** reportable: la banda casi nunca se viola.

**Los tres son ortogonales:** RAM = **régimen** (dirección discreta), PSA = **consistencia
temporal del agente**, GSO = **volatilidad del mercado** (magnitud). En el día de arriba ni
PSA ni GSO son el protagonista; RAM lo es.

### ¿Por qué la volatilidad va **anualizada**?

El GARCH da una $\sigma$ **diaria**. Se anualiza con $\sigma_{\text{anual}}=\sigma_{\text{diaria}}\cdot\sqrt{252}$
(252 = días de bolsa/año; el $\sqrt{}$ sale de que la **varianza** escala lineal con el tiempo
si los retornos son ~i.i.d., así que la desviación típica escala con $\sqrt{T}$). Dos razones:

1. **Unidad estándar e interpretable.** `target_vol = 0.10` = "quiero ~10 % de volatilidad
   **anual**" lo entiende cualquiera; un 0.63 % diario no dice nada (el VIX, etc., van anualizados).
2. **Coherencia de unidades en la banda.** $b_t=\text{target\_vol}/\sigma_t$ exige numerador y
   denominador en las mismas unidades; como `target_vol` es anual, $\sigma_t$ debe serlo. Si
   mezclaras $\sigma$ diaria con target anual, la banda erraría por un factor $\sqrt{252}\approx16$.

#### Qué tiene que ver esto con el *sizing* del agente

El *sizing* del agente es $w_t$ (su `size`): **cuánto apuesta**, como fracción del capital —no
en euros— con signo = dirección y módulo = convicción. La clave es que el **riesgo** que asume no
es el tamaño de la apuesta a secas, sino el tamaño **multiplicado por lo nervioso que está el
mercado**:

$$\sigma_{\text{posición}}=|w_t|\cdot\sigma_t.$$

Apostar el 100 % ($w_t=1$) sobre un activo plácido ($\sigma_t=8\%$) arriesga *menos* que apostar
el 50 % ($w_t=0.5$) sobre uno que se mueve un 40 %. Por eso el GSO no fija el tamaño: fija el
**riesgo objetivo** `target_vol` (presupuesto de volatilidad) y de ahí **deduce** el tamaño máximo
coherente, igualando el riesgo de la posición al presupuesto:

$$|w_t|\cdot\sigma_t=\text{target\_vol}\ \Longrightarrow\ \underbrace{|w_t|}_{\text{banda }b_t}=\frac{\text{target\_vol}}{\sigma_t}.$$

Eso es literalmente `bound = min(1, target_vol / sigma_t_annualized)` en `gso_detector`. El GSO
compara el `size` del agente contra esa banda; si el agente se pasa (sobreexposición), recorta.
Esto es **volatility targeting**: apuestas menos cuando el mercado está más volátil, para que el
riesgo de tu P&L se mantenga ~constante.

Aquí es donde la división **obliga** a misma escala: `target_vol` se fija con sentido económico
("quiero 10 % **anual**"), que es inevitablemente anual. Para que $b_t$ salga adimensional —una
fracción de capital— $\sigma_t$ debe ser anual también. Si metieras la $\sigma$ **diaria**
($\approx0.0095$) contra el target anual ($0.10$), saldría $b_t=0.10/0.0095\approx10$, y
`min(1, 10)=1` **siempre** → el GSO **no saltaría jamás**.

> **Matiz para el tribunal.** Como en $\text{target\_vol}/\sigma_t$ **ambos** llevan el mismo
> $\sqrt{252}$, el factor se **cancela**: la banda no depende de anualizar... *siempre que seas
> consistente*. El error real no es "anualizar o no", es **mezclar escalas** entre el target y
> $\sigma_t$. La anualización es la convención que garantiza que los dos lados hablan el mismo
> idioma — y, de paso, que `target_vol` se lea como un riesgo interpretable.
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. La intervención: de la decisión del agente a la posición final

Una vez los detectores hablan, la capa de intervención transforma $s_t\mapsto\tilde
s_t$. Tres modos:

- **warn**: $\tilde s_t=s_t$ — no toca nada, solo registra (esto es la base de M5).
- **reduce**: $\tilde s_t=s_t\cdot(1-\text{factor})$ — atenúa.
- **override-C** (el canónico de M8): si RAM dispara, **voltea** hacia el régimen con
  la banda GARCH: $\tilde s_t=\text{signo}_{\text{régimen}}\cdot b_t$.

Corremos los tres modos sobre el **mismo día** y vemos qué posición sale de cada uno:
""")

code(r"""
from strata.strata import StrataSupervisor

print(f"{'modo':10s} | {'RAM sev':8s} | {'posición final w_t':>18s} | acción | ¿intervino?")
print("-" * 72)
for mode, kw in [("warn", {}), ("reduce", {"reduce_mode": "bucket"}),
                 ("override", {"override_variant": "C"})]:
    sup = StrataSupervisor(mode=mode, **kw)
    d = sup.supervise(agent, market_state, sizing_history)
    print(f"{mode:10s} | {d.detectors['ram'].severity:8s} | {d.final_size:+18.4f} | "
          f"{d.final_action:5s} | {d.was_intervened}")

print()
print("Lectura: el agente quería long +0.30.")
print("  warn      -> deja pasar: w = +0.30 (esto es M5)")
print("  reduce    -> lo encoge a cash: w = 0.00")
print("  override  -> lo VOLTEA a short: w = -0.43  (= regime_sign * bound; esto es M8)")
print("  donde bound = min(1, 0.10/0.233) = 0.4292")
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. EL CUADRO CLAVE — por qué M5, M8 y M10 *sí* son comparables

Esta es la duda que hay que matar. *"Si M8 da una posición y M10 da una probabilidad,
¿no estamos comparando cosas distintas?"* **No.** Los tres dan **una posición**. Lo
único que cambia es **cómo** la calculan. La probabilidad de M10 es un **paso
intermedio** (su borrador), no su salida.

Tomemos el **mismo día** (agente: long +0.30; régimen: Crisis):

| | **M5** | **M8 (STRATA)** | **M10 (XGBoost)** |
|---|---|---|---|
| **¿Cómo decide?** | copia al agente | corrige al agente con estadística | calcula $p_1=P(r_{t+1}>0)$ |
| **Paso intermedio** | — | RAM ve Crisis ≠ long → override-C | $p_1=0.30$ (modelo entrenado) |
| **Cómo fija el signo** | signo del agente (+) | $\text{signo}_{\text{régimen}}=-1$ | $\text{signo}(p_1-0.5)=-1$ |
| **SALIDA = posición $w_t$** | **+0.30** | **−0.43** | **−1** |
| **¿Qué *tipo* de objeto es?** | posición | posición | posición |

Las tres columnas terminan en **la misma fila final: una posición**. Esa fila es la
única que se compara. La probabilidad $p_1$ vive *dentro* de M10 y se tira después de
producir el signo — igual que el régimen HMM vive dentro de M8 y no se compara contra
nada.

**Analogía.** Examen diario de una pregunta: *"¿long o short SPY, y cuánto?"*
- M5 responde lo que dice el agente.
- M8 responde corrigiendo al agente con estadística.
- M10 responde calculando una probabilidad y quedándose con su signo.

Los tres entregan **la misma hoja de respuestas** (una posición). El profesor corrige
las tres con el **mismo criterio**. La probabilidad de M10 es su *borrador*: no se
entrega, no se compara. Comparar borradores no tendría sentido; comparamos hojas
entregadas.
""")

code(r"""
import numpy as np

def posicion_M5(agent):
    "Copia al agente. Sin supervisión."
    return agent.size

def posicion_M8(agent, market_state, sizing_history):
    "STRATA override-C: devuelve directamente una posición."
    sup = StrataSupervisor(mode="override", override_variant="C")
    return sup.supervise(agent, market_state, sizing_history).final_size

def posicion_M10(p1, sigma, regime_factor=1.0):
    "XGBoost: la probabilidad p1 es un PASO INTERMEDIO hacia la posición."
    direccion = np.sign(p1 - 0.5)          # <- aquí p1 se convierte en +1/-1 y se 'tira'
    return float(direccion)                 # mapeo direccional canónico (BITACORA 2026-06-08)

p1_del_modelo = 0.30   # M10 cree que es POCO probable que SPY suba mañana
w5  = posicion_M5(agent)
w8  = posicion_M8(agent, market_state, sizing_history)
w10 = posicion_M10(p1_del_modelo, market_state["garch_vol_annualized"])

print("Las tres estrategias, el mismo día, devuelven UNA POSICIÓN:")
print(f"  w_M5  = {w5:+.4f}   (long: copia al agente)")
print(f"  w_M8  = {w8:+.4f}   (short: STRATA volteó al agente)")
print(f"  w_M10 = {w10:+.4f}   (short: signo(p1-0.5) con p1={p1_del_modelo})")
print()
print("Mismo tipo de objeto -> comparables. p1 fue solo el borrador de M10.")
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. ¿Qué es exactamente M10 y qué devuelve?

M10 es un **meta-learner XGBoost**, un clasificador binario entrenado sobre el target
$y_{t+1}=\mathbb 1\{r_{t+1}>0\}$ con 22 *features*:

$$\underbrace{5\text{ personalidades}\times(\text{sign},\text{size},\text{conf})}_{15}
\ +\ \underbrace{\text{ram, psa, gso}}_{3}\ +\
\underbrace{\text{calm, stress, crisis, }\sigma}_{4}\ =\ 22$$

**M10 devuelve, en dos planos:**

1. **Crudo:** un vector de probabilidades $p_1=P(r_{t+1}>0)$, **una por día** del OOS.
   Esto es lo *único* que el XGBoost produce de verdad.
2. **Operativo:** esas $p_1$ convertidas en **posiciones** $w_t$ (paso 5), que es lo que
   se compara con M8.

Y al ser un clasificador, **se le pueden medir métricas de clasificación** (log-loss,
AUC, Brier…) que a M8 *no* tienen sentido, porque M8 no produce ninguna probabilidad.

> **El número que importa:** el log-loss de M10 es ≈ **0.914**. El de una moneda
> (predecir siempre $p_1=0.5$) es $-\ln 0.5=0.693$. **M10 clasifica peor que una
> moneda.** Eso *no* es un fallo del TFG: es la prueba de que ni un XGBoost con 22
> features predice la dirección de SPY a un día — y por eso **M10 no bate a M8**.
""")

code(r"""
import numpy as np
logloss_trivial = -np.log(0.5)
logloss_M10 = 0.914   # mediana OOF del notebook canónico §11 (≈ 0.91)
print(f"log-loss de una moneda (p1=0.5 siempre) = {logloss_trivial:.3f}")
print(f"log-loss de M10                         = {logloss_M10:.3f}")
print(f"-> M10 es PEOR que la moneda por {logloss_M10 - logloss_trivial:+.3f}")
print("Y su accuracy direccional = 0.539 < trivial 0.569: ni siquiera bate a 'comprar siempre'.")
print("La dirección de SPY a 1 día es casi impredecible — aunque M10 SÍ intente predecirla.")
""")

# ---------------------------------------------------------------------------
md(r"""
**Quién predice y quién no (la asimetría clave).** STRATA/M8 **no predice nada**: es una
regla que decide una posición. **M10 sí predice** — es un modelo entrenado que pronostica
$p_1=P(\uparrow)$ y de ahí deriva la posición. Lo bonito de la tesis: el que **sí** predice
(M10) **no le gana** al que **no** predice (M8), y encima su pronóstico es malo (accuracy
0.539 < trivial 0.569). Por eso M10 no predice "una posición" directamente: predice la
**dirección** (probabilidad) y la posición es post-procesado.

| | ¿Predice? | Qué hace | Salida que se compara |
|---|---|---|---|
| **M8** | **No** | regla: corrige al agente con el régimen | posición $w$ |
| **M10** | **Sí** | modelo: pronostica $p_1=P(\uparrow)$ → posición | posición $w$ |
""")

# ---------------------------------------------------------------------------
md(r"""
### De $p_1$ a la posición: la regla del signo

M10 da un número $p_1\in(0,1)$ = probabilidad de que SPY suba mañana. Se convierte en
posición $w$ comparándola con **0.5**:

$$w=\operatorname{signo}(p_1-0.5)=\begin{cases}+1 & p_1>0.5\ \ (\text{cree que sube}\to\text{largo})\\ -1 & p_1<0.5\ \ (\text{cree que baja}\to\text{corto})\\ 0 & p_1=0.5\ \ (\text{indiferente})\end{cases}$$

El **0.5** es el umbral natural de un clasificador binario. La posición sale **±1** (apuesta
entera): solo importa **de qué lado de 0.5** cae $p_1$, no cuánto.
""")

code(r"""
import numpy as np
import pandas as pd

ejemplos = [0.81, 0.55, 0.50, 0.43, 0.20]
tab = pd.DataFrame({
    "p1 = P(sube)": ejemplos,
    "p1 − 0.5": [round(p - 0.5, 2) for p in ejemplos],
    "w direccional = signo(p1−0.5)": [int(np.sign(round(p - 0.5, 4))) for p in ejemplos],
    "(alternativa) w continua = 2·p1−1": [round(2 * p - 1, 2) for p in ejemplos],
})
print("De la probabilidad p1 a la posición:")
print(tab.to_string(index=False))
print()
print("Direccional (la canónica): apuesta entera ±1, tira la confianza.")
print("Continua (alternativa): conserva la confianza (p1=0.55 -> +0.10; p1=0.81 -> +0.62).")
""")

md(r"""**¿Por qué la direccional y no la continua?** (1) Es el **"+1/−1"** que pide el tutor;
(2) es la salida natural de un clasificador binario; (3) hace la comparación con M8 un duelo
justo **signo contra signo**. Con la continua M10 daba Sharpe 0.51 (< M8); con la direccional,
0.71 ≈ M8 — decisión documentada en BITACORA para que no parezca cherry-picking.
""")

md(r"""### ¿Por qué un ML (M10) no bate a la regla a mano (M8)? ¿Y por qué la continua dio menos Sharpe?

**Mito a desmontar:** *"el machine learning siempre gana a una regla a mano."* No es cierto.
Un ML gana solo si (a) hay un patrón rico que aprender, (b) hay datos suficientes y limpios, y
(c) **lo que el modelo optimiza coincide con lo que mides.** Aquí fallan los tres:

1. **El XGBoost optimiza log-loss, NO Sharpe.** M10 se entrena para clasificar bien el signo
   (minimizar log-loss sobre $y_{t+1}$). El paso $p_1\to$ posición es **post-procesado que el
   modelo nunca optimizó**. "Ser ML" no garantiza buen Sharpe: optimizó otra cosa.

2. **Por qué la continua hace daño.** $w=2p_1-1$ apuesta **grande** cuando el modelo está muy
   seguro ($p_1$ cerca de 0 o 1) y pequeño cuando duda. Pero su confianza está **mal
   calibrada** (log-loss 0.91 > 0.69): cuando está muy seguro **no acierta más**. Concentrar
   exposición en los días de alta convicción **dispara la varianza del P&L sin subir la
   media** → como Sharpe = media/desviación, el **Sharpe cae**. La direccional (±1 siempre)
   reparte la exposición por igual → varianza más estable → mejor Sharpe.

3. **M8 no paga "gap de sobreajuste".** Es una regla fija: no se entrena, no generaliza mal.
   M10 se valida out-of-fold con N≈400 (muestra pequeña, etiquetas ruidosas) → paga un coste
   de generalización. López de Prado dixit: con pocos datos y señal débil, un modelo complejo
   no recupera una ventaja estable; una regla simple bien motivada lo iguala o lo supera.

**Lo definitivo:** incluso con el mapeo bueno (direccional), M10 solo **empata** a M8
(0.71 vs 0.75, DM p=0.61). El mapeo continuo solo hizo que M10 *se viera* peor; arreglarlo lo
lleva al empate. **En ningún caso el ML bate a la regla** — porque apenas hay señal direccional
que extraer, y eso *es* la tesis de universalidad.

> *"El XGBoost no pierde por ser ML ni ganaría por serlo: optimiza log-loss, no Sharpe, y con
> la dirección a un día casi impredecible no hay ventaja extra. Por eso, con el mapeo justo,
> empata a mi regla; nunca la bate."*
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. Cómo la posición se convierte en dinero: el backtest

**¿Qué es el P&L?** *Profit and Loss* ("Pérdidas y Ganancias") = cuánto **ganas o pierdes**.
El P&L de cada día es la posición de hoy por el retorno de mañana:

$$\text{P\&L}_t=\tilde s_t\cdot r_{t+1}$$

= la **fracción de tu capital** que ganaste o perdiste ese día. Es **positivo cuando aciertas
la dirección** (largo y el mercado sube, o corto y baja) y negativo cuando te equivocas — por
eso acertar el **signo** es lo que cuenta. Acumulando el P&L día a día sobre €1000 sale la
**equity curve** (los "€1000 → €903 / €1069" de antes).

El backtest es **contabilidad**, no una compra real, con **desfase causal** para no mirar el
futuro (`signal_lag=1`): la posición decidida **hoy** gana el retorno de **mañana**. STRATA
produce $\tilde s_t$; el retorno futuro $r_{t+1}$ entra **solo aquí**, en la contabilidad,
nunca en la decisión.
""")

code(r"""
# Cerramos el día ilustrativo: ¿qué pasó si mañana SPY bajó 1.2 %?
r_manana = -0.012

for nombre, w in [("M5 ", w5), ("M8 ", w8), ("M10", w10)]:
    acerto_signo = np.sign(w) == np.sign(r_manana)
    pnl = w * r_manana
    print(f"{nombre}: w={w:+.3f}  ¿acertó dirección? {str(acerto_signo):5s}  P&L = {pnl*100:+.3f} %")

print()
print("SPY bajó. El agente (M5) iba long -> falló y perdió.")
print("M8 y M10 iban short -> acertaron la dirección y ganaron.")
print("ESTE es el 'rescate': STRATA volteó una apuesta perdedora del agente.")
print("(un día no prueba nada; se mide sobre ~400 días con tests pareados.)")
""")

# ---------------------------------------------------------------------------
md(r"""
## 8. Qué se compara, y con qué test

Todo se compara **sobre las posiciones / su P&L**, nunca sobre la $p_1$ de M10:

| Comparación | Sobre qué | Test | Criterio |
|---|---|---|---|
| **M8 vs M5** | ¿STRATA acierta el signo más que el agente solo? | **McNemar** pareado | $p<0.10$ |
| **M10 vs M8** | ¿el XGBoost universal bate a la regla a mano? | **Diebold-Mariano** | $p>0.10$ ⇒ empatan |
| atribución | ¿qué detector aporta el P&L? | descomposición | se espera RAM ≈ todo |

La hipótesis del TFG **se confirma** si: McNemar M8≻M5 es significativo **y** DM dice
que M10 **no** bate a M8. Es decir: la regla estadística simple rescata al agente, y un
ML con todo dentro no lo hace mejor.
""")

# ---------------------------------------------------------------------------
md(r"""
### 8.1 ¿Cómo mide Diebold-Mariano que dos P&L son "indistinguibles"?

DM no compara dos números agregados (dos Sharpes, dos P&L totales): compara los dos modelos
**día a día, de forma pareada**, sobre las mismas fechas.

**Serie de diferencia.** La "pérdida" de cada día es el retorno con signo negativo, $L_t=-r_t$
(perder = retorno negativo). La diferencia diaria es
$$d_t = L^{M10}_t - L^{M8}_t = r^{M8}_t - r^{M10}_t,$$
o sea, cuánto gana M8 *de más* que M10 ese día. Bajo $H_0:\ \mathbb E[d_t]=0$ (rinden igual),
$$\mathrm{DM}=\frac{\bar d}{\sqrt{s_d^2/n}}\ \sim\ \mathcal N(0,1),$$
con $\bar d$ la media de las diferencias y $s_d^2$ su varianza (horizonte $h=1$: la corrección
HAC de Newey-West solo entra para $h>1$). Se lee el $p$ a dos colas.

**Por qué DM y no comparar los dos Sharpes sueltos.** (1) Es **pareado**: ambos modelos viven
los **mismos días**, así que al restar día a día **el movimiento común del mercado se cancela** y
queda solo la diferencia atribuible a los modelos → mucha más potencia. (2) Testea la **diferencia
directa**, no un cociente (el Sharpe es un cociente con distribución muestral incómoda).

**Lo que salió y la lectura honesta.** $\mathrm{DM}(M10\text{ vs }M8)=-0.43$, $p=0.666$: **no se
rechaza $H_0$** → no hay diferencia *detectable* en el P&L diario. Cautela clave: *no rechazar* **no
es** *demostrar que son iguales* (eso sería afirmar la nula). Por eso se corre **además TOST**
(test de equivalencia), que dio $p=0.42$ → **tampoco** demuestra equivalencia. La frase exacta:
*"indistinguibles en P&L (DM p=0.67); con N≈400 no hay potencia para afirmar equivalencia formal"*,
nunca *"probadas iguales"*. Y M10 **sí** gana en lo que importa: accuracy 0.539 vs 0.436.
""")

# ---------------------------------------------------------------------------
md(r"""
## 9. ¿Predice alguna variable sola la dirección? El descriptivo

Antes de cualquier modelo, el tutor pidió un **descriptivo**: para cada variable continua,
un histograma coloreado por el signo de $r_{t+1}$ (verde = sube, rojo = baja) con el corte
de un árbol de profundidad 1. Pregunta: *"cuando la variable vale X, ¿mañana tiende a subir
o a bajar?"*. La métrica resumen es `acc` = acierto direccional usando **solo** esa variable,
y el listón es el **trivial = 0.569** (la proporción de días al alza: si dices "sube" siempre,
aciertas el 56.9 %). Una variable informa solo si supera ese listón **con holgura**.

(Cifras reales del notebook canónico §6; aquí solo se muestran para entenderlas.)
""")

code(r"""
import pandas as pd
TRIVIAL = 0.569   # proporción de días al alza en el OOS (listón a batir)

desc = pd.DataFrame([
    ("size agente", 0.5985, 0.61, 0.44, "agente corto → sube 61 %; largo → 44 %  (¡va a contramano!)"),
    ("P(Crisis)",   0.5935, 0.46, 0.61, "tras algo de crisis tiende a SUBIR (rebote del OOS)"),
    ("σ_t GARCH",   0.5910, 0.45, 0.60, "más volatilidad → tiende a SUBIR (rebote)"),
    ("PSA score",   0.5860, 0.58, 0.00, "el lado alto son 7 días → no concluyente"),
    ("P(Calma)",    0.5686, 0.66, 0.51, "≈ moneda"),
    ("P(Estrés)",   0.5686, 0.52, 0.66, "≈ moneda"),
    ("RAM score",   0.5686, 0.56, 0.89, "lado alto = 9 días → no concluyente"),
    ("GSO score",   0.5686, 0.57, 1.00, "casi constante → no informa"),
    ("conf agente", 0.5686, 0.53, 0.59, "≈ moneda"),
], columns=["variable", "acc univar.", "% alza si baja", "% alza si alta", "lectura"])
desc["supera trivial"] = desc["acc univar."] - TRIVIAL
print(f"Listón trivial = {TRIVIAL}.  Ninguna variable lo bate con holgura (la mejor, 0.599).")
desc
""")

md(r"""**Conclusión honesta:** ninguna variable, por sí sola, predice la dirección de mañana
—justo lo que dijo el tutor: *"predecir la dirección del mercado es muy difícil"*—. La única
separación nítida y con sentido es que **el propio agente va a contramano** (se pone corto
antes de los días que suben). Por eso el valor **no** está en que una variable adivine el
signo, sino en (a) **condicionar** con el régimen —STRATA— o (b) **combinar** todas las
variables —M10—.
""")

# ---------------------------------------------------------------------------
md(r"""
## 10. RAM no es un predictor: es un *gate*

En el descriptivo anterior `RAM` sale plano (acc = trivial). ¿RAM no sirve? Sí sirve, pero
lo estábamos midiendo con la **pregunta equivocada**. RAM no intenta predecir el mercado;
es un **gate condicionado al agente**: marca los días en que el **régimen contradice** la
apuesta del agente. La pregunta correcta es:

> *"En los días en que RAM dispara ($\ge\tau$), ¿acierta más la dirección seguir al **régimen**
> que seguir al **agente**?"*

(Igual que `GSO` controla la **magnitud** y `PSA` la **estabilidad**: ninguno es un predictor
de signo, por eso los tres salen planos en el descriptivo estándar.)
""")

code(r"""
# Cifras reales del notebook canónico §6 ("descriptivo correcto para RAM").
gate = pd.DataFrame([
    ("RAM < τ (no dispara)", 258, 0.403, 0.473),
    ("RAM ≥ τ (dispara)",    121, 0.413, 0.587),
], columns=["grupo", "n días", "acierto SEGUIR AGENTE", "acierto SEGUIR RÉGIMEN"])
print("Cuando RAM dispara, seguir al régimen acierta 58.7 % vs 41.3 % del agente: +17 puntos.")
print("Ese volteo (override-C) es el 'rescate'. Es la versión dibujada del McNemar M8 vs M5.")
gate
""")

md(r"""**Lectura del gate.** El agente acierta ~40 % en ambos grupos (es malo siempre, < 50 %
—la premisa del TFG—). Lo que cambia es que **en los días que RAM marca, el régimen sí
acierta** (58.7 %), mientras que cuando RAM no dispara el régimen no aporta (47.3 %). Por eso
solo se interviene cuando RAM $\ge\tau$. *Cautela:* es **descriptivo** sobre el mismo OOS, no
prueba independiente; la prueba formal es el McNemar pareado (§8).
""")

# ---------------------------------------------------------------------------
md(r"""
## 11. Cómo usan ese *gate* M8 y M10 (la clave de la tesis)

- **M8** escribe el gate **a mano**: *si* RAM $\ge\tau$, voltea hacia el régimen. Dos piezas:
  RAM = *cuándo* actuar; régimen (Calma/Crisis) = *hacia dónde*.
- **M10** no tiene gate: el XGBoost recibe `ram_score` y las probabilidades de régimen como
  *features* (entre 22) y **aprende solo** a combinarlas. El "descriptivo" de M10 es el
  **SHAP**: cuánto pesa cada *feature* en su predicción.

Si M10 redescubre el gate, sus *features* más importantes deberían ser las de STRATA y
régimen —no las personalidades del agente—. Y eso es justo lo que pasa:
""")

code(r"""
# Cifras reales del notebook canónico §11 (SHAP pooled out-of-fold + ablación).
shap_top = pd.DataFrame([
    ("ram_score",        "STRATA",       0.43),
    ("garch_sigma",      "régimen",      0.43),
    ("psa_score",        "STRATA",       0.38),
    ("crisis_prob",      "régimen",      0.36),
    ("stress_prob",      "régimen",      0.31),
    ("calm_prob",        "régimen",      0.31),
    ("cathie_wood_conf", "personalidad", 0.21),
], columns=["feature", "familia", "mean|SHAP|"])
print("Top features de M10: las 6 primeras son STRATA + régimen.")
print("La primera personalidad del agente aparece en el puesto 7 (la mitad de peso que ram_score).")
print()
print("Ablación: quitando STRATA/régimen y dejando solo las 15 features del agente,")
print("  Sharpe de M10:  +0.64  ->  +0.21   (la señal estaba en STRATA/régimen)")
print("M10 NO bate a M8 (Diebold-Mariano p=0.61): empatan, redescubriendo el mismo gate.")
shap_top
""")

md(r"""**La frase que cierra el proyecto:** *"En M8 el gate lo escribo yo; en M10 nadie lo
escribe, pero el SHAP demuestra que el XGBoost le da el máximo peso justo a `ram_score` y al
régimen — redescubre mi regla desde los datos, y aun así no la bate. Eso confirma que la
señal útil es la que STRATA codifica explícitamente, no las personalidades del agente."*
""")

# ---------------------------------------------------------------------------
md(r"""
## 12. ¿Predigo dirección o tamaño? B&H, y cómo demuestro que mejoro al agente

**Cuidado con una confusión típica.** Tu salida es una posición
$w_t=\underbrace{\text{signo}}_{\text{dirección}}\times\underbrace{\text{magnitud}}_{\text{tamaño}}$.
STRATA **no predice** ninguna de las dos: las **decide por regla**.

- **No predices la dirección del mercado** — es casi imposible (lo demuestra M10).
- **Tampoco "predices el tamaño"** — el tamaño lo **calculas** a partir de la volatilidad.
- **Tu mejora viene de la DIRECCIÓN**: corriges el **signo** del agente con el régimen
  (accuracy $0.384\to0.436$). El tamaño es gestión de riesgo, no es donde está el valor.

> Frase correcta: *"No predigo el mercado; corrijo la dirección de un agente que va
> sistemáticamente al revés, y dimensiono la apuesta por riesgo."*

### Qué es el "tamaño" en economía

$w\in[-1,+1]$ = **peso de la posición** = fracción del capital expuesta (exposición /
apalancamiento). $+1$ todo largo, $-1$ todo corto, $0$ fuera. En dinero: $w=0.13$ → ese día
pones el **13 % de tus €1000** en esa dirección; $\text{P\&L}=w\cdot r_{t+1}\cdot$ capital.

La magnitud sale de $b_t=\min(1,\ \text{target\_vol}/\sigma_t)$ = **volatility targeting**:
apuestas **menos cuando el mercado está más volátil**, para que el riesgo de tu P&L se
mantenga ~constante (objetivo 10 % anual). Es gestión de riesgo estándar, no una predicción.

### Diferencia con Buy & Hold (B&H)

- **B&H**: $w\equiv+1$ **siempre** (todo largo, nunca cambia) = "comprar el índice y esperar".
- **M8**: $w$ cambia de signo y tamaño cada día, usando agente + régimen.
- **Hecho honesto (dilo tú primero):** en este OOS **B&H gana a todos** (€1000→€1317 vs M8
  €1069). **Ningún sistema bate al mercado.** Pero **eso no es lo que demuestras**: B&H ni usa
  el agente. Tu tesis es **rescatar al agente** (M8 €1069 > M5 €903), no batir al mercado.
""")

code(r"""
import pandas as pd
# Cifras reales del notebook canónico §7/§10 (se muestran; el cálculo vive allí).

# (1) ¿Mejora M8 sobre M5? Acierto direccional y dinero.
acc = pd.DataFrame({"accuracy (signo)": [0.384, 0.436], "hit_rate": [0.377, 0.431],
                    "€1000 →": [903.5, 1068.9]}, index=["M5 (agente solo)", "M8 (STRATA)"])
print("(1) Acierto direccional y dinero:"); print(acc.to_string()); print()

# (2) Matrices de confusión de la DIRECCIÓN (real vs predicho).
cm5 = pd.DataFrame([[36, 192], [40, 133]], index=["real ↑", "real ↓"], columns=["pred ↑", "pred ↓"])
cm8 = pd.DataFrame([[107, 121], [90, 83]], index=["real ↑", "real ↓"], columns=["pred ↑", "pred ↓"])
print("(2) M5 predice ↓ casi siempre (325/401): sesgo corto en mercado alcista → falla.")
print(cm5.to_string()); print("    M8 reequilibra (197↑ / 204↓):"); print(cm8.to_string()); print()

# (3) La PRUEBA pareada: McNemar M8 vs M5 (sobre los 121 días en que difieren).
print("(3) McNemar M8 vs M5 — la prueba que importa:")
print("    de 121 días en que difieren: M8 ARREGLA 71, ESTROPEA 50  →  χ²=3.31, p=0.069")
print("    permutación por bloques p=0.044  |  Deflated Sharpe 0.106")
print("    → a α=0.10 rechaza H0 (rescate); a α=0.05 borderline. Honesto, no slam-dunk.\n")

# (4) La escalera del mecanismo: ¿de dónde viene el rescate?
esc = pd.DataFrame({"€1000 →": [903.5, 932.2, 1068.9],
                    "qué hace": ["copia al agente", "ENCOGE la apuesta (no cambia signo)",
                                 "VOLTEA la dirección al régimen"]},
                   index=["M5", "M7 (reduce)", "M8 (override)"])
print("(4) Escalera M5→M7→M8: el rescate viene de CAMBIAR LA DIRECCIÓN, no de arriesgar menos:")
print(esc.to_string())
""")

md(r"""**Lectura tangible para la defensa.** Lo que demuestra que mejoras al agente, de menor
a mayor rigor: (1) la **accuracy del signo** sube ($0.384\to0.436$); (2) la **matriz de
confusión** lo hace visible (el agente apostaba "baja" 325/401 días, M8 reequilibra); (3) el
**McNemar pareado** es la prueba formal (arregla 71, estropea 50, p=0.069); (4) la **escalera
M5→M7→M8** aísla el mecanismo: encoger (M7) no basta, **voltear la dirección (M8) sí**. El
€1000→ es la ilustración económica que el tribunal "ve", pero la **prueba** es el McNemar.
""")

# ---------------------------------------------------------------------------
md(r"""
## 13. La hipótesis falsable (lo que el TFG demuestra o refuta)

> *Filtrar/atenuar las decisiones de un agente LLM con detectores estadísticos
> clásicos rescata al agente cuando pierde y acierta la dirección < 50 %.*

Falsable en tres niveles:

1. **Estadístico:** McNemar M8 vs M5 con $p<0.10$.
2. **Mecánico:** atribución del P&L a cada detector (se espera que RAM domine).
3. **Universalidad:** un XGBoost-CPCV (M10) **no** debe batir a M8 (DM $p>0.10$), y SHAP
   debe señalar las *features* de STRATA como las informativas.

Y una regla de falsación pre-registrada, el **`prior-flip`**: si el signo de la
calibración por régimen ≠ el signo en OOS, se documenta como caso donde la técnica
*no* funciona. Eso blinda contra el p-hacking.
""")

# ---------------------------------------------------------------------------
md(r"""
## 14. El HMM da VOLATILIDAD, no dirección (filtrado, leverage, prior-flip)

Pregunta recurrente: *"si el HMM me da el régimen de cada día, ¿no sirve eso para
predecir mucho?"*. Sirve — es la señal más informativa de M10 (SHAP) — pero con **dos
límites duros** que conviene recitar, porque son justo lo que un tribunal ataca.

### 14.1 Filtrado vs suavizado (la trampa de look-ahead)

Hay dos formas de "etiquetar el régimen del día $t$", y solo una es legal:

- **Suavizado** $P(\text{estado}_t\mid \text{TODA la serie})$ (Viterbi / posterior completo):
  para decidir $t$ mira días **posteriores** a $t$. Sirve para *describir* el pasado, pero
  como señal de trading es **look-ahead**: en tiempo real no tienes el futuro.
- **Filtrado** $P(\text{estado}_t\mid \text{datos}\le t)$: solo pasado. **Es lo único legal**, y
  es lo que STRATA usa (`predict_proba_filtered`). El régimen sirve, pero **solo el filtrado**.

### 14.2 El HMM modela volatilidad; la dirección solo llega por el *leverage effect*

El HMM se ajusta sobre $(r_t,\ \text{RV}_{21})$ → sus estados son regímenes de **volatilidad**
(Calma/Estrés/Crisis), no de signo. Saber que estás en alta vol informa del **tamaño** del
movimiento, no de su **dirección**. La dirección solo aparece **vía leverage effect** (Black 1976;
Christie 1982): en índices, alta vol coincide con caídas. Y eso es **contemporáneo** (mismo día),
no predictivo. Al hacerlo causal (régimen de hoy → retorno de mañana), la señal direccional casi
se evapora. Retorno medio por régimen filtrado (`experiments/regime_direction_table.py`):

| | SPY mismo-día (calib) | SPY día-sig (calib→oos) | SMCI mismo-día (calib→oos) | SMCI día-sig (calib→oos) |
|---|---|---|---|---|
| Calma | +0.00054 | +0.00032 → +0.00012 | −0.00034 → +0.00191 | −0.00006 → **+0.02546** |
| Estrés | +0.00017 | +0.00033 → +0.00082 | +0.00143 → +0.00328 | +0.00117 → **−0.00384** |
| Crisis | ≈0 | +0.00015 → +0.00329 | +0.00278 → −0.00238 | +0.00248 → **−0.00003** |

**SPY:** el signo causal **transfiere** (todo positivo: deriva alcista, leverage débil pero coherente).
**SMCI:** el signo causal **FLIPEA** en los tres regímenes calib→OOS — el régimen no predice dirección
fuera de muestra. Esto es el **`prior-flip`**, la regla de falsación pre-registrada: donde el signo de
calibración ≠ el de OOS, se documenta como caso donde la técnica *no* aplica. SPY (leverage) funciona;
SMCI/TSLA/UNG (sin leverage) no. Ese es el **dominio de validez**.

### 14.3 ¿Y si hacemos que el HMM prediga dirección? (la pregunta del tutor)

Se puede, de tres formas: (a) cambiar las features de emisión por direccionales (momentum/signo);
(b) HMM multivariante con una emisión direccional añadida; (c) regímenes con **media** propia
(Hamilton 1989, *mean-switching*) — que es **lo que STRATA ya hace** con el signo data-driven
($\mu_k$ por estado) + `prior-flip`. **Pero todas chocan con el mismo muro:** la volatilidad es
**predecible** (clustering, por eso el GARCH funciona); la dirección a un día es **casi una
martingala** (eficiencia de mercado). Un HMM ajustado a dirección **sobreajusta el pasado y no
transfiere** — evidencia directa: la trendiness en calibración no predice el beneficio del momentum
en OOS (Spearman de −0.55 a +0.45, nunca significativo sobre 10 activos), y el `prior-flip` muestra
que el signo por régimen solo transfiere en SPY.

**Consecuencias de hacerlo direccional:** pierdes la coherencia con vol/GARCH y, sobre todo, **el relato
del leverage effect** (tu aportación teórica); RAM deja de ser "desajuste con el régimen de volatilidad";
y ganas riesgo de overfitting sin una señal direccional fiable a cambio. Por eso el diseño actual es el
**inteligente**: predice lo predecible (volatilidad) y cosecha dirección **solo donde la economía la
regala** (leverage). El "HMM direccional", en su versión defendible (medias por régimen), ya está hecho,
y su límite es justo el `prior-flip`.

""")

# ---------------------------------------------------------------------------
md(r"""
## 14b. El embargo del walk-forward: por qué **embargo = 1** (no 5)

**Decisión (2026-06-17):** en la validación walk-forward de M10 uso **embargo = 1 día**, no 5.
Aquí está el porqué, atado a la literatura, porque es justo el detalle que un tribunal ataca.

### Qué es el embargo (y qué NO es)

Al validar sin mirar el futuro hay **dos** mecanismos distintos, que conviene no confundir
(López de Prado 2018, cap. 7, §7.4):

- **Purga (*purging*).** Quita del entrenamiento las observaciones cuya **etiqueta se solapa en el
  tiempo** con la del test. Su tamaño = **horizonte de la etiqueta**.
- **Embargo (*embargoing*).** Quita, *además*, unas pocas observaciones **inmediatamente posteriores**
  al test, para cortar la **autocorrelación residual** en la frontera. López de Prado lo fija como una
  fracción pequeña del total: *"A small value $h\approx 0.01\,T$ often suffices"*.

Lo esencial: **ambos existen porque en K-fold / CPCV los folds de test tienen entrenamiento ANTES y
DESPUÉS** (estructura *interleaved*, bidireccional). El embargo blinda ese borde posterior.

### Por qué en *mi* validación el número correcto es 1

Mi validación **no es K-fold ni CPCV**: es **walk-forward de origen móvil** (*rolling-origin*,
Tashman 2000), donde **el test es siempre futuro respecto al entrenamiento** → el solape
"entrenamiento *después* del test" que motiva el embargo **no existe por construcción**. Solo queda el
solape de la **etiqueta**, y mi etiqueta tiene **horizonte 1 día**:
$$ y_t = \mathbf{1}[\,r_{t+1} > 0\,]. $$
La etiqueta de $t$ solo ocupa hasta $t+1$ → la purga necesaria es de **1 día**. El **embargo $\geq 5$**
de CLAUDE.md §4 es la regla de **CPCV** (folds bidireccionales) y de **etiquetas multi-día**
(triple-barrier): **otro régimen**, no el mío. Cierre (Bergmeir, Hyndman & Koo 2018): con **residuos
no correlados**, la validación con hueco mínimo es **válida**; el único solape mecánico —la etiqueta
$t+1$— se elimina con **embargo = 1**.

### Frase lista para defender

> *"El embargo $\geq 5$ es una recomendación calibrada para Purged/Combinatorial K-Fold con folds
> interleaved y etiquetas multi-día (López de Prado 2018, §7.4), no para evaluación walk-forward de
> origen móvil con etiqueta de horizonte 1. En rolling-origin (Tashman 2000) el test es siempre futuro
> respecto al entrenamiento, lo que elimina por construcción el solape bidireccional que motiva el
> embargo; el único solape residual —la etiqueta $y_t=\mathbf{1}[r_{t+1}>0]$— se purga con embargo = 1.
> La validez de la validación con hueco mínimo bajo residuos no correlados está en Bergmeir, Hyndman &
> Koo (2018)."*

Apoyos sobre el **tamaño** del hueco en datos dependientes: *h-block* (Burman, Chow & Nolan 1994)
introduce eliminar $h$ vecinos para datos dependientes; la idea de ligar $h$ a la **estructura de
dependencia** (y *hv-block*) es de Racine (2000); Bergmeir & Benítez (2012) respalda empíricamente el
buen comportamiento de la CV en series temporales.

### Honestidad (esto va conmigo, no contra mí)

embargo = 1 es **corrección del protocolo**, no un truco para "sacar significancia":

- **Sí** mejora la accuracy **nominal** en SMCI: $0.524$ (embargo 5) $\to \mathbf{0.552}$ (embargo 1),
  con posiciones equilibradas (47 % corto, 48 % de días alcistas) — no es "ponerse corto a un activo
  que cae".
- **No** crea significancia. El único $p<0.05$ aparece **solo** en embargo = 1 (pico aislado: embargo
  0 y 2, igual de válidos, dan $p\approx0.12$–$0.13$); no sobrevive la corrección por multiplicidad del
  barrido (Bonferroni-5: $0.047\times5\approx0.24$) ni el Holm de la familia {vs M5, M8, B&H}. Se
  reporta como **sensibilidad**, no como hallazgo confirmatorio.

**El barrido completo (embargo 1,2,3,5,10,21) confirma que la accuracy es RUIDO, no señal**
(`experiments/embargo_sweep.py`): el rango entre embargos es $0.032$ en SPY y $0.040$ en SMCI, ambos
$\approx$ **1 desviación binomial** ($\pm0.0316$ para $n\approx250$). Y van en **direcciones opuestas**:
en SPY el "mejor" es embargo 5, en SMCI es embargo 1 — se contradicen, luego no hay regla que extraer.
La causa mecánica: quitar 4 días de entrenamiento desplaza $p_1$ en $\sim0.06$ y **voltea el signo del
10 % de las posiciones**. Que un cambio tan pequeño mueva tanto es, en sí, **la prueba de que M10 no
tiene señal direccional** (un modelo con señal sería estable). Cambiar la semilla mueve la accuracy lo
mismo que cambiar el embargo.

> **Regla:** elijo embargo = 1 **por principio** (horizonte = 1), justificado a priori — *no* por su
> p-valor. Y digo yo misma que la significancia no sobrevive. Esa es la defensa sólida.
""")

# Celda ilustrativa: el indexado del walk-forward con embargo 1 vs 5 (sin datos, solo la mecánica).
code(r'''
# Mecánica del embargo en el walk-forward (ilustrativo, sin datos):
# para predecir el bloque que empieza en `start`, entreno con [:start - embargo].
STEP = 21
start = 171                                   # un reentreno cualquiera
for embargo in (5, 1):
    tr_end = start - embargo                  # última fila de train = tr_end - 1
    ultima_etiqueta_usa = (tr_end - 1) + 1    # y_t usa r_{t+1}  (horizonte 1)
    primer_ret_test = start + 1               # la fila `start` se evalúa contra r_{start+1}
    gap = primer_ret_test - ultima_etiqueta_usa
    print(f"embargo={embargo}: train=[:{tr_end}]  predice=[{start}:{start+STEP}]  "
          f"última etiqueta de train usa r_{ultima_etiqueta_usa}, primer retorno de test r_{primer_ret_test}"
          f"  -> gap={gap}d {'(sin solape)' if gap >= 1 else '(SOLAPE)'}")
# Con horizonte 1, embargo=1 ya deja gap>=1 -> sin solape de etiquetas -> sin fuga.
''')

# ---------------------------------------------------------------------------
md(r"""
## 14c. El momentum y M10 sobre SPY: por qué el momentum NO entra al modelo

Exploración (2026-06-17) de si añadir *momentum* a M10 mejora, y si se puede decidir **a priori**.
Conclusión: el momentum **no es un componente desplegable**, y el M10 canónico no tiene alfa direccional.

### El cuadro de M10 canónico (ALL22, sin momentum) sobre SPY — embargo = 1

`experiments/spy_m10_full_report.py` (OOS 2025-05→2026-05, $n=251$):

| modelo | accuracy | Sharpe | equity | maxDD | AUC | log-loss | Brier |
|---|---|---|---|---|---|---|---|
| M5 (agente) | 0.367 | −2.73 | 0.932 | −0.069 | — | — | — |
| M8 (STRATA) | 0.442 | **+1.60** | **1.097** | −0.060 | — | — | — |
| M10 (ALL22) | 0.494 | −0.60 | 0.920 | −0.161 | 0.531 | 0.856 | 0.308 |
| B&H | 0.566 | +2.20 | 1.302 | −0.098 | — | — | — |

Tests de M10: vs M5 **McNemar $p=0.007$** (corrige al agente); vs M8 $p=0.29$ (universalidad); vs B&H
$p=0.13$; **sign vs 0.5 $p=0.90$**; IC95 del exceso de accuracy $[-0.058,+0.042]$ (cruza el 0).

Lectura: el M10 desplegable es **una moneda que pierde dinero** (acc 0.494, AUC 0.53, log-loss $>0.693$,
equity $<1$). Su único valor —igual que el de M8— es **corregir al agente** ($p=0.007$), no generar alfa.
Y la regla determinista **M8 le gana al ML** económicamente (Sharpe $+1.60$ vs $-0.60$): universalidad.

### El momentum: significativo en SPY, pero no robusto ni justificable a priori

Con momentum, SPY/aug subía a accuracy $0.59$ y era significativo vs azar — pero:

1. **No bate a B&H** y el motor era el momentum, no STRATA (`spy_momentum_ablation.py`): el momentum solo
   da Sharpe alto con accuracy de moneda (0.52); STRATA+régimen añade los puntos significativos.
2. **No se puede decidir a priori si meterlo.** Una regla "mete momentum si funcionó el último año"
   acierta 7/10 (`momentum_decision_rule.py`) **pero no es robusta**: el supuesto (que el rendimiento del
   momentum persiste) **es falso** — sobre 708 puntos de 24 años, Spearman señal↔resultado $=0.03$
   (`momentum_rule_robustness.py`); y el 7/10 oscila entre 2/10 y 8/10 según parámetros (ruido).
3. **A horizonte mensual** (Moskowitz, Ooi & Pedersen 2012) el momentum es real en calibración (acc 0.55)
   pero **no transfiere** al OOS (`momentum_tsmom_monthly.py`): OOS corto y alcista, B&H gana.

> **Frase:** *"El momentum no entra al modelo: demuestro (708 puntos, sin look-ahead) que su beneficio no
> persiste, así que no hay regla a priori para incluirlo. El M10 desplegable es STRATA puro, que no
> predice dirección mejor que el azar; su valor es corregir al agente ($p=0.007$), no batir al mercado."*

""")

# ---------------------------------------------------------------------------
md(r"""
## 14d. El ensemble de M10: qué es, por qué ayuda y por qué es lícito

El M10 desplegable no es **un** XGBoost, es el **promedio de 10**. Esto conviene tenerlo clavado para la
defensa.

### Qué es exactamente

Un XGBoost tiene **azar dentro**: cada árbol ve solo el 80 % de las filas (`subsample=0.8`) y el 80 % de las
features (`colsample_bytree=0.8`), elegidas según una **semilla**. Cambias la semilla → modelo distinto →
`p1` distinto. Ese azar es **ruido**, no información. El ensemble es la receta más simple:

1. Entreno **10 XGBoost idénticos**, cambiando solo la semilla (42, 43, …, 51).
2. Promedio sus probabilidades: $p_1^{\text{ens}} = \frac{1}{10}\sum_k p_1^{(k)}$.
3. Posición = $\text{signo}(p_1^{\text{ens}} - 0.5)$.

Mismas 22 features, mismo walk-forward, mismo embargo. **Apuesta todos los días** (cobertura 100 %), igual
que B&H.

### Por qué ayuda (reducción de varianza)

Al promediar, la **señal** (lo que dicen las features) es común a las 10 y se conserva; el **ruido** del
muestreo aleatorio es distinto en cada una y se **cancela parcialmente**. Resultado: una $p_1$ más **estable**
→ sube algo la accuracy (0.52 → 0.552) y, sobre todo, mejora el Sharpe (0.85 → 1.84) y la equity (1.45× →
3.24×), porque las posiciones dan menos vaivenes. La celda de abajo lo ilustra con números.

### Por qué es lícito (y no es trampa)

- **Principio de *bagging*** (Breiman 1996): *"the vital element is the instability of the prediction method;
  if perturbing the learning set can cause significant changes in the predictor, bagging can improve
  accuracy."* Promediar versiones inestables reduce el componente de **varianza** sin añadir sesgo.
- **Aleatorización del aprendiz** (Dietterich 2000): mi variabilidad es la **semilla** (mismo dato, distinto
  submuestreo interno) → *seed averaging*. Matiz honesto: el *bagging* clásico remuestrea los **datos**
  (bootstrap); cito Breiman por el **principio**, Dietterich por la aleatorización del aprendiz.
- **Sin look-ahead:** cada uno de los 10 se entrena **solo con el pasado** (mismo WF). Promediar no mira el
  futuro.
- **Sin cherry-pick:** promedio **las 10** semillas; **no elijo la mejor** (eso sí sería p-hacking).

### Honestidad (lo dices tú)

El ensemble **reduce ruido, no crea señal**. Si la dirección diaria de SMCI es casi aleatoria, promediar no
inventa información → la ganancia de accuracy es **modesta** y la significancia no sobrevive (DSR=0.72<0.95).
Pero como decisión metodológica es impecable: es la **única** palanca que mejora sin tocar la cobertura ni
mirar el futuro. *(Respaldo completo en `decisiones_respaldadas_literatura.md` §2.)*

> **Frase:** *"El M10 promedia 10 XGBoost que solo difieren en la semilla; reduce la varianza de la predicción
> siguiendo el principio del bagging [Breiman 1996], sin remuestrear datos —seed averaging, aleatorización del
> aprendiz [Dietterich 2000]—, sin look-ahead y sin elegir la mejor semilla. Mejora accuracy y, sobre todo,
> Sharpe/equity; pero reduce ruido, no crea señal."*
""")

code(r'''
# Ilustración del ensemble: promediar estimaciones ruidosas reduce su dispersión (no necesita datos reales).
import numpy as np
rng = np.random.default_rng(0)
p_true = 0.55                       # "señal" verdadera (prob. de subida de un día)
n_dias, n_seeds = 2000, 10
# Cada semilla estima p_true con ruido (el azar interno del XGBoost):
ruido = rng.normal(0, 0.08, size=(n_dias, n_seeds))
p_por_semilla = p_true + ruido      # estimación de 1 sola semilla, por día
p_ensemble = p_por_semilla.mean(axis=1)   # promedio de las 10 semillas, por día

print(f"Desviación típica de la estimación con 1 sola semilla : {p_por_semilla[:, 0].std():.4f}")
print(f"Desviación típica de la estimación con ensemble de 10  : {p_ensemble.std():.4f}")
print(f"Reducción de varianza ~ 1/sqrt(10) = {1/np.sqrt(n_seeds):.2f}  -> menos decisiones volcadas por ruido")
print("La señal (0.55) se conserva; el ruido del muestreo se cancela. Eso es el ensemble.")
''')

# ---------------------------------------------------------------------------
md(r"""
## 14e. Contra qué se mide la accuracy: el test binomial vs *no-information rate*

Para decir que un modelo "acierta", hay que contrastarlo contra el baseline correcto. El ingenuo (0.5) es
**demasiado fácil** cuando las clases están desbalanceadas; el correcto es la **clase mayoritaria**.

### Qué es

- **Baseline de no-habilidad = clase mayoritaria (regla ZeroR** [Witten et al. 2016]**):** predecir siempre
  la dirección dominante. Su accuracy = **no-information rate (NIR)** = frecuencia de la clase más frecuente
  = $\max(\%\text{suben}, \%\text{bajan})$ [Kuhn 2008].
- **El test:** binomial unilateral — `binomtest(aciertos, n, p=NIR, alternative="greater")`.
  - **H0:** accuracy del modelo $\leq$ NIR (no tiene habilidad por encima de predecir la clase dominante).
  - Rechazar H0 ⇒ el modelo tiene **habilidad real**, no solo el sesgo de la clase mayoritaria.

### Por qué es más honesto que el sign test vs 0.5

Porque **0.5 no es "sin habilidad" si las clases están desbalanceadas.** En SMCI bajan más días de los que
suben, así que **"siempre corto" ya saca 0.516 gratis**. Comparar contra 0.5 regala esa diferencia y hace que
el modelo parezca mejor de lo que es. La celda lo muestra: el mismo M10 (0.552) pasa de "casi significativo"
(vs 0.5) a "no significativo" (vs NIR).

> **Frase:** *"No contrasto la accuracy contra 0.5 (la moneda), sino contra el no-information rate —la
> frecuencia de la clase mayoritaria, regla ZeroR [Witten et al. 2016; Kuhn 2008]— mediante un test binomial
> unilateral. Es el baseline de no-habilidad correcto en clases desbalanceadas y, por tanto, más exigente:
> en SMCI lo adopto aun sabiendo que reduce la significancia de mi modelo (de 0.057 a 0.141)."*

**Matiz:** este test es de **una muestra** (accuracy del modelo vs un umbral). Es complementario a **McNemar /
block-permutation vs B&H**, que son **pareados** (comparan dos estrategias día a día). Juntos responden
"¿mejor que el no-skill?" y "¿mejor que esta estrategia concreta?".
""")

code(r'''
# El mismo modelo, dos baselines: 0.5 (moneda, fácil) vs NIR=clase mayoritaria (correcto, exigente).
from scipy.stats import binomtest
n, aciertos = 250, 138            # M10 en SMCI: accuracy 0.552 sobre 250 días
frac_up = 0.484                   # % días que suben -> baja la mayoría
nir = max(frac_up, 1 - frac_up)   # no-information rate = clase mayoritaria ("siempre corto")
p_05  = binomtest(aciertos, n, 0.5, alternative="greater").pvalue
p_nir = binomtest(aciertos, n, nir, alternative="greater").pvalue
print(f"accuracy M10 = {aciertos/n:.3f}   |   NIR (clase mayoritaria) = {nir:.3f}  ('siempre corto')")
print(f"  vs 0.5  (moneda, baseline INCORRECTO/fácil): p = {p_05:.3f}   -> parece casi significativo")
print(f"  vs NIR  (clase mayoritaria, baseline CORRECTO): p = {p_nir:.3f}   -> NO significativo (honesto)")
print(f"  la 'habilidad gratis' por el desbalance = NIR - 0.5 = {nir-0.5:.3f}  (lo que el test vs 0.5 te regala)")
''')

# ---------------------------------------------------------------------------
md(r"""
## 15. Checklist — lo que debes saber recitar

1. **STRATA no predice $r_{t+1}$.** Es $f:(\text{decisión}_t,\text{mercado}_t)\to
   \text{posición}_t$, determinista. El retorno futuro solo entra en el backtest
   ($w_t\cdot r_{t+1}$, `signal_lag=1`).
2. **Input de STRATA** = tupla **agregada** del agente + estado de mercado (régimen,
   $\sigma$) + historial de sizing. Las 5 personalidades sueltas **no** entran (solo en M10).
3. **Output de STRATA** = `SupervisedDecision`; lo operativo es **`final_size`** $=w_t$.
4. **M5, M8 y M10 devuelven los tres una posición $w_t$.** Por eso son comparables.
   La $p_1$ de M10 es un **paso intermedio**, no su salida.
5. **Ninguna variable sola predice la dirección** (descriptivo: nadie bate el trivial 0.569
   con holgura). La única señal limpia es que **el agente va a contramano**.
6. **RAM es un *gate*, no un predictor.** En los días que dispara, seguir al régimen acierta
   58.7 % vs 41.3 % del agente. override-C voltea: $w_t=\text{signo}_{\text{régimen}}\cdot b_t$.
   Ese volteo es el "rescate". (`GSO` controla magnitud, `PSA` estabilidad.)
7. **M8 escribe el gate a mano; M10 lo aprende.** El SHAP de M10 pone arriba `ram_score` y el
   régimen (las personalidades, relegadas); la ablación sin STRATA hunde el Sharpe (+0.64→+0.21).
   M10 **redescubre** el gate y **no lo bate** (DM p=0.61).
8. **M10 clasifica peor que una moneda** (log-loss 0.914 > 0.693). Eso *confirma* la hipótesis
   de universalidad: la dirección a un día es casi impredecible.
9. **Se compara posición contra posición** (McNemar sobre el signo, DM sobre el P&L),
   nunca probabilidad contra posición.
10. **No predigo dirección ni tamaño**: produzco una posición. Su **signo** lo corrijo con el
    régimen (ahí está el valor: accuracy $0.384\to0.436$); su **magnitud** la fijo por
    volatilidad (vol-targeting, gestión de riesgo).
11. **No bato al mercado; rescato al agente.** B&H gana (€1317) — pero no usa el agente. Mi
    tesis es M8 (€1069) > M5 (€903), probado con McNemar (p=0.069, borderline — lo digo yo).
""")

# ---------------------------------------------------------------------------
nb["cells"] = C
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logic_esential.ipynb")
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Escrito: {out}  ({len(C)} celdas)")
