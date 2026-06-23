# Canal régimen (price-only) en universo HELD-OUT — resultado [verificado]

> Exploratorio. NO canónico hasta validar (regla `trabajo-exploratorio-aislado`). Pre-registro:
> BITACORA 2026-06-23. Script: `experiments/regime_channel_heldout.py`. Output:
> `outputs/experiments/regime_channel_heldout.json`. 12 activos con precio pero SIN agente
> (AMD, ARKK, COIN, GME, INTC, META, NFLX, PLTR, PYPL, RIOT, SHOP, SNOW), nunca usados para
> construir la ley naturaleza→canal.

## Qué se preguntó
¿La ley *ventaja del régimen ∝ leverage* (panel-15: r=−0.55) **generaliza** fuera del panel, en un
universo de precio que no costó inferencia del agente? Y de paso: ¿cuánto **rota** el régimen-solo y
sobrevive al **coste** (hueco "producción")?

## Resultado honesto (no es un win nuevo, y no debería serlo)

**1. El universo held-out casi no tiene leverage fuerte.** Rango `leverage_corr` ∈ [−0.058, +0.083]
(solo ARKK roza −0.05). Son acciones individuales y proxies meme/cripto (GME, RIOT, COIN, SNOW, PLTR…),
justo los activos donde la teoría (Black 1976; Christie 1982) dice que el régimen **no** es direccional.
Los activos de leverage fuerte son estructuralmente los **ETFs de índice amplio** (SPY/QQQ/DIA/IWM/XLF/
XLK) — y esos **ya están en el panel**. No se puede fabricar el canal con single stocks.

**2. Donde el leverage es nulo, el régimen colapsa a B&H — y es inofensivo.** META, NFLX, SHOP:
ningún estado tiene media negativa → el régimen **nunca se pone corto** → `régimen ≡ B&H`, turnover=0.
El mecanismo del criterio de alcance se ve en crudo: sin leverage no hay señal de corto.

**3. Donde sí apuesta direccionalmente con leverage débil, pierde** (como predice la teoría):
INTC net −0.91, COIN −1.12, RIOT −1.34, PYPL −0.45 (acumulado OOS). Sharpe del grupo disperso ≈ 0.

**4. La ley conserva el signo pooled, pierde potencia.** Pooled con los 15 (n=27): `leverage_corr ↔
Sharpe régimen` r=−0.27 (p=0.18), **mismo signo** que el panel-15 (−0.55) pero no significativo. Añadir
12 puntos apelotonados en leverage≈0 no da potencia: la relación vive en el **rango (spread)** de
leverage, no en el número de activos. En el held-out solo, lev↔Sharpe es +0.10 (p=0.74, ruido: no hay
spread) y lev↔ventaja-acc −0.31 (p=0.32, signo correcto, no sig.). **No hay prior-flip con señal** — es
nulo, no inversión.

**5. Coste/turnover (hueco #2).** El régimen-solo **apenas siente el coste**: net acumulado a 1bp vs
10bp se mueve <1–3% (INTC −0.906→−0.937; GME +0.807→+0.776). El break-even mediano es alto o, en los
perdedores, 0bp (no hay edge ni gratis). **Implicación para producción:** la amenaza de coste **no** es
la capa de régimen/riesgo (rota por estado, persiste), sino el **aprendiz que voltea a diario** (M10/
AutoML). El análisis net-of-cost del notebook debe centrarse ahí.

## Lectura para el TFG
- **No** convierte esto en "STRATA funciona en más sitios". Lo honesto: **el canal es una propiedad del
  leverage de índice amplio**, empíricamente reconfirmada en 12 nombres frescos (generaliza la
  limitación de la decisión #16, ahora con evidencia fuera de muestra).
- **Refuerza por qué el panel son índices/ETFs amplios:** ahí vive la región de leverage fuerte.
- **Aporta al hueco de costes:** la capa de riesgo es cost-robusta; el foco net-of-cost es el ML diario.

## Qué NO hacer
- No meterlo como resultado positivo en el notebook canónico.
- No re-buscar accuracy vs ZeroR (no se buscaba; el contraste era relacional + riesgo/coste).
- No agrandar el panel con estos nombres: son la región débil, diluyen la ley.
