# Glosario y notación canónica de la memoria

Terminología y símbolos **únicos** para toda la memoria. `redactor-tesis` y `narrativa-coherencia`
los respetan: si el marco teórico llama $\sigma_t$ a la volatilidad GARCH, el marco práctico no
puede llamarla $v_t$. Coherente con el código (`core/`, `strata/`) y con `preamble.tex`.

## Notación matemática

| Símbolo | LaTeX | Significado |
|---|---|---|
| $r_t^{\log}$ | `\reglog` | log-retorno del día $t$ |
| $r_{t+1}$ | `\rnext` | retorno del día siguiente (el que se predice/contabiliza) |
| $\sigma_t$ | `\sigmat` | volatilidad condicional (anualizada) del GARCH(1,1)-t |
| $\mu_k$ | `\muk` | media de retorno del régimen $k$ |
| $\gamma^{f}_t(k)$ | `\gammaf` | posterior **filtrado** del HMM: $P(z_t=k\mid x_{1:t})$ (causal) |
| $z_t$ | | estado oculto (régimen) en $t$ |
| $w_t$ | | posición / peso de la cartera en $t$, $w_t\in[-1,+1]$ |
| $\tau$ | | umbral (gate) de RAM; canónico $\tau=0.5$ |
| $\alpha,\beta,\omega$ | | parámetros GARCH ($\alpha+\beta<1$ estacionariedad) |

## Términos del proyecto (usar SIEMPRE igual)

- **STRATA** — *Statistical Trading Real-time Audit*; capa de supervisión estadística.
- **Régimen** — estado del HMM: **Calma**, **Estrés**, **Crisis** (3 estados, $K=3$).
- **RAM** (*Regime-Action Mismatch*) — detector de coherencia acción↔régimen (HMM).
- **PSA** (*Position Sizing Anomaly*) — detector de cambio anómalo de sizing del agente (BOCPD).
- **GSO** (*GARCH-bounded Sizing Override*) — detector de sobreexposición vs volatilidad (GARCH).
- **override-C** — variante de intervención que voltea la posición hacia la dirección del régimen.
- **gate $\tau$** — umbral por encima del cual RAM interviene.
- **leverage effect** — correlación negativa entre retorno y volatilidad en índices.
- **volatility targeting** — dimensionar la posición para una volatilidad objetivo.

## Nombres de las estrategias (no renombrar)

- **M5** — el agente LLM sin supervisar (la víctima).
- **M7** — *reduce* (encoge la apuesta, no cambia el signo).
- **M8** — STRATA en override-C (el rescate, la regla a mano).
- **M10** — meta-learner XGBoost con CPCV (la referencia de ML).
- **B&H** — *Buy & Hold* (mercado pasivo).

## Convenciones de escritura

- Decimales con coma en el texto en español (0,5), pero el código y los JSON usan punto.
- Cifras siempre desde `tesis/tables/*.tex` (autogeneradas) o JSON; nunca a mano.
- Citación numérica `[n]`.
