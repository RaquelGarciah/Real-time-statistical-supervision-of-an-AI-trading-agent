# STRATA — Kit semilla

Este es el **kit semilla** para arrancar el proyecto STRATA limpio desde cero, conservando lo que es irrecuperable o caro: caches de inferencias LLM ya pagadas, modelos calibrados, contexto del proyecto anterior y conocimiento sintetizado.

**No es una réplica del proyecto anterior.** Los experimentos, notebooks, figuras y pipeline se rediseñan en el nuevo proyecto con ayuda de agentes especializados (ver `AGENTES_SUGERIDOS.md`).

---

## Cómo bootstrapear el nuevo proyecto

```bash
# 1. Copia este kit como semilla
cp -R /Users/Raquel/Desktop/STRATA_kit/ /Users/Raquel/Desktop/STRATA/
cd /Users/Raquel/Desktop/STRATA/

# 2. Inicializa git y submódulo del agente
git init
git submodule add https://github.com/virattt/ai-hedge-fund.git agent/ai_hedge_fund

# 3. Entorno Python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Variables de entorno
cp .env.example .env
# editar .env con OPENROUTER_API_KEY, FINANCIAL_DATASETS_API_KEY, GOOGLE_API_KEY

# 5. Test smoke: 18 suites deben quedar verdes desde el primer commit
pytest tests/ -v

# 6. Configurar agentes (ver AGENTES_SUGERIDOS.md)
mkdir -p .claude/agents/
# crear los ficheros descritos en AGENTES_SUGERIDOS.md
```

---

## Orden de lectura recomendado

| # | Fichero | Propósito | Tiempo |
|---|---|---|---|
| 1 | `CLAUDE.md` | Constitución del nuevo proyecto: qué es, qué no es, cómo trabajar con agentes | 10 min |
| 2 | `CONOCIMIENTO_ACUMULADO.md` | Síntesis del proyecto anterior. Lectura obligatoria antes de cualquier experimento | 10 min |
| 3 | `DECISIONES_ESENCIALES.md` | Las 12 decisiones vivas a 2026-06-07 (las que sobreviven al pivot final) | 5 min |
| 4 | `LECCIONES_APRENDIDAS.md` | Errores cometidos y cómo NO repetirlos | 10 min |
| 5 | `ENTREGABLES.md` | Qué exige el tutor para la defensa | 5 min |
| 6 | `RESULTADOS_OBJETIVO.md` | Cifras actuales como referencia a replicar/superar | 5 min |
| 7 | `AGENTES_SUGERIDOS.md` | Arquitectura de agentes que ayudan a hacer cada cosa bien | 15 min |

Total: ~60 minutos para tener el contexto completo antes de tocar código.

---

## Qué está incluido

| Carpeta | Tamaño | Por qué se incluye |
|---|---:|---|
| `core/` | <1M | Primitivas matemáticas testadas (HMM, GARCH, BOCPD, CPCV, stats, metrics, backtest). Reescribirlas no añade rigor |
| `strata/` | <1M | Detectores RAM/PSA/GSO + capa de intervención |
| `agent/` | <1M | Wrapper + cliente LLM + patches macro/precio/stats. **Sin** el submódulo `ai_hedge_fund/` |
| `tests/` | <1M | 18 suites verdes |
| `cache/agent/` | 16M | Decisiones del Portfolio Manager por activo, 10 tickers, 2024-10-01 → 2026-05-19 |
| `cache/models/` | 44K | HMM/GARCH/BOCPD pickles, thresholds STRATA, calibración |
| `cache/llm/` | 156M | Inferencias LLM individuales por personalidad (no se versiona en git) |
| `data/` | 170M | Parquets de yfinance (SPY + panel + factores) |
| `_archivo_proyecto_anterior/` | 1.4M | BITACORA + decisiones + chats + transcripciones + outputs canónicos como referencia |

**Total kit:** ~345 MB.

---

## Qué NO está incluido (y por qué)

- `experiments/` — los rediseña `@disenador-experimentos` desde cero con rigor.
- `notebooks/` — el nuevo notebook canónico lo hace el agente, math-first.
- `viz/` — los gráficos se regeneran junto con los experimentos.
- `pipeline.py` — el nuevo pipeline se diseña con la nueva estructura.
- `live/` — el modo live se rehace con la nueva arquitectura.
- `outputs/figures/` — regenerable.
- `.git/` — el nuevo proyecto inicializa su propio repo.
- Submódulo `ai_hedge_fund/` — se re-añade con `git submodule add`.

---

## Verificación rápida del kit

```bash
du -sh .
ls cache/agent/ | wc -l                                 # esperado: 10
ls cache/agent/SPY/ | wc -l                             # esperado: ~401
ls cache/models/                                        # HMM, GARCH, BOCPD, thresholds
find core/ strata/ agent/ tests/ -name "*.py" | wc -l   # esperado: ~50
ls *.md                                                 # 8 MDs raíz
ls _archivo_proyecto_anterior/                          # BITACORA + docs + outputs_canonicos
```

---

## Workflow del nuevo proyecto (resumen)

1. Tienes una pregunta de investigación nueva.
2. Pregunta a `@asesor-historico`: "¿se intentó algo parecido?". Recibirás cita de BITACORA + decisión tomada.
3. Pide a `@disenador-experimentos` el pre-registro del experimento (metodología, criterios de éxito, citas).
4. `@rigor-matematico` audita el diseño antes de ejecutar.
5. `@ejecutor-experimentos` lo corre y guarda outputs.
6. `@rigor-matematico` audita los resultados.
7. `@bitacora` decide si entra a BITACORA.
8. `@narrativa-coherencia` propaga al notebook y a las decisiones.
9. `@defensa-tutor` anticipa las objeciones del tribunal.

Nunca se ejecuta un experimento sin pasar por los pasos 1–4. Eso es lo que en el proyecto anterior no se hizo y costó 4 notebooks + 12 scripts de tuning huérfanos.
