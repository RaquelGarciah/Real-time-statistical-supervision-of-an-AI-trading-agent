# Estado del arte / motivación — Supervisión de agentes de trading con IA

> **EXPLORATORIO — material de insumo para el redactor de la memoria.** No es texto canónico.
> Recopila el discurso profesional (LinkedIn) y la literatura académica detrás de él sobre
> *supervisión de agentes de trading basados en LLMs*, para usarse en **introducción** y/o
> **estado del arte**. El redactor toma lo que considere; cada afirmación va atribuida.
>
> **Norma de uso (importante).** LinkedIn se cita como **literatura gris** = evidencia de lo
> que la industria *piensa y demanda*, NUNCA como prueba de un hecho técnico. Todo hecho
> empírico/técnico se apoya en el **paper** correspondiente. Antes de pasar cualquier entrada
> a `tesis/bibliography.bib` debe verificarla `@experto-citas` (existencia real, no alucinada).

---

## 0. Cómo se obtuvo y límites

Barrido sobre contenido público de LinkedIn (`site:linkedin.com`) + apertura de los artículos
largos ("Pulse") accesibles + rastreo de los papers a los que esos posts remiten (arXiv/SSRN),
que es donde está la sustancia citable. LinkedIn no permite leer el cuerpo completo de los
posts sin sesión: cuando solo se dispone del extracto del buscador se marca como **[extracto]**.
Fecha de acceso: junio de 2026.

---

## 1. El marco profesional dominante: *"guardrails, but no handcuffs"*

**Idea (literatura gris).** El consenso 2025-2026 entre profesionales de finanzas no es
"agente autónomo sí/no", sino **autonomía acotada con humano en el lazo**: la IA detecta
señales, pero los comités de riesgo fijan los límites y el gestor decide ejecución, tamaño y
salida. Se recomienda monitorización continua, trazas de auditoría, *listas de usos
preaprobados* y *exit conditions* programadas que escalen a intervención humana. La consigna
que circula entre directivos de banca es literalmente *"guardrails but no handcuffs… always a
human on top and in the loop"*.

**Citas LinkedIn.**
- Gartner / discurso CFO sobre agentic AI en finanzas con checkpoints y human-in-the-loop. **[extracto]**
- *AI Agents vs. Deterministic Workflows* — C. Dkhil: las transacciones financieras exigen
  flujos deterministas; los agentes "con guardrails" se reservan a síntesis/recomendación. **[extracto]**

**Lectura para STRATA.** La industria pide supervisión continua y trazable, pero la resuelve
con **gobernanza humana**, no con un supervisor estadístico automático y falsable. Ese es el
hueco que ocupa STRATA. → §7.

---

## 2. Comportamiento de los agentes LLM de trading (núcleo de la motivación)

**Idea (hecho técnico, respaldado por paper).** Los LLM **no maximizan beneficio: ejecutan
instrucciones con fidelidad**, incluso cuando eso conduce a pérdidas. Responden a la dinámica
de mercado, pero como múltiples agentes actúan de forma **correlacionada**, pueden introducir
**riesgo sistémico** (burbujas, infrarreacción). El entorno de prueba es un libro de órdenes
persistente con clearing de equilibrio.

**Cita académica.** Lopez-Lira (2025), *Can Large Language Models Trade? Testing Financial
Theories with LLM Agents in Market Simulations* (arXiv:2504.10789; SSRN 5217340).

**Cita LinkedIn (puente divulgativo).** Nam Nguyen, *LLMs as Trading Agents*, glosando a
Lopez-Lira: "los LLM no buscan maximizar beneficio; priorizan ejecutar fielmente las
instrucciones" y su uso masivo "podría introducir nuevos riesgos sistémicos por actuar de modo
correlacionado".

**Lectura para STRATA.** Respalda la premisa de la hipótesis: un agente que sigue su "carácter"
rígidamente necesita un supervisor exógeno que lo corrija cuando cambia el régimen. Justifica el
sentido de RAM (coherencia con el régimen) y PSA (coherencia temporal de la opinión).

---

## 3. El objeto que STRATA supervisa: agentes multi-persona / debate

**Idea (hecho técnico + fenómeno de industria).** La frontera "sofisticar el agente" va por
sistemas multi-agente basados en roles/personas que debaten antes de decidir. Es exactamente la
familia a la que pertenece el AI Hedge Fund que STRATA envuelve (5 personalidades: Buffett,
Wood, Druckenmiller, Burry, Ackman).

**Citas académicas.**
- Xiao, Sun, Luo & Wang (2024), *TradingAgents: Multi-Agents LLM Financial Trading Framework*
  (arXiv:2412.20138). Roles especializados (analistas fundamentales/sentimiento/técnicos,
  investigadores bull/bear, **equipo de riesgo**, trader) que sintetizan en debate; reportan
  mejora en retorno acumulado, Sharpe y máximo drawdown. **Ya está en `bibliography.bib` como
  `xiao2024tradingagents`** (año 2024, no 2025 — preprint de dic-2024).
- Zhao, Lyu, Jones, Garber, Pasquali & Mehta (2025) [afiliación BlackRock] — *AlphaAgents: Large
  Language Model based Multi-Agents for Equity Portfolio Constructions* (arXiv:2508.11152).
  Agentes por rol que debaten sobre evidencia recuperada (RAG); salidas "auditable y grounded",
  condicionadas a la tolerancia al riesgo del inversor (risk-seeking / neutral / averse).
  *(No citar "BlackRock" como autor: es la afiliación.)*

> **⚠ Riesgo de redescubrimiento (blindaje obligatorio en la memoria).** El "risk team" de
> TradingAgents es un **agente LLM deliberativo interno** que razona sobre riesgo en lenguaje
> natural y emite sugerencias narrativas no vinculantes; **no usa ningún modelo estadístico
> formal** (ni HMM, ni BOCPD, ni GARCH), ni contraste pareado, ni hipótesis falsables. STRATA es
> lo contrario: **supervisor exógeno** cuya intervención no depende de la salida lingüística del
> agente, sino de detectores formales con criterios pre-registrados y contraste pareado (McNemar,
> DM), y cuya corrección es vinculante (`override`) o cuantificable (`reduce`). AlphaAgents ajusta
> a la *tolerancia al riesgo del inversor* (un parámetro de preferencia), no detecta coherencia
> interna. Frase lista para el redactor en §7.

**Citas LinkedIn (fenómeno viral / divulgación).**
- Matt Dancho, *"This guy built an AI Hedge Fund. With 6 AI Agents"* (sobre `virattt/ai-hedge-fund`). **[extracto]**
- Siraj Raval, *"I Built a Hedge Fund Run by AI Agents"*. **[extracto]**
- Bimsara Walallawita, post sobre `virattt/ai-hedge-fund`. **[extracto]**

**Lectura para STRATA.** El ecosistema celebra el agente-persona y publica resultados, pero
**ninguno audita estadísticamente si la decisión es coherente con régimen o volatilidad**.
Útil para situar STRATA como capa ortogonal de supervisión, no como otro agente.

---

## 4. Calibración y exceso de confianza del LLM (objeción anticipada del tribunal)

**Idea (hecho técnico, respaldado por papers).** Los LLM no solo se equivocan: lo hacen con
**más confianza que un humano**, y los modelos de razonamiento mejoran el acierto pero empeoran
en "saber cuándo están adivinando". Esto es relevante si STRATA usa (o decide no usar) las
probabilidades direccionales del agente.

**Citas académicas.**
- Sung, Fleisig, Hou, Upadhyay & Boyd-Graber (2025), *GRACE: A Granular Benchmark for Evaluating
  Model Calibration against Human Calibration* (arXiv:2502.19684). Los modelos son más
  sobreconfiados que los humanos en respuestas incorrectas e infraconfiados en las correctas.
- Damani, Puri, Slocum, Shenfeld, Choshen, Kim & Andreas (2025), *Beyond Binary Rewards:
  Training LMs to Reason About Their Uncertainty* (RLCR; arXiv:2507.16806). El RL ordinario
  degrada la calibración; añadir una recompensa tipo Brier la recupera sin perder accuracy.

**Citas LinkedIn (divulgación del problema).**
- *LLMs as Judges: Overconfidence and Calibration*. **[extracto]**
- *Apple Researchers: LLMs Overconfident Despite Uncertainty*. **[extracto]**

**Lectura para STRATA.** La objeción "¿están calibradas las probabilidades del LLM?" es real y
está documentada — no es un invento defensivo. Permite blindar la sección de supuestos: STRATA
no confía ciegamente en la confianza autoinformada del agente, sino que la contrasta contra
detectores estadísticos exógenos.

---

## 5. La pieza más cercana a STRATA en el discurso: *"statistical validation"* como guardrail

**Idea (literatura gris).** En los catálogos de *guardrails* para LLMs en finanzas aparece,
como uno más entre once, la *statistical validation*: que los outputs "se alineen con normas
estadísticas y modelos realistas para que las predicciones sigan ancladas a la realidad". Pero
queda en **eslogan**: no se especifica qué norma, qué test ni cómo se mide la corrección.

**Cita LinkedIn.** Alan Milligan, *Navigating the New Frontier: Effective Guardrail
Implementations for LLMs in Financial Services* (artículo Pulse, accesible completo). Lista 11
guardrails; el único del tipo que aborda STRATA es *statistical validation*. Los demás son
PII/compliance/topic-constraint/provenance.

**Lectura para STRATA.** Es la mejor frase de posicionamiento: STRATA **instancia ese eslogan**
en detectores falsables (HMM de régimen, BOCPD, GARCH-t) con contraste pareado (McNemar,
Diebold-Mariano). La distancia entre "alinear con normas estadísticas" y una implementación
falsable es, literalmente, la contribución del TFG.

---

## 6. Riesgo sistémico y regulación (para limitaciones, no para promesas)

**Idea (hecho técnico + discurso).** Si muchas instituciones despliegan arquitecturas
multi-agente LLM **similares**, las señales correlacionadas pueden **amplificar** la volatilidad
en lugar de amortiguarla; los marcos regulatorios (IMF, CFTC, EU AI Act) están calibrados para
IA de un solo agente y no capturan la coordinación emergente.

**Citas académicas / think-tank** (los tres IDs de arXiv confirmados por `@experto-citas`):
- Aldridge et al. (2026), *Agentic Artificial Intelligence in Finance: A Comprehensive Survey*
  (arXiv:2604.21672). *(Listas de coautores difieren entre fuentes: citar "Aldridge et al.")*
- Kurshan, Balch & Byrd (2025), *The Agentic Regulator: Risks for AI in Finance and a Proposed
  Agent-based Framework for Governance* (arXiv:2512.11933).
- Gong (2026), *AI Agents in Financial Markets: Architecture, Applications, and Systemic
  Implications* (arXiv:2603.13942; también en *FinTech* 5(2):34, doi:10.3390/fintech5020034).
- Atlantic Council, *Agentic AI opens the door to weaponizing financial systems* (John James &
  Alia Brahimi). **Opinión / think-tank, NO fuente empírica**: usar solo como discurso. El hecho
  de riesgo sistémico debe apoyarse en Aldridge et al. 2026, Kurshan et al. 2025 o Gong 2026.

**Lectura para STRATA.** El alcance de STRATA es supervisión **single-agent**; el riesgo
correlacionado multi-agente es otra escala. Reconocerlo explícitamente como limitación refuerza
la honestidad del trabajo (coherente con la regla `prior-flip` y la filosofía de rigor).

---

## 6.bis Literatura ausente a integrar (detectada por `@revisor-bibliografico`)

Referencias que **faltaban** y que el redactor debería incorporar al estado del arte. Las que ya
están en `bibliography.bib` se marcan; las nuevas requieren verificación de `@experto-citas`.

**Prioridad alta:**
- **Li, Z. et al. (2026), *Behavioral Consistency Validation for LLM Agents: An Analysis of
  Trading-Style Switching through Stock-Market Simulation* (arXiv:2602.07023).** El trabajo más
  cercano a la motivación del **PSA**: valida si los agentes LLM cambian de estilo de forma
  coherente con drivers de finanzas conductuales (aversión a pérdidas, herding) en simulación del
  S&P 500. Diferencia con STRATA: Li et al. validan *ex-post* por simulación; STRATA lo detecta en
  tiempo real y *actúa*. **Cita casi obligatoria** (un tribunal podría usarla para preguntar qué
  añade STRATA). ✓ VERIFICADA — ya en `bibliography.bib` como `li2026behavioral`.
- **Yan et al. (2025), *TradeTrap* (arXiv:2512.02261, ya en `.bib` como `yan2025tradetrap`).** Los
  agentes LLM de trading son manipulables y fallan sistemáticamente → refuerza la premisa de que
  hace falta supervisión exógena. Solo mencionarlo en el texto.
- **López de Prado (2018), meta-labeling (ya en `.bib` como `lopezdeprado2018`).** Antecedente
  arquitectónico de STRATA (modelo secundario que filtra al primario). Citarlo explícitamente
  ancla la credibilidad matemática. Ver frase en §7.

**Prioridad media:**
- **Nelson (1991), EGARCH** y **Glosten, Jagannathan & Runkle (1993), GJR-GARCH.** Si se motiva el
  GSO con el leverage effect, el tribunal preguntará por qué GARCH(1,1) simétrico y no asimétrico.
  Citarlas como alternativas consideradas y descartadas (parsimonia, estabilidad en ventanas
  largas, suficiencia para señal de sizing). ✓ VERIFICADAS — ya en `bibliography.bib` como
  `nelson1991` y `glosten1993`.
- **Guidolin & Timmermann (2007) (ya en `.bib`).** Referencia canónica de régimen/HMM en
  asignación de activos; citarla junto a Hamilton (1989) para anclar el **RAM** en literatura
  financiera, no solo de NLP.

**Prioridad baja:** `wang2025agentspec` (enforcement simbólico en runtime — diferenciarse:
STRATA impone inferencia estadística, no reglas simbólicas) y `flehmig2025reliability`
(reliability monitoring de sistemas agénticos). Ambas ya en `.bib`.

---

## 7. Cómo aterriza en STRATA (frases listas para el redactor)

Borradores de párrafo (atribuidos, sin presentar opinión de LinkedIn como hecho propio):

> *"Existe una demanda profesional explícita de supervisión continua y trazable de agentes de
> trading basados en LLMs [Milligan 2025; Gartner], pero el sector la articula como gobernanza
> humana —la consigna 'guardrails but no handcuffs'— y, a lo sumo, como el objetivo de 'alinear
> los outputs con normas estadísticas' [Milligan 2025], sin especificar qué norma, qué contraste
> ni cómo se mide la corrección. STRATA instancia ese objetivo en detectores estadísticos
> falsables con contraste pareado."*

> *"La literatura sobre agentes LLM de trading documenta que estos ejecutan instrucciones con
> fidelidad antes que maximizar beneficio [Lopez-Lira 2025] y que su confianza autoinformada
> está sistemáticamente mal calibrada [GRACE, Sung et al. 2025]. Ambos hechos motivan un
> supervisor exógeno que no dependa de la propia certeza del agente."*

> *"Los marcos multi-agente recientes [TradingAgents, Xiao et al. 2024; AlphaAgents, Zhao et al.
> 2025] enriquecen la deliberación del agente, pero no incorporan una auditoría estadística en
> tiempo de ejecución de la coherencia de cada decisión con el régimen y la volatilidad
> observados: ese es el espacio que ocupa STRATA."*

> **(Blindaje frente a la objeción del 'risk team').** *"A diferencia del risk team de
> TradingAgents [Xiao et al. 2024], que es un agente LLM deliberativo interno al sistema, STRATA
> es un supervisor estadístico exógeno cuya mecánica de intervención no depende de la salida
> lingüística del agente supervisado, sino de detectores formales (HMM de régimen, BOCPD, GARCH-t)
> con hipótesis pre-registradas y contraste pareado."*

> **(Ancla metodológica).** *"Arquitectónicamente, STRATA instancia la lógica del meta-labeling
> [López de Prado 2018]: un modelo secundario que filtra o atenúa las señales del modelo primario.
> La novedad es trasladar esa lógica del dominio de señales cuantitativas al de un agente LLM
> multi-persona, haciéndola falsable mediante contrastes estadísticos explícitos."*

---

## 8. Bibliografía

### 8.1 Literatura gris (LinkedIn) — APA-español

- Milligan, A. (2024). *Navigating the New Frontier: Effective Guardrail Implementations for
  LLMs in Financial Services* [Artículo]. LinkedIn. Recuperado en junio de 2026 de
  https://www.linkedin.com/pulse/navigating-new-frontier-effective-guardrail-llms-alan-milligan-vdqre
  *(Fecha de publicación real: feb-2024, confirmada.)*
- Nguyen, N. (2025). *LLMs as Trading Agents: A New Approach to Financial Markets* [Publicación].
  LinkedIn. https://www.linkedin.com/posts/namnguyento_trading-artificialintelligence-portfoliomanagement-activity-7354546060897251329-8nx_
- Ralph, G. (2025). *AI-Powered Risk Management: Transforming Hedge Funds* [Artículo]. LinkedIn.
  https://www.linkedin.com/pulse/ai-powered-risk-management-transforming-hedge-fund-george-ralph-citp-riuje
- Dkhil, C. (2025). *AI Agents vs. Deterministic Workflows* [Artículo]. LinkedIn.
  https://www.linkedin.com/pulse/ai-agents-vs-deterministic-workflows-chiheb-dkhil-vlsve
- Dancho, M. (2024). *This guy built an AI Hedge Fund. With 6 AI Agents* [Publicación]. LinkedIn.
  https://www.linkedin.com/posts/mattdancho_this-guy-built-an-ai-hedge-fund-with-6-activity-7272285382921064449-cGqT
- Raval, S. (2023). *I Built a Hedge Fund Run by AI Agents* [Publicación]. LinkedIn.
  https://www.linkedin.com/posts/sirajraval_i-built-a-hedge-fund-run-by-ai-agents-activity-7067252972128243714-PJNZ

### 8.2 Académicas / institucionales — APA-español

- Lopez-Lira, A. (2025). *Can Large Language Models Trade? Testing Financial Theories with LLM
  Agents in Market Simulations*. arXiv:2504.10789 (SSRN 5217340).
- Xiao, Y., Sun, E., Luo, D., & Wang, W. (2024). *TradingAgents: Multi-Agents LLM Financial
  Trading Framework*. arXiv:2412.20138. *(Ya en `bibliography.bib`: `xiao2024tradingagents`.)*
- Zhao, T., Lyu, J., Jones, S., Garber, H., Pasquali, S., & Mehta, D. (2025). *AlphaAgents: Large
  Language Model based Multi-Agents for Equity Portfolio Constructions*. arXiv:2508.11152. [BlackRock]
- Sung, Y. Y., Fleisig, E., Hou, Y., Upadhyay, I., & Boyd-Graber, J. L. (2025). *GRACE: A
  Granular Benchmark for Evaluating Model Calibration against Human Calibration*. arXiv:2502.19684.
- Damani, M., Puri, I., Slocum, S., Shenfeld, I., Choshen, L., Kim, Y., & Andreas, J. (2025).
  *Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty* (RLCR). arXiv:2507.16806.
- Aldridge, I., et al. (2026). *Agentic Artificial Intelligence in Finance: A Comprehensive
  Survey*. arXiv:2604.21672.
- Kurshan, E., Balch, T., & Byrd, D. (2025). *The Agentic Regulator: Risks for AI in Finance and
  a Proposed Agent-based Framework for Governance*. arXiv:2512.11933.
- Gong, H. (2026). *AI Agents in Financial Markets: Architecture, Applications, and Systemic
  Implications*. arXiv:2603.13942 (doi:10.3390/fintech5020034).
- Atlantic Council (2026). *Agentic AI opens the door to weaponizing financial systems* (J. James
  & A. Brahimi). https://www.atlanticcouncil.org/dispatches/agentic-ai-opens-the-door-to-weaponizing-financial-systems/

### 8.3 Bloque BibTeX (VERIFICADO por `@experto-citas`, jun-2026)

> `xiao2024tradingagents` ya existe en `bibliography.bib` — **no duplicar**, reusar esa clave.
> El resto son adiciones netas. Antes de pegar, confirmar que ninguna clave colisiona.

```bibtex
@misc{milligan2024guardrails,
  author       = {Milligan, Alan},
  title        = {Navigating the New Frontier: Effective Guardrail Implementations for {LLMs} in Financial Services},
  year         = {2024},
  howpublished = {LinkedIn},
  note         = {Recuperado en junio de 2026},
  url          = {https://www.linkedin.com/pulse/navigating-new-frontier-effective-guardrail-llms-alan-milligan-vdqre}
}

@misc{nguyen2025llmtraders,
  author       = {Nguyen, Nam},
  title        = {{LLMs} as Trading Agents: A New Approach to Financial Markets},
  year         = {2025},
  howpublished = {LinkedIn},
  note         = {Recuperado en junio de 2026},
  url          = {https://www.linkedin.com/posts/namnguyento_trading-artificialintelligence-portfoliomanagement-activity-7354546060897251329-8nx_}
}

@misc{dancho2024aihedgefund,
  author       = {Dancho, Matt},
  title        = {This guy built an {AI} Hedge Fund. With 6 {AI} Agents},
  year         = {2024},
  howpublished = {LinkedIn},
  note         = {Recuperado en junio de 2026},
  url          = {https://www.linkedin.com/posts/mattdancho_this-guy-built-an-ai-hedge-fund-with-6-activity-7272285382921064449-cGqT}
}

@article{lopezlira2025cantrade,
  author  = {Lopez-Lira, Alejandro},
  title   = {Can Large Language Models Trade? Testing Financial Theories with {LLM} Agents in Market Simulations},
  journal = {arXiv preprint arXiv:2504.10789},
  year    = {2025},
  note    = {SSRN 5217340}
}

% TradingAgents: usar la clave existente xiao2024tradingagents del .bib (año 2024). No re-crear.

@article{zhao2025alphaagents,
  author  = {Zhao, Tianjiao and Lyu, Jingrao and Jones, Stokes and Garber, Harrison and Pasquali, Stefano and Mehta, Dhagash},
  title   = {{AlphaAgents}: Large Language Model based Multi-Agents for Equity Portfolio Constructions},
  journal = {arXiv preprint arXiv:2508.11152},
  year    = {2025},
  note    = {BlackRock}
}

@article{sung2025grace,
  author  = {Sung, Yoo Yeon and Fleisig, Eve and Hou, Yu and Upadhyay, Ishan and Boyd-Graber, Jordan Lee},
  title   = {{GRACE}: A Granular Benchmark for Evaluating Model Calibration against Human Calibration},
  journal = {arXiv preprint arXiv:2502.19684},
  year    = {2025}
}

@article{damani2025rlcr,
  author  = {Damani, Mehul and Puri, Isha and Slocum, Stewart and Shenfeld, Idan and Choshen, Leshem and Kim, Yoon and Andreas, Jacob},
  title   = {Beyond Binary Rewards: Training {LMs} to Reason About Their Uncertainty},
  journal = {arXiv preprint arXiv:2507.16806},
  year    = {2025}
}

@article{aldridge2026agenticsurvey,
  author  = {Aldridge, Irene and others},
  title   = {Agentic Artificial Intelligence in Finance: A Comprehensive Survey},
  journal = {arXiv preprint arXiv:2604.21672},
  year    = {2026}
}

@article{kurshan2025agenticregulator,
  author  = {Kurshan, Eren and Balch, Tucker and Byrd, David},
  title   = {The Agentic Regulator: Risks for {AI} in Finance and a Proposed Agent-based Framework for Governance},
  journal = {arXiv preprint arXiv:2512.11933},
  year    = {2025}
}

@article{gong2026aiagents,
  author  = {Gong, Hui},
  title   = {{AI} Agents in Financial Markets: Architecture, Applications, and Systemic Implications},
  journal = {arXiv preprint arXiv:2603.13942},
  year    = {2026},
  note    = {Tambi\'en en FinTech 5(2):34, doi:10.3390/fintech5020034}
}
```

---

## 9. Estado de verificación y pendientes

**Verificado por `@experto-citas` (jun-2026):** 0 citas alucinadas, 0 IDs falsos. Correcciones
ya aplicadas a este documento: año de TradingAgents (2024), autores de AlphaAgents (Zhao et al.,
no "BlackRock"), autores de RLCR (lista completa), año de Milligan (2024), confirmados los tres
IDs de §6, matizada Atlantic Council como opinión.

**Revisado por `@revisor-bibliografico` (jun-2026):**
- Único solapamiento con `bibliography.bib`: TradingAgents → reusar `xiao2024tradingagents`, no
  duplicar (ya corregido en §8.3).
- Riesgo de redescubrimiento principal: el "risk team" de TradingAgents → blindaje añadido en §3
  y §7.
- Literatura ausente prioritaria añadida en §6.bis (Li et al. 2026, meta-labeling, EGARCH/GJR).

**Referencias NUEVAS de §6.bis ya verificadas y añadidas a `bibliography.bib`** (jun-2026):
`li2026behavioral`, `nelson1991`, `glosten1993`. Sin colisión de claves.

**Pendiente antes de pasar al `.bib` las de LinkedIn/agentes (§8.3):** insertar las entradas
verificadas de §8.3 cuando el redactor decida usarlas (las académicas de §6.bis ya están dentro).

**Recordatorio de uso:** los ítems **[extracto]** y toda cita de LinkedIn valen solo como
evidencia de *discurso de industria* (intro/estado del arte), nunca como afirmación técnica.
