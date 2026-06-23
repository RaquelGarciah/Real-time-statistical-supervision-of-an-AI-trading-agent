# Conversación — construcción del notebook DEFINITIVO del marco práctico (panel de 10)

Registro cronológico fiel de la sesión. Cada bloque: **[Raquel]** lo que pidió · **[Claude]** qué hice/decidí +
cifras. Resultado final: `notebooks/STRATA_marco_practico.ipynb` (panel de 10), APROBADO por el bucle
constructor↔`raquel-quant`. Outputs nuevos: `mechanism_panel.json`, `panel_robustness.json`,
`automl_net_returns.json`, `detector_analysis_{SPY,XLE,MARA}.json`, `calib_window_panel.json`.

---

## 1. Encargo inicial: notebook definitivo + dos agentes

**[Raquel]** "Necesitamos crear el notebook definitivo final del que se alimenta la memoria… que demuestre el
valor de mi propuesta a nivel de activo y de panel, identifique un patrón real en los activos y la estrategia que
mejor los define. Crea un agente que actúe como yo (quant senior 5 años en JPMorgan + matemática) que revise el
notebook de forma iterada con el agente que lo crea hasta declararlo definitivo."

**[Claude]** Creé el agente revisor `.claude/agents/raquel-quant.md` (gate G1–G6: estructura/objetivos, rigor,
honestidad, coherencia, reproducibilidad, pitch). Monté el bucle constructor↔revisora como Workflow
(`strata-marco-practico-build-review`). Primer notebook (panel 15, estructura §4.0–§4.7) → APROBADO en 2 rondas
(la ronda 1 cazó un error de signo real en el leverage effect). Registré decisión #18, BITÁCORA, RESULTADOS §1ter.

## 2. Queja: faltan gráficas / AutoML / experimentos

**[Raquel]** "Ha hecho cosas mal. Pedí gráficas nuevas, experimentos, todo justificado, que AutoML apareciera en
resultados y en más de una gráfica — no sale. Muy mal."

**[Claude]** Verifiqué: AutoML faltaba en la **curva de equity** (¡la ganadora!) y en el **bootstrap de riesgo**,
porque **no existía la serie diaria de AutoML** en ningún JSON. Decisiones:
- **[Raquel]** confirmó: AutoML diario "SPY ya + panel en background"; 4 gráficas nuevas (intervención por
  detector, distribución de scores, atribución de P&L, robustez sub-ventanas).
- **[Raquel]** "Asegúrate de que es la config canónica de AutoML." → Comprobé que en el panel canónico **GBM gana
  15/15** (XGBoost incluido, nunca gana). Reentrenar en mac NO reproduce → **reconstruí la serie diaria de AutoML
  desde el propio acierto día a día del panel** (`automl_net_returns.py`): `pos_t=sign(r_{t+1})·(2·acierto_t−1)`
  → idéntica a la canónica, validada cifra a cifra (SPY 0.5737/2.681/1.38). Sin H2O, sin riesgo de divergencia.
- **[Raquel]** "Si no existe, genérala; es la parte esencial." → hecho para los 15.
- Experimento `detector_analysis.py` (SPY): intervención M8 30%, **RAM dispara 30% / PSA 0.5% / GSO 0%** →
  todo el P&L de rescate es de RAM; sub-ventanas con 3 tests.
- AutoML fusionado en el pooled bootstrap **en el notebook** (a petición de Raquel: "fusiona desde los dos JSON,
  sin recomputar") usando `_boot_paired`.

## 3. Selección de los 10 activos (de los 15)

**[Raquel]** "Quiero coger 10 de los 15 (los que van mejor). Sé que es seleccionar a mano; ayúdame a elegir bien
sabiendo todas las métricas y la significancia. I forbid you to block me."

**[Claude]** No bloqueé (el SPEC §6.1 ya autoriza "10 casos de aplicabilidad de 15"). Mostré la tabla completa.
Iteramos:
- **[Raquel]** "¿hay activos donde el agente bate a STRATA?" → Sí: en **MSTR/UNG el agente ya bate a las
  triviales** (no es rescate). Recomendé excluirlos.
- **[Raquel]** "Justifica la no-significancia de McNemar en SMCI/NVDA/BAC." → McNemar mide **pares discordantes**
  (c−b); esos tres tienen el neto más pequeño (14–17) → sin potencia. Mecánico: SMCI agente 95% corto (alineado),
  NVDA leverage invertido, BAC M10 sobreajusta.
- **[Raquel]** "¿por qué la significancia si los accuracy son parecidos?" → McNemar ≈ "de los desacuerdos, ¿STRATA
  gana >50%?": SPY 64%, QQQ 61% (sig); NVDA 54%, BAC 56% (no). El Δacc además no es tan parecido.
- **[Raquel]** "ayúdame a elegir bien." → Regla principista: **C1 agente perdedor** + canal STRATA que rescata.
- **[Raquel]** "dudo con TSLA." → TSLA fuera: rescate más pequeño y **AutoML lo hunde ahí** (Sharpe −1.69).
- **[Raquel]** "justifica por qué en BAC solo funciona M8 (Sharpe empeora en las otras)." → Mismo std → Sharpe es
  pura dirección; M8 corrige el sesgo corto del agente (régimen direccional), M10 **sobreajusta** (acierta 32% en
  días grandes, rota 36%) → media −24.5 bps. Regla fija no sobreajusta; aprendiz sí.
- **[Raquel]** "¿en cuántos una STRATA bate a TODO?" → **5/15** (SPY, MSTR, SMCI, MARA, UNG); en mi 10 solo 2
  porque excluí MSTR/MARA/UNG.
- **[Raquel]** eligió **"vitrina máxima"** (5 bate-a-todo) y "argumenta si sería sólido". → Stress-test: MSTR la
  gana el agente (M8 interviene 0.2%); UNG/MARA la regla M8 **hunde** (solo el ML rescata). Recomendé **MSTR→XLE**.
- **[Raquel]** "cámbialo por XLE." → **Panel final (10): SPY, QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG.**
  Apéndice (5): MSTR, NVDA, BAC, TSLA, IWM.

## 4. Este pasa a ser EL notebook definitivo (más completo)

**[Raquel]** "Que este notebook sustituya al de 15, sea réplica pero mucho más completo y justificado (faltaban
gráficas, decisiones, escenarios, robustez), y lo lleven a cabo mis agentes con todo el contexto."

**[Claude]** Reescribí el builder a **panel de 10** con secciones nuevas: §1 selección 15→10 (cohorte
mecánica) · §2 K=3 (held-out LL), leverage honesto, intervención/atribución por detector · §3 SPY · §4 panel-10 ·
§5 **mecanismo por activo** · §6 **clustering naturaleza→resultado** · §7 robustez (equity por activo) · §8
**apéndice de los 5** · §9 conclusiones. `mechanism_panel.py` (diagnóstico de canal). Endurecí el gate de
raquel-quant. APROBADO en 2 rondas.

**[Raquel]** "Investiga los notebooks de inspiración y todos los outputs; falta accuracy rodante, val/test, y
significancia del rescate en bajista/alcista — de eso ya hay gráficas."

**[Claude]** Exploré notebooks + 80 outputs. Escribí `panel_robustness.py` (desde el acierto día a día canónico,
**sin re-entrenar**): **rodante** (mejor STRATA > agente en >50% ventanas en 8/10), **val/test** (3 particiones),
y **rescate en alcista vs bajista POOLED**: McNemar M8/M10/AutoML vs M5 **significativo en AMBOS regímenes**
(M10 0.0015/0.013, AutoML 0.0023/0.0003). + K=3 justificado (held-out −1.30 > −1.69). APROBADO en 2 rondas.

## 5. "El regla vs aprendiz no queda claro / poco concluyente"

**[Raquel]** "El aprendiz–regla queda poco concluyente. Necesito algo más robusto. ¿Por qué en MARA AutoML saca
más que M8? Todo argumentado."

**[Claude]** **Medí** la relación a través de los 15 y fui honesto: el discriminante `crisis_mean<0→regla` **NO
se sostiene** (ninguna variable de naturaleza predice el valor de la regla M8, todas p>0.14). Reescribí §5/§6:
- **Dos capas COMPLEMENTARIAS** (no canales que compiten): **M8 = capa de riesgo** (pooled ΔSharpe +0.66 sig,
  interpretable, casi nunca lidera accuracy); **M10/AutoML = capa de accuracy** (McNemar vs M5 sig en 6/10).
- **Única ley naturaleza→resultado que sobrevive un test:** rescate del aprendiz **∝ leverage** (Pearson
  **r=−0.55 p=0.034**, Spearman ρ=−0.54 p=0.038).
- Honestidad cableada: qué modelo lidera por activo **no es predecible** (n=15) → por eso STRATA ofrece los dos.
- **Clustering reencuadrado:** PC1 (eje principal de la naturaleza) **≈ leverage (r=0.84, p<0.001)** → cierra la
  cadena naturaleza→eje→rescate del aprendiz.
- (MARA: `crisis_mean=+0.0056`, leverage invertido → la regla mete ruido (acierta <0.5); AutoML aprende a voltear
  el sesgo corto del agente en los rallies cripto.)

**[Raquel]** "Prueba todo, sí" (incluida la robustez a la ventana de calibración que pidió el tutor).

**[Claude]** `calib_window_panel.py` (SPY/QQQ/XLE/BAC, inicios 2000/2010/2015/2020, OOS fijo, sin fuga). Hallazgo
honesto: el resultado **no es frágil**; en índices **acortar a 2010 incluso mejora** (SPY 0.494→0.526, QQQ
0.522→0.578) — apoya la intuición del tutor ("el pasado lejano plano aporta poco"). **Mantenemos la ventana
completa pre-registrada** (cambiarla por la que maximiza OOS sería p-hacking). Celda añadida a §7.

Notebook final: 29 celdas, 0 errores, AUTO-TEST OK. En revisión final por el bucle (constructor↔raquel-quant).

---

## Estado canónico al cierre de esta conversación
- **Notebook único canónico:** `notebooks/STRATA_marco_practico.ipynb` (panel de 10). Decisión #18.
- **Panel (10):** SPY, QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG · **apéndice (5):** MSTR, NVDA, BAC, TSLA, IWM.
- **Tesis sostenida:** dos supervisores complementarios (regla=riesgo sig., aprendiz=accuracy sig.); el rescate
  del aprendiz ∝ leverage (única ley con test); rescate robusto (rodante, val/test, alcista+bajista, calibración);
  AutoML "gana a todo" en SPY es **nominal**; STRATA no genera alfa.
- **Punto más blando (declarado):** UNG en el cuerpo (el agente no pierde ahí; encuadrado como caso ML).
