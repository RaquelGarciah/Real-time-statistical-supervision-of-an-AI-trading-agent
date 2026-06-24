# Q&A de defensa — TFG STRATA

> Preparado para el tribunal y para el tutor **Dani** (especialista en series temporales).
> Ordenado de **mayor a menor probabilidad**, en tres niveles. Cada pregunta lleva: **(P)** por qué es probable,
> **(R)** respuesta defendible, y **cifra→fuente** (los JSON canónicos del notebook y `MARCO_PRACTICO_CONTEXTO.md` §IX).
>
> **Nota de honestidad:** algunas cifras de los *back-up* (p. ej. SPY α+β≈0,997, o JSON como `net_of_cost_panel.json`,
> `panel_robustness.json`, `alfa_beta_lectura.json`) provienen de chats del proyecto anterior y conviene
> re-verificarlas contra el notebook actual antes de citarlas en sala. Las del cuerpo principal sí están ancladas.

---

## Las 5 preguntas MÁS probables (preparar estas como sea)

1. **"Pierdes contra 'siempre uno'. ¿Para qué sirve si no bate a la trivial?"** → A1
2. **"¿Cómo validas que tu método es mejor? La curva por encima no demuestra nada."** (literal de Dani) → A2
3. **"Un XGBoost con todas las probabilidades juntas siempre batirá tu regla a mano. ¿Por qué tu capa?"** (literal) → A3
4. **"Los umbrales los has elegido tú. ¿Por qué 0,50 y no 0,49? Eso es olfato, no rigor."** (literal) → A4 / B3
5. **"Lánzalo en distintos periodos y activos. ¿No tuviste suerte con la ventana?"** (literal) → A5

**Frase-cierre que sintetiza todo:** *"El agente pierde (0,366, sign test p<0,001); STRATA lo rescata con
significancia en accuracy (McNemar vs M5 p=0,0002) y en riesgo (pooled-10 ΔSharpe excluye 0); el aprendiz
redescubre nuestras señales (SHAP 0,66, 10/10); no batimos al mercado y no era el objetivo."*

**Tono para Dani:** no maneja el Sharpe; entiende **la curva de equity y la matriz de confusión**. Liderar con
esas dos; el Sharpe siempre con su test detrás (DM, DSR). Cada elección (umbral, K, override) anclada en criterio
ex-ante o literatura, nunca en "salía mejor".

---

## Nivel ALTA probabilidad

### A1. El listón de la trivial (ataque nº1 de Dani)
**(P)** Obsesión declarada del tutor; es el sesgo que el propio TFG denuncia.
**(R)**
- Reconocer: no batimos a la clase mayoritaria con significancia. SPY AutoML 0,574 vs ZeroR/B&H 0,566, pero **McNemar AutoML vs ZeroR p=0,90** → con ~250 días no hay potencia. Lo decimos nosotros primero.
- Reencuadre clave: **el listón no es la trivial, es el agente.** La hipótesis es rescatar a un agente que pierde y acierta <0,5. Eso sí se prueba: agente 0,366 (sign test vs 0,5 **p<0,001**); McNemar mejor derivada **vs M5 p=0,0002** (AutoML), 0,0074 (M10), 0,051 (M8).
- Cierre: batir a B&H nunca fue el objetivo. La victoria nominal 4/10 es valor en punto, línea futura.

cifra→fuente: `automl_runs/panel_mm25_..._seed42.json` (McNemar), s1_spy (sign test).
**BACK-UP:** la nominalidad es de **potencia, no ausencia de señal**: en SMCI/MARA/UNG la derivada saca Sharpe positivo donde B&H pierde → compatible con valor direccional.

### A2. "¿Cómo validas que es mejor? La curva por encima no demuestra nada"
**(P)** Textual: *"Sharpe positivo y curva mejor no demuestra nada… parece que vendes humo"*.
**(R)**
- Reconocer: una curva por encima no es prueba; por eso no validamos con la curva.
- Validamos con **tests pareados pre-registrados**: McNemar (vs M5), bootstrap estacionario del ΔSharpe (Politis-Romano, bloque √n), TOST, DiD. Cada cifra con su test e IC al lado.
- En su idioma: matriz de confusión SPY (**71 aciertos / 50 fallos** de la regla; el agente al revés) + curva de equity de las 6 estrategias. El Sharpe con su test (DM, DSR), no suelto.

cifra→fuente: `spy_intervention_anatomy.json` (71/50, P&L +0,312); `bullbear_confirmatory.json` (POOLED10).
**BACK-UP:** pre-registro en BITACORA (H0, estadístico, criterio de éxito y fracaso) antes de mirar resultados. La regla `prior-flip` y la falsación SPY-bajista (ΔSharpe −1,49) documentan cuándo NO funciona.

### A3. "Un XGBoost con todo siempre batirá tu regla a mano"
**(P)** Objeción literal y reiterada de Dani; es el corazón del nivel 3.
**(R)**
- Reconocer: técnicamente tiene razón, y **hicimos justo ese experimento** (M10 ensemble + AutoML-H2O). Usamos el aprendiz como juez, no discutimos contra él.
- El aprendiz **se apoya en STRATA**: cuota SHAP de los detectores **>0,5 en 10/10, media 0,66**; redescubre `ram_score`, `psa_score`, `garch_sigma` como top. No inventa señal nueva.
- Al **quitar** los detectores, **degrada** (AutoML 0,574→0,550 sin PSA+GSO). La señal informativa es la de STRATA.
- Matiz (TOST): el aprendiz **bate modestamente a la regla en accuracy** (+0,021 M10, +0,034 AutoML, IC90>0) por interacciones no lineales; en riesgo son indistinguibles.

cifra→fuente: `decision_automl_prep.json` (SHAP 0,66); `automl_ablation_detectors.json` (0,574→0,550); `equivalence_tost.json`.
**BACK-UP:** interpretabilidad y desplegabilidad. El umbral aprendido por XGBoost no es estable en el tiempo; los de STRATA son fijos. El leader de AutoML cambia en cada reentreno; la regla opera desde el día 1.

### A4. El nivel 3: el aprendiz BATE a M8 en accuracy, ¿no contradice la hipótesis?
**(P)** El pre-registro pedía que M10 **no** batiera a M8; el TOST lo refuta.
**(R)**
- Reconocer: la cláusula de **equivalencia** se refuta en su letra; el aprendiz supera a la regla en accuracy.
- Separar las dos cosas del enunciado: (a) **atribución** —el valor viene de STRATA, no de la fuerza bruta— es la **sustantiva**, y se **confirma** (SHAP >0,5 en 10/10 + ablación degrada); (b) **equivalencia** era la prueba indirecta, y se refuta a favor del aprendiz.
- La hipótesis **se sostiene** porque la sostiene la atribución. Que el aprendiz afine la regla un punto montándose sobre las mismas señales **refuerza** la tesis.

cifra→fuente: `decision_automl_prep.json` (SHAP 0,66); `equivalence_tost.json`.
**BACK-UP:** el criterio de fracaso pre-registrado del nivel 3 era "SHAP<0,5 o ablación que no degrade". **Ninguno ocurre** → pasa por su criterio propio.

### A5. "Lánzalo en distintos periodos. ¿No tuviste suerte con la ventana?"
**(P)** Textual de Dani.
**(R)**
- Reconocer: limitación real, **una sola ventana OOS** y **un solo agente**.
- Lo que sí hicimos: (a) **walk-forward de origen rodante** con reentreno cada 21 días (~250 orígenes); (b) robustez por **sub-ventana y partición** (60/40, 70/30, 80/20, burn-in 150); (c) robustez a la **ventana de calibración**; (d) por **régimen** (bull/bear): aguanta en los dos.
- El panel de **10 activos** es la otra forma de "distintos lanzamientos": transversal (mejora al agente en 10/10).

cifra→fuente: `bullbear_confirmatory.json` (régimen, Holm); robustez → notebook.
**BACK-UP:** reportar la **falsación SPY-bajista** (ΔSharpe −1,49) demuestra que no escondemos periodos donde falla.

### A6. La grieta de Bonferroni de M8 (cota −0,047)
**(R)**
- Reconocer: bajo Bonferroni (m=3), **M8 no sobrevive**: cota inferior **−0,047**, justo por debajo de 0. El ΔSharpe simple +0,60 [0,05, 1,22] excluye 0 pero no resiste el ajuste.
- Reencuadre: **no tumba la tesis** porque los aprendices **sí pasan** (M10 +1,12 cota +0,258; AutoML +1,08 cota +0,261). El rescate de riesgo significativo lo sostienen ellos; M8 lo sostiene la accuracy y la interpretabilidad.

cifra→fuente: `bullbear_confirmatory.json` (POOLED10).
**BACK-UP:** el valor de M8 es el **mecanismo interpretable** (100% del P&L de rescate de SPY del canal RAM, día 1), no la mayor potencia.

### A7. El pooled-10: ¿es lícito apilar? n efectivo, IC optimista
**(R)**
- Reconocer: apilar trata los días como independientes y no lo son (SPY-QQQ >0,9); el **n efectivo < nominal (2493) y el IC queda algo optimista**. Lo declaramos.
- Lo controlamos: **bootstrap estacionario de bloque medio √n** (Politis-Romano), bloques contiguos, no iid.
- Aun así, **M10 y AutoML excluyen 0 con holgura** (Bonferroni +0,258/+0,261). Solo M8 vive en el filo → consistente.

cifra→fuente: `bullbear_confirmatory.json`; caveat corr cruzada en s2_panel/s4_limites.
**BACK-UP:** el rescate **no vive solo en el pooled**: también por régimen (Holm) y en accuracy por activo (10/10).

### A8. Por qué SPY como caso central (leverage effect, proxy direccional)
**(R)**
- El régimen **no predice** el signo de mañana; **coincide** con la caída el mismo día. En SPY el leverage effect (Black 1976; Christie 1982) hace que vol alta ↔ caídas → el régimen lleva contenido de riesgo direccional. Frac. al alza del día siguiente ≈0,5 incluso en Crisis: el valor es **disciplinar el riesgo, no anticipar**.
- Por eso SPY es el mejor escaparate, y la asunción **no se traslada** a stocks de leverage débil (documentado, CLAUDE.md §3) — eso lo mide la ley del leverage.

cifra→fuente: `regime_direction_table.json`.
**BACK-UP:** SPY central es **ex-ante por naturaleza** (índice = leverage fuerte), no por significancia; el panel cubre el espectro a propósito.

### A9. Look-ahead / contaminación temporal del LLM
**(P)** Línea roja técnica de Dani.
**(R)**
- Causalidad estricta: **`signal_lag=1`**, posición de t × retorno de t+1, nunca el del propio día.
- Walk-forward: **embargo=1** (horizonte 1, rolling-origin; Tashman 2000, LdP 2018), purge, sin KFold (única excepción M3). Test `test_no_leakage.py` en CI.
- LLM: OOS arranca **2024-10-01**, posterior al corte de DeepSeek V3 → no pudo memorizar. Calibración (2000→2024-09) y OOS nunca se solapan.

cifra→fuente: s1_spy §protocolo; 03_marco_teorico §hmm-filtrado.
**BACK-UP:** STRATA usa el **posterior filtrado** (solo pasado), nunca el suavizado (que vería el futuro); el suavizado solo en calibración offline.

### A10. El HMM, ¿usa el suavizado y mete look-ahead?
**(R)** En **explotación** solo el **posterior filtrado** γ_t = P(z_t | x_{1:t}), función del pasado (Corolario de causalidad). En **calibración** (Baum-Welch, offline, una vez) sí se usa el suavizado, legítimo porque no decide en tiempo real. Viterbi solo para la lectura interpretativa en calibración.
cifra→fuente: 03_marco_teorico §hmm-filtrado, §hmm-em.

---

## Nivel MEDIA probabilidad

### M1. La ley del leverage con n=10 (p=0,093)
**(R)** r=−0,56 (p=0,093), ρ=−0,59 (p=0,074): **tendencia al α=0,10** pre-registrado, no p<0,05 ni LOO. Con 10 puntos la correlación es tan fuerte como en un universo mayor; lo que pierde es potencia. 9/10 la siguen (ROKU excepción). La naturaleza ordena el rescate del **aprendiz**, no el de la regla (p>0,14).
cifra→fuente: `leverage_law_panel10.json`.
**BACK-UP:** el clustering recupera el mismo eje sin etiquetas de rescate (Rand=1,0; PC1≈leverage r=0,84) → evidencia convergente.

### M2. ¿Qué añade el clustering? Exploratorio con n=10
**(R)** Exploratorio, no confirmatorio. Aporta: **converge con la ley del leverage sin ver una métrica de rescate.** Cuatro métodos coinciden (Rand=1,0, silhouette 0,55); PC1 de la naturaleza recupera el leverage (r=0,84). Cierra la cadena naturaleza→eje→rescate por vía independiente. El cluster predice el **tipo** de canal, no el modelo exacto.
cifra→fuente: `cluster_panel10.json`, `mechanism_panel.json`.

### M3. SHAP sobre árbol vs ensemble; cuota STRATA
**(R)** Cada cuota etiquetada con su modelo y ventana: ensemble M10 (n=251) **0,71**; leader AutoML (n=401) **0,565 árbol / 0,564 permutación**. TreeSHAP nativo, eficiencia comprobada (|ΣSHAP+base−logit(p)|<1e-6). Se acompaña de permutation importance y ablación (vía no-SHAP) → tres métodos apuntan igual.
cifra→fuente: `decision_automl_prep.json`.

### M4. Reproducibilidad de AutoML (leader cambiante)
**(R)** Cambia de leader en cada reentreno → lo blindamos: **`max_models=25` (determinista), nunca por tiempo** (`max_runtime_secs` dio 3 leaders con la misma semilla → corregido), semilla 42, DeepLearning excluido, Purged K-fold emb=1, auto-test que cruza cada cifra con su JSON. La pieza estable de producción es **M8**; AutoML es el techo de accuracy, no indispensable.
cifra→fuente: MARCO §III.10; auto-memory `automl-h2o-reproducibilidad`.

### M5. Por qué GARCH(1,1) simétrico y no EGARCH/GJR
**(P)** Dani dio clases de GARCH; preguntará por la asimetría.
**(R)** El leverage effect lo capturamos en el **régimen del HMM** y en `leverage_corr` (Black), no en la ecuación de varianza. El GARCH solo estima **σ_t** para GSO (sizing); para eso GARCH(1,1)-t es el estándar (Bollerslev 1986) con mínimos parámetros y la t por curtosis. EGARCH/GJR afinaría el nivel de σ_t pero no cambia un detector de **banda**. Además GSO apenas dispara, así que la asimetría no es el cuello de botella.
cifra→fuente: 03_marco_teorico §garch; `spy_intervention_anatomy.json` (GSO inerte).
**BACK-UP:** α+β<1 (estacionariedad) se verifica; modelo bien especificado para su función.

### M6. Por qué K=3 estados
**(R)** Criterio doble ex-ante: held-out por observación **−1,30 (K=3) > −1,69 (K=2)**, **BIC 18.775 < 24.131**; e interpretabilidad (Calma/Estrés/Crisis). El held-out es sobre un tramo posterior, no premia el sobreajuste.
cifra→fuente: `k_selection.json`.

### M7. El alfa: ¿4/10 en Sharpe es señal o azar?
**(R)** No es titular por **disciplina anti-data-snooping**: la ganadora cambia por activo (a posteriori) y son ~250 días → el sesgo que el TFG denuncia. El **DSR** castiga el número de pruebas. Distinción honesta: en **SPY** el Sharpe es **beta** (OOS alcista); en **SMCI/MARA/UNG** (leverage débil/invertido) el pasivo pierde y la derivada saca Sharpe positivo yendo defensiva → compatible con **valor direccional**, nominal y a posteriori. Por eso línea futura con contraste pre-registrado.
cifra→fuente: `automl_net_returns.json`, `bullbear_confirmatory.json`.
**BACK-UP:** que SPY sea beta y SMCI/MARA/UNG parezcan alfa encaja con la ley del leverage → no es ad-hoc.

### M8. Generalización: un solo agente, una sola ventana
**(R)** Reconocer sin rodeos. Mitiga: panel de 10, walk-forward ~250 orígenes, robustez por régimen y calibración. La aportación central es **metodológica** (protocolo interpretable, pre-registrado, causalidad estricta), trasladable aunque las cifras sean de este agente. Repetir sobre otros agentes/LLMs = línea futura.
cifra→fuente: conclusiones §Limitaciones / §Líneas futuras.

### M9. Calibración de las probabilidades del LLM (sobreconfianza tipo GRACE)
**(R)** No dependemos de ella: STRATA trata al agente como **caja negra**, no usa su `confidence` como probabilidad calibrada. RAM compara la **dirección** con el régimen; el sizing lo da GSO vía σ_t. En el aprendiz, **ninguna personalidad llega al top de SHAP**; las informativas son las de STRATA → la mala calibración del LLM no es la palanca.
cifra→fuente: `decision_automl_prep.json` (top SHAP).
**BACK-UP:** precisamente porque el agente **no es fiable** lo supervisamos desde fuera con un modelo del mercado, en vez de fiarnos de su coherencia interna.

### M10. DiD: ¿es el estimador econométrico?
**(R)** Es un **contraste propio** de diferencia-en-diferencias, no el estimador de tratamiento causal. ΔΔSharpe entre los dos aprendices entre los dos regímenes; lo común se cancela, queda la complementariedad. Pre-registrado. Resultado pooled **+1,37, IC95 [0,20, 2,60], p=0,008** (excluye 0); fenómeno de panel, no en SPY solo.
cifra→fuente: `regime_did_learners.json`.

---

## Nivel BAJA probabilidad

### B1. Por qué BOCPD (PSA) si apenas dispara
**(R)** PSA/GSO disparan poco como reglas (posiciones casi constantes), pero sus **scores continuos informan al aprendiz** (ablación sin PSA+GSO: 0,574→0,550). Detector bien fundado (Adams-MacKay 2007, online, causal); su inactividad como regla es un hallazgo que reportamos.

### B2. Override vs modos intermedios (reduce/abstención)
**(R)** Sobre días en mercado (n=232): **override 0,478** > abstención 0,401 > reducir 0,397 (≈agente); Sharpe override −0,46 vs −2,10/−2,75. Corregir activamente la dirección gana a solo evitar riesgo. El prior del signo es **data-driven por activo** (media del régimen), nunca hardcoded.
cifra→fuente: `spy_intervention_variants.json`.

### B3. Sensibilidad a umbrales: ¿anti-tuning real? (respuesta dura al "¿0,50 o 0,49?")
**(R)** El barrido es una **meseta**: M8 acc 0,466 en τ=0,3 y 0,478 desde τ=0,5, sin saltos; el umbral del aprendiz también plano en ~0,50. **Da igual 0,49 o 0,50**: estamos en meseta, no en un punto afortunado. Los umbrales de PSA/GSO son **cuantiles de calibración (P95)**, data-driven, justo lo que Dani sugería (sacarlo de un histograma, no a ojo).
cifra→fuente: `psa_gso_threshold_sensitivity.json`, `spy_intervention_variants.json`.

### B4. Costes de transacción / rotación
**(R)** M8 rota **61** (< agente 105); el Sharpe del aprendiz se mantiene positivo hasta ~10 pb. Override solo actúa donde RAM dispara (compuerta) → rota poco, no es alta frecuencia. *(Re-verificar cifras contra notebook.)*

### B5. Por qué embargo=1 y no 5 (CPCV)
**(R)** Walk-forward desplegable con **embargo=1** porque el **horizonte de etiqueta es 1** y el origen es rodante (Tashman 2000, LdP §7.4). El embargo=5 es de folds bidireccionales de CPCV; el rolling-origin no los tiene. Pre-registrado (DECISIONES #15). Única excepción a "no KFold": M3, deliberada.

### B6. Dos ventanas (n≈401 vs n≈251): ¿peras con manzanas?
**(R)** Al contrario: **nunca las mezclamos.** OOS completo (401) para las que no entrenan (M5/M8/ZeroR/B&H); desplegable (251, tras burn-in 150) para las que sí (M10/AutoML). Mezclarlas sería el error; por eso se separan, y cada tabla etiqueta su ventana.

### B7. El gate RAM (r=0,93): ¿correlación trivial?
**(R)** No tautológico: mide que la **intensidad de la intervención escala con la discrepancia agente↔régimen** (r=0,93, p<0,001) → la regla está **dirigida**, no actúa al azar. Cuando RAM dispara, seguir el régimen bate a seguir al agente en **6/10**.
cifra→fuente: `spy_panel_gate_descriptive.json`.

### B8. ¿Por qué AI Hedge Fund / por qué 5 personalidades?
**(R)** Réplica de un sistema multiagente publicado (Singh 2024); 5 personalidades sobre un LLM coordinadas por un gestor de cartera. No condiciona el método: STRATA supervisa cualquier salida w_t∈[−1,1]; el agente es el caso de estudio, no la aportación. SHAP: ninguna personalidad individual es informativa.

### B9. ¿Sin APIs de pago compromete el agente?
**(R)** OpenRouter free (DeepSeek/gpt-oss) a temperatura 0 → reproducible, coste cero. La baja calidad del agente es lo que hace interesante el rescate. El cutoff del LLM (anterior a 2024-10) es lo que permite un OOS sin look-ahead.

---

*Fuente: `defensa-tutor` (2026-06-24), anclado en `MARCO_PRACTICO_CONTEXTO.md`, los capítulos y las transcripciones del tutor. Falta: revisar si el estado del arte (cap. 1–2) genera preguntas de novedad bibliográfica.*
