# Revisión iterada del notebook definitivo del marco práctico (panel de 10)

Bucle constructor ↔ revisora (`raquel-quant`, quant senior + matemática) sobre
`notebooks/STRATA_marco_practico.ipynb`, hasta declararlo definitivo contra el gate G1–G6 (+ G1b/G6b).
**Cierre: APROBADO sin condiciones en la ronda 2** (2026-06-23). Notebook: 27 celdas de código, 0 errores,
auto-test verde.

## Alcance del notebook (panel de 10, mucho más completo)
Réplica del análisis de 15 pero más completo y con todo justificado: §1 datos/protocolo (15→10 cohorte de
aplicabilidad; 5 en apéndice) · §2 mecánica ex-ante (HMM **K=3 justificado por verosimilitud held-out**,
GARCH-t, BOCPD, leverage honesto, intervención/scores/atribución por detector) · §3 caso SPY (AutoML gana
nominal; equity con AutoML; McNemar honesto; SHAP) · §4 panel-10 (ablación, SHAP, heatmap, **pooled bootstrap
con AutoML**) · §5 **mecanismo por activo** (dos supervisores; discriminante `crisis_mean`; casos trabajados
QQQ=régimen / MARA=ML) · §6 **clustering que afirma naturaleza→canal** (tabla cruzada) · §7 robustez (equity por
activo, **accuracy rodante**, **val/test 3 particiones**, **rescate significativo en alcista Y bajista pooled**,
suite SMCI, techo ZeroR) · §8 apéndice de los 5 excluidos · §9 conclusiones.

## Ronda 1 — APROBADO CON CONDICIONES (6 fixes)
1. **§5 caso régimen:** XLE estaba etiquetado como canal régimen pero `mechanism_panel.json` lo marca ML
   (crisis_mean≈0). Sustituido por **QQQ** (crisis_mean<0, canal régimen, M8 acierto 0.62, McNemar M8/M10 vs M5
   0.051/0.036); XLE reencuadrado como **frontera**.
2. **Criterio cuerpo/apéndice:** reformulado como **cohorte pre-registrada ilustrativa del mecanismo** (no de
   significancia per-activo), con assert que reproduce PANEL10/EXCL5; BAC justificado (rescate nominal p=0.198).
3. **Headline de riesgo pooled:** **pooled-15 canónico (n=3751)** + **pooled-10 (n=2493) como sensibilidad**,
   coherente con RESULTADOS §1ter y BITÁCORA.
4. **Conclusiones §9:** numeración duplicada (dos "7") corregida → 1–8, mapeo O6/O7 explícito.
5. **Clustering:** declarado el desacuerdo de **Spectral (Rand 0.401)**; consenso KMeans/Ward/GMM (Rand 1.0).
6. **Apéndice:** corregido el mislabel "MSTR/UNG-like" (UNG está en el **cuerpo**, no en EXCL5).

## Ronda 2 — APROBADO (sin condiciones)
Los 6 fixes verificados contra builder, notebook reejecutado y JSON fuente. Cifras load-bearing trazadas:
SPY McNemar AutoML/M10/M8 vs M5 = 0.0002/0.0074/0.0509; **pooled-15 ΔSharpe +0.66 IC[0.225,1.157] sig**,
ΔmaxDD +0.24 IC[0.017,0.445] sig; **bull/bear pooled M10/AutoML vs M5 significativo en AMBOS regímenes**;
rolling mejor-STRATA>M5 en 8/10; K=3 held-out −1.301 > −1.693. Honestidad intacta (G3): las únicas frases
fuertes son negaciones de over-claim. Veredicto: defendible ante el tribunal.

## Punto más blando (anotado por la revisora, no bloqueante)
**UNG** está en el cuerpo pero el agente NO pierde a las triviales ahí (M5 0.510 > trivial 0.486); se encuadra
como caso ML donde STRATA defiere. Es el punto del split que un tribunal podría cuestionar. Defendible con el
mecanismo, pero si se quiere un panel sin ninguna arista, UNG es el candidato a intercambiar (p.ej. por BAC/NVDA,
rescate de riesgo con mecanismo nombrable).
