ya tengo espacio
Bash
IN
cd /Users/Raquel/Desktop/tradingSupervisor_ML_IA_messy
df -h /private/tmp | tail -1
echo "--- caches OK ---"
echo "UNG=$(ls cache/agent/UNG/|wc -l|tr -d ' ')  MSTR=$(ls cache/agent/MSTR/|wc -l|tr -d ' ')"

OUT
/dev/disk3s5   460Gi   432Gi   981Mi   100%    2.6M   10M   20%   /System/Volumes/Data
--- caches OK ---
UNG=401  MSTR=401
Espacio justo (981Mi libres) pero suficiente. Sigo. Estructura del notebook:

Bash
IN
cd /Users/Raquel/Desktop/tradingSupervisor_ML_IA_messy
.venv/bin/python - <<'PY'
import json
nb=json.load(open("notebooks/strata_tfg.ipynb")); cells=nb["cells"]
print("total cells:", len(cells))
for i,c in enumerate(cells):
    s="".join(c["source"]); h=s.lstrip().splitlines()[0] if s.strip() else ""
    if (c["cell_type"]=="markdown" and h.startswith(("## ","### "))) or (c["cell_type"]=="code" and any(k in s for k in ("**Lectura XLE.","PANEL = {","DEEP_DIVE","NEW_TICKERS","Series de sizing de XLE"))):
        print(f"  {i:3} {c['cell_type'][:2]:2}  | {h[:75]}")
PY

OUT
total cells: 78
    1 ma  | ## 0. Configuración y semilla
    3 ma  | ## 1. Datos de mercado
    5 ma  | ## 2. Calibración (2000-01-01 → 2024-09-30)
    9 ma  | ## 3. Los tres detectores y sus umbrales (calibrados por activo)
   11 ma  | ## 4. El agente: AI Hedge Fund (5 personalidades)
   13 ma  | ## 5. Las nueve configuraciones (M1–M9)
   19 ma  | ## 6. Resultados comparativos
   21 ma  | ## 7. El sesgo del *machine learning*: KFold vs CPCV (M3 vs M4)
   23 ma  | ## 8. Significancia estadística
   26 ma  | ## 9. Ablación de STRATA y techo de supervisión
   28 ma  | ### 9.1 Ablación en modo *override*
   30 ma  | ## 10. Estudio de caso: una intervención de M8 (*override*)
   32 ma  | ## 11. Conclusiones sobre SPY
   33 ma  | ## 12. Extensión multi-activo: el *leverage effect* y el régimen por activo
   36 ma  | ### 12.1 Comparativa M1–M9 sobre NVDA
   43 ma  | ## 13. Tercer activo: BAC (financiero, *leverage effect* clásico)
   45 ma  | ### 13.1 Comparativa M1–M9 sobre BAC
   52 ma  | ## 14. Cuarto activo: TSLA (*growth* de beta alta)
   54 ma  | ### 14.1 Comparativa M1–M9 sobre TSLA
   61 ma  | ## 15. Quinto activo: XLE (energía, sector cíclico)
   63 ma  | ### 15.1 Comparativa M1–M9 sobre XLE
   67 co  | # Series de sizing de XLE (peso decidido por configuración).
   70 ma  | ## 16. Panel multi-activo: la ventaja condicional de STRATA
   71 co  | # Panel multi-activo: M1–M9 por activo con TODOS sus modelos propios (HMM, 
   74 ma  | ### 16.1 Significancia por activo: Diebold-Mariano y Deflated Sharpe
   75 co  | # Matriz Diebold-Mariano 9×9 por activo (réplica de §8) + DSR por configura
   76 ma  | ### 16.2 Lectura del panel
   77 ma  | ## 17. Reproducibilidad
Estructura clara. Voy a:

Construir las 18 celdas (9 UNG + 9 MSTR) con la misma plantilla que TSLA/XLE (incluida la columna Ret@σBH y la matriz DM).
Insertarlas entre la Lectura de XLE (c69) y el header del panel (c70).
Renumerar panel §16→§18 y repro §17→§19; sembrar UNG/MSTR en PANEL y añadirlos a DEEP_DIVE.
Bash
IN
cd /Users/Raquel/Desktop/tradingSupervisor_ML_IA_messy
.venv/bin/python - <<'PYEOF'
import json
NB="notebooks/strata_tfg.ipynb"
nb=json.load(open(NB)); cells=nb["cells"]
def S(i): return "".join(cells[i]["source"])
def setsrc(i,s): cells[i]["source"]=s.splitlines(keepends=True)
def repl(i,old,new,n=1):
    s=S(i); c=s.count(old); assert c==n, f"cell {i}: found {c} of snippet (want {n})"
    setsrc(i,s.replace(old,new)); print(f"edited cell {i}")
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)}
def md(s):   return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}

# Anchor checks
assert S(69).startswith("**Lectura XLE."), "cell 69 not XLE lectura"
assert S(70).startswith("## 16. Panel multi-activo"), "cell 70 not panel header"
assert "PANEL = {" in S(71)
assert S(74).startswith("### 16.1"), "cell 74 not 16.1"
assert "DEEP_DIVE" in S(75)
assert S(76).startswith("### 16.2")
assert S(77).startswith("## 17. Reproducibilidad")

# ---- IN-PLACE EDITS on panel/repro (indices stable; do first) ----
repl(70, "## 16. Panel multi-activo: la ventaja condicional de STRATA",
         "## 18. Panel multi-activo: la ventaja condicional de STRATA")
repl(70,
"Las inmersiones SPY (§1–§11), NVDA (§12), BAC (§13), TSLA (§14) y XLE (§15)\nentran ya calculadas; el panel añade cualquier otro *ticker* con caché (objetivo: hasta 10 activos).",
"Las inmersiones SPY (§1–§11), NVDA (§12), BAC (§13), TSLA (§14), XLE (§15), UNG (§16) y MSTR (§17)\nentran ya calculadas; el panel añade cualquier otro *ticker* con caché (objetivo: hasta 10 activos).")

repl(71,
'''    "XLE":  {"nr": nrx, "common": common_xle,  "leverage_bps": [m*1e4 for m in _rdm_xle],  "rdir": RDIR_xle, "ram_act": float("nan")},
}''',
'''    "XLE":  {"nr": nrx, "common": common_xle,  "leverage_bps": [m*1e4 for m in _rdm_xle],  "rdir": RDIR_xle, "ram_act": float("nan")},
    "UNG":  {"nr": nru, "common": common_ung,  "leverage_bps": [m*1e4 for m in _rdm_u],    "rdir": RDIR_u,   "ram_act": float("nan")},
    "MSTR": {"nr": nrm, "common": common_mstr, "leverage_bps": [m*1e4 for m in _rdm_m],    "rdir": RDIR_m,   "ram_act": float("nan")},
}''')

repl(74, "### 16.1 Significancia por activo", "### 18.1 Significancia por activo")
repl(74,
"Cada inmersión (SPY §8, NVDA, BAC, TSLA, XLE) ya incluye su **matriz 9×9 de p-valores de\nDiebold-Mariano**",
"Cada inmersión (SPY §8, NVDA, BAC, TSLA, XLE, UNG, MSTR) ya incluye su **matriz 9×9 de p-valores de\nDiebold-Mariano**")

repl(75,
'DEEP_DIVE = {"SPY", "NVDA", "BAC", "TSLA", "XLE"}',
'DEEP_DIVE = {"SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR"}')

repl(76, "### 16.2 Lectura del panel", "### 18.2 Lectura del panel")
repl(77, "## 17. Reproducibilidad", "## 19. Reproducibilidad")
print("in-place edits OK")

# ---- BUILD NEW SECTIONS ----
def asset_section(num, tk, suf, nr_var, com_var, rows_var, title_tail, intro, lectura_placeholder):
    h_md = md(f"## {num}. {title_tail}\n\n{intro}\n")
    hmm_code = code(f'''# {tk} con modelos PROPIOS (HMM + GARCH del activo), simétrico con SPY/NVDA/BAC/TSLA/XLE.
{tk.lower()}_px_full = load_market_data("{tk}", CALIB_START, END_DATE)["Close"]
{tk.lower()}_ret_full = log_returns({tk.lower()}_px_full)

# HMM de 3 estados entrenado sobre la PROPIA serie de {tk}.
feats_{suf}_full = pd.DataFrame({{"ret_log": {tk.lower()}_ret_full}})
feats_{suf}_full["rv_21_ann"] = feats_{suf}_full["ret_log"].rolling(21).std() * np.sqrt(252)
feats_{suf}_full = feats_{suf}_full.dropna()
feats_{suf}_calib = feats_{suf}_full.loc[:CALIB_END]
hmm_{suf} = RegimeHMM().fit(feats_{suf}_calib[["ret_log", "rv_21_ann"]].to_numpy())
RDIR_{suf}, _rdm_{suf} = regime_dir_from_calib(hmm_{suf}, feats_{suf}_calib)   # prior de RAM propio
states_{suf}_full = pd.Series(hmm_{suf}.predict_states(feats_{suf}_full[["ret_log", "rv_21_ann"]].to_numpy()),
                           index=feats_{suf}_full.index)
print("HMM {tk} — diagonal transmat:", np.round(np.diag(hmm_{suf}.transition_matrix), 3))

feats_{suf} = feats_{suf}_full.reindex(oos_dates).ffill()
proba_{suf} = hmm_{suf}.predict_proba(feats_{suf}[["ret_log", "rv_21_ann"]].to_numpy())
states_{suf} = pd.Series(hmm_{suf}.predict_states(feats_{suf}[["ret_log", "rv_21_ann"]].to_numpy()), index=oos_dates)
regime_{suf} = states_{suf}.map(REGIME_FACTOR)

{tk.lower()}_ret_spx = {tk.lower()}_ret_full.reindex(feats_full.index)
{tk.lower()}_ret_own = {tk.lower()}_ret_full.reindex(states_{suf}_full.index)
by_regime_{suf} = pd.DataFrame({{
    "S&P (rég. S&P)":   [feats_full["ret_log"][states_full == s].mean()*1e4 for s in (0, 1, 2)],
    "{tk} (rég. S&P)":  [{tk.lower()}_ret_spx[states_full == s].mean()*1e4 for s in (0, 1, 2)],
    "{tk} (rég. {tk})": [{tk.lower()}_ret_own[states_{suf}_full == s].mean()*1e4 for s in (0, 1, 2)],
}}, index=["Calma", "Estrés", "Crisis"])
print("\\nRetorno diario medio por régimen (bps):"); print(by_regime_{suf}.round(1).to_string())

{suf}_act_spx = ram_activation("{tk}", proba_smoothed)
{suf}_act_own = ram_activation("{tk}", proba_{suf})
print(f"\\nActivación de RAM (medium+):  {tk}(rég. S&P)={{{suf}_act_spx:.1%}}  {tk}(rég. propio)={{{suf}_act_own:.1%}}")
print(f"Prior RAM reasignado por activo — {tk} Crisis = "
      f"{{'long(+1)' if RDIR_{suf}[2] > 0 else 'short(−1)'}} (calib {{_rdm_{suf}[2]*1e4:+.1f}} bps) ⇒ "
      f"{{'leverage invertido' if RDIR_{suf}[2] > 0 else 'leverage clásico'}}.")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
by_regime_{suf}.plot(kind="bar", ax=axes[0], color=[PALETTE[1], PALETTE[5], PALETTE[2]])
axes[0].axhline(0, color="0.5", lw=0.6); axes[0].set_title("{tk} — retorno medio por régimen (bps)")
axes[0].set_xticklabels(by_regime_{suf}.index, rotation=0); axes[0].legend(fontsize=7)
axes[1].bar(["{tk}\\n(rég. S&P)", "{tk}\\n(rég. propio)"], [{suf}_act_spx, {suf}_act_own],
            color=[PALETTE[5], PALETTE[2]])
axes[1].set_title("{tk} — activación de RAM (flag medium+)"); axes[1].set_ylabel("frecuencia")
plt.tight_layout(); plt.show()
''')
    sub_md = md(f'''### {num}.1 Comparativa M1–M9 sobre {tk}

Comparativa completa con los componentes propios de {tk} (HMM, GARCH, umbrales recalibrados y prior de RAM
re-signado sin look-ahead) y la señal direccional de H2O del S&P (M3/M4/M9). Se añade la **matriz 9×9 de
Diebold-Mariano** y el DSR de cada configuración. Sin *cross-check* oficial (el experimento canónico es SPY).
''')
    m_code = code(f'''# --- {tk}: GARCH propio, umbrales por activo y las nueve configuraciones ---
garch_{suf} = GARCHModel().fit({tk.lower()}_ret_full.loc[:CALIB_END].dropna())
ret_{suf} = {tk.lower()}_ret_full.reindex(oos_dates).ffill()
sigma_{suf} = garch_{suf}.forecast_path(ret_{suf})
THR_{suf}, thr_{suf}_vals = calibrate_thresholds(hmm_{suf}, garch_{suf}, feats_{suf}_calib)
magnitude_{suf} = (TARGET_VOL/sigma_{suf}).clip(0, 1) * regime_{suf}
print(f"GARCH {tk}: alpha={{garch_{suf}.alpha:.3f}} beta={{garch_{suf}.beta:.3f}} nu={{garch_{suf}.nu:.2f}}")
print(f"Umbrales {tk}: PSA P95={{thr_{suf}_vals['psa']['p95']:.4f}}  GSO P95={{thr_{suf}_vals['gso']['p95']:.4f}}")

agent_{suf} = load_agent_decisions("{tk}", oos_dates)
agent_size_{suf} = pd.Series({{d: float(v.get("size", 0.0)) for d, v in agent_{suf}.items()}}).reindex(oos_dates)
pag_{suf} = pd.Series({{d: _agent_p1(v) for d, v in agent_{suf}.items()}}).reindex(oos_dates)

{nr_var} = {{}}
{nr_var}["m1_buy_and_hold"]   = run_backtest(ret_{suf}, pd.Series(1.0, index=oos_dates))
{nr_var}["m2_bh_garchhmm"]    = run_backtest(ret_{suf}, magnitude_{suf})
{nr_var}["m3_ml_naive"]       = run_backtest(ret_{suf}, np.clip(2*p3-1, -1, 1))
{nr_var}["m4_ml_strata"]      = run_backtest(ret_{suf}, np.clip(2*p4-1, -1, 1)*magnitude_{suf})
p9{suf} = pd.Series(np.where(~pag_{suf}.isna(), 0.6*p4b+0.4*pag_{suf}, p4b), index=oos_dates)
{nr_var}["m9_ml_ai"]          = run_backtest(ret_{suf}, np.clip(2*p9{suf}-1, -1, 1)*magnitude_{suf})
{nr_var}["m5_agent_alone"]    = run_backtest(ret_{suf}, agent_size_{suf}.dropna())
{nr_var}["m6_strata_warn"]    = run_backtest(ret_{suf}, agent_size_{suf}.reindex(oos_dates).fillna(0.0))
{nr_var}["m7_strata_reduce"]  = run_backtest(ret_{suf}, supervised_sizes("reduce", psa_signal="cp_prob_delta",
                                        psa_hazard=1/60, agent_dec=agent_{suf}, sig=sigma_{suf}, thr=THR_{suf},
                                        hmm_src=hmm_{suf}, feats_src=feats_{suf}, proba_src=proba_{suf}, regime_dir=RDIR_{suf}))
{nr_var}["m8_strata_override"]= run_backtest(ret_{suf}, supervised_sizes("override", override_variant="C",
                                        regime_mode="filtered", agent_dec=agent_{suf}, sig=sigma_{suf}, thr=THR_{suf},
                                        hmm_src=hmm_{suf}, feats_src=feats_{suf}, proba_src=proba_{suf}, regime_dir=RDIR_{suf}))
{com_var} = sorted(set.intersection(*[set({nr_var}[c].dropna().index) for c in CONFIGS]))
print(f"{tk} — ventana común: {{len({com_var})}} sesiones; decisiones agente: {{len(agent_{suf})}}")
''')
    t_code = code(f'''# Tabla de métricas 9x8 para {tk} (con Ret@σBH).
vol_bh = ann_vol({nr_var}["m1_buy_and_hold"].reindex({com_var}))   # vol de B&H del activo
{rows_var} = []
for c in CONFIGS:
    r = {nr_var}[c].reindex({com_var})
    {rows_var}.append({{"Config": LABELS[c], "Sharpe": sharpe(r), "DSR": dsr(sharpe(r), 9, len({com_var})),
                    "MaxDD": max_dd(equity(r)), "Calmar": calmar(r),
                    "Retorno": total_return(r), "Ret@σBH": ret_at_vol(r, vol_bh),
                    "Vol.": ann_vol(r), "Hit": hit_rate(r)}})
tabla_{suf} = pd.DataFrame({rows_var}).set_index("Config")
(tabla_{suf}.style.format({{"Sharpe":"{{:+.3f}}","DSR":"{{:.3f}}","MaxDD":"{{:.1%}}","Calmar":"{{:+.2f}}",
                        "Retorno":"{{:+.1%}}","Ret@σBH":"{{:+.1%}}","Vol.":"{{:.1%}}","Hit":"{{:.1%}}"}})
        .background_gradient(subset=["Sharpe"], cmap="RdYlGn")
        .set_caption("Métricas M1–M9 sobre {tk} (causal, neto de costes)"))
''')
    e_code = code(f'''# Curvas de capital y drawdowns de {tk}.
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for i, c in enumerate(CONFIGS):
    eq = equity({nr_var}[c].reindex({com_var}))
    axes[0].plot(eq.index, eq, color=PALETTE[i], lw=1.1, label=LABELS[c])
    axes[1].plot(eq.index, eq/eq.cummax()-1, color=PALETTE[i], lw=0.8)
axes[0].axhline(1, color="0.7", lw=0.5); axes[0].set_title("{tk} — curvas de capital (base 1.0)")
axes[0].legend(ncol=3, fontsize=8, loc="upper left")
axes[1].set_title("{tk} — drawdowns"); axes[1].set_ylabel("DD")
plt.tight_layout(); plt.show()
''')
    s_code = code(f'''# Series de sizing de {tk} (peso decidido por configuración).
weights_{suf} = {{
    "m2_bh_garchhmm": magnitude_{suf}, "m4_ml_strata": np.clip(2*p4-1, -1, 1)*magnitude_{suf},
    "m5_agent_alone": agent_size_{suf},
    "m8_strata_override": supervised_sizes("override", override_variant="C", regime_mode="filtered",
                                           agent_dec=agent_{suf}, sig=sigma_{suf}, thr=THR_{suf},
                                           hmm_src=hmm_{suf}, feats_src=feats_{suf}, proba_src=proba_{suf},
                                           regime_dir=RDIR_{suf}),
}}
fig, ax = plt.subplots(figsize=(11, 3.2))
for i, (c, w) in enumerate(weights_{suf}.items()):
    ax.plot(oos_dates, pd.Series(w).reindex(oos_dates).values, lw=0.8,
            color=PALETTE[[1, 3, 4, 7][i]], label=LABELS[c])
ax.axhline(0, color="0.6", lw=0.5); ax.set_title("{tk} — series de sizing (selección de configuraciones)")
ax.set_ylabel("peso"); ax.legend(ncol=4, fontsize=8); plt.tight_layout(); plt.show()
''')
    dm_code = code(f'# Matriz de p-valores Diebold-Mariano para {tk} (réplica de §8).\n_ = plot_dm_matrix({nr_var}, {com_var}, "{tk}")\n')
    l_md = md(lectura_placeholder)
    return [h_md, hmm_code, sub_md, m_code, t_code, e_code, s_code, dm_code, l_md]

ung = asset_section(
    16, "UNG", "u", "nru", "common_ung", "rows_u",
    "Sexto activo: UNG (gas natural, B&H catastrófico por *contango*)",
    "UNG (United States Natural Gas Fund) es un ETF de gas natural muy volátil y con **B&H históricamente "
    "pésimo** por el *contango* (decadencia estructural de los futuros que replica). Lo añadimos como prueba "
    "del ángulo *complementario* al de TSLA: en TSLA el M8 capturaba *melt-ups* gracias al leverage **invertido**; "
    "aquí el preview de calibración salió **clásico** (Crisis bajista) y B&H **catastrófico** (Sharpe negativo, "
    "drawdown muy profundo), así que la pregunta es si STRATA bate a B&H **por la otra vía**: ponerse corto en "
    "Crisis con el prior clásico bien signado sobre un activo que se desploma.",
    "**Lectura UNG.** _(se completa tras la ejecución con las cifras del activo.)_\n")

mstr = asset_section(
    17, "MSTR", "m", "nrm", "common_mstr", "rows_m",
    "Séptimo activo: MSTR (proxy de bitcoin, calibración dominada por la era pre-2020)",
    "MSTR (MicroStrategy) actúa desde 2020 como **proxy de bitcoin** (tesorería en BTC). La intuición sugería "
    "leverage invertido (melt-ups de growth/cripto), pero el preview revela que **su calibración 2000–2024 está "
    "dominada por la era software/puntocom** —incluido el −99 % de 2000—, así que el régimen de máxima "
    "volatilidad sale **bajista** (clásico). Lo añadimos para responder: con un B&H **muy débil en el OOS** "
    "(Sharpe ~0,14, drawdown −85 %), ¿el agente y STRATA superan al largo pasivo aprovechando ese mismo "
    "mecanismo de corto en Crisis?",
    "**Lectura MSTR.** _(se completa tras la ejecución con las cifras del activo.)_\n")

# ---- SPLICE: insert ung + mstr between cell 69 (XLE Lectura) and 70 (panel header) ----
new = cells[:70] + ung + mstr + cells[70:]
nb["cells"] = new
print(f"spliced. {len(cells)} -> {len(new)} cells")

# compile check
src="\n\n".join("".join(c["source"]) for c in new if c["cell_type"]=="code")
compile(src,"<nb>","exec"); print("compile OK")

json.dump(nb, open(NB,"w"), indent=1, ensure_ascii=False)
print("saved")
PYEOF

OUT
edited cell 70
edited cell 70
edited cell 71
edited cell 74
edited cell 74
edited cell 75
edited cell 76
edited cell 77
in-place edits OK
spliced. 78 -> 96 cells
compile OK
saved
Splice + edits + compile OK. Re-ejecuto el notebook.

Update Todos

Add UNG + MSTR sections + panel updates (splice + compile OK)

Re-execute notebook; verify cross-check + new outputs

Fill UNG/MSTR Lecturas with real numbers; update BITACORA/memory
Bash
IN
cd /Users/Raquel/Desktop/tradingSupervisor_ML_IA_messy
PYTHONUNBUFFERED=1 .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=2400 notebooks/strata_tfg.ipynb 2>&1

OUT
Command running in background with ID: b5bn3th7m. Output is being written to: /private/tmp/claude-502/-Users-Raquel-Desktop-tradingSupervisor-ML-IA-messy/efb5368f-c860-4660-9dc5-1cc560673695/tasks/b5bn3th7m.output. You will be notified when it completes. To check interim output, use Read on that file path.
Notebook re-ejecutándose (ahora 7 activos: SPY/NVDA/BAC/TSLA/XLE/UNG/MSTR + panel). ~20 min. Te aviso cuando termine y leo las cifras de UNG/MSTR para rellenar las Lecturas.

<task-notification>
<task-id>b5bn3th7m</task-id>
<tool-use-id>toolu_0192m6z4X2BmZuVpahNt6VPU</tool-use-id>
<output-file>/private/tmp/claude-502/-Users-Raquel-Desktop-tradingSupervisor-ML-IA-messy/efb5368f-c860-4660-9dc5-1cc560673695/tasks/b5bn3th7m.output</output-file>
<status>completed</status>
<summary>Background command "cd /Users/Raquel/Desktop/tradingSupervisor_ML_IA_messy
PYTHONUNBUFFERED=1 .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=2400 notebooks/strata_tfg.ipynb 2&gt;&amp;1" completed (exit code 0)</summary>
</task-notification>
Done. Let me verify cross-check + extract UNG/MSTR results.

Bash
IN
cd /Users/Raquel/Desktop/tradingSupervisor_ML_IA_messy
.venv/bin/python - <<'PY'
import json, re
nb=json.load(open("notebooks/strata_tfg.ipynb")); cells=nb["cells"]
def out(c):
    t=[]
    for o in c.get("outputs",[]):
        if not isinstance(o,dict): continue
        if o.get("output_type")=="stream": t.append("".join(o.get("text",[])))
        elif o.get("output_type") in ("execute_result","display_data"):
            d=o.get("data",{})
            if "text/plain" in d: t.append("".join(d["text/plain"]))
    return "".join(t)
def html(c):
    for o in c.get("outputs",[]):
        if isinstance(o,dict) and "text/html" in o.get("data",{}): return "".join(o["data"]["text/html"])
    return ""
def find(pred): return [i for i,c in enumerate(cells) if c["cell_type"]=="code" and pred("".join(c["source"]))]

# cross-check
for i in find(lambda s:"Cross-check" in s and "Reproducción fiel" in s):
    for ln in out(cells[i]).splitlines():
        if "Δmax" in ln or "fiel" in ln: print("[xcheck]", ln.strip())

# leverage + activación + ventana
for tk in ("UNG","MSTR"):
    for i in find(lambda s,k=tk: f"HMM {k} —" in s):
        for ln in out(cells[i]).splitlines():
            if any(w in ln for w in("Prior RAM","Activación de RAM","Retorno diario")):
                print(f"[{tk}]", ln.strip())
    for i in find(lambda s,k=tk: f"GARCH {k}:" in s):
        for ln in out(cells[i]).splitlines():
            if "ventana común" in ln or "GARCH" in ln or "Umbrales" in ln: print(f"[{tk}]", ln.strip())

# panel scatter print
for i in find(lambda s:"Correlación (calidad B&H" in s):
    print("\n[PANEL scatter]")
    print(out(cells[i]).strip())

# extract M1-M9 tables
def rows(h):
    return [(re.sub(r'\s+',' ',m.group(1)).strip(),
             re.findall(r'<td[^>]*>([^<]*)</td>', m.group(2)))
            for m in re.finditer(r'row_heading[^>]*>(M\d[^<]*)</th>(.*?)(?=row_heading|</tbody>)', h, re.S)]
def cols(h):
    return re.findall(r'col_heading[^>]*>([^<]*)</th>', h)
for tk in ("UNG","MSTR"):
    for c in cells:
        h=html(c)
        if f"Métricas M1–M9 sobre {tk}" in h:
            print(f"\n=== {tk}  cols={cols(h)}")
            for lab,v in rows(h): print(f"  {lab:14} {v}")
            break
PY

OUT
[xcheck] Cross-check (Δmax net_return vs JSON oficial)
[xcheck] M1 B&H         Δmax=3.35e-07  ✓
[xcheck] M2 GARCH×HMM   Δmax=3.27e-07  ✓
[xcheck] M3 ML KFold    Δmax=6.63e-08  ✓
[xcheck] M4 ML CPCV     Δmax=5.58e-08  ✓
[xcheck] M5 Agente      Δmax=7.90e-08  ✓
[xcheck] M6 warn        Δmax=7.90e-08  ✓
[xcheck] M7 reduce      Δmax=2.68e-04  ✓
[xcheck] M8 override    Δmax=3.27e-07  ✓
[xcheck] M9 ML+IA       Δmax=2.09e-08  ✓
[xcheck] Reproducción fiel verificada (todos < 5e-3).
[UNG] Retorno diario medio por régimen (bps):
[UNG] Activación de RAM (medium+):  UNG(rég. S&P)=41.9%  UNG(rég. propio)=9.7%
[UNG] Prior RAM reasignado por activo — UNG Crisis = short(−1) (calib -19.8 bps) ⇒ leverage clásico.
[UNG] GARCH UNG: alpha=0.069 beta=0.924 nu=13.46
[UNG] Umbrales UNG: PSA P95=0.0058  GSO P95=6.6777
[UNG] UNG — ventana común: 401 sesiones; decisiones agente: 401
[MSTR] Retorno diario medio por régimen (bps):
[MSTR] Activación de RAM (medium+):  MSTR(rég. S&P)=41.9%  MSTR(rég. propio)=0.0%
[MSTR] Prior RAM reasignado por activo — MSTR Crisis = short(−1) (calib -10.4 bps) ⇒ leverage clásico.
[MSTR] GARCH MSTR: alpha=0.029 beta=0.971 nu=3.33
[MSTR] Umbrales MSTR: PSA P95=0.0053  GSO P95=13.3411
[MSTR] MSTR — ventana común: 401 sesiones; decisiones agente: 401

[PANEL scatter]
<Figure size 750x500 with 1 Axes>Correlación (calidad B&H, ventaja STRATA): -0.54  (pendiente negativa ⇒ STRATA aporta donde el largo pasivo falla)
M8 supera a B&H en 4/7: UNG, TSLA, BAC, NVDA
M8 supera a la mejor clásica en 2/7: UNG, TSLA

=== UNG  cols=['Sharpe', 'DSR', 'MaxDD', 'Calmar', 'Retorno', 'Ret@σBH', 'Vol.', 'Hit']
  M1 B&H         ['-0.364', '0.000', '-65.6%', '-0.54', '-50.5%', '-50.5%', '63.9%', '49.6%']
  M2 GARCH×HMM   ['-0.467', '0.000', '-6.3%', '-0.33', '-3.3%', '-55.1%', '4.3%', '23.2%']
  M3 ML KFold    ['-0.548', '0.000', '-11.2%', '-0.39', '-6.8%', '-58.9%', '7.5%', '52.1%']
  M4 ML CPCV     ['-0.918', '0.000', '-0.8%', '-0.59', '-0.7%', '-72.0%', '0.5%', '25.4%']
  M5 Agente      ['+0.333', '1.000', '-6.5%', '+0.32', '+3.3%', '+1.7%', '6.9%', '49.1%']
  M6 warn        ['+0.308', '1.000', '-6.5%', '+0.29', '+3.0%', '-0.9%', '6.8%', '48.6%']
  M7 reduce      ['+0.399', '1.000', '-2.6%', '+0.41', '+1.7%', '+8.8%', '2.8%', '48.1%']
  M8 override    ['+0.183', '0.983', '-6.5%', '+0.16', '+1.6%', '-12.8%', '6.9%', '48.1%']
  M9 ML+IA       ['+0.159', '0.952', '-1.2%', '+0.13', '+0.3%', '-15.9%', '1.0%', '24.2%']

=== MSTR  cols=['Sharpe', 'DSR', 'MaxDD', 'Calmar', 'Retorno', 'Ret@σBH', 'Vol.', 'Hit']
  M1 B&H         ['+0.246', '1.000', '-81.5%', '-0.16', '-19.9%', '-19.9%', '83.8%', '47.1%']
  M2 GARCH×HMM   ['-1.395', '0.000', '-10.4%', '-0.46', '-7.4%', '-91.3%', '3.4%', '26.7%']
  M3 ML KFold    ['+0.420', '1.000', '-14.2%', '+0.27', '+6.1%', '-0.6%', '10.0%', '49.9%']
  M4 ML CPCV     ['+0.098', '0.669', '-0.7%', '+0.06', '+0.1%', '-34.3%', '0.4%', '28.7%']
  M5 Agente      ['-0.193', '0.000', '-11.6%', '-0.17', '-3.1%', '-56.3%', '8.3%', '52.9%']
  M6 warn        ['-0.134', '0.000', '-10.8%', '-0.13', '-2.3%', '-52.7%', '8.3%', '52.6%']
  M7 reduce      ['-0.262', '0.000', '-5.1%', '-0.19', '-1.5%', '-60.2%', '3.4%', '52.6%']
  M8 override    ['-0.040', '0.010', '-10.1%', '-0.07', '-1.1%', '-46.5%', '8.3%', '52.9%']
  M9 ML+IA       ['+1.882', '1.000', '-0.5%', '+2.82', '+2.1%', '+614.4%', '0.7%', '32.2%']
Cross-check verde (Δ 3.27e-7). Resultados muy interesantes — la hipótesis original era equivocada en el mecanismo pero acertada en UNG por otra vía. Voy a rellenar las Lecturas.

Bash
IN
cd /Users/Raquel/Desktop/tradingSupervisor_ML_IA_messy
.venv/bin/python - <<'PYEOF'
import json
NB="notebooks/strata_tfg.ipynb"
nb=json.load(open(NB)); cells=nb["cells"]
def setsrc(c,s): c["source"]=s.splitlines(keepends=True)

UNG_LEC = """**Lectura UNG.** UNG confirma la hipótesis "**el agente bate a B&H**" — pero **no** por el mecanismo de leverage invertido que se postulaba (en UNG el leverage es **clásico**, Crisis −19,8 bps en calibración), sino por la **vía complementaria**: B&H es tan **catastrófico** (Sharpe **−0,36**, retorno **−50,5 %**, MaxDD **−66 %**) que cualquier disciplina razonable lo supera.

- **La clásica y el ML también pierden** sobre un activo que solo cae: M2 −0,47, M3 −0,55, M4 −0,92. El *sizing* por régimen no salva nada cuando la dirección es siempre bajista.
- **El agente crudo (M5) sale POSITIVO** (Sharpe **+0,33**, retorno +3,3 %), un hecho **único en todo el panel** (en el resto el agente es negativo). El contexto macro/sentimiento lo orienta a posiciones cortas o flat la mayor parte del tiempo, lo que en gas natural en *contango* es lo correcto.
- **STRATA queda positiva en sus tres modos:** M6 +0,31, **M7 reduce +0,40** (la mejor configuración del activo), M8 override **+0,18**. La intervención atenuante (reduce) saca lo mejor del agente; el override va corto en Crisis con el prior clásico bien signado y captura las caídas.
- **UNG es el segundo activo del panel (con TSLA) donde M8 supera a la mejor clásica.** A riesgo de B&H (`Ret@σBH`) M7 da **+8,8 %** frente a **−50,5 %** de B&H; M8 da −12,8 % (igualmente muy por encima de B&H aunque por debajo de M7 a vol común).

Es la **otra cara** del caso TSLA: TSLA = STRATA captura *melt-ups* con leverage invertido; UNG = STRATA evita *cuchillos cayendo* con leverage clásico. Ambos extremos del mecanismo condicional del panel se hacen visibles.
"""

MSTR_LEC = """**Lectura MSTR.** MSTR es el **contraejemplo** más interesante del panel: ni la clásica ni STRATA consiguen rescatar al activo. El motivo es un **cambio estructural no recogido en la calibración** —MSTR fue una empresa de software 2000–2020 (incluido el −99 % puntocom) y desde 2020 actúa como **proxy de bitcoin** (tesorería en BTC); el HMM calibrado 2000–2024 está dominado por la era pre-2020, así que su Crisis sale **clásica** (−10,4 bps) cuando en el OOS bitcoin-proxy la realidad es otra.

- **B&H paradójico:** Sharpe **+0,25** pero **retorno −19,9 % y MaxDD −81,5 %** (vol del 84 %: la dispersión erosiona el retorno compuesto pese al Sharpe positivo).
- **M2 GARCH×HMM colapsa: −1,40.** El *sizing* por régimen, calibrado al carácter equivocado del activo, sale completamente fuera de fase.
- **El agente y STRATA no remontan:** M5 −0,19, M7 −0,26, **M8 override −0,04** (esencialmente flat). El *override* no rescata porque el régimen calibrado señala "corto en Crisis" en días que en realidad son de melt-up bitcoin.
- **Una excepción: M9 ML+IA** sale con Sharpe **+1,88** (DSR 1,00). Pero hay que leerla con cautela: opera a vol minúscula (0,7 %), su retorno bruto es solo +2,1 % y el `Ret@σBH = +614 %` es **artefacto del escalado lineal** (k ≈ 120 sobre una serie casi plana amplifica cualquier movimiento). El Sharpe es real, el retorno reescalado pierde significado a vol tan baja.

**Conclusión MSTR.** Cuando los **regímenes calibrados no reflejan el carácter actual del activo** (cambio estructural reciente), ningún componente de STRATA puede arreglarlo: M2 falla por sizing fuera de fase, M8 falla por prior mis-signado para la nueva era. Es un límite honesto del marco —el HMM asume estacionariedad en el sentido del régimen, y MSTR la viola tras 2020—. Como contraste, refuerza la lectura del panel: la ventaja condicional de STRATA depende de que la calibración del régimen siga siendo válida en el OOS.
"""

done=[]
for c in cells:
    if c["cell_type"]!="markdown": continue
    s="".join(c["source"])
    if s.startswith("**Lectura UNG."):  setsrc(c, UNG_LEC);  done.append("UNG")
    elif s.startswith("**Lectura MSTR."): setsrc(c, MSTR_LEC); done.append("MSTR")
assert set(done)=={"UNG","MSTR"}, done
json.dump(nb, open(NB,"w"), indent=1, ensure_ascii=False)
print("Lecturas:", done)
PYEOF

OUT
Lecturas: ['UNG', 'MSTR']
Ahora BITACORA + memoria.

Edit BITACORA.md
Added 24 lines
Read panel-multiactivo-status.md
Write panel-multiactivo-status.md
49 lines
Edit MEMORY.md
Modified
Update Todos

Add UNG + MSTR sections + panel updates

Re-execute notebook; cross-check verified

Fill UNG/MSTR Lecturas; BITACORA + memory updated
Terminado. Resumen del trabajo completo:

Resultados (cross-check verde, panel 7 activos)
M5 agente	M8 override	vs B&H	Lectura
UNG	+0,33 ⭐	+0,18 (M7: +0,40)	bate B&H (−0,36) y bate clásica	Caso "cuchillo cayendo"
MSTR	−0,19	−0,04 (flat)	NO bate B&H (+0,25)	Contraejemplo: cambio estructural
TSLA	−0,64	+1,137	bate B&H y clásica	Caso "melt-up" (leverage invertido)
Las dos vías de la "ventaja condicional"
La hipótesis original (leverage invertido) era falsa en los 3 candidatos, pero el resultado refina y refuerza la tesis del panel: STRATA puede ser alfa por dos vías complementarias allí donde B&H falla:

TSLA → melt-up capture con leverage invertido (M8 va largo en Crisis).
UNG → cuchillo cayendo con leverage clásico (M8 va corto en Crisis sobre activo que se desploma; el agente, casi siempre cauto, acierta sin ayuda — único M5 positivo del panel).
Y MSTR delimita el alcance: cuando la calibración del régimen no representa la era OOS del activo (software 2000–2020 → bitcoin-proxy 2020+), ningún componente de STRATA puede corregirlo. Es un contraejemplo defendible del marco.

Panel actualizado
Correlación (calidad B&H, ventaja STRATA) = −0,54 (más fuerte que −0,51 con 5).
M8 supera a B&H en 4/7 (UNG, TSLA, BAC, NVDA); a la mejor clásica en 2/7 (UNG, TSLA).
Estado
Notebook 96 celdas, nbconvert verde, cross-check verde (Δ 3,27e-7), pytest 106 verde.
Lecturas UNG/MSTR escritas con cifras reales; BITACORA entrada [2026-05-26]; memoria actualizada.
Sin commitear. Faltan IWM/KRE/AMD/INTC/PYPL por caché si quieres completar los 10.
README.md
no veo lo ultimo en el notebook
Probablemente el IDE tenía abierto el .ipynb antes de mi edición y no se ha recargado. Voy a verificar que el archivo en disco tiene UNG y MSTR, y revisar el notebooks/README.md que has abierto.

Bash
IN
cd /Users/Raquel/Desktop/tradingSupervisor_ML_IA_messy
echo "=== notebook en disco — mtime + tamaño ==="
ls -la --time-style=+'%H:%M:%S' notebooks/strata_tfg.ipynb 2>/dev/null || stat -f '%Sm %z bytes  %N' notebooks/strata_tfg.ipynb
echo
echo "=== ¿están UNG y MSTR? ==="
.venv/bin/python - <<'PY'
import json
nb=json.load(open("notebooks/strata_tfg.ipynb"))
for i,c in enumerate(nb["cells"]):
    s="".join(c["source"]).lstrip()
    if c["cell_type"]=="markdown" and s.startswith(("## ","### ")):
        h=s.splitlines()[0]
        if any(k in h for k in("UNG","MSTR","16.","17.","18.","19.")):
            print(f"  c{i:3}  {h[:80]}")
PY
echo
echo "=== ¿hay notebooks/README.md? ==="
ls notebooks/README.md 2>/dev/null && echo "--- contenido ---" && cat notebooks/README.md

OUT
=== notebook en disco — mtime + tamaño ===
May 26 08:05:34 2026 3556426 bytes  notebooks/strata_tfg.ipynb

=== ¿están UNG y MSTR? ===
  c 70  ## 16. Sexto activo: UNG (gas natural, B&H catastrófico por *contango*)
  c 72  ### 16.1 Comparativa M1–M9 sobre UNG
  c 79  ## 17. Séptimo activo: MSTR (proxy de bitcoin, calibración dominada por la era p
  c 81  ### 17.1 Comparativa M1–M9 sobre MSTR
  c 88  ## 18. Panel multi-activo: la ventaja condicional de STRATA
  c 92  ### 18.1 Significancia por activo: Diebold-Mariano y Deflated Sharpe
  c 94  ### 18.2 Lectura del panel
  c 95  ## 19. Reproducibilidad

=== ¿hay notebooks/README.md? ===
notebooks/README.md
--- contenido ---
# Notebooks

Esta carpeta aloja el **notebook atómico end-to-end** del TFG: un único cuaderno autocontenido
que se ejecuta de arriba a abajo y reproduce todo el análisis, parametrizado por un solo `TICKER`
(`SPY`, `NVDA`, o cualquier activo con caché de agente). El contrato (qué recalcula, qué lee de
caché, qué descarga) está en [`../docs/notebook_contract.md`](../docs/notebook_contract.md).

`_archive/` conserva los 4 notebooks temáticos previos (calibración, comparativa, STRATA,
extensión NVDA) como referencia; quedan superados por el notebook único.
El notebook en disco está bien y completo:

Modificado a las 08:05 de hoy (después de la última re-ejecución).
Tiene ## 16. UNG, ## 17. MSTR, panel reubicado en ## 18, reproducibilidad en ## 19.
El IDE no recargó el archivo. Lo más probable: tienes el .ipynb abierto desde antes y VS Code/Jupyter te muestra la versión cacheada. Cierra la pestaña del notebook y vuélvelo a abrir (o Cmd⇧P → "Revert File"). Si te sale un aviso "el archivo ha cambiado en disco", acepta recargar.

Después deberías ver las nuevas secciones:


## 14. Cuarto activo: TSLA …
## 15. Quinto activo: XLE …
## 16. Sexto activo: UNG (gas natural, B&H catastrófico por *contango*)
   ### 16.1 Comparativa M1–M9 sobre UNG
## 17. Séptimo activo: MSTR (proxy de bitcoin, …)
   ### 17.1 Comparativa M1–M9 sobre MSTR
## 18. Panel multi-activo: la ventaja condicional de STRATA
   ### 18.1 Significancia por activo: …
   ### 18.2 Lectura del panel
## 19. Reproducibilidad
(El notebooks/README.md que has abierto es solo el README de la carpeta, no afecta — no hace falta tocarlo.)

strata_tfg.ipynb
guarda toda la conversacion y almacenala en chats/organize_notebook.md
Bash
IN
cd /Users/Raquel/Desktop/tradingSupervisor_ML_IA_messy
ls chats/ 2>/dev/null && echo "--- existing chat files ---" && wc -l chats/*.md 2>/dev/null

OUT
ENOSPC: no space left on device, open '/private/tmp/claude-502/-Users-Raquel-Desktop-tradingSupervisor-ML-IA-messy/efb5368f-c860-4660-9dc5-1cc560673695/tasks/b8k4dhnj1.output'
El disco está al 100 % (205 MB libres) — por eso Bash falla al crear su fichero de output en /private/tmp. Voy a escribir el chat directamente con Write (no usa /private/tmp).

Write organize_notebook.md
209 lines
Guardado en chats/organize_notebook.md. Es un log completo del hilo desde 2026-05-21 a hoy: RAM re-signed, marco teórico + explicación, TSLA/XLE + DM en todos, Ret@σBH + decisión diferida, UNG/MSTR + el contraejemplo MSTR, la saga del watchdog/caffeinate, y una nota al final con la continuación posterior (SMCI/ROKU/MARA) que documenta la memoria.

Aviso importante: sigues con el disco al 100 % (205 MB libres de 460 GB). Eso bloquea el caching de SMCI/ROKU/MARA y también hace fallar mis Bash (necesitan /private/tmp). Cuando liberes unos cuantos GB, la memoria panel-multiactivo-status tiene el comando exacto para reanudar el driver con caffeinate + watchdog.