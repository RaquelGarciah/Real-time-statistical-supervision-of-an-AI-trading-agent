#!/usr/bin/env bash
# Corre Optuna sobre la semilla de AutoML para un ticker y AVISA por Telegram al terminar.
# Uso: run_optuna_seed.sh <TICKER> <TOKEN> <CHAT_ID>
TK="$1"; T="$2"; C="$3"
cd ~/STRATA_kit || exit 1
. .venv/bin/activate
OUT="outputs/experiments/automl_runs/optuna_SEED_phacking_${TK}_mm20_GBM-SE_emb1_AUC_x86.json"
python experiments/optuna_automl_seed_spy.py --ticker "$TK" --trials 25 --out "$OUT" > "/tmp/optuna_${TK}.log" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  curl -s "https://api.telegram.org/bot$T/sendMessage" --data-urlencode "chat_id=$C" \
    --data-urlencode "text=⚠️ Optuna $TK FALLÓ (exit $RC). Revisa /tmp/optuna_${TK}.log en la VM." >/dev/null
  exit $RC
fi
RES=$(python - "$OUT" "$TK" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); tk = sys.argv[2]
ts = d["trials"]; b = max(ts, key=lambda r: r["acc"])
beat = sum(1 for r in ts if r["acc"] > r["zeror"])
accs = sorted(r["acc"] for r in ts)
print(f"✅ Optuna semilla {tk} TERMINADO (mm20 GBM+SE, x86, {len(ts)} trials).\n"
      f"Mejor: seed={b['seed']} acc={b['acc']:.4f} vs ZeroR={b['zeror']:.4f} -> bate={b['acc']>b['zeror']}.\n"
      f"Trials que baten ZeroR: {beat}/{len(ts)}. Rango {accs[0]:.4f}-{accs[-1]:.4f}.\n"
      f"(exploratorio/p-hacking, no reportable)")
PY
)
curl -s "https://api.telegram.org/bot$T/sendMessage" --data-urlencode "chat_id=$C" --data-urlencode "text=$RES" >/dev/null
