#!/usr/bin/env bash
# Watcher de progreso -> Telegram. Uso: tg_watch.sh <TOKEN> <CHAT_ID>
# Manda update cada 5 min SOLO si cambió algo, y un mensaje final al terminar Optuna.
T="$1"; C="$2"
send(){ curl -s "https://api.telegram.org/bot$T/sendMessage" --data-urlencode "chat_id=$C" --data-urlencode "text=$1" >/dev/null; }
LAST=""
while true; do
  P=$(grep -c "^=== " /tmp/panel.log 2>/dev/null || echo 0)
  PR=$(pgrep -f automl_m10.py >/dev/null && echo run || echo done)
  O=$(grep -c "^trial" /tmp/optuna.log 2>/dev/null || echo 0)
  B=$(grep -oE "acc=[0-9.]+" /tmp/optuna.log 2>/dev/null | sort -t= -k2 -n | tail -1)
  Z=$(grep -c BATE /tmp/optuna.log 2>/dev/null || echo 0)
  if grep -q "^BEST:" /tmp/optuna.log 2>/dev/null; then
     send "✅ Optuna TERMINADO. $(grep '^BEST:' /tmp/optuna.log | tail -1) · trials que baten ZeroR: $Z/$O (exploratorio/p-hacking, no reportable)"
     break
  fi
  CUR="panel $P/15($PR) · optuna $O/25 best=$B · baten ZeroR $Z"
  if [ "$CUR" != "$LAST" ]; then send "📈 $CUR"; LAST="$CUR"; fi
  sleep 300
done
