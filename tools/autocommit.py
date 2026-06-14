#!/usr/bin/env python3
"""Auto-commit + push de la tesis.

Vigila la carpeta y, cuando detecta cambios guardados, hace ``git commit`` y
``git push`` automáticamente a GitHub (en la rama actual). Pensado para escribir
la tesis en local sin pensar en git: guardas el .tex y se sube solo.

Uso:
    python tools/autocommit.py            # vigila solo tesis/ (recomendado)
    python tools/autocommit.py --all      # vigila todo el repo
    python tools/autocommit.py --interval 30

Déjalo corriendo en una terminal mientras escribes; Ctrl+C para parar.
Respeta .gitignore: NO sube .env ni las muestras de estilo (están ignoradas).
"""
from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="vigilar todo el repo (por defecto solo tesis/)")
    ap.add_argument("--interval", type=int, default=15, help="segundos entre comprobaciones")
    args = ap.parse_args()

    paths = [] if args.all else ["tesis/"]
    rama = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    ambito = "todo el repo" if args.all else "tesis/"
    print(f"[autocommit] vigilando {ambito} en la rama '{rama}'. Guarda y se sube solo. Ctrl+C para parar.")

    while True:
        try:
            if git("status", "--porcelain", *paths).stdout.strip():
                git("add", *(["-A"] if args.all else paths))
                if git("diff", "--cached", "--quiet").returncode != 0:  # hay algo staged
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                    git("commit", "-m", f"wip(tesis): autosave {ts}")
                    push = git("push")
                    estado = "subido a GitHub" if push.returncode == 0 else f"push pendiente ({push.stderr.strip()[:70]})"
                    print(f"[autocommit] {ts}  commit hecho · {estado}")
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[autocommit] parado. Tus últimos cambios guardados ya están commiteados.")
            return
        except Exception as e:  # red caída, etc.: reintenta
            print(f"[autocommit] aviso: {e}; reintento en {args.interval}s")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
