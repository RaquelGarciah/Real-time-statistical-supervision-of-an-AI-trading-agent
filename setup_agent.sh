#!/usr/bin/env bash
# Bootstrap del módulo agent/ para STRATA_kit.
#
# Idempotente: revisa qué falta, instala lo necesario y deja un smoke test.
# Si ya está todo en su sitio, sale verde sin tocar nada.

set -euo pipefail

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_HEDGE_FUND_REPO="https://github.com/virattt/ai-hedge-fund.git"
AI_HEDGE_FUND_HASH="e06b186510cf64e1991951da36da1a4b6ad3cead"
SUBMODULE_PATH="agent/ai_hedge_fund"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
red()    { printf "\033[0;31m%s\033[0m\n" "$*"; }
green()  { printf "\033[0;32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[0;33m%s\033[0m\n" "$*"; }
blue()   { printf "\033[0;34m%s\033[0m\n" "$*"; }

step() { echo; blue "==> $*"; }

# ---------------------------------------------------------------------------
# 1. Verificar contexto
# ---------------------------------------------------------------------------
cd "$PROJECT_ROOT"

step "1. Verificando contexto del proyecto"

if [[ ! -d "agent" ]]; then
    red "ERROR: no existe agent/. Este script asume que ya estás en STRATA_kit."
    exit 1
fi

for f in agent/wrapper.py agent/llm_client.py agent/_macro_patch.py \
         agent/_price_patch.py agent/_stats_patch.py; do
    if [[ ! -f "$f" ]]; then
        red "ERROR: falta $f"
        exit 1
    fi
done
green "  agent/ patches OK"

for f in core/macro_features.py core/data.py; do
    if [[ ! -f "$f" ]]; then
        red "ERROR: falta $f"
        exit 1
    fi
done
green "  core/ helpers OK"

mkdir -p cache/agent cache/llm
green "  cache/ directorios OK"

# ---------------------------------------------------------------------------
# 2. Submódulo ai_hedge_fund
# ---------------------------------------------------------------------------
step "2. Verificando submódulo agent/ai_hedge_fund"

if [[ -d "$SUBMODULE_PATH/.git" ]] || [[ -f "$SUBMODULE_PATH/.git" ]]; then
    current_hash=$(git -C "$SUBMODULE_PATH" rev-parse HEAD 2>/dev/null || echo "")
    if [[ "$current_hash" == "$AI_HEDGE_FUND_HASH" ]]; then
        green "  submódulo ya en hash correcto ($AI_HEDGE_FUND_HASH)"
    else
        yellow "  submódulo presente pero en hash distinto: $current_hash"
        yellow "  esperado: $AI_HEDGE_FUND_HASH"
        yellow "  haciendo checkout al hash correcto..."
        git -C "$SUBMODULE_PATH" fetch --all --tags
        git -C "$SUBMODULE_PATH" checkout "$AI_HEDGE_FUND_HASH"
        green "  submódulo ahora en hash $AI_HEDGE_FUND_HASH"
    fi
else
    yellow "  submódulo no existe — clonando..."
    # Si esto es un repo git, añadir como submódulo. Si no, clonar plain.
    if git rev-parse --git-dir >/dev/null 2>&1; then
        git submodule add "$AI_HEDGE_FUND_REPO" "$SUBMODULE_PATH" 2>/dev/null || \
            git clone "$AI_HEDGE_FUND_REPO" "$SUBMODULE_PATH"
    else
        git clone "$AI_HEDGE_FUND_REPO" "$SUBMODULE_PATH"
    fi
    git -C "$SUBMODULE_PATH" checkout "$AI_HEDGE_FUND_HASH"
    green "  submódulo clonado y en hash $AI_HEDGE_FUND_HASH"
fi

# ---------------------------------------------------------------------------
# 3. Dependencias Python del submódulo
# ---------------------------------------------------------------------------
step "3. Verificando dependencias del submódulo"

PY_CMD=".venv/bin/python"
PIP_CMD=".venv/bin/pip"

if [[ ! -x "$PY_CMD" ]]; then
    yellow "  .venv no encontrado en STRATA_kit, usando python3 del sistema"
    PY_CMD="python3"
    PIP_CMD="python3 -m pip"
fi

# Las dependencias clave de ai_hedge_fund se instalan vía pyproject.toml o requirements
if [[ -f "$SUBMODULE_PATH/pyproject.toml" ]]; then
    if $PY_CMD -c "import langchain_openai" 2>/dev/null && \
       $PY_CMD -c "import langgraph" 2>/dev/null; then
        green "  dependencias clave (langchain_openai, langgraph) ya instaladas"
    else
        yellow "  instalando dependencias del submódulo (puede tardar 2-3 min)..."
        # Intentar con poetry primero, fallback a pip
        if command -v poetry >/dev/null 2>&1; then
            (cd "$SUBMODULE_PATH" && poetry install --no-root 2>&1 | tail -5) || \
                $PIP_CMD install langchain langchain-openai langgraph pydantic
        else
            $PIP_CMD install langchain langchain-openai langgraph pydantic
        fi
        green "  dependencias instaladas"
    fi
else
    yellow "  pyproject.toml no encontrado en submódulo, instalando mínimo vía pip"
    $PIP_CMD install langchain langchain-openai langgraph pydantic
fi

# ---------------------------------------------------------------------------
# 4. Variables de entorno
# ---------------------------------------------------------------------------
step "4. Verificando .env"

if [[ ! -f ".env" ]]; then
    yellow "  .env no existe, creando plantilla..."
    cat > .env <<'EOF'
# OBLIGATORIO: clave de OpenRouter (https://openrouter.ai)
OPENROUTER_API_KEY=

# OPCIONAL: solo para enriquecer SPY con datos macro (gratis)
FINANCIAL_DATASETS_API_KEY=
EOF
    red "  RELLENA .env con tu OPENROUTER_API_KEY antes de continuar"
    exit 1
fi

# Leer la key (sin imprimirla)
if grep -qE "^OPENROUTER_API_KEY=.+" .env; then
    green "  OPENROUTER_API_KEY presente en .env"
else
    red "  OPENROUTER_API_KEY ausente o vacío en .env"
    exit 1
fi

# ---------------------------------------------------------------------------
# 5. Smoke test
# ---------------------------------------------------------------------------
step "5. Smoke test: importar agent.wrapper.run_agent"

if $PY_CMD -c "
import sys
sys.path.insert(0, '.')
try:
    from agent.wrapper import run_agent
    print('OK')
except Exception as e:
    print(f'FAIL: {type(e).__name__}: {e}')
    sys.exit(1)
" 2>&1 | tail -3; then
    green "  import OK"
else
    red "  FAIL: revisa el error arriba"
    exit 1
fi

# ---------------------------------------------------------------------------
# 6. Resumen
# ---------------------------------------------------------------------------
step "Resumen"
green "  Bootstrap completado."
echo
echo "  Para generar una decisión nueva:"
echo "    .venv/bin/python -c \"from agent.wrapper import run_agent; print(run_agent('MARA', '2026-06-21'))\""
echo
echo "  La decisión se cachea en: cache/agent/<TICKER>/<TICKER>_<date>.json"
echo "  Las llamadas LLM crudas se cachean en: cache/llm/ (por hash del prompt)"
echo
echo "  Para bulk backtest (si copiaste experiments/m5_agent_alone.py):"
echo "    .venv/bin/python -m experiments.m5_agent_alone --ticker MARA --end-date 2026-05-11"
