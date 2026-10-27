#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f "$HOME/myenv/bin/activate" ]]; then
  source "$HOME/myenv/bin/activate"
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
else
  echo "Activa un entorno Python e instala requirements.txt antes de iniciar."
  exit 1
fi
if ! python -c 'import flask, pandas, numpy, sklearn, joblib' >/dev/null 2>&1; then
  echo "Faltan dependencias. Ejecuta: python -m pip install -r requirements.txt"
  exit 1
fi
exec python app.py "$@"
