#!/bin/zsh
cd "$(dirname "$0")"
if [[ -x "${HOME}/miniforge3/bin/python" ]]; then
  PY="${HOME}/miniforge3/bin/python"
else
  PY="$(command -v python3)"
fi
if [[ -d "${PWD}/vendor" ]]; then
  export PYTHONPATH="${PWD}/vendor${PYTHONPATH:+:$PYTHONPATH}"
fi
exec "$PY" -m streamlit run app.py --server.port 8501
