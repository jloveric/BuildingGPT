#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLOUDCOMPY_ROOT="/opt/cloudcompy/current"
PY312_BIN="${REPO_ROOT}/third_party/python312/python/bin"
VENV_ACTIVATE="${REPO_ROOT}/.venv/bin/activate"

if [[ ! -d "${CLOUDCOMPY_ROOT}/cloudComPy" ]]; then
  echo "Missing CloudComPy runtime: ${CLOUDCOMPY_ROOT}/cloudComPy" >&2
  echo "Install CloudComPy under /opt/cloudcompy/current first." >&2
  exit 1
fi

# Activate CloudComPy runtime env directly (avoid envCloudComPy.sh basename bug).
export PATH="${CLOUDCOMPY_ROOT}/bin:${CLOUDCOMPY_ROOT}/cloudComPy:${PATH}"
if [[ -x "${PY312_BIN}/python3.12" ]]; then
  export PATH="${PY312_BIN}:${PATH}"
  export UV_PYTHON="${PY312_BIN}/python3.12"
fi
export PYTHONPATH="${CLOUDCOMPY_ROOT}:${CLOUDCOMPY_ROOT}/cloudComPy:${CLOUDCOMPY_ROOT}/cloudComPy/doc/PythonAPI_test:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CLOUDCOMPY_ROOT}/cloudComPy:${CLOUDCOMPY_ROOT}/cloudComPy/plugins/CC:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="${CLOUDCOMPY_ROOT}/cloudComPy/plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="${CLOUDCOMPY_ROOT}/cloudComPy/plugins/platforms"
export QT_QPA_PLATFORM=offscreen
export CLOUDCOMPY_ENV_ACTIVATED=1
export LC_NUMERIC=C

if [[ -f "${VENV_ACTIVATE}" ]]; then
  source "${VENV_ACTIVATE}"
fi

cd "${REPO_ROOT}"
echo "Activated CloudComPy environment in ${REPO_ROOT}"
echo "Python: $(python -V 2>&1)"
echo "Use 'exit' to leave this shell."

exec bash --noprofile --norc -i
