#!/usr/bin/env bash
# Run the data_loader script and capture logs in artifacts/logs

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_ACTIVATE="${REPO_ROOT}/.venv/bin/activate"
LOG_DIR="${REPO_ROOT}/artifacts/logs"
LOG_FILE="${LOG_DIR}/data_loader.log"
DATA_LOADER="${REPO_ROOT}/src/data_loader.py"

# Activate the virtual environment if it exists
if [[ -f "${VENV_ACTIVATE}" ]]; then
  source "${VENV_ACTIVATE}"
fi

mkdir -p "${LOG_DIR}"
python "${DATA_LOADER}" >> "${LOG_FILE}" 2>&1
