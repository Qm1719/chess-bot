#!/usr/bin/env bash
set -euo pipefail

# Run ops.selftrain with env loaded, suitable for cron.
# Intended usage on VPS:
#   chmod +x /path/to/submission\ v5/ops/run_selftrain_cron.sh
#   crontab -e
#   30 0 * * * /path/to/submission\ v5/ops/run_selftrain_cron.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
V5_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${V5_ROOT}"

# Load env vars (tokens, dirs, caps). Prefer selftrain.env, fallback to .env.
if [[ -f "${V5_ROOT}/selftrain.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${V5_ROOT}/selftrain.env"
  set +a
elif [[ -f "${V5_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${V5_ROOT}/.env"
  set +a
fi

mkdir -p "${V5_ROOT}/data"

LOG="${V5_ROOT}/data/selftrain_cron.log"
echo "[$(date -Is)] selftrain cron start" >> "${LOG}"

# Use python3 by default; override with SELFTRAIN_PYTHON in env if needed.
PY="${SELFTRAIN_PYTHON:-python3}"
${PY} -m ops.selftrain >> "${LOG}" 2>&1

echo "[$(date -Is)] selftrain cron end (exit=$?)" >> "${LOG}"
