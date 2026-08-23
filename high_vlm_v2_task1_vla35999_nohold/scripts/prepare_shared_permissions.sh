#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${EXP_ROOT}/records" "${EXP_ROOT}/runtime_assets" "${EXP_ROOT}/runs"
CURRENT_USER="$(id -un)"

for root in \
  "${EXP_ROOT}/records" \
  "${EXP_ROOT}/runtime_assets" \
  "${EXP_ROOT}/runs"; do
  find "${root}" -type d -user "${CURRENT_USER}" -exec chmod 2775 {} +
  find "${root}" -type f -user "${CURRENT_USER}" -exec chmod g+rw {} +
done

echo "shared experiment outputs prepared for group $(stat -c %G "${EXP_ROOT}")"
