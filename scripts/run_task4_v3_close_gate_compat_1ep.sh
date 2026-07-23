#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TASK4_DRAWER_FORWARD_ADVANCE_GUARD=0
export DRAWER_CLOSE_REQUIRE_STAGE=0
exec bash "${ROOT}/scripts/run_task_20ep.sh" 4 "${NUM_TRIALS:-1}" "${SEED:-108}" "${PORT:-9656}"
