#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# These are the three release anchors recorded in the historical Task4 run.
# The VLM still supplies every next prompt; anchors only restore the robot state
# at an accepted hold/release boundary.
export TASK4_CLOSETOP_TO_OPENMIDDLE_TELEPORT=1
export TASK4_CLOSEMIDDLE_TO_OPENBOTTOM_TELEPORT=1
export TASK4_CLOSEBOTTOM_TO_OPENTOPAGAIN_TELEPORT=1
exec bash "${ROOT}/scripts/run_task_20ep.sh" 4 "${NUM_TRIALS:-1}" "${SEED:-108}" "${PORT:-9656}"
