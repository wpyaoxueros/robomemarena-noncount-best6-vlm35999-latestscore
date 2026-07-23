#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="task4_v3_seed108_${STAMP}"
LOG="${ROOT}/logs/${SESSION}.log"

tmux new-session -d -s "${SESSION}" \
  "bash -lc 'srun -p acd_ue --exclude=ACD1-1 --gres=gpu:2 -c 16 --mem=245760M --job-name=${SESSION} bash -lc \"cd ${ROOT} && STAMP=${STAMP} SEED=108 NUM_TRIALS=1 PORT=9658 bash scripts/run_task4_v3_close_gate_compat_1ep.sh\" 2>&1 | tee -a ${LOG}'"

printf 'session=%s\nlog=%s\n' "${SESSION}" "${LOG}"
