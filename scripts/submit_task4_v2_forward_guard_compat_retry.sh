#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="task4_v2_seed108_retry_${STAMP}"
LOG="${ROOT}/logs/${SESSION}.log"

tmux new-session -d -s "${SESSION}" \
  "bash -lc 'srun --immediate=600 -p acd_ue --exclude=ACD1-1 --gres=gpu:2 -c 16 --mem=245760M --time=00:01:00 --job-name=${SESSION}_probe bash -lc \"hostname\" && srun -p acd_ue --exclude=ACD1-1 --gres=gpu:2 -c 16 --mem=245760M --job-name=${SESSION} bash -lc \"cd ${ROOT} && STAMP=${STAMP} SEED=108 NUM_TRIALS=1 PORT=9657 bash scripts/run_task4_v2_forward_guard_compat_1ep.sh\" 2>&1 | tee -a ${LOG}'"

printf 'session=%s\nlog=%s\n' "${SESSION}" "${LOG}"
