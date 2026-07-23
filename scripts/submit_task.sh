#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723"
TASK_ID="${1:?usage: submit_task.sh TASK_ID [NUM_TRIALS] [SEED]}"
NUM_TRIALS="${2:-20}"
SEED="${3:-104}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="nc6_t${TASK_ID}_${STAMP}"
JOB="nc6_t${TASK_ID}_${STAMP}"
LOG="${ROOT}/logs/${SESSION}.log"
PORT="$((9400 + TASK_ID))"

tmux new-session -d -s "${SESSION}" \
  "bash -lc 'srun -p acd_u --gres=gpu:2 -c 16 --mem=245760M --job-name=${JOB} bash -lc \"${ROOT}/scripts/run_task_20ep.sh ${TASK_ID} ${NUM_TRIALS} ${SEED} ${PORT}\" 2>&1 | tee -a ${LOG}'"

echo "${SESSION}" | tee "${ROOT}/logs/task${TASK_ID}.session"
echo "${LOG}" | tee "${ROOT}/logs/task${TASK_ID}.launcher_log"
