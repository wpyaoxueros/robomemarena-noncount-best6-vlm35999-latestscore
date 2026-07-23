#!/usr/bin/env bash
set -euo pipefail
ROOT="/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723"
for task in 4 5 11 14 17 19; do
  ssh zzhang510@localhost "${ROOT}/scripts/submit_task.sh ${task} 20 104"
  sleep 2
done
