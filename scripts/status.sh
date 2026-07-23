#!/usr/bin/env bash
set -euo pipefail
ROOT="/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723"

printf 'task\tcompleted\tsuccess\tavg_score_pct\tlatest_run\n'
for task in 4 5 11 14 17 19; do
  run="$(find "${ROOT}/outputs/task${task}" -mindepth 1 -maxdepth 1 -type d -name '*latestd9*' 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "${run}" ]]; then
    printf '%s\t0\t0\t0.0\t-\n' "${task}"
    continue
  fi
  mapfile -t scores < <(rg --no-filename '\[OFFICIAL_SCORE\]' "${run}/task${task}"/ep*/sync_vlm.log 2>/dev/null || true)
  completed="${#scores[@]}"
  if [[ "${completed}" -eq 0 ]]; then
    printf '%s\t0\t0\t0.0\t%s\n' "${task}" "${run}"
    continue
  fi
  printf '%s\n' "${scores[@]}" | awk -v task="${task}" -v run="${run}" '
    {
      score=0; success=0
      if (match($0, /average_score_pct=[0-9.]+/)) {
        score=substr($0, RSTART+18, RLENGTH-18)+0
      }
      if ($0 ~ /stage_success=1/) success=1
      n++; sum+=score; ok+=success
    }
    END { printf "%s\t%d\t%d\t%.1f\t%s\n", task, n, ok, sum/n, run }
  '
done
