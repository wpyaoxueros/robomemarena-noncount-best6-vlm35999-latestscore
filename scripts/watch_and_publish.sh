#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723"
cd "${ROOT}"
export all_proxy=socks5://localhost:9632

while true; do
  ./scripts/status.sh > LIVE_STATUS.tsv.tmp
  mv LIVE_STATUS.tsv.tmp LIVE_STATUS.tsv

  published=0
  for task in 4 5 11 14 17 19; do
    marker="logs/task${task}.published"
    if [[ -e "${marker}" ]]; then
      published=$((published + 1))
      continue
    fi
    run="$(find "outputs/task${task}" -mindepth 1 -maxdepth 1 -type d -name '*latestd9*' | sort | tail -n 1 || true)"
    [[ -n "${run}" ]] || continue
    summary="${run}/official_task_summary.tsv"
    [[ -s "${summary}" ]] || continue
    trials="$(awk -F '\t' 'NR==2 {print $2}' "${summary}")"
    [[ "${trials}" == "20" ]] || continue

    result_dir="results/task${task}"
    mkdir -p "${result_dir}"
    cp "${run}/official_episodes.tsv" "${result_dir}/"
    cp "${run}/official_task_summary.tsv" "${result_dir}/"
    cp "${run}/official_summary.json" "${result_dir}/"
    cp "${run}/run_manifest.env" "${result_dir}/"
    printf '%s\n' "${run}" > "${result_dir}/output_path.txt"
    find "${run}/videos/task${task}" -maxdepth 1 -type f -name '*.mp4' -printf '%p\n' 2>/dev/null | sort > "${result_dir}/video_paths.txt" || true

    git add "${result_dir}" LIVE_STATUS.tsv
    git commit -m "results(task${task}): publish VLM+35999 latest-score 20ep"
    git push origin main
    date -Is > "${marker}"
    published=$((published + 1))
  done

  if [[ "${published}" -eq 6 ]]; then
    exit 0
  fi
  sleep 60
done
