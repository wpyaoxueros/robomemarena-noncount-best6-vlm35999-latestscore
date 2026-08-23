#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-/data/user/hlei573/openpi/checkpoints/pi05_libero_robomemarena_fullvlm_v2_noflip_dataset/fullvlm_v2_robomemarena_noflip_v2_bs128_4gpu_20260507_183338/35999}"
VLM_ADAPTER="${VLM_ADAPTER:-${EXP_ROOT}/runtime_assets/high_vlm_v2_26tasks_lora_r32_final_adapter}"
VLM_BASE="${VLM_BASE:-/data/user/jwen341/model_lib/Qwen3-VL-8B-Instruct}"
TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH:-${EXP_ROOT}/runtime_assets/libero_fork/libero}"
EVAL_PY="${EVAL_PY:-/data/user/hlei573/openpi_inference/.venv/bin/python}"
RUN_ID="${RUN_ID:-task1_gt_replay_runtime_diag_$(date +%Y%m%d_%H%M%S)}"
PROBE_DIR="${EXP_ROOT}/records/probes/${RUN_ID}"
LAUNCH_LOG="${EXP_ROOT}/records/launcher_logs/${RUN_ID}.log"
PARTITIONS=(acd_u acd_ue emergency_acd)

[[ "$(whoami)" == "hzhang061" ]] || { echo "launch from hzhang061" >&2; exit 1; }
[[ -w "${EXP_ROOT}/records" && -w "${EXP_ROOT}/runs" ]] || { echo "shared outputs are not writable" >&2; exit 1; }

mkdir -p "${PROBE_DIR}/frozen_scripts" "$(dirname "${LAUNCH_LOG}")"
cp "${SCRIPT_DIR}/probe_gpu_environment.sh" "${PROBE_DIR}/frozen_scripts/"
cp "${SCRIPT_DIR}/run_task1_gt_replay_runtime_diagnostic.sh" "${PROBE_DIR}/frozen_scripts/"
chmod 755 "${PROBE_DIR}/frozen_scripts/"*.sh
sha256sum "${PROBE_DIR}/frozen_scripts/"*.sh > "${PROBE_DIR}/frozen_scripts/SHA256SUMS"
FROZEN_PROBE="${PROBE_DIR}/frozen_scripts/probe_gpu_environment.sh"
FROZEN_RUNNER="${PROBE_DIR}/frozen_scripts/run_task1_gt_replay_runtime_diagnostic.sh"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

printf 'launch_utc=%s\nunix_user=%s\nrun_id=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(whoami)" "${RUN_ID}"
export EXP_ROOT VLA_CHECKPOINT VLM_ADAPTER VLM_BASE TARGET_LIBERO_PATH EVAL_PY PYTHONNOUSERSITE=1

max_mem_mb() {
  sinfo -N -h -p "$1" -o '%m' | awk '$1 ~ /^[0-9]+$/ {if ($1 > max) max=$1} END {if (max > 0) print max; else exit 1}'
}

run_with_escalation() {
  local label="$1" time_limit="$2"
  shift 2
  local partition max_mem mem_mb marker log rc
  for partition in "${PARTITIONS[@]}"; do
    max_mem="$(max_mem_mb "${partition}")"
    for mem_mb in "${max_mem}" 163840; do
      marker="${PROBE_DIR}/${label}_${partition}_${mem_mb}M.allocated"
      log="${PROBE_DIR}/${label}_${partition}_${mem_mb}M.log"
      set +e
      srun --partition="${partition}" --nodes=1 --ntasks=1 --gres=gpu:1 \
        --cpus-per-task=12 --mem="${mem_mb}M" --time="${time_limit}" \
        --job-name="hv2_t1_${label}_${RUN_ID: -6}" --immediate=20 \
        bash -lc "printf 'allocated\n' > '${marker}'; exec \"\$@\"" bash "$@" \
        > >(tee -a "${log}") 2>&1
      rc=$?
      set -e
      if [[ -e "${marker}" ]]; then
        [[ "${rc}" -eq 0 ]] || return "${rc}"
        return 0
      fi
    done
  done
  return 1
}

run_with_escalation one_gpu_probe 00:05:00 \
  env EXPECTED_GPU_COUNT=1 EXP_ROOT="${EXP_ROOT}" VLA_CHECKPOINT="${VLA_CHECKPOINT}" \
    VLM_ADAPTER="${VLM_ADAPTER}" VLM_BASE="${VLM_BASE}" \
    TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH}" EVAL_PY="${EVAL_PY}" "${FROZEN_PROBE}"

run_with_escalation formal_shape_probe 00:05:00 \
  env EXPECTED_GPU_COUNT=1 EXP_ROOT="${EXP_ROOT}" VLA_CHECKPOINT="${VLA_CHECKPOINT}" \
    VLM_ADAPTER="${VLM_ADAPTER}" VLM_BASE="${VLM_BASE}" \
    TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH}" EVAL_PY="${EVAL_PY}" "${FROZEN_PROBE}"

run_with_escalation runtime_diag 02:00:00 \
  env RUN_ID="${RUN_ID}" EXP_ROOT="${EXP_ROOT}" AUDIT_SCRIPT_DIR="${SCRIPT_DIR}" \
    VLM_ADAPTER="${VLM_ADAPTER}" VLM_BASE="${VLM_BASE}" \
    TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH}" EVAL_PY="${EVAL_PY}" "${FROZEN_RUNNER}"

echo "diagnostic launcher complete run_id=${RUN_ID}"
