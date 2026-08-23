#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-/data/user/hlei573/openpi/checkpoints/pi05_libero_robomemarena_fullvlm_v2_noflip_dataset/fullvlm_v2_robomemarena_noflip_v2_bs128_4gpu_20260507_183338/35999}"
VLM_ADAPTER="${VLM_ADAPTER:-${EXP_ROOT}/runtime_assets/high_vlm_v2_26tasks_lora_r32_final_adapter}"
VLM_BASE="${VLM_BASE:-/data/user/jwen341/model_lib/Qwen3-VL-8B-Instruct}"
TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH:-${EXP_ROOT}/runtime_assets/libero_fork/libero}"
EVAL_PY="${EVAL_PY:-/data/user/hlei573/openpi_inference/.venv/bin/python}"
EVALUATOR="${EVALUATOR:-${EXP_ROOT}/source/vlm_ft/eval_three_tasks.py}"
CONTROLLER_STABLE_VLM_CALLS="${CONTROLLER_STABLE_VLM_CALLS:-1}"
CONTROL_CONTRACT="${CONTROL_CONTRACT:-synchronous VLM prompt -> VLA actions; no hold/release/anchor/oracle/GT replay}"
AUDIT_EXTENSION="${AUDIT_EXTENSION:-}"
HV2_EXP_ROOT="${HV2_EXP_ROOT:-${EXP_ROOT}}"
RUN_ID="${RUN_ID:-task1_seed104_nohold_$(date +%Y%m%d_%H%M%S)}"
LAUNCH_LOG_DIR="${EXP_ROOT}/records/launcher_logs"
PROBE_DIR="${EXP_ROOT}/records/probes/${RUN_ID}"
LAUNCH_LOG="${LAUNCH_LOG_DIR}/${RUN_ID}.log"
PARTITIONS=(acd_u acd_ue emergency_acd)

if [[ "$(whoami)" != "hzhang061" ]]; then
  echo "launch this script from the hzhang061 login shell" >&2
  exit 1
fi

if [[ ! -w "${EXP_ROOT}/records" || ! -w "${EXP_ROOT}/runs" ]]; then
  echo "shared experiment outputs are not writable by $(whoami); run prepare_shared_permissions.sh as the owner" >&2
  exit 1
fi

mkdir -p "${LAUNCH_LOG_DIR}" "${PROBE_DIR}"
FROZEN_SCRIPT_DIR="${PROBE_DIR}/frozen_scripts"
mkdir -p "${FROZEN_SCRIPT_DIR}"
cp "${SCRIPT_DIR}/probe_gpu_environment.sh" "${FROZEN_SCRIPT_DIR}/probe_gpu_environment.sh"
cp "${SCRIPT_DIR}/run_task1_seed104_nohold_inside_allocation.sh" \
  "${FROZEN_SCRIPT_DIR}/run_task1_seed104_nohold_inside_allocation.sh"
FROZEN_PROBE="${FROZEN_SCRIPT_DIR}/probe_gpu_environment.sh"
FROZEN_RUNNER="${FROZEN_SCRIPT_DIR}/run_task1_seed104_nohold_inside_allocation.sh"
FROZEN_FILES=("${FROZEN_PROBE}" "${FROZEN_RUNNER}")
if [[ -n "${AUDIT_EXTENSION}" ]]; then
  FROZEN_EXTENSION="${FROZEN_SCRIPT_DIR}/$(basename "${EVALUATOR}")"
  cp "${EVALUATOR}" "${FROZEN_EXTENSION}"
  EVALUATOR="${FROZEN_EXTENSION}"
  AUDIT_EXTENSION="${FROZEN_EXTENSION}"
  FROZEN_FILES+=("${FROZEN_EXTENSION}")
fi
chmod 755 "${FROZEN_FILES[@]}"
sha256sum "${FROZEN_FILES[@]}" > "${FROZEN_SCRIPT_DIR}/SHA256SUMS"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

printf 'launch_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'unix_user=%s\n' "$(whoami)"
printf 'run_id=%s\n' "${RUN_ID}"

export EXP_ROOT VLA_CHECKPOINT VLM_ADAPTER VLM_BASE TARGET_LIBERO_PATH EVAL_PY
export EVALUATOR CONTROLLER_STABLE_VLM_CALLS CONTROL_CONTRACT AUDIT_EXTENSION
export HV2_EXP_ROOT
export PYTHONNOUSERSITE=1

run_with_escalation() {
  local label="$1"
  local gpus="$2"
  local cpus="$3"
  local mem="$4"
  local time_limit="$5"
  shift 5
  local partition marker attempt_log rc

  for partition in "${PARTITIONS[@]}"; do
    marker="${PROBE_DIR}/${label}_${partition}.allocated"
    attempt_log="${PROBE_DIR}/${label}_${partition}.log"
    rm -f "${marker}"
    echo "request label=${label} partition=${partition} gpus=${gpus} cpus=${cpus} mem=${mem}"
    set +e
    srun \
      --partition="${partition}" \
      --nodes=1 \
      --ntasks=1 \
      --gres="gpu:${gpus}" \
      --cpus-per-task="${cpus}" \
      --mem="${mem}" \
      --time="${time_limit}" \
      --job-name="hv2_t1_${label}_${RUN_ID: -6}" \
      --immediate=20 \
      bash -lc "printf 'allocated\\n' > '${marker}'; exec \"\$@\"" bash "$@" \
      > >(tee -a "${attempt_log}") 2>&1
    rc=$?
    set -e

    if [[ -e "${marker}" ]]; then
      if [[ "${rc}" -ne 0 ]]; then
        echo "allocated command failed label=${label} partition=${partition} rc=${rc}" >&2
        return "${rc}"
      fi
      echo "request passed label=${label} partition=${partition}"
      SELECTED_PARTITION="${partition}"
      return 0
    fi
    echo "no allocation within 20 seconds on ${partition}; escalating"
  done

  echo "unable to allocate ${label} on any permitted partition" >&2
  return 1
}

EXPECTED_GPU_COUNT=1 run_with_escalation \
  one_gpu_probe 1 4 16384M 00:05:00 \
  env EXPECTED_GPU_COUNT=1 EXP_ROOT="${EXP_ROOT}" VLA_CHECKPOINT="${VLA_CHECKPOINT}" \
    VLM_ADAPTER="${VLM_ADAPTER}" VLM_BASE="${VLM_BASE}" \
    TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH}" EVAL_PY="${EVAL_PY}" \
    EVALUATOR="${EVALUATOR}" CONTROLLER_STABLE_VLM_CALLS="${CONTROLLER_STABLE_VLM_CALLS}" \
    CONTROL_CONTRACT="${CONTROL_CONTRACT}" AUDIT_EXTENSION="${AUDIT_EXTENSION}" \
    HV2_EXP_ROOT="${HV2_EXP_ROOT}" \
    "${FROZEN_PROBE}"

EXPECTED_GPU_COUNT=2 run_with_escalation \
  two_gpu_shape_probe 2 16 163840M 00:05:00 \
  env EXPECTED_GPU_COUNT=2 EXP_ROOT="${EXP_ROOT}" VLA_CHECKPOINT="${VLA_CHECKPOINT}" \
    VLM_ADAPTER="${VLM_ADAPTER}" VLM_BASE="${VLM_BASE}" \
    TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH}" EVAL_PY="${EVAL_PY}" \
    EVALUATOR="${EVALUATOR}" CONTROLLER_STABLE_VLM_CALLS="${CONTROLLER_STABLE_VLM_CALLS}" \
    CONTROL_CONTRACT="${CONTROL_CONTRACT}" AUDIT_EXTENSION="${AUDIT_EXTENSION}" \
    HV2_EXP_ROOT="${HV2_EXP_ROOT}" \
    "${FROZEN_PROBE}"

run_with_escalation \
  formal_eval 2 16 163840M 08:00:00 \
  env RUN_ID="${RUN_ID}" VLA_CHECKPOINT="${VLA_CHECKPOINT}" \
    VLM_ADAPTER="${VLM_ADAPTER}" VLM_BASE="${VLM_BASE}" \
    TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH}" EVAL_PY="${EVAL_PY}" \
    EVALUATOR="${EVALUATOR}" CONTROLLER_STABLE_VLM_CALLS="${CONTROLLER_STABLE_VLM_CALLS}" \
    CONTROL_CONTRACT="${CONTROL_CONTRACT}" AUDIT_EXTENSION="${AUDIT_EXTENSION}" \
    HV2_EXP_ROOT="${HV2_EXP_ROOT}" \
    env EXP_ROOT="${EXP_ROOT}" AUDIT_SCRIPT_DIR="${SCRIPT_DIR}" "${FROZEN_RUNNER}"

echo "launcher complete run_id=${RUN_ID} formal_partition=${SELECTED_PARTITION}"
