#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TASK_ID="${TASK_ID:?TASK_ID is required}"
if [[ ! "${TASK_ID}" =~ ^([1-9]|1[0-9]|2[0-6])$ ]]; then
  echo "TASK_ID must be in 1..26; got ${TASK_ID}" >&2
  exit 1
fi
SEED="${SEED:-104}"
MAX_STEPS="${MAX_STEPS:-2500}"
EXPECTED_UNIX_USER="${EXPECTED_UNIX_USER:-$(whoami)}"
EXCLUDE_NODES="${EXCLUDE_NODES:-ACD1-8,ACD1-11,ACD1-31,ACD1-54}"

VLA_CHECKPOINT="${VLA_CHECKPOINT:-/data/user/hlei573/openpi/checkpoints/pi05_libero_robomemarena_fullvlm_v2_noflip_dataset/fullvlm_v2_robomemarena_noflip_v2_bs128_4gpu_20260507_183338/35999}"
VLM_ADAPTER="${VLM_ADAPTER:-${EXP_ROOT}/runtime_assets/high_vlm_v2_26tasks_lora_r32_final_adapter}"
VLM_BASE="${VLM_BASE:-/data/user/jwen341/model_lib/Qwen3-VL-8B-Instruct}"
TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH:-${EXP_ROOT}/runtime_assets/libero_fork/libero}"
EVAL_PY="${EVAL_PY:-/data/user/hlei573/openpi_inference/.venv/bin/python}"
RUN_ID="${RUN_ID:-task${TASK_ID}_seed${SEED}_highvlmv2_cc156e5_$(date +%Y%m%d_%H%M%S)}"
LAUNCH_LOG_DIR="${EXP_ROOT}/records/launcher_logs"
PROBE_DIR="${EXP_ROOT}/records/probes/${RUN_ID}"
LAUNCH_LOG="${LAUNCH_LOG_DIR}/${RUN_ID}.log"

if [[ "$(whoami)" != "${EXPECTED_UNIX_USER}" ]]; then
  echo "launch expected ${EXPECTED_UNIX_USER}, got $(whoami)" >&2
  exit 1
fi
if [[ ! -w "${EXP_ROOT}/records" || ! -w "${EXP_ROOT}/runs" ]]; then
  echo "shared experiment outputs are not writable by $(whoami)" >&2
  exit 1
fi

mkdir -p "${LAUNCH_LOG_DIR}" "${PROBE_DIR}"
FROZEN_SCRIPT_DIR="${PROBE_DIR}/frozen_scripts"
mkdir -p "${FROZEN_SCRIPT_DIR}"
cp "${SCRIPT_DIR}/probe_gpu_environment.sh" "${FROZEN_SCRIPT_DIR}/probe_gpu_environment.sh"
cp "${SCRIPT_DIR}/run_all26_one_ep_inside_allocation.sh" \
  "${FROZEN_SCRIPT_DIR}/run_all26_one_ep_inside_allocation.sh"
cp "${EXP_ROOT}/extensions/eval_all26_physical_two_call.py" \
  "${FROZEN_SCRIPT_DIR}/eval_all26_physical_two_call.py"
cp "${EXP_ROOT}/extensions/eval_two_call_prompt_commit.py" \
  "${FROZEN_SCRIPT_DIR}/eval_two_call_prompt_commit.py"
chmod 755 "${FROZEN_SCRIPT_DIR}"/*.sh "${FROZEN_SCRIPT_DIR}"/*.py
sha256sum \
  "${FROZEN_SCRIPT_DIR}/probe_gpu_environment.sh" \
  "${FROZEN_SCRIPT_DIR}/run_all26_one_ep_inside_allocation.sh" \
  "${FROZEN_SCRIPT_DIR}/eval_all26_physical_two_call.py" \
  "${FROZEN_SCRIPT_DIR}/eval_two_call_prompt_commit.py" \
  > "${FROZEN_SCRIPT_DIR}/SHA256SUMS"
FROZEN_PROBE="${FROZEN_SCRIPT_DIR}/probe_gpu_environment.sh"
FROZEN_RUNNER="${FROZEN_SCRIPT_DIR}/run_all26_one_ep_inside_allocation.sh"
FROZEN_EVALUATOR="${FROZEN_SCRIPT_DIR}/eval_all26_physical_two_call.py"
FROZEN_TWO_CALL="${FROZEN_SCRIPT_DIR}/eval_two_call_prompt_commit.py"

exec > >(tee -a "${LAUNCH_LOG}") 2>&1
printf 'launch_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'unix_user=%s\n' "$(whoami)"
printf 'task_id=%s\n' "${TASK_ID}"
printf 'seed=%s\n' "${SEED}"
printf 'max_steps=%s\n' "${MAX_STEPS}"
printf 'exclude_nodes=%s\n' "${EXCLUDE_NODES}"
printf 'run_id=%s\n' "${RUN_ID}"

export EXP_ROOT VLA_CHECKPOINT VLM_ADAPTER VLM_BASE TARGET_LIBERO_PATH EVAL_PY
export PYTHONNOUSERSITE=1

PARTITIONS=(acd_u acd_ue emergency_acd)
SRUN_EXCLUDE=()
if [[ -n "${EXCLUDE_NODES}" ]]; then
  SRUN_EXCLUDE=("--exclude=${EXCLUDE_NODES}")
fi
SELECTED_PARTITION=""

run_probe() {
  local label="$1"
  local gpus="$2"
  local cpus="$3"
  local mem="$4"
  local time_limit="$5"
  shift 5
  local partition request_log rc
  SELECTED_PARTITION=""
  for partition in "${PARTITIONS[@]}"; do
    request_log="${PROBE_DIR}/${label}_${partition}.log"
    echo "probe label=${label} partition=${partition} gpus=${gpus} cpus=${cpus} mem=${mem}"
    set +e
    srun \
      --immediate=20 \
      --partition="${partition}" \
      --nodes=1 \
      --ntasks=1 \
      --gres="gpu:${gpus}" \
      --cpus-per-task="${cpus}" \
      --mem="${mem}" \
      --time="${time_limit}" \
      "${SRUN_EXCLUDE[@]}" \
      --job-name="hv2_t${TASK_ID}_${label}_${RUN_ID: -6}" \
      bash -lc 'exec "$@"' bash "$@" \
      > >(tee -a "${request_log}") 2>&1
    rc=$?
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      SELECTED_PARTITION="${partition}"
      echo "probe passed label=${label} partition=${partition}"
      return 0
    fi
    echo "probe unavailable label=${label} partition=${partition} rc=${rc}"
  done
  echo "all partitions rejected probe label=${label}" >&2
  return 1
}

run_probe \
  one_gpu_probe 1 4 81920M 00:05:00 \
  env EXPECTED_GPU_COUNT=1 EXP_ROOT="${EXP_ROOT}" VLA_CHECKPOINT="${VLA_CHECKPOINT}" \
    VLM_ADAPTER="${VLM_ADAPTER}" VLM_BASE="${VLM_BASE}" \
    TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH}" EVAL_PY="${EVAL_PY}" \
    "${FROZEN_PROBE}"
ONE_GPU_PARTITION="${SELECTED_PARTITION}"

run_probe \
  two_gpu_shape_probe 2 16 163840M 00:05:00 \
  env EXPECTED_GPU_COUNT=2 EXP_ROOT="${EXP_ROOT}" VLA_CHECKPOINT="${VLA_CHECKPOINT}" \
    VLM_ADAPTER="${VLM_ADAPTER}" VLM_BASE="${VLM_BASE}" \
    TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH}" EVAL_PY="${EVAL_PY}" \
    "${FROZEN_PROBE}"
FORMAL_PARTITION="${SELECTED_PARTITION}"

echo "formal request partition=${FORMAL_PARTITION} task=${TASK_ID} seed=${SEED}"
srun \
  --partition="${FORMAL_PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:2 \
  --cpus-per-task=16 \
  --mem=163840M \
  --time=08:00:00 \
  "${SRUN_EXCLUDE[@]}" \
  --job-name="hv2_t${TASK_ID}_formal_${RUN_ID: -6}" \
  bash -lc 'exec "$@"' bash \
  env TASK_ID="${TASK_ID}" RUN_ID="${RUN_ID}" VLA_CHECKPOINT="${VLA_CHECKPOINT}" \
    SEED="${SEED}" MAX_STEPS="${MAX_STEPS}" EXPECTED_UNIX_USER="${EXPECTED_UNIX_USER}" \
    VLM_ADAPTER="${VLM_ADAPTER}" VLM_BASE="${VLM_BASE}" \
    TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH}" EVAL_PY="${EVAL_PY}" \
    EVALUATOR="${FROZEN_EVALUATOR}" TWO_CALL_EXTENSION="${FROZEN_TWO_CALL}" \
    CONTROLLER_STABLE_VLM_CALLS=2 HV2_EXP_ROOT="${EXP_ROOT}" \
    EXP_ROOT="${EXP_ROOT}" AUDIT_SCRIPT_DIR="${SCRIPT_DIR}" "${FROZEN_RUNNER}"

echo "launcher complete task=${TASK_ID} seed=${SEED} run_id=${RUN_ID} one_gpu_partition=${ONE_GPU_PARTITION} formal_partition=${FORMAL_PARTITION}"
