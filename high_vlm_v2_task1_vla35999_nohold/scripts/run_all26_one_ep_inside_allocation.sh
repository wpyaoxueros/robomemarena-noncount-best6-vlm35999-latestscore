#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="${EXP_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PACK_ROOT="$(cd "${EXP_ROOT}/.." && pwd)"
AUDIT_SCRIPT_DIR="${AUDIT_SCRIPT_DIR:-${EXP_ROOT}/scripts}"
TASK_ID="${TASK_ID:?TASK_ID is required}"
if [[ ! "${TASK_ID}" =~ ^([1-9]|1[0-9]|2[0-6])$ ]]; then
  echo "TASK_ID must be in 1..26; got ${TASK_ID}" >&2
  exit 1
fi
SEED="${SEED:-104}"
MAX_STEPS="${MAX_STEPS:-2500}"
EXPECTED_UNIX_USER="${EXPECTED_UNIX_USER:?EXPECTED_UNIX_USER is required}"
CONTROLLER_STABLE_VLM_CALLS="${CONTROLLER_STABLE_VLM_CALLS:-2}"

OPENPI_ROOT="${OPENPI_ROOT:-/data/user/hlei573/openpi}"
OPENPI_INFERENCE_ROOT="${OPENPI_INFERENCE_ROOT:-/data/user/hlei573/openpi_inference}"
VLA_PY="${VLA_PY:-${OPENPI_ROOT}/.venv/bin/python}"
EVAL_PY="${EVAL_PY:-${OPENPI_INFERENCE_ROOT}/.venv/bin/python}"
VLA_CONFIG="${VLA_CONFIG:-pi05_libero_robomemarena_fullvlm_v2_noflip_dataset}"
VLA_CHECKPOINT="${VLA_CHECKPOINT:-${OPENPI_ROOT}/checkpoints/pi05_libero_robomemarena_fullvlm_v2_noflip_dataset/fullvlm_v2_robomemarena_noflip_v2_bs128_4gpu_20260507_183338/35999}"
VLM_BASE="${VLM_BASE:-/data/user/jwen341/model_lib/Qwen3-VL-8B-Instruct}"
VLM_ADAPTER="${VLM_ADAPTER:-${EXP_ROOT}/runtime_assets/high_vlm_v2_26tasks_lora_r32_final_adapter}"
TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH:-${EXP_ROOT}/runtime_assets/libero_fork/libero}"
REFERENCE_RUNTIME="${EXP_ROOT}/source/eval_robomem/robomemarena_official/evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference/eval_tasks2_26_vlm_vla.py"
TASK_CONFIG="${EXP_ROOT}/source/eval_robomem/robomemarena_official/evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference/fullvlm_v2_26_memory_tasks.json"
EVALUATOR="${EVALUATOR:-${EXP_ROOT}/extensions/eval_all26_physical_two_call.py}"
TWO_CALL_EXTENSION="${TWO_CALL_EXTENSION:-${EXP_ROOT}/extensions/eval_two_call_prompt_commit.py}"
OFFICIAL_SCORER="${EXP_ROOT}/official_snapshot_cc156e5/evaluation_benchmark/scripts/task2_26_reference_stage.py"
OFFICIAL_SCORER_SHA256="4eb949049b3175df01e8c632a6159a4b65bf9e2a667f6cbe612132ba5e7e0b99"
OFFICIAL_BDDL_DIR="${EXP_ROOT}/official_snapshot_cc156e5/evaluation_benchmark/bddl"
OFFICIAL_SHA256SUMS="${EXP_ROOT}/official_snapshot_cc156e5/SHA256SUMS"
RUN_ID="${RUN_ID:-task${TASK_ID}_seed${SEED}_highvlmv2_cc156e5_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${EXP_ROOT}/runs/${RUN_ID}"
PORT="${PORT:-$((18000 + ${SLURM_JOB_ID:-0} % 20000))}"

if [[ "$(whoami)" != "${EXPECTED_UNIX_USER}" ]]; then
  echo "formal runner expected ${EXPECTED_UNIX_USER}, got $(whoami)" >&2
  exit 1
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "formal runner requires a Slurm allocation" >&2
  exit 1
fi
VISIBLE_GPU_COUNT="$(PYTHONNOUSERSITE=1 "${EVAL_PY}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${VISIBLE_GPU_COUNT}" -ne 2 ]]; then
  echo "formal runner requires exactly two visible GPUs; got ${VISIBLE_GPU_COUNT}" >&2
  exit 1
fi

shopt -s nullglob
TASK_BDDLS=("${OFFICIAL_BDDL_DIR}/${TASK_ID}_"*.bddl)
shopt -u nullglob
if [[ "${#TASK_BDDLS[@]}" -ne 1 ]]; then
  echo "expected exactly one Task${TASK_ID} official BDDL, got ${TASK_BDDLS[*]:-none}" >&2
  exit 1
fi
TASK_BDDL="${TASK_BDDLS[0]}"

for required in \
  "${VLA_PY}" \
  "${EVAL_PY}" \
  "${VLA_CHECKPOINT}/assets/robomemarena_fullvlm_v2_noflip_dataset_v2/norm_stats.json" \
  "${VLM_BASE}/config.json" \
  "${VLM_ADAPTER}/adapter_model.safetensors" \
  "${TARGET_LIBERO_PATH}/libero/envs" \
  "${REFERENCE_RUNTIME}" \
  "${TASK_CONFIG}" \
  "${EVALUATOR}" \
  "${TWO_CALL_EXTENSION}" \
  "${OFFICIAL_SCORER}" \
  "${OFFICIAL_SHA256SUMS}" \
  "${TASK_BDDL}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required path: ${required}" >&2
    exit 1
  fi
done

ACTUAL_SCORER_SHA256="$(sha256sum "${OFFICIAL_SCORER}" | awk '{print $1}')"
if [[ "${ACTUAL_SCORER_SHA256}" != "${OFFICIAL_SCORER_SHA256}" ]]; then
  echo "official scorer hash mismatch: ${ACTUAL_SCORER_SHA256}" >&2
  exit 1
fi
(cd "$(dirname "${OFFICIAL_SHA256SUMS}")" && sha256sum -c "$(basename "${OFFICIAL_SHA256SUMS}")")

"${AUDIT_SCRIPT_DIR}/verify_source_snapshot.sh" --against-source
RUNNER="${BASH_SOURCE[0]}" EXTENSION="${EVALUATOR}" "${AUDIT_SCRIPT_DIR}/audit_nohold_contract.sh"
RUNNER="${BASH_SOURCE[0]}" EXTENSION="${TWO_CALL_EXTENSION}" "${AUDIT_SCRIPT_DIR}/audit_nohold_contract.sh"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/hf_cache" "${RUN_ROOT}/libero_config"
chmod 2775 "${RUN_ROOT}" "${RUN_ROOT}/logs" "${RUN_ROOT}/hf_cache" "${RUN_ROOT}/libero_config"
SERVER_LOG="${RUN_ROOT}/logs/vla_server.log"
EVAL_LOG="${RUN_ROOT}/logs/evaluator.log"
COMMAND_FILE="${RUN_ROOT}/formal_command.txt"
MANIFEST="${RUN_ROOT}/run_manifest.json"

"${EVAL_PY}" - "${RUN_ROOT}/libero_config/config.yaml" "${TARGET_LIBERO_PATH}" <<'PY'
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2]).resolve()
datasets = root.parent / "datasets"
datasets.mkdir(parents=True, exist_ok=True)
values = {
    "assets": root / "assets",
    "bddl_files": root / "bddl_files",
    "benchmark_root": root,
    "datasets": datasets,
    "init_states": root / "init_files",
}
config_path.write_text(
    "".join(f"{key}: {value}\n" for key, value in values.items()),
    encoding="utf-8",
)
PY

export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export OPENPI_ROOT
export OPENPI_INFERENCE_ROOT
export TARGET_LIBERO_PATH
export LIBERO_CONFIG_PATH="${RUN_ROOT}/libero_config"
export HF_HOME="${RUN_ROOT}/hf_cache"
export HV2_EXP_ROOT="${EXP_ROOT}"
export TWO_CALL_EXTENSION_PATH="${TWO_CALL_EXTENSION}"
export CONTROLLER_STABLE_VLM_CALLS

SOURCE_COMMIT="$(git -C "${PACK_ROOT}" rev-parse HEAD)"
RUNNER_SHA256="$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
EVALUATOR_SHA256="$(sha256sum "${EVALUATOR}" | awk '{print $1}')"
TWO_CALL_SHA256="$(sha256sum "${TWO_CALL_EXTENSION}" | awk '{print $1}')"
NORM_SHA256="$(sha256sum "${VLA_CHECKPOINT}/assets/robomemarena_fullvlm_v2_noflip_dataset_v2/norm_stats.json" | awk '{print $1}')"
ADAPTER_SHA256="$(sha256sum "${VLM_ADAPTER}/adapter_model.safetensors" | awk '{print $1}')"
BDDL_SHA256="$(sha256sum "${TASK_BDDL}" | awk '{print $1}')"
SLURM_JOB_RECORD="$(scontrol show job -o "${SLURM_JOB_ID}")"
export SLURM_JOB_RECORD

"${EVAL_PY}" - "${MANIFEST}" <<PY
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
slurm_record = os.environ.get("SLURM_JOB_RECORD", "")
account_match = re.search(r"(?:^| )Account=([^ ]+)", slurm_record)
partition_match = re.search(r"(?:^| )Partition=([^ ]+)", slurm_record)
payload = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "status": "RUNNING",
    "unix_user": subprocess.check_output(["whoami"], text=True).strip(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_account": account_match.group(1) if account_match else None,
    "partition": partition_match.group(1) if partition_match else None,
    "slurm_job_record": slurm_record,
    "hostname": socket.gethostname(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "nvidia_smi_gpu_list": subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines(),
    "producer_commit": "${SOURCE_COMMIT}",
    "frozen_runner": "${BASH_SOURCE[0]}",
    "frozen_runner_sha256": "${RUNNER_SHA256}",
    "frozen_extension": "${EVALUATOR}",
    "frozen_extension_sha256": "${EVALUATOR_SHA256}",
    "frozen_two_call_extension": "${TWO_CALL_EXTENSION}",
    "frozen_two_call_extension_sha256": "${TWO_CALL_SHA256}",
    "task": ${TASK_ID},
    "seed": ${SEED},
    "max_steps": ${MAX_STEPS},
    "controller_stable_vlm_calls": ${CONTROLLER_STABLE_VLM_CALLS},
    "control_contract": "two consecutive fresh VLM prompts -> VLA35999 actions; no hold/release/anchor/oracle/GT replay",
    "vla": {
        "config": "${VLA_CONFIG}",
        "checkpoint": "${VLA_CHECKPOINT}",
        "norm_sha256": "${NORM_SHA256}",
    },
    "vlm": {
        "architecture": "high_vlm_v2",
        "base": "${VLM_BASE}",
        "adapter": "${VLM_ADAPTER}",
        "adapter_model_sha256": "${ADAPTER_SHA256}",
    },
    "official_scoring": {
        "repository": "https://github.com/OpenHelix-Team/RoboMemArena",
        "commit": "cc156e519990ae43cf3b64281a548724f428fbbd",
        "scorer": "${OFFICIAL_SCORER}",
        "scorer_sha256": "${ACTUAL_SCORER_SHA256}",
        "bddl": "${TASK_BDDL}",
        "bddl_sha256": "${BDDL_SHA256}",
        "optional_final_stage": "declared only by frozen official scorer",
    },
    "reference_runtime": "${REFERENCE_RUNTIME}",
    "task_config": "${TASK_CONFIG}",
    "libero": "${TARGET_LIBERO_PATH}",
    "port": ${PORT},
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

(
  cd "${OPENPI_ROOT}"
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONFAULTHANDLER=1 \
    "${VLA_PY}" -u scripts/serve_policy.py --port "${PORT}" \
      policy:checkpoint \
      --policy.config="${VLA_CONFIG}" \
      --policy.dir="${VLA_CHECKPOINT}"
) > "${SERVER_LOG}" 2>&1 &
server_pid=$!

server_ready=0
for _ in $(seq 1 450); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "VLA server exited before becoming ready" >&2
    tail -100 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if "${EVAL_PY}" - "${PORT}" <<'PY' >/dev/null 2>&1
import socket
import sys

with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
    pass
PY
  then
    server_ready=1
    break
  fi
  sleep 2
done
if [[ "${server_ready}" -ne 1 ]]; then
  echo "timed out waiting for VLA server on port ${PORT}" >&2
  exit 1
fi

eval_cmd=(
  "${EVAL_PY}" -u "${EVALUATOR}"
  --tasks "${TASK_ID}"
  --vlm-checkpoint "${VLM_ADAPTER}"
  --processor-dir "${VLM_ADAPTER}"
  --vlm-architecture high_vlm_v2
  --vlm-base-model "${VLM_BASE}"
  --output-dir "${RUN_ROOT}/eval"
  --reference-runtime "${REFERENCE_RUNTIME}"
  --task-config "${TASK_CONFIG}"
  --host 127.0.0.1
  --port "${PORT}"
  --action-source vla
  --vlm-device cuda:0
  --seeds "${SEED}"
  --max-steps "${MAX_STEPS}"
  --num-steps-wait 10
  --replan-steps 5
  --resize-size 256
  --vlm-interval 5
  --n-recent 5
  --k-max 8
  --d-merge 6
  --no-async-vlm
  --use-wrist
  --use-keyframe-memory
  --save-video
  --save-vlm-images
)
printf '%q ' env CUDA_VISIBLE_DEVICES=1 HV2_EXP_ROOT="${EXP_ROOT}" \
  TWO_CALL_EXTENSION_PATH="${TWO_CALL_EXTENSION}" \
  CONTROLLER_STABLE_VLM_CALLS="${CONTROLLER_STABLE_VLM_CALLS}" \
  "${eval_cmd[@]}" > "${COMMAND_FILE}"
printf '\n' >> "${COMMAND_FILE}"

set +e
CUDA_VISIBLE_DEVICES=1 HV2_EXP_ROOT="${EXP_ROOT}" \
  TWO_CALL_EXTENSION_PATH="${TWO_CALL_EXTENSION}" \
  CONTROLLER_STABLE_VLM_CALLS="${CONTROLLER_STABLE_VLM_CALLS}" PYTHONFAULTHANDLER=1 \
  "${eval_cmd[@]}" > "${EVAL_LOG}" 2>&1
eval_rc=$?
set -e

"${EVAL_PY}" - "${MANIFEST}" "${eval_rc}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
rc = int(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
payload["exit_code"] = rc
payload["status"] = "COMPLETED" if rc == 0 else "FAILED"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ "${eval_rc}" -ne 0 ]]; then
  tail -160 "${EVAL_LOG}" >&2 || true
  exit "${eval_rc}"
fi

echo "TASK${TASK_ID}_SEED${SEED}_EVAL_COMPLETE run_root=${RUN_ROOT}"
