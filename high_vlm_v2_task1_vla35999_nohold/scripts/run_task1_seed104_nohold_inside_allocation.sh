#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="${EXP_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
REPO_ROOT="$(cd "${EXP_ROOT}/.." && pwd)"
AUDIT_SCRIPT_DIR="${AUDIT_SCRIPT_DIR:-${EXP_ROOT}/scripts}"
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
EVALUATOR="${EXP_ROOT}/source/vlm_ft/eval_three_tasks.py"
RUN_ID="${RUN_ID:-task1_seed104_nohold_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${EXP_ROOT}/runs/${RUN_ID}"
PORT="${PORT:-$((18000 + ${SLURM_JOB_ID:-0} % 20000))}"

if [[ "$(whoami)" != "hzhang061" ]]; then
  echo "formal runner must execute as hzhang061, got $(whoami)" >&2
  exit 1
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "formal runner requires a Slurm allocation" >&2
  exit 1
fi
VISIBLE_GPU_COUNT="$(PYTHONNOUSERSITE=1 "${EVAL_PY}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${VISIBLE_GPU_COUNT}" -ne 2 ]]; then
  echo "formal runner requires exactly two visible GPUs" >&2
  echo "torch visible GPU count: ${VISIBLE_GPU_COUNT}" >&2
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi

for required in \
  "${VLA_PY}" \
  "${EVAL_PY}" \
  "${VLA_CHECKPOINT}/assets/robomemarena_fullvlm_v2_noflip_dataset_v2/norm_stats.json" \
  "${VLM_BASE}/config.json" \
  "${VLM_ADAPTER}/adapter_model.safetensors" \
  "${TARGET_LIBERO_PATH}/libero/envs" \
  "${REFERENCE_RUNTIME}" \
  "${TASK_CONFIG}" \
  "${EVALUATOR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required path: ${required}" >&2
    exit 1
  fi
done

"${AUDIT_SCRIPT_DIR}/verify_source_snapshot.sh"
"${AUDIT_SCRIPT_DIR}/audit_nohold_contract.sh"

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

SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
RUNNER_SHA256="$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
NORM_SHA256="$(sha256sum "${VLA_CHECKPOINT}/assets/robomemarena_fullvlm_v2_noflip_dataset_v2/norm_stats.json" | awk '{print $1}')"
ADAPTER_SHA256="$(sha256sum "${VLM_ADAPTER}/adapter_model.safetensors" | awk '{print $1}')"
SLURM_JOB_RECORD="$(scontrol show job -o "${SLURM_JOB_ID}")"
export SLURM_JOB_RECORD

"${EVAL_PY}" - "${MANIFEST}" <<PY
import json
import os
import pathlib
import socket
import subprocess
import sys
import re
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
slurm_record = os.environ.get("SLURM_JOB_RECORD", "")
account_match = re.search(r"(?:^| )Account=([^ ]+)", slurm_record)
partition_match = re.search(r"(?:^| )Partition=([^ ]+)", slurm_record)
payload = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "status": "RUNNING",
    "unix_user": os.environ.get("USER") or subprocess.check_output(["whoami"], text=True).strip(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_account": account_match.group(1) if account_match else os.environ.get("SLURM_JOB_ACCOUNT"),
    "partition": partition_match.group(1) if partition_match else os.environ.get("SLURM_JOB_PARTITION"),
    "slurm_job_record": slurm_record,
    "hostname": socket.gethostname(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "nvidia_smi_gpu_list": subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines(),
    "producer_commit": "${SOURCE_COMMIT}",
    "frozen_runner": "${BASH_SOURCE[0]}",
    "frozen_runner_sha256": "${RUNNER_SHA256}",
    "task": 1,
    "seed": 104,
    "control_contract": "synchronous VLM prompt -> VLA actions; no hold/release/anchor/oracle/GT replay",
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
    "copied_evaluator": "${EVALUATOR}",
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
  --tasks 1
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
  --seeds 104
  --max-steps 2500
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
printf '%q ' env CUDA_VISIBLE_DEVICES=1 "${eval_cmd[@]}" > "${COMMAND_FILE}"
printf '\n' >> "${COMMAND_FILE}"

set +e
CUDA_VISIBLE_DEVICES=1 PYTHONFAULTHANDLER=1 "${eval_cmd[@]}" > "${EVAL_LOG}" 2>&1
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

echo "TASK1_EVAL_COMPLETE run_root=${RUN_ROOT}"
