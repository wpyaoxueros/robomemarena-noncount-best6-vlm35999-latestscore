#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="${EXP_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
REPO_ROOT="$(cd "${EXP_ROOT}/.." && pwd)"
AUDIT_SCRIPT_DIR="${AUDIT_SCRIPT_DIR:-${EXP_ROOT}/scripts}"
EVAL_PY="${EVAL_PY:-/data/user/hlei573/openpi_inference/.venv/bin/python}"
VLM_BASE="${VLM_BASE:-/data/user/jwen341/model_lib/Qwen3-VL-8B-Instruct}"
VLM_ADAPTER="${VLM_ADAPTER:-${EXP_ROOT}/runtime_assets/high_vlm_v2_26tasks_lora_r32_final_adapter}"
TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH:-${EXP_ROOT}/runtime_assets/libero_fork/libero}"
GT_DATA_ROOT="${GT_DATA_ROOT:-/data/user/jwen341/dataset/robomemarena_fullvlm_v2_official_remote_20260711_raw}"
REFERENCE_RUNTIME="${EXP_ROOT}/source/eval_robomem/robomemarena_official/evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference/eval_tasks2_26_vlm_vla.py"
TASK_CONFIG="${EXP_ROOT}/source/eval_robomem/robomemarena_official/evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference/fullvlm_v2_26_memory_tasks.json"
EVALUATOR="${EXP_ROOT}/source/vlm_ft/eval_three_tasks.py"
RUN_ID="${RUN_ID:-task1_gt_replay_runtime_diag_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${EXP_ROOT}/runs/${RUN_ID}"

if [[ "$(whoami)" != "hzhang061" ]]; then
  echo "diagnostic runner must execute as hzhang061" >&2
  exit 1
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "diagnostic runner requires a Slurm allocation" >&2
  exit 1
fi
if [[ "$(PYTHONNOUSERSITE=1 "${EVAL_PY}" -c 'import torch; print(torch.cuda.device_count())')" -ne 1 ]]; then
  echo "diagnostic runner requires exactly one visible GPU" >&2
  exit 1
fi

for required in \
  "${VLM_BASE}/config.json" \
  "${VLM_ADAPTER}/adapter_model.safetensors" \
  "${TARGET_LIBERO_PATH}/libero/envs" \
  "${GT_DATA_ROOT}/1_cookies_tomato_basket_dataset/subtask_data/pick_cookies_0_seed104_task1.hdf5" \
  "${REFERENCE_RUNTIME}" \
  "${TASK_CONFIG}" \
  "${EVALUATOR}"; do
  test -e "${required}" || { echo "missing required path: ${required}" >&2; exit 1; }
done

"${AUDIT_SCRIPT_DIR}/verify_source_snapshot.sh"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/hf_cache" "${RUN_ROOT}/libero_config"
chmod 2775 "${RUN_ROOT}" "${RUN_ROOT}/logs" "${RUN_ROOT}/hf_cache" "${RUN_ROOT}/libero_config"

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
config_path.write_text("".join(f"{key}: {value}\n" for key, value in values.items()), encoding="utf-8")
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
export TARGET_LIBERO_PATH
export LIBERO_CONFIG_PATH="${RUN_ROOT}/libero_config"
export HF_HOME="${RUN_ROOT}/hf_cache"

MANIFEST="${RUN_ROOT}/run_manifest.json"
EVAL_LOG="${RUN_ROOT}/logs/evaluator.log"
COMMAND_FILE="${RUN_ROOT}/formal_command.txt"
SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
RUNNER_SHA256="$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
SLURM_JOB_RECORD="$(scontrol show job -o "${SLURM_JOB_ID}")"
export SLURM_JOB_RECORD

"${EVAL_PY}" - "${MANIFEST}" <<PY
import json
import os
import pathlib
import socket
from datetime import datetime, timezone

path = pathlib.Path("${MANIFEST}")
payload = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "status": "RUNNING",
    "diagnostic_only": True,
    "diagnostic_question": "Does the copied high-vlm-v2 runtime survive 300 steps / 60 VLM calls without the VLA server or VLA-generated trajectory?",
    "unix_user": os.environ.get("USER"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_job_record": os.environ.get("SLURM_JOB_RECORD"),
    "hostname": socket.gethostname(),
    "producer_commit": "${SOURCE_COMMIT}",
    "frozen_runner": "${BASH_SOURCE[0]}",
    "frozen_runner_sha256": "${RUNNER_SHA256}",
    "action_source": "gt-replay",
    "trajectory_only": True,
    "max_steps": 300,
    "seed": 104,
    "vlm_adapter": "${VLM_ADAPTER}",
    "gt_data_root": "${GT_DATA_ROOT}",
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

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
  --action-source gt-replay
  --gt-data-root "${GT_DATA_ROOT}"
  --trajectory-only
  --vlm-device cuda:0
  --seeds 104
  --max-steps 300
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
printf '%q ' env CUDA_VISIBLE_DEVICES=0 PYTHONFAULTHANDLER=1 "${eval_cmd[@]}" > "${COMMAND_FILE}"
printf '\n' >> "${COMMAND_FILE}"

set +e
CUDA_VISIBLE_DEVICES=0 "${eval_cmd[@]}" > "${EVAL_LOG}" 2>&1
eval_rc=$?
set -e

"${EVAL_PY}" - "${MANIFEST}" "${eval_rc}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
payload["exit_code"] = int(sys.argv[2])
payload["status"] = "COMPLETED" if int(sys.argv[2]) == 0 else "FAILED"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ "${eval_rc}" -ne 0 ]]; then
  tail -200 "${EVAL_LOG}" >&2 || true
  exit "${eval_rc}"
fi
echo "TASK1_GT_REPLAY_RUNTIME_DIAGNOSTIC_COMPLETE run_root=${RUN_ROOT}"
