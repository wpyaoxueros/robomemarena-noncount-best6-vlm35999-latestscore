#!/usr/bin/env bash
set -euo pipefail

EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:?set EXPECTED_GPU_COUNT}"
EXP_ROOT="${EXP_ROOT:?set EXP_ROOT}"
VLA_CHECKPOINT="${VLA_CHECKPOINT:?set VLA_CHECKPOINT}"
VLM_ADAPTER="${VLM_ADAPTER:?set VLM_ADAPTER}"
VLM_BASE="${VLM_BASE:?set VLM_BASE}"
TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH:?set TARGET_LIBERO_PATH}"
EVAL_PY="${EVAL_PY:-/data/user/hlei573/openpi_inference/.venv/bin/python}"

printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'unix_user=%s\n' "$(whoami)"
printf 'hostname=%s\n' "$(hostname)"
printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-missing}"
printf 'slurm_account=%s\n' "${SLURM_JOB_ACCOUNT:-unknown}"
printf 'slurm_partition=%s\n' "${SLURM_JOB_PARTITION:-unknown}"
printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

test -r "${VLA_CHECKPOINT}/assets/robomemarena_fullvlm_v2_noflip_dataset_v2/norm_stats.json"
test -r "${VLM_ADAPTER}/adapter_model.safetensors"
test -r "${VLM_BASE}/config.json"
test -d "${TARGET_LIBERO_PATH}/libero/envs"
mkdir -p "${EXP_ROOT}/records/probes/write_test"
probe_file="${EXP_ROOT}/records/probes/write_test/${SLURM_JOB_ID:-noslurm}_$(whoami)"
printf 'write-ok\n' > "${probe_file}"
rm -f "${probe_file}"

PYTHONNOUSERSITE=1 "${EVAL_PY}" - "${EXPECTED_GPU_COUNT}" <<'PY'
import json
import sys

import torch

expected = int(sys.argv[1])
actual = torch.cuda.device_count()
if actual != expected:
    raise SystemExit(f"expected {expected} visible GPUs, found {actual}")
rows = []
for index in range(actual):
    rows.append(
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "bf16": bool(torch.cuda.is_bf16_supported()),
        }
    )
print(json.dumps({"gpu_count": actual, "gpus": rows}, sort_keys=True))
PY

echo "GPU_PROBE_PASS count=${EXPECTED_GPU_COUNT}"

