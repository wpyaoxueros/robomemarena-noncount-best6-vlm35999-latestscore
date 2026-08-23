#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RECORDS_DIR="${EXP_ROOT}/records"
ASSET_ROOT="${EXP_ROOT}/runtime_assets"

VLM_ADAPTER_SOURCE="${VLM_ADAPTER_SOURCE:-/data/user/jwen341/openpi_rm/output/high_vlm_v2_26tasks_lora_r32_8gpu_499178/final_adapter}"
LIBERO_SOURCE="${LIBERO_SOURCE:-/data/user/jwen341/openpi_rm/eval_robomem/robomemarena_official/evaluation_benchmark/libero_fork/libero}"
VLM_ADAPTER_COPY="${ASSET_ROOT}/high_vlm_v2_26tasks_lora_r32_final_adapter"
LIBERO_COPY="${ASSET_ROOT}/libero_fork/libero"

for required in \
  "${VLM_ADAPTER_SOURCE}/adapter_model.safetensors" \
  "${VLM_ADAPTER_SOURCE}/high_vlm_v2_config.json" \
  "${LIBERO_SOURCE}/libero/envs"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required runtime source: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${RECORDS_DIR}" "${VLM_ADAPTER_COPY}" "${LIBERO_COPY}"

# Do not use archive owner/group preservation: copied assets must inherit the
# experiment's irpn group rather than their source account's primary group.
rsync -rltp "${VLM_ADAPTER_SOURCE}/" "${VLM_ADAPTER_COPY}/"
rsync -rltp "${LIBERO_SOURCE}/" "${LIBERO_COPY}/"
find "${ASSET_ROOT}" -type d -exec chmod 2775 {} +
find "${ASSET_ROOT}" -type f -exec chmod g+r {} +

hash_tree() {
  local root="$1"
  local output="$2"
  (
    cd "${root}"
    find . -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
      | sort -z \
      | xargs -0 sha256sum
  ) > "${output}"
}

hash_tree "${VLM_ADAPTER_SOURCE}" "${RECORDS_DIR}/high_vlm_v2_adapter_source.sha256"
hash_tree "${VLM_ADAPTER_COPY}" "${RECORDS_DIR}/high_vlm_v2_adapter_copy.sha256"
hash_tree "${LIBERO_SOURCE}" "${RECORDS_DIR}/libero_source.sha256"
hash_tree "${LIBERO_COPY}" "${RECORDS_DIR}/libero_copy.sha256"

cmp "${RECORDS_DIR}/high_vlm_v2_adapter_source.sha256" \
  "${RECORDS_DIR}/high_vlm_v2_adapter_copy.sha256"
cmp "${RECORDS_DIR}/libero_source.sha256" "${RECORDS_DIR}/libero_copy.sha256"

{
  printf 'asset\tsource\tcopy\tverification\n'
  printf 'high_vlm_v2_adapter\t%s\t%s\tsha256_tree_equal\n' \
    "${VLM_ADAPTER_SOURCE}" "${VLM_ADAPTER_COPY}"
  printf 'libero_runtime\t%s\t%s\tsha256_tree_equal\n' \
    "${LIBERO_SOURCE}" "${LIBERO_COPY}"
} > "${RECORDS_DIR}/runtime_asset_provenance.tsv"

echo "runtime asset copy verified: ${ASSET_ROOT}"

