#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

(
  cd "${EXP_ROOT}"
  sha256sum -c SOURCE_MANIFEST.sha256
)

if [[ "${1:-}" != "--against-source" ]]; then
  echo "source manifest verified"
  exit 0
fi

JROOT="${JROOT:-/data/user/jwen341/openpi_rm}"

compare() {
  local copied="$1"
  local original="$2"
  if ! cmp -s "${EXP_ROOT}/${copied}" "${original}"; then
    echo "source mismatch: ${copied} != ${original}" >&2
    exit 1
  fi
  printf 'SAME\t%s\t%s\n' "${copied}" "${original}"
}

for name in \
  eval_three_tasks.py \
  high_vlm_v2_components.py \
  primitive_weighted_loss.py \
  prompt_contract_26.py \
  semantic_metrics.py \
  task_semantics.py \
  handoff_metrics.py \
  training_components.py; do
  compare "source/vlm_ft/${name}" "${JROOT}/vlm_ft/${name}"
done
compare "source/vlm_ft/README.upstream.md" "${JROOT}/vlm_ft/README.md"

for name in \
  eval_common.py \
  eval_task1_qwen3_async_openpi_inference_vla_cam.py \
  keyframe_selection.py \
  retry_tasks2_26_stage_from_anygrasp.py \
  robocerebra_adapter.py \
  task_prompts.py; do
  compare \
    "source/eval_robomem/robomemarena_official/evaluation_benchmark/openpi_minimal_runtime/${name}" \
    "${JROOT}/eval_robomem/robomemarena_official/evaluation_benchmark/openpi_minimal_runtime/${name}"
done

for name in eval_tasks2_26_vlm_vla.py fullvlm_v2_26_memory_tasks.json; do
  compare \
    "source/eval_robomem/robomemarena_official/evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference/${name}" \
    "${JROOT}/eval_robomem/robomemarena_official/evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference/${name}"
done

compare \
  "source/eval_robomem/robomemarena_official/bddl/1_cookies_tomato_basket.bddl" \
  "${JROOT}/eval_robomem/robomemarena_official/bddl/1_cookies_tomato_basket.bddl"

echo "source snapshot is byte-identical to the read-only source tree"
