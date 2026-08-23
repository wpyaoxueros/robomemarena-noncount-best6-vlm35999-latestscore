#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVALUATOR="${EXP_ROOT}/source/vlm_ft/eval_three_tasks.py"
RUNNER="${EXP_ROOT}/scripts/run_task1_seed104_nohold_inside_allocation.sh"

for forbidden in \
  'ORACLE_(HOLD|FORCE|STAGE|PROMPT|NEXT)' \
  '(OBJECT|ROBOT)_ANCHOR' \
  'AUTO_RELEASE' \
  'STAGE_LOCK' \
  'REQUIRE_FORWARD' \
  'hold_steps' \
  'auto_release'; do
  if grep -En "${forbidden}" "${EVALUATOR}" "${RUNNER}"; then
    echo "forbidden control mechanism found: ${forbidden}" >&2
    exit 1
  fi
done

grep -Fq 'parser.add_argument("--action-source", choices=("vla", "gt-replay"), default="vla")' \
  "${EVALUATOR}"
grep -Fq 'prompt_for_vla = current_prompt or planner.default_subtask_prompt' "${EVALUATOR}"
grep -Fq 'output = client.infer(element)' "${EVALUATOR}"
grep -Eq -- '--action-source[[:space:]]+vla' "${RUNNER}"

if grep -En -- '--action-source[[:space:]]+gt-replay|--trajectory-only' "${RUNNER}"; then
  echo "runner enables GT replay or trajectory-only control" >&2
  exit 1
fi

echo "PASS: VLM prompt -> VLA action path; no hold/release/anchor/oracle control"
