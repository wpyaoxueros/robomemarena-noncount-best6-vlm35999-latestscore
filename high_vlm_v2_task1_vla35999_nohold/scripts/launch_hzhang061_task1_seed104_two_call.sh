#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export EVALUATOR="${EXP_ROOT}/extensions/eval_two_call_prompt_commit.py"
export CONTROLLER_STABLE_VLM_CALLS=2
export CONTROL_CONTRACT="two consecutive VLM prompts -> VLA actions; no hold/release/anchor/oracle/GT replay"
export AUDIT_EXTENSION="${EVALUATOR}"
export RUN_ID="${RUN_ID:-task1_seed104_two_call_$(date +%Y%m%d_%H%M%S)}"

exec "${SCRIPT_DIR}/launch_hzhang061_task1_seed104.sh"
