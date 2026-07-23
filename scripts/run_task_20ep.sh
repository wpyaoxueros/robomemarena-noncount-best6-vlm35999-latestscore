#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723"
TASK_ID="${1:?usage: run_task_20ep.sh TASK_ID [NUM_TRIALS] [SEED] [PORT]}"
NUM_TRIALS="${2:-20}"
SEED="${3:-104}"
PORT="${4:-$((9300 + TASK_ID))}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

VLA_POLICY="/data/user/hlei573/openpi/checkpoints/pi05_libero_robomemarena_fullvlm_v2_noflip_dataset/fullvlm_v2_robomemarena_noflip_v2_bs128_4gpu_20260507_183338/35999"
VLA_REPO_ID="/data/user/hlei573/.cache/huggingface/lerobot/lhs/robomemarena_fullvlm_v2_noflip_dataset_v2"
TARGETS="${ROOT}/config/tasks2_26_endpose_targets_seed100_199.json"
DRAWER_PASSAGE="${ROOT}/config/drawer_passage_counts_task4full_plus_alltasks_20260627.json"

MAX_STEPS=2200
REPLAN_STEPS=10
PASSAGE="${DRAWER_PASSAGE}"
TOL_FILE=""
ANCHORS=""
COMPLETED_MODE="completed_struct"
TASK_TEXT_MODE="english_reference_no_candidate"
DRAWER_OPEN=0.10
DRAWER_CLOSE=0.08
DRAWER_DEBUG=0
PICK_GRIPPER=1
PICK_LIFT=1
DRAWER_GUARD=1
HOLD_BLOCK_PAST=0

case "${TASK_ID}" in
  4)
    VLM_CKPT="/data/user/hlei573/vla_memory_experiments/english_ref_vlm26/drawer_success20_reval_20260630/task4/vlm_eval_ready/task4_20260630_160758/dropo3_ckpt1000_topagain_f50_stagegate_success20_retry2_ckpt1000"
    MAX_STEPS=2000
    REPLAN_STEPS=10
    PASSAGE="${ROOT}/config/drawer_passage_counts_task4_openmiddle1_20260627.json"
    TOL_FILE="${ROOT}/config/task4_success20_tol_overrides.json"
    DRAWER_CLOSE=0.002
    DRAWER_DEBUG=10
    ;;
  5)
    VLM_CKPT="/data/user/hlei573/vla_memory_experiments/english_ref_vlm26/task5_fixeval_task4replica4ep_20260628_220923/dropo3_eval_artifacts/vlm_eval_ready/task5_20260628_220923_dropo3_4ep/task05_fixreplica_dropo3_4ep_ckpt1000"
    MAX_STEPS=2000
    REPLAN_STEPS=5
    TOL_FILE="${ROOT}/config/task5_tol_overrides.json"
    ANCHORS="${ROOT}/config/drawer_release_anchor_rules_seed104_v2_task5_firstclose.json"
    DRAWER_CLOSE=0.002
    DRAWER_DEBUG=10
    ;;
  11)
    VLM_CKPT="/data/user/hlei573/vla_memory_experiments/english_ref_vlm26/task11_ct2om_f0_fixspec_eval_20260630_085935/vlm_eval_ready/task11_20260630_085935/task11_ct2om_f0_fixspec_ckpt500"
    ANCHORS="${ROOT}/config/task11_release_anchor_rules_seed104_closetop_to_openmiddle_f0.json"
    ;;
  14)
    VLM_CKPT="/data/user/hlei573/vla_memory_experiments/english_ref_vlm26/output_shared_20260702_20260702_140540_task14_choco_pickpersist/eval_artifacts/vlm_eval_ready/task14_task14_english_ref_20260702_140740_ckpt1000_20260702_151642/task14_english_ref_20260702_140740_ckpt1000"
    ;;
  17)
    VLM_CKPT="/data/user/hlei573/vla_memory_experiments/english_ref_vlm26/task17_f250_openhold_completed_placebutter1_eval4_20260704_004406/eval_artifacts/vlm_eval_ready/task17_20260704_004406_openhold_completed_placebutter1/task17_f250_openhold_completed_placebutter1_ckpt1000"
    PASSAGE="${ROOT}/config/task17_passage_counts_placebutter1_20260704.json"
    ANCHORS="${ROOT}/config/task17_release_anchor_rules_f250_f0_chocf200.json"
    ;;
  19)
    VLM_CKPT="/data/user/hlei573/openpi_inference/output/tasks4_26_noorder_base_eval_artifacts/vlm_eval_ready/task19_task19_noorder_adaptive_20260621_044113_ckpt500_20260621_064018/task19_noorder_adaptive_20260621_044113_ckpt500"
    MAX_STEPS=2000
    REPLAN_STEPS=5
    PASSAGE="__NONE__"
    COMPLETED_MODE="off"
    TASK_TEXT_MODE="no_label_no_order"
    DRAWER_GUARD=0
    PICK_GRIPPER=0
    PICK_LIFT=0
    ;;
  *)
    echo "unsupported task: ${TASK_ID}" >&2
    exit 2
    ;;
esac

for required in \
  "${VLA_POLICY}/params" \
  "${VLA_REPO_ID}/norm_stats.json" \
  "${VLM_CKPT}/model.safetensors" \
  "${TARGETS}" \
  "${ROOT}/official_snapshot/evaluation_benchmark/scripts/task2_26_reference_stage.py"; do
  [[ -e "${required}" ]] || { echo "missing required path: ${required}" >&2; exit 3; }
done

RUN_ID="task${TASK_ID}_vlm35999_latestd9_20ep_seed${SEED}_${STAMP}"
OUT_ROOT="${ROOT}/outputs/task${TASK_ID}/${RUN_ID}"
mkdir -p "${OUT_ROOT}/logs"

export EVAL_PY="${ROOT}/evaluators/eval_tasks2_26_sync_endpose_hold_officialscore.py"
export TASKS2_26_BASE_EVAL_PY="/data/user/hlei573/tmp/rma_refeval_fresh_20260513_052445/RoboMemArena/evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference/eval_tasks2_26_vlm_vla.py"
export ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR="${ROOT}/official_snapshot/evaluation_benchmark/scripts"
export TARGET_LIBERO_PATH="${ROOT}/official_snapshot/evaluation_benchmark/libero_fork"
export TASK_CONFIG="${ROOT}/official_snapshot/evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference/fullvlm_v2_26_memory_tasks.json"
export TASKS_JSON="[${TASK_ID}]" NUM_TRIALS SEED MAX_STEPS REPLAN_STEPS PORT RUN_ID OUT_ROOT
export VLM_CKPT VLA_POLICY VLA_REPO_ID
export VLA_CONFIG="pi05_libero_robomemarena_fullvlm_v2_noflip_dataset"
export VLA_SERVER_PY="${ROOT}/scripts/serve_policy_custom_repo.py"
export VLA_ACTION_TARGET_MODE=raw DISABLE_OUTPUT_NORMALIZE=1
export ENDPOSE_HOLD_TARGETS_JSON="${TARGETS}"
export ENDPOSE_TARGET_PASSAGE_COUNTS_JSON="${PASSAGE}"
export ENDPOSE_HOLD_POS_TOL=0.06 ENDPOSE_HOLD_EEF_DEFAULT_TOL=0.06
export ENDPOSE_HOLD_EEF_P95_EXTRA_TOL=0.02 ENDPOSE_HOLD_EEF_TOL_CAP=0.08
export ENDPOSE_HOLD_MIN_ACTIVE_STEPS=20 ENDPOSE_HOLD_CONSECUTIVE=2
export POST_HOLD_RELEASE_VLA_STEPS=30 STRICT_HOLD_RELEASE_NEXT=0
export PREVENT_SUBTASK_REGRESSION=1 REGRESSION_GUARD_AFTER_HOLD_RELEASE=1
export HOLD_RELEASE_BLOCK_PAST_SUBTASKS="${HOLD_BLOCK_PAST}"
export DRAWER_FORWARD_ADVANCE_GUARD="${DRAWER_GUARD}"
export DRAWER_OPEN_STAGE_THRESH="${DRAWER_OPEN}" DRAWER_CLOSE_STAGE_THRESH="${DRAWER_CLOSE}"
export DRAWER_STAGE_DEBUG_INTERVAL="${DRAWER_DEBUG}"
export ENDPOSE_PICK_GRIPPER_GATE="${PICK_GRIPPER}" ENDPOSE_PICK_OBJECT_LIFT_GATE="${PICK_LIFT}"
export ENDPOSE_PICK_OBJECT_LIFT_DELTA=0.01
export VLM_TASK_TEXT_MODE="${TASK_TEXT_MODE}" VLM_COMPLETED_SUBTASKS_MODE="${COMPLETED_MODE}"
export SUBTASK_RELEASE_ANCHORS_JSON="${ANCHORS}"
export ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE="${TOL_FILE}"
export OPENPI_PYTHON="/data/user/hlei573/openpi/.venv/bin/python3"
export INFER_PYTHON="/data/user/hlei573/openpi_inference/.venv/bin/python"

{
  echo "task_id=${TASK_ID}"
  echo "num_trials=${NUM_TRIALS}"
  echo "seed=${SEED}"
  echo "vla_policy=${VLA_POLICY}"
  echo "vla_repo_id=${VLA_REPO_ID}"
  echo "vlm_ckpt=${VLM_CKPT}"
  echo "official_commit=d9f83ac5182e25ad7f0a301a77a0b667f2392df1"
  echo "official_stage_sha256=0ab5e19cb7b90844b86fe04a76facc0364af55f1e841c4754aa675404a318538"
  echo "launcher_sha256=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
  echo "unix_user=$(id -un)"
  echo "unix_groups=$(id -Gn)"
  env | sort
} > "${OUT_ROOT}/run_manifest.env"

exec "${ROOT}/evaluators/run_tasks2_26_sync_hold_eval_customrepo.sh"
