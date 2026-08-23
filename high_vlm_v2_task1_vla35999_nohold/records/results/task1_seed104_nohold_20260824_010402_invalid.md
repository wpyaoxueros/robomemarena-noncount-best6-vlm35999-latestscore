# Task1 seed104 no-hold invalid run receipt

- Run: `task1_seed104_nohold_20260824_010402`
- Slurm: Job `537544`, `hzhang061`, account `hzhang061`, `acd_u`, `ACD1-11`, two H100 80GB GPUs.
- Status: `INVALID_INFRASTRUCTURE_EXIT`; do not include in success-rate statistics.
- Exit: evaluator `SIGABRT`; parent allocation step ultimately exited `2:0`.
- Last rollout step: `245`.
- Verified behavior before exit: physical `pick cookies` completed at `t=127`; VLM autonomously selected `place cookies into basket`; no hold, release, anchor, oracle, GT replay, or trajectory-only control was active.
- Missing required artifacts: no episode summary and no MP4, so the run is not complete.

## Evidence

- `sync_vlm.log`: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_seed104_nohold_20260824_010402/eval/task1/ep000/sync_vlm.log`, SHA-256 `8539356538c55f23d43e70918fe35dc7fa56af34865bbcc36bbd9d282dd7d1a4`.
- `sync_vlm_trace.jsonl`: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_seed104_nohold_20260824_010402/eval/task1/ep000/sync_vlm_trace.jsonl`, SHA-256 `061790cd6933786444ad02d3df5bdaca733c87d9b486c322fb4db8fe818079c7`.
- `semantic_events.jsonl`: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_seed104_nohold_20260824_010402/eval/task1/ep000/semantic_events.jsonl`, SHA-256 `471b27d87965413dc5a38e516fb579561e9846cee326b278e10982745ed940dd`.
- `semantic_state.jsonl`: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_seed104_nohold_20260824_010402/eval/task1/ep000/semantic_state.jsonl`, SHA-256 `7914145c6ac2c3a5ecfd507beb672bcb361602d45ce7a997930aea78844a6444`.
- Last main frame: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_seed104_nohold_20260824_010402/eval/task1/ep000/vlm_inputs/t0245/recent_05_agentview_place_cookies_into_basket.png`, SHA-256 `1c9d1c32e5e271b52664ff99501ae915a3464905b29e1b35cc212f9644817528`.
- Last wrist frame: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_seed104_nohold_20260824_010402/eval/task1/ep000/vlm_inputs/t0245/recent_05_wrist_place_cookies_into_basket.png`, SHA-256 `685f4ce38bd30b45a71d1e53df972f57e3df425bab591d9b9f72fc9fed79f064`.
