# Task1 seed104 no-hold frozen retry receipt

- Run: `task1_seed104_nohold_retry1_20260824_013208`
- Slurm: Job `537682`, `hzhang061`, account `hzhang061`, `acd_u`, `ACD1-11`, two H100 80GB GPUs.
- Producer commit: `27467402a6467cae54aa29f8acf7cfa00bd9c9b2`.
- Frozen runner SHA-256: `87beaf3cb9f2b1c20b58008d9592bdc9ea786e5f88b6d4a71965bbbe7ea44342`.
- Status: `INVALID_NATIVE_ABORT`; do not include in success-rate statistics.
- Exit: evaluator `SIGABRT`, code `134`; manifest status `FAILED`.
- Last rollout step: `250`.
- Verified behavior before exit: physical `pick cookies` completed at `t=127`; VLM autonomously retained `place cookies into basket` from `t=200`; no hold, release, anchor, oracle, GT replay, or trajectory-only control was active.
- Missing required artifacts: no episode summary and no MP4.

## Evidence

- `sync_vlm.log`: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_seed104_nohold_retry1_20260824_013208/eval/task1/ep000/sync_vlm.log`, SHA-256 `8cb41d9adbab4a867ae098d71c0aa333ce72b918ad9e8c70546dc5ca4198b7d1`.
- `sync_vlm_trace.jsonl`: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_seed104_nohold_retry1_20260824_013208/eval/task1/ep000/sync_vlm_trace.jsonl`, SHA-256 `0ee48a72d3ce350cf28c395390cb7355e7140eb908b2e92157281ae2498a02ea`.
- `semantic_state.jsonl`: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_seed104_nohold_retry1_20260824_013208/eval/task1/ep000/semantic_state.jsonl`, SHA-256 `4cba010c3c81c62f2041a3eab10825ca882961b0641a39e21c4905e2101aab05`.
- Last main frame: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_seed104_nohold_retry1_20260824_013208/eval/task1/ep000/vlm_inputs/t0250/recent_05_agentview_place_cookies_into_basket.png`, SHA-256 `81c59497671548c41744e8a4a9b50f695903c3ed99227a638726de3535738cf4`.
- Last wrist frame: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_seed104_nohold_retry1_20260824_013208/eval/task1/ep000/vlm_inputs/t0250/recent_05_wrist_place_cookies_into_basket.png`, SHA-256 `ea9f2f7307a87aa66f8116147d6e4c0af8d9d0be257d0139f4b39f9bdd622dee`.
- `run_manifest.json`: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_seed104_nohold_retry1_20260824_013208/run_manifest.json`, SHA-256 `20ecc5709072f49ea81cea22558ca5c987ccdec77a003af4ba1bab23d99276c9`.
