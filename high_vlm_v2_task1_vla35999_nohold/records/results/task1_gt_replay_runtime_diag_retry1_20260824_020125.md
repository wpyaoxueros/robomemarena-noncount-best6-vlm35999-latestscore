# Task1 high-vlm-v2 GT-replay runtime diagnostic receipt

- Run: `task1_gt_replay_runtime_diag_retry1_20260824_020125`.
- Status: `COMPLETED_DIAGNOSTIC_ONLY`; do not include in autonomous Task1 success-rate statistics.
- Slurm gates: Job `537846` one-GPU probe and Job `537850` formal-shape probe passed as Unix user/account `hzhang061` on `acd_ue`.
- Diagnostic allocation: Job `537853`, `hzhang061`, account `hzhang061`, partition `acd_u`, node `ACD1-6`, one H100 80GB, 160 GiB RAM.
- Producer commit: `8b5e79d13ffe9f003095a0205096d385a042534b`.
- Frozen runner SHA-256: `f44063ba29641c7003aec3892d65b2ee915c6048b903b92987ebcbad470f15e8`.
- Controlled path: copied high-vlm-v2 checkpoint and evaluator, synchronous VLM every five steps, wrist plus keyframe memory, LIBERO, image saving, and video writing.
- Removed path: no VLA server and no VLA-generated actions; `action_source=gt-replay`, `trajectory_only=true`.
- Result: process exit code `0`, 300 steps, 60 VLM calls, latest input `t=295`, summary and both videos written.
- Interpretation: the VLM/runtime path survived beyond the two autonomous aborts near step 250. The remaining fault boundary is the VLA server/action path or the physical state produced by VLA actions.

## Evidence

- Run manifest: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_gt_replay_runtime_diag_retry1_20260824_020125/run_manifest.json`, SHA-256 `afd111319bd7b4926425bafbc25a014f21356d888025521e20cf6feecdd6ad22`.
- Evaluator log: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_gt_replay_runtime_diag_retry1_20260824_020125/logs/evaluator.log`, SHA-256 `bc00cb2cb2270e9ff65d645565c42840447018087eb007f01071b3a97765a84b`.
- Summary: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_gt_replay_runtime_diag_retry1_20260824_020125/eval/summary.tsv`, SHA-256 `9a0545c4eaac2a206856f45c59b1bc3e9f5e7a0f38d26b3bf785bab54321a5b7`.
- Aggregate: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_gt_replay_runtime_diag_retry1_20260824_020125/eval/aggregate.json`, SHA-256 `efa6425213edd04012a9e49aff2f809061c438e378fb8094f0da31e80d02c581`.
- Main video: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_gt_replay_runtime_diag_retry1_20260824_020125/eval/videos/task1_ep000_seed104_main.mp4`, SHA-256 `a78565c6835b19220820f0cc1b2f8d50acb8061a2a2c74ab4081603d92358f41`.
- Wrist video: `/data/user/hlei573/vla_memory_experiments/repro_eval_packs/noncount_best6_latestscore_20ep_20260723/high_vlm_v2_task1_vla35999_nohold/runs/task1_gt_replay_runtime_diag_retry1_20260824_020125/eval/videos/task1_ep000_seed104_wrist.mp4`, SHA-256 `292c5bbca122a612c327b55566f46c9ffc6cdb3a837aa858a80996d6d425c2ba`.
