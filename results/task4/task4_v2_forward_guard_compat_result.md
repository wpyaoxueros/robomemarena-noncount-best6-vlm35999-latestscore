# Task4 v2 Forward-Guard Compatibility Replay Result

## Scope

This is the committed one-episode test described in
`task4_v2_forward_guard_compat_PRE_RUN.md`.

- Seed: `108`
- Only changed variable: `TASK4_DRAWER_FORWARD_ADVANCE_GUARD=0`
- VLM/VLA checkpoints, norm, targets, passage requirements, tolerances,
  `REPLAN_STEPS=10`, rollout path, and latest official scorer were unchanged.

## Result

The episode completed in 237.78 seconds with latest official `CSR=25.0` and
`TSR=0.0`.

Official stages:

1. `01_Open_Top_Drawer=Y`
2. `02_Close_Top_Drawer=Y`
3. `03_Open_Middle_Drawer=N`
4. Remaining stages `N`

The VLM autonomously alternated between `close the top drawer` and
`open middle drawer`. Disabling the forward guard therefore removed the
specific forward-switch block, but did not make the middle drawer open. The
remaining blocker is the separate close/open end-pose stage gating and the
post-close physical state, not the forward guard by itself.

## Evidence

- Summary: `outputs/task4/task4_vlm35999_latestd9_20ep_seed108_20260724_024626/summary.tsv`
- Main video: `outputs/task4/task4_vlm35999_latestd9_20ep_seed108_20260724_024626/videos/task4/task4_failure_ep0.mp4`
- Wrist video: `outputs/task4/task4_vlm35999_latestd9_20ep_seed108_20260724_024626/videos/task4/task4_failure_ep0_wrist.mp4`
- Trace: `outputs/task4/task4_vlm35999_latestd9_20ep_seed108_20260724_024626/task4/ep0/sync_vlm.log`

This is a valid latest-score diagnostic result, not a reproduction success.
