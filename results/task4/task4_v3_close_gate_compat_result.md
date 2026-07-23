# Task4 v3 Close-Gate Compatibility Replay Result

## Scope

Seed108 one-episode compatibility diagnostic using commit `12beb31`.

- `TASK4_DRAWER_FORWARD_ADVANCE_GUARD=0`
- `DRAWER_CLOSE_REQUIRE_STAGE=0`
- VLM/VLA, norm, targets, passages, replan, seed and latest official scorer
  stayed fixed.

## Result

The rollout reproduced the historical-style handoff:

1. top open hold at `t=183`
2. VLM-selected release to `close the top drawer`
3. close-top hold at `t=308`
4. VLM-selected release to `open middle drawer`

However, this advance happened before the top drawer was physically closed.
The policy then moved away while requesting `open bottom drawer`.

Latest official result: `CSR=12.5`, `TSR=0.0`.
Only `01_Open_Top_Drawer` was true; `02_Close_Top_Drawer` and all subsequent
stages were false.

## Conclusion

The strict close-stage gate is not the source of a recoverable success: turning
it off restores the legacy prompt handoff but breaks physical stage completion.
The next reproduction step must recover and compare the original high-score
Task4 manifest and its complete rollout state alignment, rather than relaxing
another stage predicate.

## Evidence

- Summary: `outputs/task4/task4_vlm35999_latestd9_20ep_seed108_20260724_025838/summary.tsv`
- Main video: `outputs/task4/task4_vlm35999_latestd9_20ep_seed108_20260724_025838/videos/task4/task4_failure_ep0.mp4`
- Wrist video: `outputs/task4/task4_vlm35999_latestd9_20ep_seed108_20260724_025838/videos/task4/task4_failure_ep0_wrist.mp4`
- Trace: `outputs/task4/task4_vlm35999_latestd9_20ep_seed108_20260724_025838/task4/ep0/sync_vlm.log`
