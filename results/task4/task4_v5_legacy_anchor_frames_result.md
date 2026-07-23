# Task4 v5 Historical Anchor-Frame Replay Result

## Result

The corrected historical anchor frames `20/20/50` were applied. The trace
confirmed the first release anchor at frame 20, but latest official scoring
still produced `CSR=12.5` and `TSR=0.0`: only `01_Open_Top_Drawer` was true.

This confirms that restoring the missing historical release anchors alone does
not reproduce a latest-stage success. The Task4 v5 evidence is retained, but
Task4 iteration is paused while the counting-task line is prioritized.

## Evidence

- Summary: `outputs/task4/task4_vlm35999_latestd9_20ep_seed108_20260724_040410/summary.tsv`
- Main video: `outputs/task4/task4_vlm35999_latestd9_20ep_seed108_20260724_040410/videos/task4/task4_failure_ep0.mp4`
- Wrist video: `outputs/task4/task4_vlm35999_latestd9_20ep_seed108_20260724_040410/videos/task4/task4_failure_ep0_wrist.mp4`
- Trace: `outputs/task4/task4_vlm35999_latestd9_20ep_seed108_20260724_040410/task4/ep0/sync_vlm.log`
