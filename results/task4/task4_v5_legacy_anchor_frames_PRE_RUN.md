# Task4 v5 Exact Historical Anchor-Frame Replay

v5 corrects the invalid v4 defaults. It enables the three historical release
anchors with exactly the frames recorded in the old seed108 trace:

- close top -> open middle: frame 20
- close middle -> open bottom: frame 20
- close bottom -> open top again: frame 50

All policy, VLM, norm, hold, target, passage, replan and latest official
scoring settings remain the same as the historical-compatible v4 replay.
