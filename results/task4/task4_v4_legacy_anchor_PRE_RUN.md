# Task4 v4 Historical Release-Anchor Replay

## Evidence From The Historical Run

The historical seed108 goal-success trace used three release anchors. At
`t=124`, after VLM output `open middle drawer`, it applied
`legacy::close the top drawer->open middle drawer` from
`open_middle_drawer_2_seed104_task4.hdf5`, frame 20. The same run recorded two
additional close-to-open release anchors for middle and bottom drawers.

## v4 Scope

Enable only the three recorded `TASK4_*_TELEPORT` release-anchor flags.
The VLM still generates each next prompt. Hold/release identifies timing; the
anchor restores the robot joint/gripper state to the historical training
trajectory boundary. VLA/VLM, norm, targets, passages, replan, strict drawer
guards, seed108 and the latest official scorer remain unchanged.

## Acceptance

The trace must show all enabled release-anchor rules and the latest official
score. This is an exact historical rollout-state compatibility test, not an
oracle next-prompt test.
