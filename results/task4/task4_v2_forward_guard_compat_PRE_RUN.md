# Task4 v2 Forward-Guard Compatibility Replay

## Hypothesis

The historical seed108 success trace released `close the top drawer` to
`open middle drawer` at t=124 while the close-stage predicate was false. The
current replay instead blocks that VLM-selected forward handoff from t=212
onward, leaving the policy at close-top until timeout.

## Single Change

Set `TASK4_DRAWER_FORWARD_ADVANCE_GUARD=0` for Task4 only. The VLM and VLA
checkpoints, norm, Task4 targets, passage file, tolerance override, replan=10,
seed108, original rollout evaluator and latest official scorer are unchanged.

## Acceptance

This is a one-episode diagnostic. The log must show a close-top release to the
VLM-proposed `open middle drawer` rather than a forward-switch block. The
latest official score remains the only reported score.
