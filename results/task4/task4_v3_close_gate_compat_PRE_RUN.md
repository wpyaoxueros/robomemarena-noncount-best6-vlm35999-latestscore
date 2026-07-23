# Task4 v3 Close-Gate Compatibility Replay

## Hypothesis

In v2, disabling only the forward-switch guard allowed the VLM to request
`open middle drawer`, but the rollout still refused to hold/release
`close the top drawer` until the close-stage predicate was true. Historical
Task4 behavior advanced after the close-top EEF target was reached even when
that strict predicate was not yet true.

## Single Additional Change

Set `DRAWER_CLOSE_REQUIRE_STAGE=0` for Task4. v3 retains v2's
`TASK4_DRAWER_FORWARD_ADVANCE_GUARD=0`. No other rollout or scorer setting
changes: VLM/VLA, norm, targets, passages, replan, seed and latest official
scoring remain fixed.

## Acceptance

The one-episode trace must show a close-top hold/release driven by EEF/passage
criteria without requiring close-stage success. The latest official result is
the sole score reported.
