# Task4 v2 Attempt 1: Infrastructure Abort

- Version commit: `41a9fb6`
- Seed: 108
- Slurm job: `433833` on `ACD1-1`
- Outcome: cancelled after 6m40s without a single `VLM @t=0` entry, action
  chunk, stage result or episode summary.

This attempt is excluded from Task4 results. The logged contract confirmed
`DRAWER_FORWARD_ADVANCE_GUARD=0`; no policy behavior was observed, so it does
not test the forward-guard hypothesis. The same v2 code and inputs must be
rerun on a different node.
