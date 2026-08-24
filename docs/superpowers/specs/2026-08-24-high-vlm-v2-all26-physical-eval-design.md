# High-VLM-V2 All-26 Physical Evaluation Design

## Goal

Run one autonomous physical episode for every RoboMemArena Task1-26 with the
same high-vlm-v2 planner and VLA checkpoint 35999, while scoring simulator state
with the current upstream RoboMemArena `main` commit
`cc156e519990ae43cf3b64281a548724f428fbbd`.

## Frozen Control Contract

- VLM base, adapter, processor, dual-camera prompt, and visual memory match the
  existing high-vlm-v2 Task1 package.
- VLA uses checkpoint `35999` and its checkpoint-local norm statistics.
- The initial VLM primitive is active immediately. A later primitive reaches
  the VLA only after two consecutive fresh VLM calls predict the same value.
- VLM supplies prompts and VLA supplies every action. There is no hold,
  release, pose/object anchor, stage-driven prompt, oracle prompt, or GT replay.
- Each task uses seed104, one episode, replan 5, VLM interval 5, and at most
  2500 environment steps.

## Official Evaluation Boundary

A new append-only snapshot directory vendors the exact upstream files needed
for physical scoring: all 26 task BDDL files, `task2_26_reference_stage.py`,
`shared_pour_counter.py`, and `eval_common.py`. Its manifest records upstream
URL, commit, and SHA-256 values. Existing historical snapshots are not edited.

An experiment-owned adapter imports the immutable high-vlm-v2 runner, registers
official Task1-26 stage predicates, points BDDL resolution at the new snapshot,
and then installs the already-tested two-call controller adapter. Missing files,
an unexpected scorer hash, absent stage specifications, or ambiguous BDDL
resolution fail before model loading. The adapter does not replace prompt
construction, image preprocessing, VLA inference, action execution, or video
writing.

Official optional final stages remain optional exactly as declared upstream.
Counting tasks use the new persistent `SharedPourCounter`; the old tilt-cycle
fallback is not allowed.

## Validation And Expansion Gate

CPU tests cover all task IDs, stage registration, optional-stage removal,
latest scorer dependency loading, two-call composition, and fail-closed hashes.
Static audits verify the copied source remains byte-identical and no forbidden
control path was introduced.

Before broad launch, one autonomous physical smoke must cross 300 steps or end
with a valid summary. A native render abort without a summary is invalid and is
retried on another node. Once the smoke passes, available borrowed-account GPU
quota is filled with independent two-GPU episodes, without touching jobs already
owned by those accounts.

## Scheduling And Evidence

`hzhang061`, `xiangqim`, and `prtroas0003` are eligible only after `whoami`, a
fresh one-GPU probe, and a fresh two-GPU formal-shape probe in that account's
own shell. Every run has a unique tmux session, Slurm job, port, output path,
frozen scripts, manifest, scorer/BDDL hashes, summary, prompt trace, logs, and
videos. Generated artifacts remain outside Git; receipts, ledgers, code, and
hash manifests are committed and pushed.

Completion means 26 valid episode summaries, one per task. Invalid native
aborts remain auditable attempts and are replaced until a valid episode exists
or available GPU access becomes genuinely blocked.
