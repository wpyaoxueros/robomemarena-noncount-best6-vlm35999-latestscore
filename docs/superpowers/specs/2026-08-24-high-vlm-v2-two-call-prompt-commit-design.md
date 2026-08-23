# High-VLM-V2 Two-Call Prompt Commit Design

## Goal

Evaluate Task1 with the existing high-vlm-v2 model and VLA checkpoint `35999`,
but require two consecutive VLM calls to predict the same new primitive before
that primitive is sent to the VLA controller.

This is a controlled A/B change against the existing no-hold Task1 run. It
does not add stage gates, ordering rules, hold/release behavior, anchors,
oracle prompts, or GT actions.

## Frozen Baseline

- Keep the byte-identical `jwen341/openpi_rm` evaluator snapshot unchanged.
- Keep the same VLM base, adapter, processor, dual-camera input, keyframe
  memory, five-step VLM interval, five recent frames, and eight-keyframe cap.
- Keep VLA checkpoint `35999`, its checkpoint-local norm statistics, five-step
  replanning, Task1 seed `104`, and the physical stage tracker.
- Keep the current no-hold/no-oracle audit contract.

## Controller State Machine

The controller begins with the planner's default first primitive. Each VLM
result retains its raw parsed primitive for metrics and trace analysis.

For control only:

1. A valid prediction equal to the active controller prompt clears any pending
   candidate.
2. A valid prediction different from the active prompt starts a candidate at
   count one.
3. A second consecutive prediction of the same candidate commits it as the
   active controller prompt.
4. A different candidate replaces the pending candidate and resets its count
   to one.
5. An inference error or empty prediction clears the pending candidate and
   leaves the active prompt unchanged.

With the frozen five-step VLM interval, a new primitive therefore needs two
matching observations separated by five environment steps before it controls
the VLA.

## Isolation

Implement the state machine in an experiment-owned evaluator adapter. The
adapter wraps the immutable evaluator's recording planner and changes only the
prompt returned to the existing action loop. It must not change raw model
generation, parsed primitive metrics, keyframe extraction, visual-memory
updates, image preprocessing, VLA actions, or physical scoring.

Each VLM trace records:

- raw model/controller candidate prompt;
- pending candidate and count;
- required confirmation count;
- whether a commit occurred on that call;
- prompt actually sent to the VLA.

## Validation

1. CPU unit tests cover first candidate, repeated candidate commit, candidate
   replacement, regression to active prompt, empty/error output, and episode
   reset.
2. Existing source-manifest and no-hold audits must continue to pass.
3. Run one Task1 seed104 episode with the same baseline configuration and only
   the two-call adapter enabled.
4. Compare physical stages, raw VLM predictions, committed controller prompts,
   summary, and main/wrist videos against the existing `1/4` no-hold baseline.

## Evidence And Git

Record the implementation commit, frozen runner hash, actual Unix user,
Slurm account/job/node/GPU list, checkpoint and norm hashes, exact command,
summary, prompt trace, and video paths. Generated videos and logs remain
outside Git; receipts and code are committed.
