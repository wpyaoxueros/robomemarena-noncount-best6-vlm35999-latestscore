# High-VLM-V2 Task2/3/4 Physical Evaluation Design

## Goal

Run Task2, Task3, and Task4 with the same high-vlm-v2 prompt-to-VLA control
path already frozen for Task1, while scoring real simulator state with the
official RoboMemArena snapshot at commit
`d9f83ac5182e25ad7f0a301a77a0b667f2392df1`.

## Frozen Control Contract

- VLM base and adapter are identical to the Task1 no-hold runs.
- VLA is checkpoint `35999` with its checkpoint-local norm statistics.
- The high-vlm-v2 planner uses the exact training-time dual-camera prompt.
- Each parsed VLM primitive is sent directly to the VLA websocket server.
- There is no hold, release, pose/object anchor, oracle prompt, stage-driven
  prompt, or GT action replay.
- The copied `source/` tree remains byte-identical to the jwen341 source.

## Architecture

An experiment-owned Python entrypoint imports the immutable
`source/vlm_ft/eval_three_tasks.py` module, then registers Task2/3/4 physical
scoring at module boundaries before calling its existing `main()` function.
The adapter does not replace the planner, prompt builder, VLA client, action
loop, image preprocessing, or video writer.

The adapter loads BDDL files and stage predicates only from the tracked
`official_snapshot/` tree. Missing BDDL files, scorer files, or a mismatched
official scorer SHA-256 abort before model loading.

Task2 and Task3 use the two official placement stages. Task4 uses the first
eight ordered official stages; `09_Close_Top_Drawer_Final` remains recorded as
the official optional final stage and is not required for success. Physical
handoff-window metrics remain explicitly unavailable for these three adapter
tasks because the immutable high-vlm-v2 source defines VHWS only for Tasks
1/17/20/22. This does not affect physical stage success.

## Files

- `high_vlm_v2_task1_vla35999_nohold/extensions/eval_task234_physical.py`:
  external adapter and fail-closed official snapshot validation.
- `high_vlm_v2_task1_vla35999_nohold/tests/test_task234_physical_extension.py`:
  CPU unit tests with fake official stage objects.
- `high_vlm_v2_task1_vla35999_nohold/scripts/run_task234_seed104_nohold_inside_allocation.sh`:
  one-task/two-GPU formal runner.
- `high_vlm_v2_task1_vla35999_nohold/scripts/launch_hzhang061_task234_seed104.sh`:
  fresh probes and formal Slurm request.

## Execution And Evidence

Task2, Task3, and Task4 first run as independent `seed104` one-episode jobs.
Each job receives a unique run ID, port, probe logs, frozen scripts, manifest,
summary, VLM trace, main/wrist video, and artifact hashes. The launch request
and subsequent result are appended to the experiment ledger and Codex outbox,
then committed and pushed without adding generated videos or checkpoints.
