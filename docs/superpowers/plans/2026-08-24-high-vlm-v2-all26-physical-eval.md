# High-VLM-V2 All-26 Physical Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one valid autonomous physical episode for each RoboMemArena Task1-26 with high-vlm-v2, VLA35999, two-call prompt confirmation, and upstream commit `cc156e519990ae43cf3b64281a548724f428fbbd` scoring.

**Architecture:** Vendor a minimal append-only upstream scoring snapshot and compose two experiment-owned adapters around the immutable high-vlm-v2 runner: all-task physical registration followed by two-call controller confirmation. Validate on CPU and with a 300-step physical smoke before filling borrowed-account quota with independent two-GPU jobs.

**Tech Stack:** Python 3.10, pytest, Bash, Git, Slurm `srun`, tmux, Qwen3-VL, OpenPI websocket policy, LIBERO/MuJoCo.

---

### Task 1: Freeze The Current Upstream Scoring Contract

**Files:**
- Create: `high_vlm_v2_task1_vla35999_nohold/official_snapshot_cc156e5/UPSTREAM.md`
- Create: `high_vlm_v2_task1_vla35999_nohold/official_snapshot_cc156e5/SHA256SUMS`
- Create: `high_vlm_v2_task1_vla35999_nohold/official_snapshot_cc156e5/evaluation_benchmark/`

- [ ] Vendor the 26 top-level task BDDL files plus latest scorer, shared pour
  counter, and runtime `eval_common.py` from upstream commit `cc156e5`.
- [ ] Record exact URL, commit, file list, and SHA-256 values.
- [ ] Verify there is exactly one BDDL for every task ID and that no historical
  snapshot changed.
- [ ] Commit and push the append-only snapshot.

### Task 2: Test And Implement All-Task Physical Registration

**Files:**
- Create: `high_vlm_v2_task1_vla35999_nohold/extensions/eval_all26_physical_two_call.py`
- Create: `high_vlm_v2_task1_vla35999_nohold/tests/test_all26_physical_extension.py`

- [ ] Write failing tests for task IDs 1-26, official required-stage names,
  optional final-stage removal, scorer/dependency hashes, and BDDL uniqueness.
- [ ] Write a composition test proving raw VLM prompts still pass through the
  existing two-call controller and no stage predicate generates a prompt.
- [ ] Implement fail-closed scorer loading, task registration, generic physical
  snapshots, BDDL redirection, and two-call adapter composition.
- [ ] Run the new tests plus existing Task234/two-call suites and require PASS.
- [ ] Run source-manifest and no-hold/no-oracle audits and require PASS.
- [ ] Commit and push implementation plus test evidence.

### Task 3: Add Reproducible One-Task And Batch Launchers

**Files:**
- Create: `high_vlm_v2_task1_vla35999_nohold/scripts/run_all26_one_ep_inside_allocation.sh`
- Create: `high_vlm_v2_task1_vla35999_nohold/scripts/launch_all26_one_ep_account.sh`
- Create: `high_vlm_v2_task1_vla35999_nohold/scripts/schedule_all26_one_ep.py`

- [ ] Parameterize one frozen two-GPU run by task, seed, port, output root,
  submitting account, partition, and excluded nodes.
- [ ] Snapshot and hash the adapter, runner, scorer, BDDL, and probe scripts in
  every request; record checkpoint-local norm hash and GPU UUIDs.
- [ ] Make the scheduler count only summary rows as valid, preserve native abort
  attempts, retry them on a different node, and never duplicate a valid task.
- [ ] Run shell syntax checks and CPU scheduler tests.
- [ ] Commit and push launch infrastructure before requesting GPUs.

### Task 4: Pass The Physical Smoke Gate

**Files:**
- Create at runtime: `high_vlm_v2_task1_vla35999_nohold/records/all26_<run_id>/`
- Create at runtime: `high_vlm_v2_task1_vla35999_nohold/runs/all26_<run_id>/`

- [ ] From `hzhang061`, verify account identity and required read/write paths.
- [ ] Complete fresh one-GPU and two-GPU shape probes.
- [ ] Run one Task1 autonomous episode with `MAX_STEPS=300`; require either a
  valid terminal summary or clean progress beyond state step 300.
- [ ] On native abort, retain evidence and retry on another node rather than
  counting a failure.
- [ ] Record and push the smoke result receipt.

### Task 5: Fill Available GPU Capacity

**Files:**
- Modify: `.vlavlm_codex_outbox.md`
- Modify: `high_vlm_v2_task1_vla35999_nohold/EXPERIMENT_LEDGER.md`

- [ ] Inspect each authorized account from its own shell without touching
  existing jobs.
- [ ] For every account used, pass its own fresh one-GPU and two-GPU shape
  probes and record logs.
- [ ] Launch independent Task1-26 episodes, at two GPUs per task, up to verified
  free quota; dispatch the next missing task as slots become available.
- [ ] Poll account-private queues and run directories; report each valid episode
  as soon as it closes.
- [ ] Retry invalid native aborts on a different node until each task has one
  valid summary or resource access is verified blocked.

### Task 6: Aggregate And Close The 26-Task Batch

**Files:**
- Create: `high_vlm_v2_task1_vla35999_nohold/records/results/all26_<run_id>.md`
- Modify: `.vlavlm_codex_outbox.md`
- Modify: `high_vlm_v2_task1_vla35999_nohold/EXPERIMENT_LEDGER.md`

- [ ] Produce a 26-row table containing task, seed, success, official stages,
  episode steps, job, node, manifest, logs, and video paths.
- [ ] Separate valid policy failures from invalid native/runtime attempts.
- [ ] Hash all summaries/manifests and record producer commits without adding
  generated videos or model files to Git.
- [ ] Commit and push the final receipt, ledger, and code status.
