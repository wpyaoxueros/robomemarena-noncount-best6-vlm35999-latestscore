# High-VLM-V2 Task2/3/4 Physical Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the frozen Task1 high-vlm-v2 evaluation package to run autonomous physical Task2/3/4 seed104 episodes without modifying copied source.

**Architecture:** Add one experiment-owned adapter that registers official physical stages and BDDL resolution around the immutable high-vlm-v2 runner. Add task-parameterized launchers that preserve the existing two-GPU VLM/VLA topology, fresh Slurm probes, manifests, and no-hold audit.

**Tech Stack:** Python 3.10, pytest, Bash, Slurm `srun`, tmux, Qwen3-VL, OpenPI websocket policy, LIBERO/MuJoCo.

---

### Task 1: Lock The Adapter Contract With Tests

**Files:**
- Create: `high_vlm_v2_task1_vla35999_nohold/tests/test_task234_physical_extension.py`
- Create: `high_vlm_v2_task1_vla35999_nohold/extensions/eval_task234_physical.py`

- [ ] Write tests asserting Task2 stage names are
  `01_Place_Butter_Basket` and `02_Place_Popcorn_Basket`.
- [ ] Write tests asserting Task3 stage names are
  `01_Place_Cream_Basket` and `02_Place_Pudding_Basket`.
- [ ] Write tests asserting Task4 requires stages 01 through 08 and declares
  `09_Close_Top_Drawer_Final` optional.
- [ ] Write a test that an official scorer SHA mismatch raises before the
  evaluator `main()` call.
- [ ] Run:
  `PYTHONPATH=high_vlm_v2_task1_vla35999_nohold/source/vlm_ft pytest -q high_vlm_v2_task1_vla35999_nohold/tests/test_task234_physical_extension.py`
  and verify the initial test fails because the adapter functions are absent.
- [ ] Implement the minimal adapter, preserving the imported planner and action
  loop and disabling only unsupported VHWS generation for Task2/3/4.
- [ ] Run the same pytest command and require all tests to pass.

### Task 2: Add Reproducible Formal Launchers

**Files:**
- Create: `high_vlm_v2_task1_vla35999_nohold/scripts/run_task234_seed104_nohold_inside_allocation.sh`
- Create: `high_vlm_v2_task1_vla35999_nohold/scripts/launch_hzhang061_task234_seed104.sh`
- Modify: `high_vlm_v2_task1_vla35999_nohold/scripts/audit_nohold_contract.sh`

- [ ] Parameterize the formal runner with `TASK_ID` restricted to `2`, `3`, or
  `4`, while keeping seed104, max steps 2500, replan 5, VLM interval 5, and
  synchronous direct prompt-to-VLA control.
- [ ] Require exactly two visible GPUs, checkpoint-local norm, copied adapter,
  frozen official scorer SHA, and task BDDL before launching models.
- [ ] Record task, seed, code hashes, official commit/hash, BDDL hash, Unix user,
  Slurm account, partition, job, node, GPU list, model paths, and control
  contract in `run_manifest.json`.
- [ ] Parameterize the outer launcher with `TASK_ID`; freeze the probe and
  formal scripts into the run record before requesting Slurm resources.
- [ ] Run `bash -n` on both launchers and run the no-hold/source snapshot audits.

### Task 3: CPU Integration Preflight

**Files:**
- Modify: `high_vlm_v2_task1_vla35999_nohold/EXPERIMENT_LEDGER.md`
- Modify: `.vlavlm_codex_outbox.md`

- [ ] Execute adapter `--help` in the OpenPI inference environment.
- [ ] Import the official scorer through the adapter and resolve Task2/3/4 BDDL
  files from `official_snapshot/evaluation_benchmark/bddl`.
- [ ] Print and compare registered stage names against the tracked official
  snapshot.
- [ ] Re-run `scripts/verify_source_snapshot.sh --against-source` and require
  byte-for-byte equality.
- [ ] Append the preflight result and evidence paths to the ledger and outbox.
- [ ] Commit and push only source, tests, launchers, docs, and lightweight
  records; leave `LIVE_STATUS.tsv` untouched.

### Task 4: Launch Three Independent Seed104 Episodes

**Files:**
- Create at runtime: `high_vlm_v2_task1_vla35999_nohold/records/probes/<run_id>/`
- Create at runtime: `high_vlm_v2_task1_vla35999_nohold/runs/<run_id>/`

- [ ] In the `hzhang061` shell, verify `whoami` and writable shared output
  directories.
- [ ] Start three unique tmux sessions for Task2, Task3, and Task4.
- [ ] For each task, complete a fresh one-GPU probe and fresh two-GPU
  request-shape probe before the formal request.
- [ ] Verify each formal job is visible in `hzhang061`'s own `squeue --me` and
  that both VLA and VLM processes occupy their assigned GPU.
- [ ] Check each run until its first VLM call and physical state row are written;
  report run ID, job ID, node, and exact evidence path.

### Task 5: Close Results Audibly

**Files:**
- Modify: `high_vlm_v2_task1_vla35999_nohold/EXPERIMENT_LEDGER.md`
- Modify: `.vlavlm_codex_outbox.md`
- Create: `high_vlm_v2_task1_vla35999_nohold/records/results/<run_id>.md`

- [ ] Count an episode only when `episode_summary.json` exists and the manifest
  is closed; an abort without summary is invalid, not a policy failure.
- [ ] Record summary, logs, main/wrist videos, VLM trace, manifest, and SHA-256
  values in the result receipt.
- [ ] Commit and push the result receipt and ledger update without generated
  artifacts.
- [ ] Report Task1 five-repeat final status alongside Task2/3/4 results without
  mixing invalid attempts into success rates.
