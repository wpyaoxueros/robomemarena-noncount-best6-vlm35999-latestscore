# High-VLM-V2 Two-Call Prompt Commit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auditable two-consecutive-call controller prompt filter and run one otherwise unchanged Task1 seed104 VLM-to-VLA35999 episode.

**Architecture:** An experiment-owned adapter wraps the immutable evaluator's recording planner. Raw VLM predictions continue into metrics and keyframe memory unchanged, while only the prompt returned to the existing VLA action loop is filtered. Existing launch scripts gain opt-in environment overrides whose defaults preserve prior behavior.

**Tech Stack:** Python 3.10, pytest, Bash, Qwen3-VL high-vlm-v2, OpenPI websocket policy, LIBERO/MuJoCo, Slurm, tmux.

---

### Task 1: Add the prompt confirmation state machine tests

**Files:**
- Create: `high_vlm_v2_task1_vla35999_nohold/tests/test_two_call_prompt_commit.py`

- [ ] Write tests that initialize the active prompt to `pick cookies` and verify:
  - first `place cookies into basket` prediction remains `pick cookies` with candidate count one;
  - the second consecutive prediction commits `place cookies into basket`;
  - a different candidate replaces the old candidate at count one;
  - returning to the active prompt clears the candidate;
  - empty/error output clears the candidate without changing the active prompt;
  - episode reset restores the default active prompt and clears all candidate state.
- [ ] Run `pytest -q high_vlm_v2_task1_vla35999_nohold/tests/test_two_call_prompt_commit.py` and require an import failure before implementation.

### Task 2: Implement the isolated evaluator adapter

**Files:**
- Create: `high_vlm_v2_task1_vla35999_nohold/extensions/eval_two_call_prompt_commit.py`

- [ ] Implement `PromptCommitDecision` and `TwoCallPromptCommitter` with the state transitions frozen in the design.
- [ ] Implement `install_extension(runner, required_calls=2)` by wrapping `runner._make_recording_planner_class`.
- [ ] In the wrapped planner, reset controller state on every `reset_episode`, call the original `infer_record`, preserve `parsed_primitive`, and replace only `VLMResult.prompt` with the active committed prompt.
- [ ] Add these fields to each result event:

```python
{
    "raw_controller_prompt": raw_prompt,
    "controller_candidate_prompt": decision.candidate_prompt,
    "controller_candidate_count": decision.candidate_count,
    "controller_required_calls": required_calls,
    "controller_prompt_committed": decision.committed,
    "accepted_controller_prompt": decision.active_prompt,
}
```

- [ ] In `main()`, fail unless `CONTROLLER_STABLE_VLM_CALLS` is a positive integer, import the immutable source evaluator, install the adapter, and call `runner.main()`.
- [ ] Run the new tests and require all cases to pass.

### Task 3: Add opt-in runner and launcher wiring

**Files:**
- Modify: `high_vlm_v2_task1_vla35999_nohold/scripts/run_task1_seed104_nohold_inside_allocation.sh`
- Modify: `high_vlm_v2_task1_vla35999_nohold/scripts/launch_hzhang061_task1_seed104.sh`
- Create: `high_vlm_v2_task1_vla35999_nohold/scripts/launch_hzhang061_task1_seed104_two_call.sh`

- [ ] Make `EVALUATOR`, `CONTROL_CONTRACT`, `CONTROLLER_STABLE_VLM_CALLS`, and `AUDIT_EXTENSION` opt-in environment variables; retain the existing evaluator and one-call behavior as defaults.
- [ ] Record the selected evaluator, stable-call count, and control contract in `run_manifest.json` and `formal_command.txt`.
- [ ] Pass the selected values through both fresh probe gates and the formal allocation without changing the VLA checkpoint, norm, seed, VLM cadence, or GPU shape.
- [ ] Add a small two-call launcher that sets:

```bash
EVALUATOR="${EXP_ROOT}/extensions/eval_two_call_prompt_commit.py"
CONTROLLER_STABLE_VLM_CALLS=2
CONTROL_CONTRACT="two consecutive VLM prompts -> VLA actions; no hold/release/anchor/oracle/GT replay"
AUDIT_EXTENSION="${EVALUATOR}"
```

- [ ] Run `bash -n` on all modified/new scripts.
- [ ] Run the no-hold audit with the adapter included and require `PASS`.

### Task 4: Validate and commit the implementation

**Files:**
- Modify: `.vlavlm_codex_outbox.md`
- Modify: `high_vlm_v2_task1_vla35999_nohold/EXPERIMENT_LEDGER.md`

- [ ] Run all extension tests and require all tests to pass.
- [ ] Run `scripts/verify_source_snapshot.sh` and confirm the 18 immutable source/runtime files remain byte-identical.
- [ ] Append implementation, tests, hashes, and controlled-difference status to the outbox and ledger.
- [ ] Commit only the adapter, tests, scripts, plan, outbox, and ledger; leave the pre-existing `LIVE_STATUS.tsv` change untouched.

### Task 5: Launch and evaluate one controlled Task1 episode

**Files:**
- Create generated receipt under: `high_vlm_v2_task1_vla35999_nohold/records/`
- Create generated run under: `high_vlm_v2_task1_vla35999_nohold/runs/`

- [ ] From the `hzhang061` shell, verify `whoami` and writable/readable paths.
- [ ] Run a fresh one-GPU probe and a fresh two-GPU request-shape probe; retain their job IDs and logs.
- [ ] Launch one Task1 seed104 episode through the two-call adapter inside tmux/Slurm.
- [ ] Monitor through a valid summary or explicit failure; do not count native aborts or header-only summaries.
- [ ] Verify every controller switch has two immediately preceding equal raw VLM prompts in `vlm_predictions.jsonl`.
- [ ] Record physical stage result, raw/committed prompt trace, main/wrist video, manifest and SHA-256 evidence in the outbox and ledger, then commit and push the receipt without generated videos.
